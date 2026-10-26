#include <fve/inspector.hpp>
#include <fve/output.hpp>
#include <fve/pipeline.hpp>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <thread>
#include <tuple>

namespace fs = std::filesystem;

namespace fve
{
namespace
{

// ── Frame sources ─────────────────────────────────────────────────────────

bool isImagePath(const fs::path &p)
{
    auto e = p.extension().string();
    std::transform(e.begin(), e.end(), e.begin(), ::tolower);
    return e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".bmp" || e == ".tif" || e == ".tiff" || e == ".webp";
}

/// Video file, camera index / stream URL, single image, or image directory.
class FrameSource
{
public:
    explicit FrameSource(const std::string &input) : name_(input)
    {
        const fs::path p(input);
        if (fs::is_directory(p))
        {
            for (const auto &e : fs::directory_iterator(p))
                if (e.is_regular_file() && isImagePath(e.path()))
                    images_.push_back(e.path().string());
            std::sort(images_.begin(), images_.end());
            if (images_.empty())
                throw std::runtime_error("no images found in directory: " + input);
            probeImage(images_.front());
            return;
        }
        if (fs::is_regular_file(p) && isImagePath(p))
        {
            images_.push_back(input);
            probeImage(input);
            return;
        }
        const bool is_index = !input.empty() && std::all_of(input.begin(), input.end(), ::isdigit);
        if (!is_index && !fs::exists(p) && input.find("://") == std::string::npos)
            throw std::runtime_error("input not found: " + input);
        if (is_index)
        {
            cap_.open(std::stoi(input));
            live_ = true;
        }
        else
        {
            cap_.open(input);
            live_ = input.find("://") != std::string::npos;
        }
        if (!cap_.isOpened())
            throw std::runtime_error("could not open input: " + input);
        fps_ = cap_.get(cv::CAP_PROP_FPS);
        if (!(fps_ > 0.0 && fps_ < 1000.0))
            fps_ = 25.0;
        frame_count_ = live_ ? -1 : static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
        size_ = {static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT))};
    }

    bool read(cv::Mat &out)
    {
        if (!images_.empty())
        {
            while (next_ < images_.size())
            {
                out = cv::imread(images_[next_++], cv::IMREAD_COLOR);
                if (!out.empty())
                    return true;
                std::cerr << "warning: unreadable image skipped: " << images_[next_ - 1] << '\n';
            }
            return false;
        }
        if (!cap_.read(out) || out.empty())
            return false;
        if (out.type() == CV_8UC1)
            cv::cvtColor(out, out, cv::COLOR_GRAY2BGR);
        else if (out.type() == CV_8UC4)
            cv::cvtColor(out, out, cv::COLOR_BGRA2BGR);
        return true;
    }

    bool isImages() const noexcept { return !images_.empty(); }
    bool isLive() const noexcept { return live_; }
    double fps() const noexcept { return fps_; }
    int frameCount() const noexcept { return frame_count_; }
    cv::Size size() const noexcept { return size_; }
    const std::string &name() const noexcept { return name_; }

private:
    void probeImage(const std::string &path)
    {
        const cv::Mat m = cv::imread(path, cv::IMREAD_COLOR);
        if (m.empty())
            throw std::runtime_error("could not read image: " + path);
        size_ = m.size();
        fps_ = 1.0;
        frame_count_ = static_cast<int>(images_.size());
    }

    std::string name_;
    cv::VideoCapture cap_;
    std::vector<std::string> images_;
    std::size_t next_{0};
    bool live_{false};
    double fps_{25.0};
    int frame_count_{-1};
    cv::Size size_;
};

// ── Concurrency helpers ───────────────────────────────────────────────────

template <typename T> class BoundedQueue
{
public:
    explicit BoundedQueue(std::size_t cap) : cap_(cap) {}
    void push(T v)
    {
        std::unique_lock lk(m_);
        not_full_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
        q_.push_back(std::move(v));
        not_empty_.notify_one();
    }
    std::optional<T> pop()
    {
        std::unique_lock lk(m_);
        not_empty_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty())
            return std::nullopt;
        T v = std::move(q_.front());
        q_.pop_front();
        not_full_.notify_one();
        return v;
    }
    void close()
    {
        std::lock_guard lk(m_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::size_t cap_;
    std::deque<T> q_;
    bool closed_{false};
    std::mutex m_;
    std::condition_variable not_empty_, not_full_;
};

/// Runs closures on background threads (1 thread = strictly ordered).
class TaskRunner
{
public:
    explicit TaskRunner(int threads, std::size_t cap = 64) : q_(cap)
    {
        for (int i = 0; i < threads; ++i)
            th_.emplace_back([this] {
                while (auto t = q_.pop())
                {
                    try
                    {
                        (*t)();
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << "background task failed: " << e.what() << '\n';
                        failed_ = true;
                    }
                }
            });
    }
    ~TaskRunner() { finish(); }
    void submit(std::function<void()> f) { q_.push(std::move(f)); }
    void finish()
    {
        q_.close();
        for (auto &t : th_)
            if (t.joinable())
                t.join();
        th_.clear();
    }
    bool failed() const noexcept { return failed_; }

private:
    BoundedQueue<std::function<void()>> q_;
    std::vector<std::thread> th_;
    std::atomic<bool> failed_{false};
};

struct Job
{
    int index{0};
    cv::Mat frame;
    bool gated{false};
};

struct Done
{
    int index{0};
    cv::Mat frame;
    bool computed{false};
    FrameResult result;
    cv::Mat annotated;
    cv::Mat heatmap;
    std::string error;
};

cv::Mat thumbnail(const cv::Mat &bgr)
{
    cv::Mat g, t;
    cv::cvtColor(bgr, g, cv::COLOR_BGR2GRAY);
    const double s = 320.0 / std::max(1, g.cols);
    cv::resize(g, t, cv::Size(), s, s, cv::INTER_AREA);
    cv::Mat out;
    cv::GaussianBlur(t, out, {3, 3}, 0);
    return out;
}

/// True when the frame differs from the anchor anywhere. Besides the global
/// mean, the difference is averaged over 8x8 blocks so one small part
/// appearing or disappearing (a few pixels of a 320 px thumbnail) is never
/// mistaken for "unchanged" — a missed change would silently reuse a stale
/// verdict.
bool thumbChanged(const cv::Mat &a, const cv::Mat &b, double mean_thr, double block_thr)
{
    if (a.size() != b.size())
        return true;
    cv::Mat d;
    cv::absdiff(a, b, d);
    if (cv::mean(d)[0] >= mean_thr)
        return true;
    cv::Mat blocks;
    cv::resize(d, blocks, cv::Size(std::max(1, d.cols / 8), std::max(1, d.rows / 8)), 0, 0, cv::INTER_AREA);
    double mx = 0.0;
    cv::minMaxLoc(blocks, nullptr, &mx);
    return mx >= block_thr;
}

void cleanImages(const fs::path &dir)
{
    if (!fs::exists(dir))
        return;
    for (const auto &e : fs::directory_iterator(dir))
        if (e.is_regular_file() && isImagePath(e.path()))
        {
            std::error_code ec;
            fs::remove(e.path(), ec);
        }
}

using Signature = std::tuple<bool, std::vector<std::string>, std::size_t>;

Signature signatureOf(const FrameResult &r)
{
    std::vector<std::string> ids;
    for (const auto &s : r.slots)
        if (s.deviation != Deviation::None)
            ids.push_back(s.slot_id + ":" + toString(s.deviation));
    std::sort(ids.begin(), ids.end());
    return {r.is_correct, ids, r.extra_boxes.size()};
}

} // namespace

int runPipeline(const EngineConfig &config, PipelineOptions o)
{
    const fs::path out_dir(o.output_dir);
    auto def = [&](std::string &p, const char *name) {
        if (p.empty())
            p = (out_dir / name).string();
    };
    def(o.csv_path, "frame_deviations.csv");
    def(o.json_path, "frame_deviations.json");
    def(o.correct_csv, "corrected_frames.csv");
    def(o.correct_json, "corrected_frames.json");
    def(o.deviated_csv, "deviated_frames.csv");
    def(o.deviated_json, "deviated_frames.json");
    def(o.render_video_path, "annotated_results.mp4");

    const fs::path corrected_dir = out_dir / "corrected", deviated_dir = out_dir / "deviated";
    const fs::path sbs_corrected = out_dir / "side_by_side" / "corrected", sbs_deviated = out_dir / "side_by_side" / "deviated";
    const fs::path heat_dir = out_dir / "heatmaps";
    fs::create_directories(corrected_dir);
    fs::create_directories(deviated_dir);
    for (const auto &d : {out_dir, corrected_dir, deviated_dir})
        cleanImages(d);
    if (o.side_by_side)
    {
        fs::create_directories(sbs_corrected);
        fs::create_directories(sbs_deviated);
        cleanImages(sbs_corrected);
        cleanImages(sbs_deviated);
    }
    if (o.heatmaps)
    {
        fs::create_directories(heat_dir);
        cleanImages(heat_dir);
    }

    FrameSource source(o.input);
    const double fps = source.fps();

    // ── Reference + calibration frames ────────────────────────────────────
    cv::Mat reference;
    std::vector<cv::Mat> good;
    int ref_index = -1; // -1 = external golden image
    std::string ref_desc;
    if (!o.reference_image.empty())
    {
        reference = cv::imread(o.reference_image, cv::IMREAD_COLOR);
        if (reference.empty())
            throw std::runtime_error("could not read reference image: " + o.reference_image);
        ref_desc = o.reference_image + " (external image)";
    }
    else
    {
        if (source.isImages() && source.frameCount() == 1)
            throw std::runtime_error("a single input image needs --reference-image");
        ref_index = std::max(0, o.reference_frame);
        FrameSource probe(o.input);
        cv::Mat f;
        int i = 0;
        while (probe.read(f))
        {
            if (i == ref_index)
            {
                reference = f.clone();
                if (source.isLive())
                    break;
            }
            else if (i > ref_index && static_cast<int>(good.size()) < config.anomaly.calibration_frames)
                good.push_back(f.clone());
            if (i >= ref_index + config.anomaly.calibration_frames)
                break;
            ++i;
        }
        if (reference.empty())
            throw std::runtime_error("reference frame " + std::to_string(ref_index) + " not found in " + o.input);
        ref_desc = "frame_" + std::to_string(ref_index) + (ref_index == 0 ? " (first frame)" : "");
    }

    const auto t_setup = std::chrono::steady_clock::now();
    Inspector inspector(config);
    inspector.setReference(reference, good);
    const auto &ref = inspector.reference();
    const std::size_t total_elements = ref.slots.size();
    const double setup_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_setup).count();

    const cv::Mat ref_annotated =
        renderAnnotations(ref.bgr, inspector.referenceResult(std::max(0, ref_index), fps), inspector.aligner(),
                          total_elements, fps);

    std::optional<PortableVideoWriter> video, sbs_video;
    const std::string sbs_path = (out_dir / "annotated_side_by_side.mp4").string();
    const bool want_video = o.render_video && !source.isImages();
    if (want_video)
    {
        video = PortableVideoWriter::open(o.render_video_path, fps, ref.size, o.codec);
        if (o.side_by_side)
            sbs_video = PortableVideoWriter::open(sbs_path, fps, {ref.size.width * 2, ref.size.height}, o.codec);
    }

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const int workers = o.workers > 0 ? o.workers : static_cast<int>(std::clamp(hw / 2u, 1u, 6u));

    if (!o.quiet)
    {
        std::cout << "\n===========================================================================\n"
                  << "FUTURE VISION ENGINE - REFERENCE INSPECTOR\n"
                  << "Input             : " << o.input << " (" << ref.size.width << "x" << ref.size.height << ", "
                  << fps << " FPS, " << source.frameCount() << " frames)\n"
                  << "Reference Baseline: " << ref_desc << " (" << total_elements << " components established)\n"
                  << "Engine            : " << inspector.describe() << "\n"
                  << "Calibration       : " << good.size() << " good frame(s) + nuisance model, setup "
                  << std::fixed << std::setprecision(0) << setup_ms << " ms\n"
                  << "Workers           : " << workers << "\n"
                  << "Output Directory  : " << o.output_dir << "\n"
                  << "===========================================================================\n\n";
    }

    // ── Threads: reader -> workers -> ordered consumer ────────────────────
    BoundedQueue<Job> jobs(static_cast<std::size_t>(workers) * 2);
    std::mutex done_m;
    std::condition_variable done_cv;
    std::map<int, Done> done;
    std::atomic<int> frames_read{0};
    std::atomic<bool> reader_finished{false}, abort{false};
    // Back-pressure: bound frames in flight so slow writers cannot grow memory.
    std::mutex flow_m;
    std::condition_variable flow_cv;
    int consumed = 0;
    const int max_in_flight = workers * 4;

    std::thread reader([&] {
        cv::Mat anchor, frame;
        int idx = 0;
        while (!abort && (o.max_frames < 0 || idx < o.max_frames) && source.read(frame))
        {
            Job j;
            j.index = idx;
            j.frame = frame.clone();
            if (o.gate_unchanged && idx != ref_index)
            {
                const cv::Mat t = thumbnail(j.frame);
                j.gated = !anchor.empty() && !thumbChanged(t, anchor, o.gate_mean_diff, o.gate_block_diff);
                if (!j.gated)
                    anchor = t;
            }
            {
                std::unique_lock lk(flow_m);
                flow_cv.wait(lk, [&] { return idx - consumed < max_in_flight || abort; });
            }
            jobs.push(std::move(j));
            ++idx;
            frames_read = idx;
        }
        reader_finished = true;
        jobs.close();
        done_cv.notify_all();
    });

    std::vector<std::thread> pool;
    for (int w = 0; w < workers; ++w)
        pool.emplace_back([&] {
            while (auto j = jobs.pop())
            {
                Done d;
                d.index = j->index;
                d.frame = std::move(j->frame);
                try
                {
                    if (!j->gated)
                    {
                        d.computed = true;
                        d.result = (j->index == ref_index) ? inspector.referenceResult(j->index, fps)
                                                           : inspector.inspect(d.frame, j->index, fps);
                        d.annotated = renderAnnotations(d.frame, d.result, inspector.aligner(), total_elements, fps);
                        if (o.heatmaps && !d.result.is_correct)
                            d.heatmap = renderHeatmap(inspector.aligner().warp(d.frame, d.result.alignment),
                                                      d.result.anomaly_map);
                    }
                }
                catch (const std::exception &e)
                {
                    d.error = e.what();
                    abort = true;
                }
                {
                    std::lock_guard lk(done_m);
                    done.emplace(d.index, std::move(d));
                }
                done_cv.notify_all();
            }
        });

    TaskRunner image_writer(2, 128);
    TaskRunner video_writer(1, 32);
    const std::vector<int> jpeg{cv::IMWRITE_JPEG_QUALITY, o.jpeg_quality};

    std::vector<FrameDeviationRecord> records;
    std::optional<Signature> last_sig;
    std::optional<FrameResult> last_result;
    int current_start = 0, current_repeat = 0, computed = 0, next = 0;
    std::string fatal;
    const auto t0 = std::chrono::steady_clock::now();

    auto finalizeRange = [&](int end_idx) {
        if (!o.deduplicate || records.empty())
            return;
        auto &r = records.back();
        r.frame_range = current_repeat > 1 ? std::to_string(current_start) + "-" + std::to_string(end_idx)
                                           : std::to_string(current_start);
        r.repeat_count = current_repeat;
        r.duration_sec = fps > 0.0 ? current_repeat / fps : 0.0;
    };

    while (true)
    {
        Done d;
        {
            std::unique_lock lk(done_m);
            done_cv.wait(lk, [&] { return done.count(next) || (reader_finished && next >= frames_read); });
            if (!done.count(next))
                break;
            d = std::move(done.at(next));
            done.erase(next);
        }
        {
            std::lock_guard lk(flow_m);
            consumed = next + 1;
        }
        flow_cv.notify_all();
        if (!d.error.empty())
        {
            fatal = d.error;
            break;
        }

        FrameResult result;
        cv::Mat annotated;
        if (d.computed || !last_result)
        {
            if (!d.computed) // gated before anything was computed: compute inline
            {
                d.result = inspector.inspect(d.frame, d.index, fps);
                d.annotated = renderAnnotations(d.frame, d.result, inspector.aligner(), total_elements, fps);
            }
            ++computed;
            result = std::move(d.result);
            annotated = std::move(d.annotated);
        }
        else
        {
            result = *last_result;
            result.frame_index = d.index;
            result.timestamp_sec = fps > 0.0 ? d.index / fps : 0.0;
            result.reused = true;
            result.is_reference = false;
            result.processing_ms = 0.0;
            annotated = renderAnnotations(d.frame, result, inspector.aligner(), total_elements, fps);
        }
        result.anomaly_map.release(); // keep memory flat; heatmap already rendered
        last_result = result;

        const cv::Mat sbs = o.side_by_side
                                ? createSideBySideView(ref_annotated, annotated, d.index, ref_index, result.is_correct)
                                : cv::Mat();
        if (video)
            video_writer.submit([&video, &sbs_video, annotated, sbs] {
                video->write(annotated);
                if (sbs_video)
                    sbs_video->write(sbs);
            });

        const auto sig = signatureOf(result);
        if (o.deduplicate && last_sig && *last_sig == sig)
        {
            ++current_repeat;
            ++next;
            continue;
        }
        finalizeRange(d.index - 1);
        current_start = d.index;
        current_repeat = 1;
        last_sig = sig;

        const std::string status = result.is_correct ? "CORRECT" : "DEVIATED";
        char name[64];
        std::snprintf(name, sizeof(name), "frame_%04d_%s.jpg", d.index, status.c_str());
        const std::string img_path = ((result.is_correct ? corrected_dir : deviated_dir) / name).string();

        auto rec = makeRecord(result, fps);
        if (o.save_images)
        {
            rec.image_path = img_path;
            image_writer.submit([img_path, annotated, jpeg] { cv::imwrite(img_path, annotated, jpeg); });
            if (o.side_by_side)
            {
                const std::string p = ((result.is_correct ? sbs_corrected : sbs_deviated) / ("sbs_" + std::string(name))).string();
                image_writer.submit([p, sbs, jpeg] { cv::imwrite(p, sbs, jpeg); });
            }
            if (!d.heatmap.empty())
            {
                const std::string p = (heat_dir / ("heat_" + std::string(name))).string();
                cv::Mat h = d.heatmap;
                image_writer.submit([p, h, jpeg] { cv::imwrite(p, h, jpeg); });
            }
        }
        records.push_back(std::move(rec));

        if (!o.quiet)
        {
            std::ostringstream detail;
            if (result.is_correct)
                detail << "All match reference";
            else
            {
                detail << result.missing_count << " missing, " << result.shifted_count << " shifted";
                if (result.shifted_count)
                    detail << " (max " << std::fixed << std::setprecision(1) << result.max_shift_px << "px)";
                detail << ", " << result.changed_count << " changed, " << result.extra_boxes.size() << " extra";
            }
            const double t = fps > 0.0 ? d.index / fps : 0.0;
            std::cout << "Frame " << std::setw(4) << std::setfill('0') << d.index << std::setfill(' ') << " (t="
                      << std::fixed << std::setprecision(2) << std::setw(6) << t << "s) "
                      << (result.is_correct ? "[CORRECT]  | " : "[DEVIATED] | ") << std::setw(2) << result.ok_count << "/"
                      << total_elements << " Correct | score " << std::setprecision(2) << result.anomaly_score << " | "
                      << detail.str() << (result.reused ? " (reused)" : "") << '\n';
        }
        ++next;
    }

    abort = true;
    flow_cv.notify_all();
    jobs.close();
    reader.join();
    for (auto &t : pool)
        t.join();
    if (!fatal.empty())
    {
        image_writer.finish();
        video_writer.finish();
        throw std::runtime_error("inspection failed: " + fatal);
    }
    finalizeRange(next - 1);
    image_writer.finish();
    video_writer.finish();
    if (video)
        video->release();
    if (sbs_video)
        sbs_video->release();

    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const int total = next;

    ReportMeta meta;
    meta.video_source = o.input;
    meta.reference_source = ref_desc;
    meta.backend = inspector.describe();
    meta.segmentation = toString(ref.segmentation);
    meta.total_elements = total_elements;
    meta.total_frames = total;
    meta.deduplicated = o.deduplicate;
    meta.elapsed_sec = elapsed;
    meta.fps_processed = elapsed > 0.0 ? total / elapsed : 0.0;

    std::vector<FrameDeviationRecord> correct, deviated;
    for (const auto &r : records)
        (r.is_correct ? correct : deviated).push_back(r);

    exportDeviationCsv(o.csv_path, records);
    exportDeviationJson(o.json_path, meta, records);
    exportDeviationCsv(o.correct_csv, correct);
    exportSubsetJson(o.correct_json, meta, correct, "corrected");
    exportDeviationCsv(o.deviated_csv, deviated);
    exportSubsetJson(o.deviated_json, meta, deviated, "deviated");

    std::vector<int> ci;
    for (const auto &r : correct)
        ci.push_back(r.frame_index);
    std::string ranges;
    for (const auto &r : computeFrameRanges(ci))
        ranges += (ranges.empty() ? "" : ", ") + r;
    std::vector<int> di;
    for (const auto &r : deviated)
        di.push_back(r.frame_index);
    std::string dranges;
    for (const auto &r : computeFrameRanges(di))
        dranges += (dranges.empty() ? "" : ", ") + r;

    const double n = std::max<std::size_t>(1, records.size());
    std::cout << std::fixed << "\n===========================================================================\n"
              << "SUMMARY OF INSPECTION RESULTS (REFERENCE: " << ref_desc << "):\n"
              << "  Total Frames Evaluated      : " << total << '\n'
              << "  Computation Executed On     : " << computed << " frames (reused " << total - computed
              << " unchanged frames)\n"
              << "  Total Unique State Frames   : " << records.size() << '\n'
              << "  CORRECTED States            : " << correct.size() << " (" << std::setprecision(1)
              << 100.0 * correct.size() / n << "%)\n"
              << "  DEVIATED States             : " << deviated.size() << " (" << 100.0 * deviated.size() / n << "%)\n"
              << "  Execution Time              : " << std::setprecision(2) << elapsed << "s (" << std::setprecision(1)
              << meta.fps_processed << " FPS)\n"
              << "---------------------------------------------------------------------------\n"
              << "  [CORRECTED] Image Folder    : " << corrected_dir.string() << "/ (" << correct.size() << " images)\n"
              << "  [CORRECTED] Frame Ranges    : " << (ranges.empty() ? "None" : ranges) << '\n'
              << "  [CORRECTED] CSV / JSON      : " << o.correct_csv << " / " << o.correct_json << '\n'
              << "  [DEVIATED]  Image Folder    : " << deviated_dir.string() << "/ (" << deviated.size() << " images)\n"
              << "  [DEVIATED]  Frame Ranges    : " << (dranges.empty() ? "None" : dranges) << '\n'
              << "  [DEVIATED]  CSV / JSON      : " << o.deviated_csv << " / " << o.deviated_json << '\n'
              << "  [ALL]       CSV / JSON      : " << o.csv_path << " / " << o.json_path << '\n';
    if (o.side_by_side)
        std::cout << "  [SIDE-BY-SIDE] Images      : " << (out_dir / "side_by_side").string() << "/\n";
    if (o.heatmaps)
        std::cout << "  [HEATMAPS]  Image Folder    : " << heat_dir.string() << "/\n";
    if (video)
        std::cout << "  [ANNOTATED VIDEO] Output    : " << o.render_video_path << '\n';
    if (sbs_video)
        std::cout << "  [SIDE-BY-SIDE VIDEO] Output : " << sbs_path << '\n';
    std::cout << "===========================================================================\n\n";

    return (image_writer.failed() || video_writer.failed()) ? 3 : 0;
}

} // namespace fve
