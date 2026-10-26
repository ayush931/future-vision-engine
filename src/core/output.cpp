#include <fve/output.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace fve
{
namespace
{
const cv::Scalar kGreen(36, 179, 36);
const cv::Scalar kRed(0, 0, 230);
const cv::Scalar kMagenta(220, 40, 220);
const cv::Scalar kOrange(0, 90, 255);
const cv::Scalar kAmber(0, 140, 255);

void badge(cv::Mat &img, cv::Point tl, const std::string &label, const cv::Scalar &bg, const cv::Scalar &fg)
{
    int baseline = 0;
    const auto ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.35, 1, &baseline);
    int y1 = std::max(0, tl.y - 14), y2 = tl.y;
    int ty = tl.y - 3;
    if (y1 == 0)
    {
        y1 = tl.y;
        y2 = tl.y + 14;
        ty = tl.y + 10;
    }
    cv::rectangle(img, {tl.x, y1}, {tl.x + ts.width + 4, y2}, bg, cv::FILLED);
    cv::putText(img, label, {tl.x + 2, ty}, cv::FONT_HERSHEY_SIMPLEX, 0.35, fg, 1, cv::LINE_AA);
}

void poly(cv::Mat &img, const std::vector<cv::Point> &p, const cv::Scalar &c, int t = 2)
{
    cv::polylines(img, std::vector<std::vector<cv::Point>>{p}, true, c, t, cv::LINE_AA);
}

cv::Point topLeft(const std::vector<cv::Point> &p)
{
    cv::Point tl = p[0];
    for (const auto &q : p)
    {
        tl.x = std::min(tl.x, q.x);
        tl.y = std::min(tl.y, q.y);
    }
    return tl;
}

void ensureParent(const std::string &path)
{
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
}

std::string joinIds(const std::vector<std::string> &ids, std::size_t max_items)
{
    std::string s;
    for (std::size_t i = 0; i < std::min(max_items, ids.size()); ++i)
        s += (i ? ", " : "") + ids[i];
    if (ids.size() > max_items)
        s += " (+" + std::to_string(ids.size() - max_items) + " more)";
    return s;
}

void writeStringArray(std::ostream &f, const std::vector<std::string> &v)
{
    f << "[";
    for (std::size_t i = 0; i < v.size(); ++i)
        f << (i ? ", " : "") << "\"" << jsonEscape(v[i]) << "\"";
    f << "]";
}

void writeFrameJson(std::ostream &f, const FrameDeviationRecord &r, bool last)
{
    f << std::fixed << std::setprecision(3);
    f << "    {\n"
      << "      \"frame_index\": " << r.frame_index << ",\n"
      << "      \"frame_range\": \"" << (r.frame_range.empty() ? std::to_string(r.frame_index) : r.frame_range) << "\",\n"
      << "      \"repeat_count\": " << r.repeat_count << ",\n"
      << "      \"timestamp_sec\": " << r.timestamp_sec << ",\n"
      << "      \"duration_sec\": " << r.duration_sec << ",\n"
      << "      \"status\": \"" << r.status << "\",\n"
      << "      \"is_correct\": " << (r.is_correct ? "true" : "false") << ",\n"
      << "      \"total_elements\": " << r.total_elements << ",\n"
      << "      \"correct_count\": " << r.correct_count << ",\n"
      << "      \"deviated_count\": " << r.deviated_count << ",\n"
      << "      \"missing_slots\": ";
    writeStringArray(f, r.missing_slots);
    f << ",\n      \"extra_slots_count\": " << r.extra_slots_count << ",\n"
      << "      \"image_path\": \"" << jsonEscape(r.image_path) << "\",\n"
      << "      \"shifted_slots\": ";
    writeStringArray(f, r.shifted_slots);
    f << ",\n      \"changed_slots\": ";
    writeStringArray(f, r.changed_slots);
    f << ",\n      \"max_shift_px\": " << r.max_shift_px << ",\n"
      << "      \"anomaly_score\": " << r.anomaly_score << ",\n"
      << "      \"mean_diff\": " << r.mean_diff << ",\n"
      << "      \"changed_pct\": " << r.changed_pct << ",\n"
      << "      \"alignment_ok\": " << (r.alignment_ok ? "true" : "false") << ",\n"
      << "      \"alignment_shift_px\": " << r.alignment_shift_px << ",\n"
      << "      \"processing_ms\": " << r.processing_ms << ",\n"
      << "      \"slot_deviations\": [";
    for (std::size_t i = 0; i < r.slot_details.size(); ++i)
    {
        const auto &d = r.slot_details[i];
        f << (i ? ", " : "") << "{\"slot_id\": \"" << jsonEscape(d.slot_id) << "\", \"deviation\": \"" << d.deviation
          << "\", \"dx\": " << d.dx << ", \"dy\": " << d.dy << ", \"ncc\": " << d.ncc << ", \"anomaly\": " << d.anomaly
          << ", \"board_ratio\": " << d.board_ratio << "}";
    }
    f << "],\n      \"extra_boxes\": [";
    for (std::size_t i = 0; i < r.extra_boxes.size(); ++i)
    {
        const auto &b = r.extra_boxes[i];
        f << (i ? ", " : "") << "[" << b.x << ", " << b.y << ", " << b.x + b.width << ", " << b.y + b.height << "]";
    }
    f << "]\n    }" << (last ? "" : ",") << "\n";
}

void writeMetaJson(std::ostream &f, const ReportMeta &m)
{
    f << "  \"video_source\": \"" << jsonEscape(m.video_source) << "\",\n"
      << "  \"reference_source\": \"" << jsonEscape(m.reference_source) << "\",\n"
      << "  \"backend\": \"" << jsonEscape(m.backend) << "\",\n"
      << "  \"segmentation\": \"" << jsonEscape(m.segmentation) << "\",\n"
      << "  \"total_elements\": " << m.total_elements << ",\n";
}
} // namespace

// ── Rendering ─────────────────────────────────────────────────────────────

cv::Mat renderAnnotations(const cv::Mat &frame, const FrameResult &r, const Aligner &aligner, std::size_t total_slots,
                          double fps)
{
    cv::Mat img = frame.clone();
    const int w = img.cols, h = img.rows;
    const auto &a = r.alignment;

    for (const auto &s : r.slots)
    {
        const auto p = aligner.toFrame(s.bounds, a);
        const cv::Point tl = topLeft(p);
        switch (s.deviation)
        {
        case Deviation::None:
            poly(img, p, kGreen);
            badge(img, tl, "CORRECT: " + s.slot_id, cv::Scalar(20, 20, 20), kGreen);
            break;
        case Deviation::Missing:
            poly(img, p, kRed);
            cv::line(img, p[0], p[2], kRed, 2, cv::LINE_AA);
            cv::line(img, p[1], p[3], kRed, 2, cv::LINE_AA);
            badge(img, tl, "DEVIATED: " + s.slot_id + " (MISSING)", cv::Scalar(0, 0, 180), cv::Scalar(255, 255, 255));
            break;
        case Deviation::Changed:
            poly(img, p, kOrange);
            cv::line(img, p[0], p[2], kOrange, 1, cv::LINE_AA);
            cv::line(img, p[1], p[3], kOrange, 1, cv::LINE_AA);
            badge(img, tl, "DEVIATED: " + s.slot_id + " (CHANGED)", cv::Scalar(0, 60, 200), cv::Scalar(255, 255, 255));
            break;
        case Deviation::Shifted:
        {
            const auto q = aligner.toFrame(s.matched_bounds, a);
            // Dashed expected position, solid actual position, arrow between centres.
            for (int k = 0; k < 4; ++k)
            {
                const cv::Point p0 = p[k], p1 = p[(k + 1) % 4];
                const double len = std::hypot(p1.x - p0.x, p1.y - p0.y);
                for (double t = 0; t < len; t += 8)
                {
                    const double t2 = std::min(len, t + 4);
                    cv::line(img, p0 + (p1 - p0) * (t / len), p0 + (p1 - p0) * (t2 / len), kMagenta, 1, cv::LINE_AA);
                }
            }
            poly(img, q, kMagenta);
            const cv::Point c0 = (p[0] + p[2]) / 2, c1 = (q[0] + q[2]) / 2;
            cv::arrowedLine(img, c0, c1, kMagenta, 2, cv::LINE_AA, 0, 0.3);
            char buf[96];
            std::snprintf(buf, sizeof(buf), "DEVIATED: %s (SHIFT %.1fpx)", s.slot_id.c_str(), s.shift());
            badge(img, topLeft(q), buf, cv::Scalar(120, 0, 120), cv::Scalar(255, 255, 255));
            break;
        }
        case Deviation::Extra:
            break;
        }
    }

    for (std::size_t i = 0; i < r.extra_boxes.size(); ++i)
    {
        const auto p = aligner.toFrame(r.extra_boxes[i], a);
        poly(img, p, kAmber);
        badge(img, topLeft(p), "DEVIATED: EXTRA_" + std::to_string(i + 1), cv::Scalar(0, 100, 200),
              cv::Scalar(255, 255, 255));
    }

    // Top HUD banner
    const int hud_h = 44;
    cv::rectangle(img, {0, 0}, {w, hud_h}, r.is_correct ? cv::Scalar(25, 140, 25) : cv::Scalar(30, 30, 200), cv::FILLED);
    const double t = fps > 0.0 ? r.frame_index / fps : 0.0;
    char hud[320];
    if (r.is_correct)
        std::snprintf(hud, sizeof(hud), "[CORRECT] FRAME %03d (t=%.2fs) | All %zu/%zu Elements OK (0 Deviations)",
                      r.frame_index, t, total_slots, total_slots);
    else
        std::snprintf(hud, sizeof(hud),
                      "[DEVIATED] FRAME %03d (t=%.2fs) | %zu/%zu Correct | %zu Deviations (%zu Missing, %zu Shifted, "
                      "%zu Changed, %zu Extra)",
                      r.frame_index, t, r.ok_count, total_slots, r.deviatedCount(), r.missing_count, r.shifted_count,
                      r.changed_count, r.extra_boxes.size());
    const double scale = std::clamp(w / 1280.0 * 0.65, 0.4, 0.9);
    cv::putText(img, hud, {16, 30}, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    if (!r.is_correct)
    {
        const int foot_h = 28;
        cv::rectangle(img, {0, h - foot_h}, {w, h}, cv::Scalar(15, 15, 15), cv::FILLED);
        std::string txt = "Deviated Elements: ";
        std::vector<std::string> parts;
        const auto miss = r.idsWith(Deviation::Missing), shf = r.idsWith(Deviation::Shifted),
                   chg = r.idsWith(Deviation::Changed);
        if (!miss.empty())
            parts.push_back("Missing: [" + joinIds(miss, 6) + "]");
        if (!shf.empty())
        {
            char buf[48];
            std::snprintf(buf, sizeof(buf), " max %.1fpx", r.max_shift_px);
            parts.push_back("Shifted: [" + joinIds(shf, 4) + "]" + buf);
        }
        if (!chg.empty())
            parts.push_back("Changed: [" + joinIds(chg, 4) + "]");
        if (!r.extra_boxes.empty())
            parts.push_back("Extra Components: " + std::to_string(r.extra_boxes.size()));
        for (std::size_t i = 0; i < parts.size(); ++i)
            txt += (i ? " | " : "") + parts[i];
        cv::putText(img, txt, {16, h - 8}, cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 200, 255), 1, cv::LINE_AA);
    }
    return img;
}

cv::Mat createSideBySideView(const cv::Mat &reference_annotated, const cv::Mat &frame_annotated, int frame_index,
                             int reference_index, bool is_correct)
{
    cv::Mat left = reference_annotated.clone(), right = frame_annotated.clone();
    if (left.size() != right.size())
        cv::resize(right, right, left.size());
    const int col_h = 32;
    cv::rectangle(left, {0, 44}, {left.cols, 44 + col_h}, cv::Scalar(40, 40, 40), cv::FILLED);
    char lbuf[128];
    std::snprintf(lbuf, sizeof(lbuf), "REFERENCE: FRAME %d - GOLDEN STATE", reference_index);
    cv::putText(left, reference_index < 0 ? "REFERENCE: GOLDEN IMAGE" : lbuf, {16, 66}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::rectangle(right, {0, 44}, {right.cols, 44 + col_h}, is_correct ? cv::Scalar(20, 120, 20) : cv::Scalar(20, 20, 160),
                  cv::FILLED);
    char rbuf[128];
    std::snprintf(rbuf, sizeof(rbuf), "INSPECTED: FRAME %03d - %s", frame_index,
                  is_correct ? "CORRECT (MATCHES REFERENCE)" : "DEVIATED FROM REFERENCE");
    cv::putText(right, rbuf, {16, 66}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    cv::Mat out;
    cv::hconcat(left, right, out);
    return out;
}

cv::Mat renderHeatmap(const cv::Mat &aligned, const cv::Mat &norm_map)
{
    cv::Mat m8, color, up, out;
    cv::Mat clipped;
    cv::min(norm_map, 2.0, clipped);
    clipped.convertTo(m8, CV_8U, 127.5);
    cv::applyColorMap(m8, color, cv::COLORMAP_JET);
    cv::resize(color, up, aligned.size(), 0, 0, cv::INTER_LINEAR);
    cv::addWeighted(aligned, 0.55, up, 0.45, 0.0, out);
    // Contour of the decision threshold.
    cv::Mat bin, bin_up;
    cv::compare(norm_map, 1.0, bin, cv::CMP_GE);
    cv::resize(bin, bin_up, aligned.size(), 0, 0, cv::INTER_NEAREST);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin_up, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(out, contours, -1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    return out;
}

// ── Reports ───────────────────────────────────────────────────────────────

std::string jsonEscape(const std::string &s)
{
    std::string o;
    o.reserve(s.size());
    for (const char c : s)
    {
        switch (c)
        {
        case '"':
            o += "\\\"";
            break;
        case '\\':
            o += "\\\\";
            break;
        case '\n':
            o += "\\n";
            break;
        case '\r':
            o += "\\r";
            break;
        case '\t':
            o += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20)
            {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                o += buf;
            }
            else
                o += c;
        }
    }
    return o;
}

std::vector<std::string> computeFrameRanges(const std::vector<int> &indices)
{
    std::vector<std::string> ranges;
    if (indices.empty())
        return ranges;
    int start = indices[0], prev = start;
    auto flush = [&] { ranges.push_back(start != prev ? std::to_string(start) + "-" + std::to_string(prev) : std::to_string(start)); };
    for (std::size_t i = 1; i < indices.size(); ++i)
    {
        if (indices[i] == prev + 1)
            prev = indices[i];
        else
        {
            flush();
            start = prev = indices[i];
        }
    }
    flush();
    return ranges;
}

void exportDeviationCsv(const std::string &path, const std::vector<FrameDeviationRecord> &records)
{
    ensureParent(path);
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("could not open CSV output path: " + path);
    // First 14 columns are identical to wisedo_fixture_vision_cpp.
    f << "frame_index,frame_range,repeat_count,timestamp_sec,duration_sec,status,is_correct,"
      << "total_elements,correct_count,deviated_count,missing_slots_count,missing_slots,"
      << "extra_slots_count,image_path,"
      << "shifted_slots_count,shifted_slots,changed_slots_count,changed_slots,max_shift_px,anomaly_score,"
      << "mean_diff,changed_pct,alignment_ok,processing_ms\n";
    auto list = [&](const std::vector<std::string> &v) {
        if (v.empty())
        {
            f << "None";
            return;
        }
        f << "\"";
        for (std::size_t i = 0; i < v.size(); ++i)
            f << (i ? "; " : "") << v[i];
        f << "\"";
    };
    f << std::fixed << std::setprecision(3);
    for (const auto &r : records)
    {
        f << r.frame_index << ',' << (r.frame_range.empty() ? std::to_string(r.frame_index) : r.frame_range) << ','
          << r.repeat_count << ',' << r.timestamp_sec << ',' << r.duration_sec << ',' << r.status << ','
          << (r.is_correct ? "True" : "False") << ',' << r.total_elements << ',' << r.correct_count << ','
          << r.deviated_count << ',' << r.missing_slots.size() << ',';
        list(r.missing_slots);
        f << ',' << r.extra_slots_count << ',' << r.image_path << ',' << r.shifted_slots.size() << ',';
        list(r.shifted_slots);
        f << ',' << r.changed_slots.size() << ',';
        list(r.changed_slots);
        f << ',' << r.max_shift_px << ',' << r.anomaly_score << ',' << r.mean_diff << ',' << r.changed_pct << ','
          << (r.alignment_ok ? "True" : "False") << ',' << r.processing_ms << '\n';
    }
}

void exportDeviationJson(const std::string &path, const ReportMeta &meta, const std::vector<FrameDeviationRecord> &records)
{
    ensureParent(path);
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("could not open JSON output path: " + path);
    int correct = 0;
    for (const auto &r : records)
        correct += r.is_correct ? 1 : 0;
    const int unique = static_cast<int>(records.size());
    f << "{\n";
    writeMetaJson(f, meta);
    f << "  \"total_video_frames\": " << (meta.total_frames > 0 ? meta.total_frames : unique) << ",\n"
      << "  \"unique_reported_frames\": " << unique << ",\n"
      << "  \"deduplicated\": " << (meta.deduplicated ? "true" : "false") << ",\n"
      << "  \"correct_frames\": " << correct << ",\n"
      << "  \"deviated_frames\": " << unique - correct << ",\n"
      << std::fixed << std::setprecision(2)
      << "  \"accuracy_pct\": " << (unique > 0 ? 100.0 * correct / unique : 0.0) << ",\n"
      << "  \"elapsed_sec\": " << meta.elapsed_sec << ",\n"
      << "  \"processing_fps\": " << meta.fps_processed << ",\n"
      << "  \"frames\": [\n";
    for (std::size_t i = 0; i < records.size(); ++i)
        writeFrameJson(f, records[i], i + 1 == records.size());
    f << "  ]\n}\n";
}

void exportSubsetJson(const std::string &path, const ReportMeta &meta, const std::vector<FrameDeviationRecord> &records,
                      const std::string &label)
{
    ensureParent(path);
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("could not open JSON output path: " + path);
    std::vector<int> idx;
    for (const auto &r : records)
        idx.push_back(r.frame_index);
    const auto ranges = computeFrameRanges(idx);
    f << "{\n";
    writeMetaJson(f, meta);
    f << "  \"status_filter\": \"" << label << "\",\n"
      << "  \"total_video_frames\": " << (meta.total_frames > 0 ? meta.total_frames : static_cast<int>(records.size())) << ",\n"
      << "  \"total_" << label << "_frames\": " << records.size() << ",\n"
      << "  \"frame_ranges\": [\n";
    for (std::size_t i = 0; i < ranges.size(); ++i)
        f << "    \"" << ranges[i] << "\"" << (i + 1 < ranges.size() ? "," : "") << "\n";
    f << "  ],\n  \"frame_indices\": [\n";
    for (std::size_t i = 0; i < idx.size(); ++i)
        f << "    " << idx[i] << (i + 1 < idx.size() ? "," : "") << "\n";
    f << "  ],\n  \"frames\": [\n";
    for (std::size_t i = 0; i < records.size(); ++i)
        writeFrameJson(f, records[i], i + 1 == records.size());
    f << "  ]\n}\n";
}

// ── Video writer ──────────────────────────────────────────────────────────

PortableVideoWriter PortableVideoWriter::open(const std::string &path, double fps, cv::Size size, std::string codec)
{
    if (fps <= 0.0 || size.width < 2 || size.height < 2 || codec.size() != 4)
        throw std::invalid_argument("invalid video output settings");
    ensureParent(path);
    PortableVideoWriter w;
    w.size_ = size;
    w.writer_.open(path, cv::VideoWriter::fourcc(codec[0], codec[1], codec[2], codec[3]), fps, size, true);
    if (!w.writer_.isOpened())
        throw std::runtime_error("OpenCV could not open a video writer with codec " + codec +
                                 "; use another codec (mp4v, avc1, MJPG) or an FFmpeg-enabled OpenCV");
    return w;
}

void PortableVideoWriter::write(const cv::Mat &frame)
{
    if (frame.size() != size_)
    {
        cv::Mat r;
        cv::resize(frame, r, size_);
        writer_.write(r);
        return;
    }
    writer_.write(frame);
}

void PortableVideoWriter::release()
{
    if (writer_.isOpened())
        writer_.release();
}

} // namespace fve
