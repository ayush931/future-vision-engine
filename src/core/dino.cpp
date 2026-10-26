#include <fve/anomaly.hpp>

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <cmath>
#include <utility>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef FVE_WITH_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace fve
{

#ifdef FVE_WITH_ONNX

namespace
{
constexpr int kPatch = 14;

Ort::Env &ortEnv()
{
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "fve");
    return env;
}

/// Row-normalised patch features of one image: N x D (CV_32F), N = gh * gw.
struct Features
{
    cv::Mat f;
};
} // namespace

struct DinoScorer::Impl
{
    DinoConfig cfg;
    std::unique_ptr<Ort::Session> session;
    std::string input_name;
    std::vector<std::string> output_names;
    std::vector<const char *> output_ptrs;
    std::string provider_used{"cpu"};
    cv::Size work_size;
    cv::Rect roi; // region of the working image fed to the model
    int in_w{0}, in_h{0}, gw{0}, gh{0}, dim{0};
    std::vector<Features> bank;

    void createSession()
    {
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (cfg.threads > 0)
            so.SetIntraOpNumThreads(cfg.threads);
        // Static shapes let the providers pre-plan memory and kernels.
        // C API: the C++ wrapper only gained this method in newer releases.
        const OrtApi &api = Ort::GetApi();
        for (const auto &[name, value] : {std::pair<const char *, int64_t>{"batch_size", 1}, {"num_channels", 3},
                                          {"height", in_h}, {"width", in_w}})
            Ort::ThrowOnError(api.AddFreeDimensionOverrideByName(so, name, value));

        provider_used = "cpu";
        std::string provider = cfg.provider;
        if (provider == "auto")
        {
#ifdef __APPLE__
            provider = "coreml";
#else
            provider = "cpu";
#endif
        }
        if (provider == "coreml")
        {
            try
            {
                so.AppendExecutionProvider("CoreML", {{"ModelFormat", "MLProgram"}, {"MLComputeUnits", "ALL"}});
                provider_used = "coreml";
            }
            catch (const std::exception &e)
            {
                std::cerr << "[dino] CoreML provider unavailable (" << e.what() << "), using CPU\n";
            }
        }
        else if (provider == "cuda")
        {
            try
            {
                OrtCUDAProviderOptions cuda{};
                so.AppendExecutionProvider_CUDA(cuda);
                provider_used = "cuda";
            }
            catch (const std::exception &e)
            {
                std::cerr << "[dino] CUDA provider unavailable (" << e.what() << "), using CPU\n";
            }
        }

        session = std::make_unique<Ort::Session>(ortEnv(), cfg.model_path.c_str(), so);
        Ort::AllocatorWithDefaultOptions alloc;
        input_name = session->GetInputNameAllocated(0, alloc).get();
        output_names.clear();
        for (std::size_t i = 0; i < session->GetOutputCount(); ++i)
            output_names.emplace_back(session->GetOutputNameAllocated(i, alloc).get());
        output_ptrs.clear();
        for (const auto &n : output_names)
            output_ptrs.push_back(n.c_str());
    }

    Features extract(const cv::Mat &bgr) const
    {
        cv::Mat resized, rgb, f;
        cv::resize(bgr, resized, {in_w, in_h}, 0, 0,
                   (in_w < bgr.cols) ? cv::INTER_AREA : cv::INTER_CUBIC);
        cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(f, CV_32FC3, 1.0 / 255.0);

        static const float mean[3] = {0.485f, 0.456f, 0.406f};
        static const float stdv[3] = {0.229f, 0.224f, 0.225f};
        std::vector<float> blob(static_cast<std::size_t>(3) * in_h * in_w);
        const std::size_t plane = static_cast<std::size_t>(in_h) * in_w;
        for (int y = 0; y < in_h; ++y)
        {
            const cv::Vec3f *p = f.ptr<cv::Vec3f>(y);
            for (int x = 0; x < in_w; ++x)
                for (int c = 0; c < 3; ++c)
                    blob[c * plane + static_cast<std::size_t>(y) * in_w + x] = (p[x][c] - mean[c]) / stdv[c];
        }

        const std::array<int64_t, 4> shape{1, 3, in_h, in_w};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input = Ort::Value::CreateTensor<float>(mem, blob.data(), blob.size(), shape.data(), shape.size());
        const char *in_names[] = {input_name.c_str()};
        auto outputs = session->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, output_ptrs.data(),
                                    output_ptrs.size());

        const int n = gh * gw;
        std::vector<cv::Mat> layers;
        for (auto &o : outputs)
        {
            const auto info = o.GetTensorTypeAndShapeInfo();
            const auto s = info.GetShape();
            if (s.size() != 3 || s[1] < n)
                continue; // e.g. a pooled output
            const int tokens = static_cast<int>(s[1]);
            const int c = static_cast<int>(s[2]);
            const int prefix = tokens - n; // CLS + register tokens
            const float *data = o.GetTensorData<float>() + static_cast<std::size_t>(prefix) * c;
            cv::Mat m(n, c, CV_32F, const_cast<float *>(data));
            cv::Mat normed(n, c, CV_32F);
            for (int i = 0; i < n; ++i)
            {
                const float *r = m.ptr<float>(i);
                double ss = 0.0;
                for (int k = 0; k < c; ++k)
                    ss += static_cast<double>(r[k]) * r[k];
                const float inv = static_cast<float>(1.0 / std::max(1e-12, std::sqrt(ss)));
                float *w = normed.ptr<float>(i);
                for (int k = 0; k < c; ++k)
                    w[k] = r[k] * inv;
            }
            layers.push_back(normed);
        }
        if (layers.empty())
            throw std::runtime_error("model produced no [1, tokens, C] patch outputs");

        Features out;
        if (layers.size() == 1)
            out.f = layers[0];
        else
        {
            // Concatenating unit vectors and rescaling by 1/sqrt(L) makes the dot
            // product equal the mean per-layer cosine similarity (Dinomaly2 map mean).
            cv::hconcat(layers, out.f);
            out.f *= 1.0 / std::sqrt(static_cast<double>(layers.size()));
        }
        return out;
    }

    cv::Mat distanceMap(const Features &q) const
    {
        cv::Mat dist(gh, gw, CV_32F);
        const int d = q.f.cols;
        const int r = cfg.search_radius;
        if (r >= 0)
        {
            cv::parallel_for_(cv::Range(0, gh), [&](const cv::Range &rows) {
                for (int i = rows.start; i < rows.end; ++i)
                    for (int j = 0; j < gw; ++j)
                    {
                        const float *qv = q.f.ptr<float>(i * gw + j);
                        float best = -1.f;
                        for (const auto &b : bank)
                            for (int ii = std::max(0, i - r); ii <= std::min(gh - 1, i + r); ++ii)
                                for (int jj = std::max(0, j - r); jj <= std::min(gw - 1, j + r); ++jj)
                                {
                                    const float *bv = b.f.ptr<float>(ii * gw + jj);
                                    float s = 0.f;
                                    for (int k = 0; k < d; ++k)
                                        s += qv[k] * bv[k];
                                    best = std::max(best, s);
                                }
                        dist.at<float>(i, j) = 1.f - best;
                    }
            });
        }
        else
        {
            cv::Mat best(q.f.rows, 1, CV_32F, cv::Scalar(-1.f));
            for (const auto &b : bank)
            {
                cv::Mat sims;
                cv::gemm(q.f, b.f, 1.0, cv::noArray(), 0.0, sims, cv::GEMM_2_T);
                cv::Mat row_max;
                cv::reduce(sims, row_max, 1, cv::REDUCE_MAX);
                cv::max(best, row_max, best);
            }
            dist = cv::Mat(1.f - best).reshape(1, gh);
        }
        return dist;
    }
};

DinoScorer::DinoScorer(DinoConfig config) : impl_(std::make_unique<Impl>())
{
    impl_->cfg = std::move(config);
}

DinoScorer::~DinoScorer() = default;

bool DinoScorer::available() noexcept { return true; }

void DinoScorer::setRoi(const cv::Rect &roi) { impl_->roi = roi; }

void DinoScorer::setReference(const cv::Mat &full_ref)
{
    auto &m = *impl_;
    m.work_size = full_ref.size();
    const cv::Rect all(0, 0, full_ref.cols, full_ref.rows);
    m.roi = (m.roi.area() > 0 ? m.roi : all) & all;
    if (m.roi.width < 28 || m.roi.height < 28)
        m.roi = all;
    const cv::Mat ref = full_ref(m.roi);
    const double s = static_cast<double>(m.cfg.short_edge) / std::min(ref.cols, ref.rows);
    m.gw = std::max(1, static_cast<int>(std::lround(ref.cols * s / kPatch)));
    m.gh = std::max(1, static_cast<int>(std::lround(ref.rows * s / kPatch)));
    m.in_w = m.gw * kPatch;
    m.in_h = m.gh * kPatch;
    m.createSession();

    m.bank.clear();
    m.bank.push_back(m.extract(ref));
    if (m.cfg.shift_augment)
    {
        // Half-patch shifts in model space give the bank sub-patch phase coverage.
        const double half = 0.5 * kPatch * ref.cols / static_cast<double>(m.in_w);
        const double shifts[4][2] = {{half, 0}, {-half, 0}, {0, half}, {0, -half}};
        for (const auto &sh : shifts)
        {
            cv::Mat t = (cv::Mat_<double>(2, 3) << 1, 0, sh[0], 0, 1, sh[1]);
            cv::Mat shifted;
            cv::warpAffine(ref, shifted, t, ref.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            m.bank.push_back(m.extract(shifted));
        }
    }
    m.dim = m.bank.front().f.cols;
}

std::vector<RawChannel> DinoScorer::score(const cv::Mat &frame) const
{
    const auto &m = *impl_;
    static const bool profile = std::getenv("FVE_PROFILE") != nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    const Features q = m.extract(frame(m.roi));
    const auto t1 = std::chrono::steady_clock::now();
    const cv::Mat patch = m.distanceMap(q);
    if (profile)
    {
        const auto t2 = std::chrono::steady_clock::now();
        std::cerr << "[dino] extract " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms, match "
                  << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms\n";
    }
    cv::Mat up;
    cv::resize(patch, up, m.roi.size(), 0, 0, cv::INTER_LINEAR);
    const double sigma = 0.3 * m.roi.width / static_cast<double>(m.gw);
    cv::Mat smooth;
    cv::GaussianBlur(up, smooth, {0, 0}, std::max(0.5, sigma));
    cv::Mat out = cv::Mat::zeros(m.work_size, CV_32F);
    smooth.copyTo(out(m.roi));
    return {{"dino", out}};
}

std::string DinoScorer::describe() const
{
    std::ostringstream os;
    os << "dinov2 " << impl_->in_w << "x" << impl_->in_h << " grid " << impl_->gw << "x" << impl_->gh << " dim "
       << impl_->dim << " bank " << impl_->bank.size() << " radius " << impl_->cfg.search_radius << " on "
       << impl_->provider_used;
    return os.str();
}

#else // !FVE_WITH_ONNX

struct DinoScorer::Impl
{
};
DinoScorer::DinoScorer(DinoConfig) {}
DinoScorer::~DinoScorer() = default;
bool DinoScorer::available() noexcept { return false; }
void DinoScorer::setReference(const cv::Mat &) { throw std::runtime_error("built without ONNX Runtime"); }
void DinoScorer::setRoi(const cv::Rect &) {}
std::vector<RawChannel> DinoScorer::score(const cv::Mat &) const
{
    throw std::runtime_error("built without ONNX Runtime");
}
std::string DinoScorer::describe() const { return "unavailable"; }

#endif

} // namespace fve
