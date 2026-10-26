#include <fve/anomaly.hpp>

#include <opencv2/imgcodecs.hpp>
#include <fve/cv_compat.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace fve
{
namespace
{
constexpr double kSsimSigma = 1.5;
const cv::Size kSsimWin{7, 7};

cv::Mat toLab32(const cv::Mat &bgr)
{
    cv::Mat blurred, f, lab;
    cv::GaussianBlur(bgr, blurred, {3, 3}, 0.8);
    blurred.convertTo(f, CV_32FC3, 1.0 / 255.0);
    cv::cvtColor(f, lab, cv::COLOR_BGR2Lab); // L in [0,100]
    return lab;
}

cv::Mat gaussian(const cv::Mat &m)
{
    cv::Mat out;
    cv::GaussianBlur(m, out, kSsimWin, kSsimSigma);
    return out;
}
} // namespace

// ── Classical scorer ──────────────────────────────────────────────────────

void ClassicalScorer::setReference(const cv::Mat &ref)
{
    ref_lab_ = toLab32(ref);
    cv::extractChannel(ref_lab_, ref_l_, 0);
    cv::meanStdDev(ref_l_, ref_mean_, ref_std_);
    ref_l_mu_ = gaussian(ref_l_);
    ref_l_sigma2_ = gaussian(ref_l_.mul(ref_l_)) - ref_l_mu_.mul(ref_l_mu_);
}

std::vector<RawChannel> ClassicalScorer::score(const cv::Mat &frame) const
{
    cv::Mat lab = toLab32(frame);
    std::vector<cv::Mat> ch;
    cv::split(lab, ch);

    // Global photometric normalisation of lightness (exposure / lighting drift).
    cv::Scalar mu, sd;
    cv::meanStdDev(ch[0], mu, sd);
    const double gain = std::clamp(ref_std_[0] / std::max(1e-3, sd[0]), 0.7, 1.4);
    ch[0] = (ch[0] - mu[0]) * gain + ref_mean_[0];
    cv::merge(ch, lab);

    // Colour: CIE76 deltaE, minimum over ±1 px reference shifts (sub-pixel jitter,
    // compression and residual registration error must not read as defects).
    cv::Mat ref_pad;
    cv::copyMakeBorder(ref_lab_, ref_pad, 1, 1, 1, 1, cv::BORDER_REPLICATE);
    cv::Mat best(lab.size(), CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
    cv::Mat diff, sq, d2;
    for (int dy = 0; dy <= 2; ++dy)
        for (int dx = 0; dx <= 2; ++dx)
        {
            cv::subtract(lab, ref_pad(cv::Rect(dx, dy, lab.cols, lab.rows)), diff);
            cv::multiply(diff, diff, sq);
            cv::transform(sq, d2, cv::Matx13f(1.f, 1.f, 1.f));
            cv::min(best, d2, best);
        }
    cv::Mat color;
    cv::sqrt(best, color);

    // Structure: 1 - SSIM on normalised lightness, halved into [0, 1].
    const cv::Mat &lc = ch[0];
    const cv::Mat mu_c = gaussian(lc);
    const cv::Mat sig_c = gaussian(lc.mul(lc)) - mu_c.mul(mu_c);
    const cv::Mat sig_rc = gaussian(lc.mul(ref_l_)) - mu_c.mul(ref_l_mu_);
    const double c1 = 1.0, c2 = 9.0; // (0.01 * 100)^2, (0.03 * 100)^2
    cv::Mat num = (2.0 * mu_c.mul(ref_l_mu_) + c1).mul(2.0 * sig_rc + c2);
    cv::Mat den = (mu_c.mul(mu_c) + ref_l_mu_.mul(ref_l_mu_) + c1).mul(sig_c + ref_l_sigma2_ + c2);
    cv::Mat ssim;
    cv::divide(num, den, ssim);
    cv::Mat structure = (1.0 - ssim) * 0.5;
    cv::max(structure, 0.0, structure);

    return {{"color", color}, {"structure", structure}};
}

// ── Helpers ───────────────────────────────────────────────────────────────

std::vector<cv::Mat> nuisanceVariants(const cv::Mat &bgr)
{
    std::vector<cv::Mat> out;
    const cv::Point2f c(bgr.cols * 0.5f, bgr.rows * 0.5f);
    auto affine = [&](double angle, double tx, double ty) {
        cv::Mat m = cv::getRotationMatrix2D(c, angle, 1.0);
        m.at<double>(0, 2) += tx;
        m.at<double>(1, 2) += ty;
        cv::Mat o;
        cv::warpAffine(bgr, o, m, bgr.size(), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        return o;
    };
    out.push_back(affine(0.0, 0.6, 0.4));
    out.push_back(affine(0.0, -0.5, 0.7));
    out.push_back(affine(0.3, 0.0, 0.0));
    out.push_back(affine(-0.3, 0.3, -0.3));

    // Each variant needs its own buffer: OpenCV writes into an existing Mat of
    // matching size, so reusing one header would overwrite earlier variants.
    cv::Mat brighter, darker;
    bgr.convertTo(brighter, -1, 1.08, -6.0);
    out.push_back(brighter);
    bgr.convertTo(darker, -1, 0.92, 6.0);
    out.push_back(darker);

    cv::RNG rng(0x5EED); // fixed seed: calibration is deterministic
    cv::Mat noise(bgr.size(), CV_16SC3);
    rng.fill(noise, cv::RNG::NORMAL, cv::Scalar::all(0.0), cv::Scalar::all(4.0)); // Scalars: a bare 0 binds to an empty InputArray
    cv::Mat noisy16;
    bgr.convertTo(noisy16, CV_16SC3);
    noisy16 += noise;
    cv::Mat noisy;
    noisy16.convertTo(noisy, CV_8UC3);
    out.push_back(noisy);

    std::vector<uchar> buf;
    cv::imencode(".jpg", affine(0.0, 0.3, 0.3), buf, {cv::IMWRITE_JPEG_QUALITY, 60});
    out.push_back(cv::imdecode(buf, cv::IMREAD_COLOR));

    cv::Mat blurred;
    cv::GaussianBlur(bgr, blurred, {0, 0}, 0.8);
    out.push_back(blurred);
    return out;
}

double topPercentMean(const cv::Mat &map, double percent, const cv::Mat &mask)
{
    CV_Assert(map.type() == CV_32F);
    std::vector<float> v;
    v.reserve(map.total());
    for (int y = 0; y < map.rows; ++y)
    {
        const float *p = map.ptr<float>(y);
        const uchar *m = mask.empty() ? nullptr : mask.ptr<uchar>(y);
        for (int x = 0; x < map.cols; ++x)
            if (!m || m[x])
                v.push_back(p[x]);
    }
    if (v.empty())
        return 0.0;
    const std::size_t k = std::max<std::size_t>(1, static_cast<std::size_t>(v.size() * percent / 100.0));
    std::nth_element(v.begin(), v.begin() + (v.size() - k), v.end());
    double s = 0.0;
    for (std::size_t i = v.size() - k; i < v.size(); ++i)
        s += v[i];
    return s / static_cast<double>(k);
}

// ── Calibrated scorer ─────────────────────────────────────────────────────

CalibratedScorer::CalibratedScorer(const EngineConfig &config, Backend backend) : cfg_(config.anomaly)
{
    const bool model = !config.dino.model_path.empty() && DinoScorer::available();
    if (backend == Backend::Auto)
        backend = model ? Backend::Hybrid : Backend::Classical;
    if ((backend == Backend::Dino || backend == Backend::Hybrid) && !model)
    {
        throw std::runtime_error(DinoScorer::available()
                                     ? "backend '" + std::string(toString(backend)) +
                                           "' needs --model <dinov2.onnx> (see tools/prepare_model.py)"
                                     : "this build has no ONNX Runtime; rebuild with onnxruntime or use --backend "
                                       "classical");
    }
    backend_ = backend;
    if (backend_ == Backend::Classical || backend_ == Backend::Hybrid)
        scorers_.push_back(std::make_unique<ClassicalScorer>());
    if (backend_ == Backend::Dino || backend_ == Backend::Hybrid)
        scorers_.push_back(std::make_unique<DinoScorer>(config.dino));
}

void CalibratedScorer::setRoi(const cv::Rect &roi)
{
    roi_ = roi;
    for (auto &s : scorers_)
        if (auto *d = dynamic_cast<DinoScorer *>(s.get()))
            d->setRoi(roi);
}

std::vector<RawChannel> CalibratedScorer::scoreAll(const cv::Mat &frame) const
{
    std::vector<RawChannel> all;
    for (const auto &s : scorers_)
    {
        auto ch = s->score(frame);
        for (auto &c : ch)
            all.push_back(std::move(c));
    }
    return all;
}

void CalibratedScorer::setReference(const cv::Mat &ref, const std::vector<cv::Mat> &good)
{
    for (auto &s : scorers_)
        s->setReference(ref);

    auto samples = nuisanceVariants(ref);
    for (const auto &g : good)
        samples.push_back(g);

    std::vector<cv::Mat> nuisance;
    channel_names_.clear();
    for (const auto &sample : samples)
    {
        const auto ch = scoreAll(sample);
        if (nuisance.empty())
        {
            for (const auto &c : ch)
            {
                nuisance.push_back(c.map.clone());
                channel_names_.push_back(c.name);
            }
        }
        else
        {
            for (std::size_t i = 0; i < ch.size(); ++i)
                cv::max(nuisance[i], ch[i].map, nuisance[i]);
        }
    }

    const double sens = std::max(0.05, cfg_.sensitivity);
    const auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7});
    thresholds_.clear();
    for (std::size_t i = 0; i < nuisance.size(); ++i)
    {
        const std::string &n = channel_names_[i];
        const double floor = (n == "color" ? cfg_.color_floor : n == "structure" ? cfg_.structure_floor : cfg_.dino_floor) / sens;
        cv::Mat t;
        cv::dilate(nuisance[i], t, kernel);
        t = t * (cfg_.margin / sens);
        cv::max(t, floor, t);
        thresholds_.push_back(t);
    }
}

cv::Mat CalibratedScorer::fuse(const std::vector<RawChannel> &channels) const
{
    cv::Mat classical, dino;
    for (std::size_t i = 0; i < channels.size(); ++i)
    {
        cv::Mat n;
        cv::divide(channels[i].map, thresholds_[i], n);
        if (channels[i].name == "dino")
            dino = n;
        else if (classical.empty())
            classical = n;
        else
            cv::max(classical, n, classical);
    }
    if (backend_ == Backend::Classical)
        return classical;
    if (backend_ == Backend::Dino)
        return dino;
    cv::Mat prod, out;
    cv::multiply(classical, dino, prod);
    cv::sqrt(prod, out);
    return out;
}

cv::Mat CalibratedScorer::normalizedMap(const cv::Mat &frame) const
{
    if (backend_ != Backend::Hybrid || cfg_.hybrid_cascade <= 0.0)
        return fuse(scoreAll(frame));

    // Cascade: classical first; the expensive model runs only where it can matter.
    auto channels = scorers_.front()->score(frame);
    cv::Mat classical;
    for (std::size_t i = 0; i < channels.size(); ++i)
    {
        cv::Mat n;
        cv::divide(channels[i].map, thresholds_[i], n);
        if (classical.empty())
            classical = n;
        else
            cv::max(classical, n, classical);
    }
    const cv::Rect area = (roi_.area() > 0 ? roi_ : cv::Rect(0, 0, classical.cols, classical.rows)) &
                          cv::Rect(0, 0, classical.cols, classical.rows);
    double mx = 0.0;
    cv::minMaxLoc(classical(area), nullptr, &mx);
    if (mx < cfg_.hybrid_cascade)
        return classical; // every value < cascade < 1: verdict cannot change
    for (std::size_t k = 1; k < scorers_.size(); ++k)
        for (auto &c : scorers_[k]->score(frame))
            channels.push_back(std::move(c));
    return fuse(channels);
}

std::string CalibratedScorer::describe() const
{
    std::ostringstream os;
    os << toString(backend_);
    for (const auto &s : scorers_)
    {
        if (auto *d = dynamic_cast<const DinoScorer *>(s.get()))
            os << " [" << d->describe() << "]";
    }
    return os.str();
}

} // namespace fve
