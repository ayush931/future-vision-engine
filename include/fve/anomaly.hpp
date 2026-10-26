#pragma once

#include <fve/config.hpp>

#include <memory>
#include <string>
#include <vector>

namespace fve
{

/// One raw anomaly channel at working resolution (CV_32F, larger = more anomalous).
struct RawChannel
{
    std::string name;
    cv::Mat map;
};

/// A reference-conditioned anomaly scorer. Inputs are already aligned to the
/// reference and resized to the working size. Implementations must be
/// thread-safe after setReference().
class AnomalyScorer
{
public:
    virtual ~AnomalyScorer() = default;
    virtual std::string name() const = 0;
    virtual void setReference(const cv::Mat &reference_work_bgr) = 0;
    virtual std::vector<RawChannel> score(const cv::Mat &frame_work_bgr) const = 0;
};

/// Model-free scorer: photometrically normalised, ±1 px shift-tolerant CIE76
/// colour difference plus local SSIM structural difference.
class ClassicalScorer final : public AnomalyScorer
{
public:
    std::string name() const override { return "classical"; }
    void setReference(const cv::Mat &reference_work_bgr) override;
    std::vector<RawChannel> score(const cv::Mat &frame_work_bgr) const override;

private:
    cv::Mat ref_lab_;     // CV_32FC3
    cv::Mat ref_l_;       // CV_32F
    cv::Mat ref_l_mu_;    // local mean (SSIM)
    cv::Mat ref_l_sigma2_;
    cv::Scalar ref_mean_, ref_std_;
};

/// DINOv2 patch-feature scorer (ONNX Runtime).
///  - Multi-layer features, each L2-normalised then fused (VisionAD, Dinomaly2).
///  - Memory bank from the reference plus half-patch shifted copies
///    (support augmentation, VisionAD / AnomalyDINO).
///  - 1-NN cosine distance, positional neighbourhood or global search
///    (AnomalyDINO; positional matching is UniVAD's component-aware idea
///    specialised to a fixed camera).
///  - Bilinear upsampling + Gaussian smoothing of the patch map (Dinomaly2).
class DinoScorer final : public AnomalyScorer
{
public:
    explicit DinoScorer(DinoConfig config);
    ~DinoScorer() override;
    std::string name() const override { return "dino"; }
    void setReference(const cv::Mat &reference_work_bgr) override;
    std::vector<RawChannel> score(const cv::Mat &frame_work_bgr) const override;

    static bool available() noexcept; // compiled with ONNX Runtime
    std::string describe() const;
    /// Restrict the model to a region of the working image (call before
    /// setReference). Fewer tokens at the same resolution: attention is quadratic.
    void setRoi(const cv::Rect &roi);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Combines scorer channels into one normalised map where 1.0 is the decision
/// threshold. Thresholds are per pixel, learned from the scorer's response to
/// nuisance variants of the reference (sub-pixel shift, rotation, gain, noise,
/// JPEG, blur) and from optional known-good frames, so edges and texture get
/// proportionally higher thresholds than flat areas.
class CalibratedScorer
{
public:
    CalibratedScorer(const EngineConfig &config, Backend backend);

    /// Inspected region of the working image (default: everything).
    void setRoi(const cv::Rect &roi);
    void setReference(const cv::Mat &reference_work_bgr, const std::vector<cv::Mat> &good_work_frames = {});
    cv::Mat normalizedMap(const cv::Mat &frame_work_bgr) const;

    Backend backend() const noexcept { return backend_; }
    std::string describe() const;

private:
    cv::Mat fuse(const std::vector<RawChannel> &channels) const;
    std::vector<RawChannel> scoreAll(const cv::Mat &frame_work_bgr) const;

    AnomalyConfig cfg_;
    Backend backend_;
    std::vector<std::unique_ptr<AnomalyScorer>> scorers_;
    std::vector<std::string> channel_names_;
    std::vector<cv::Mat> thresholds_; // per channel, CV_32F
    cv::Rect roi_;
};

/// Deterministic nuisance variants of an image used for calibration.
std::vector<cv::Mat> nuisanceVariants(const cv::Mat &bgr);

/// Mean of the top `percent` % of values in `map` where `mask` is non-zero.
double topPercentMean(const cv::Mat &map, double percent, const cv::Mat &mask = cv::Mat());

} // namespace fve
