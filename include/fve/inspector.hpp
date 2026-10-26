#pragma once

#include <fve/alignment.hpp>
#include <fve/anomaly.hpp>
#include <fve/config.hpp>

#include <memory>
#include <vector>

namespace fve
{

/// Everything learned from the reference image. Immutable once built.
struct ReferenceModel
{
    cv::Mat bgr;
    cv::Mat work_bgr;
    cv::Mat work_gray;
    cv::Size size;
    cv::Size work_size;
    double work_scale{1.0}; // work = full * work_scale
    std::vector<SlotDefinition> slots;
    std::vector<double> slot_ref_board_ratio;
    std::vector<int> slot_search_radius_work; // template search radius (work px)
    BoardParams board;
    cv::Rect board_bbox;
    SegmentationMode segmentation{SegmentationMode::None};
    cv::Mat ignore_mask_work; // 255 = ignore
};

/// Compares frames to a single reference. inspect() is const and thread-safe,
/// so frames can be processed in parallel.
class Inspector
{
public:
    explicit Inspector(EngineConfig config);
    ~Inspector();

    void setReference(const cv::Mat &reference_bgr, const std::vector<cv::Mat> &good_frames = {});
    FrameResult inspect(const cv::Mat &frame_bgr, int frame_index, double fps) const;

    /// The reference evaluated against itself (all components OK).
    FrameResult referenceResult(int frame_index, double fps) const;

    const ReferenceModel &reference() const noexcept { return ref_; }
    const Aligner &aligner() const noexcept { return aligner_; }
    const EngineConfig &config() const noexcept { return config_; }
    Backend backend() const noexcept;
    std::string describe() const;

private:
    void analyseSlots(const cv::Mat &aligned_bgr, const cv::Mat &aligned_work_gray, const cv::Mat &norm_map,
                      const cv::Mat &valid_work, FrameResult &out) const;
    void detectExtras(const cv::Mat &aligned_bgr, const cv::Mat &norm_map, const cv::Mat &valid_work,
                      FrameResult &out) const;

    EngineConfig config_;
    ReferenceModel ref_;
    Aligner aligner_;
    std::unique_ptr<CalibratedScorer> scorer_;
};

} // namespace fve
