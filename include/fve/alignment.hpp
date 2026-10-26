#pragma once

#include <fve/config.hpp>
#include <fve/cv_compat.hpp>

#include <vector>

namespace fve
{

/// Estimates a similarity transform (rotation + uniform scale + translation)
/// that maps the current frame onto the reference, so camera shake or a
/// re-seated fixture does not register as a deviation. Thread-safe after
/// setReference().
class Aligner
{
public:
    explicit Aligner(AlignmentConfig config = {});

    void setReference(const cv::Mat &reference_bgr);
    AlignmentResult estimate(const cv::Mat &frame_bgr) const;

    /// Warps `frame` into reference coordinates (no-op copy when not applied).
    cv::Mat warp(const cv::Mat &frame, const AlignmentResult &a, int interpolation = cv::INTER_LINEAR) const;
    /// Mask of reference pixels that are covered by the warped frame.
    cv::Mat validMask(const AlignmentResult &a, cv::Size size) const;

    /// Maps a reference-coordinate rectangle back into frame coordinates.
    std::vector<cv::Point> toFrame(const cv::Rect &ref_rect, const AlignmentResult &a) const;
    cv::Point2f toFrame(cv::Point2f ref_pt, const AlignmentResult &a) const;

private:
    AlignmentConfig config_;
    cv::Size ref_size_;
    double scale_{1.0};
    cv::Mat ref_small_;
    cv::Mat ref_blur_;
    std::vector<cv::KeyPoint> ref_kp_;
    cv::Mat ref_desc_;
};

} // namespace fve
