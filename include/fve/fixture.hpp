#pragma once

#include <fve/config.hpp>

#include <utility>
#include <vector>

namespace fve
{

// Board localisation and component discovery. The fixture functions are a
// faithful port of wisedo_fixture_vision_cpp so auto-discovered slot IDs match.

cv::Scalar centerMedianHsv(const cv::Mat &bgr);
std::pair<cv::Scalar, cv::Scalar> boardThresholdsFromMedian(const cv::Scalar &median);
cv::Rect findBoardBbox(const cv::Mat &hsv, const cv::Scalar &lower, const cv::Scalar &upper,
                       const cv::Rect &fallback);
double boxIoU(const cv::Rect &a, const cv::Rect &b);
/// Intersection area divided by the area of `a`.
double overlapRatio(const cv::Rect &a, const cv::Rect &b);
std::vector<cv::Rect> mergeBoxesNms(const std::vector<cv::Rect> &boxes, double iou_thresh = 0.30);
std::vector<cv::Rect> detectWhiteBoxes(const cv::Mat &bgr, cv::Scalar *out_median = nullptr,
                                       cv::Rect *out_board = nullptr);

/// Generic component segmentation: pixels far (in Lab) from the dominant
/// background colour, cleaned with morphology and split into connected parts.
std::vector<cv::Rect> detectForegroundComponents(const cv::Mat &bgr, cv::Rect *out_board = nullptr);

std::vector<SlotDefinition> gridSlots(cv::Size frame_size, std::size_t slot_count = 25);

struct DiscoveryResult
{
    std::vector<SlotDefinition> slots;
    BoardParams board;
    cv::Rect board_bbox;
    SegmentationMode used{SegmentationMode::None};
};

DiscoveryResult discoverComponents(const cv::Mat &reference_bgr, SegmentationMode mode,
                                   const BoardParams &base);

/// Fraction of `roi` (shrunk by margin_ratio) whose HSV lies in the board range.
double boardRatio(const cv::Mat &hsv, const cv::Rect &roi, const BoardParams &board);

} // namespace fve
