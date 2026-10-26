#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace fve
{

/// Per-component verdict. "None" means the component matches the reference.
enum class Deviation
{
    None,
    Missing, // component is gone (its slot now looks like background)
    Shifted, // component is present but displaced beyond tolerance (deflection)
    Changed, // component is in place but its appearance differs (damage, wrong part, flipped)
    Extra,   // something appeared where the reference has nothing
};

const char *toString(Deviation d) noexcept;

/// A component/slot defined in reference-frame pixel coordinates.
struct SlotDefinition
{
    std::string id;
    std::string type{"component"};
    cv::Rect bounds;
    std::string description;
};

/// Reference -> current frame registration. `warp` maps current-frame pixels into
/// reference coordinates (2x3, CV_64F). Identity when `applied` is false.
struct AlignmentResult
{
    bool ok{true};
    bool applied{false};
    cv::Mat warp;
    int inliers{0};
    double shift_px{0.0};
    double rotation_deg{0.0};
};

struct SlotResult
{
    std::string slot_id;
    std::string type{"component"};
    cv::Rect bounds;         // reference location (reference coordinates)
    cv::Rect matched_bounds; // best-match location in the current frame (reference coordinates)
    Deviation deviation{Deviation::None};
    bool is_present{true};
    double board_ratio{0.0}; // fraction of the slot ROI that looks like empty background
    double ncc{0.0};         // normalized cross-correlation at matched location (-1..1)
    double dx{0.0};          // deflection in px (reference coordinates, full resolution)
    double dy{0.0};
    double anomaly{0.0};     // top-decile normalized anomaly inside slot (1.0 == threshold)
    double anomaly_fraction{0.0}; // share of slot pixels above threshold
    double confidence{0.0};

    double shift() const noexcept;
};

struct FrameResult
{
    int frame_index{0};
    double timestamp_sec{0.0};
    bool is_reference{false};
    bool reused{false}; // result copied from the previous computed frame (content unchanged)
    bool is_correct{false};

    AlignmentResult alignment;
    std::vector<SlotResult> slots;
    std::vector<cv::Rect> extra_boxes; // reference coordinates

    std::size_t ok_count{0};
    std::size_t missing_count{0};
    std::size_t shifted_count{0};
    std::size_t changed_count{0};

    double anomaly_score{0.0}; // image-level score: mean of top 1% normalized map values
    double max_shift_px{0.0};
    double mean_diff{0.0};     // mean |reference - frame| over the inspected area (0..255)
    double changed_pct{0.0};   // % of pixels whose normalized anomaly >= 1
    double processing_ms{0.0};

    cv::Mat anomaly_map; // CV_32F normalized map at working resolution (1.0 == threshold)

    std::size_t deviatedCount() const noexcept;
    std::vector<std::string> idsWith(Deviation d) const;
};

/// Flat report row. Field names mirror wisedo_fixture_vision_cpp so downstream
/// consumers keep working; the tail fields are additions.
struct FrameDeviationRecord
{
    int frame_index{0};
    std::string frame_range;
    int repeat_count{1};
    double timestamp_sec{0.0};
    double duration_sec{0.0};
    std::string status{"CORRECT"};
    bool is_correct{false};
    std::size_t total_elements{0};
    std::size_t correct_count{0};
    std::size_t deviated_count{0};
    std::vector<std::string> missing_slots;
    std::vector<std::string> deviated_slots;
    std::size_t extra_slots_count{0};
    std::string image_path;
    double mean_diff{0.0};
    double changed_pct{0.0};

    // Additions
    std::vector<std::string> shifted_slots;
    std::vector<std::string> changed_slots;
    double max_shift_px{0.0};
    double anomaly_score{0.0};
    bool alignment_ok{true};
    double alignment_shift_px{0.0};
    double processing_ms{0.0};
    struct SlotDetail
    {
        std::string slot_id;
        std::string deviation;
        double dx{0.0}, dy{0.0}, ncc{0.0}, anomaly{0.0}, board_ratio{0.0};
    };
    std::vector<SlotDetail> slot_details; // deviated slots only
    std::vector<cv::Rect> extra_boxes;
};

FrameDeviationRecord makeRecord(const FrameResult &r, double fps);

} // namespace fve
