#pragma once

#include <fve/types.hpp>

#include <string>
#include <vector>

namespace fve
{

enum class Backend
{
    Auto,      // hybrid when a model is available, otherwise classical
    Classical, // shift-tolerant colour + structural (SSIM) difference, no model
    Dino,      // DINOv2 patch-feature matching (AnomalyDINO / VisionAD / Dinomaly2)
    Hybrid,    // geometric mean of classical and DINO normalized maps
};

enum class SegmentationMode
{
    Auto,       // fixture white-box discovery, then generic foreground, then grid
    Fixture,    // tan board + white components (wisedo_fixture_vision_cpp behaviour)
    Foreground, // generic: anything that differs from the dominant background colour
    Grid,       // fixed 5x5 grid over the central board area
    None,       // no components: pure anomaly detection, deviations reported as EXTRA regions
};

Backend parseBackend(const std::string &s);
SegmentationMode parseSegmentation(const std::string &s);
const char *toString(Backend b) noexcept;
const char *toString(SegmentationMode m) noexcept;

struct BoardParams
{
    bool valid{false};
    cv::Scalar hsv_lower{10, 50, 70};
    cv::Scalar hsv_upper{30, 180, 220};
    double missing_threshold{0.40};
    double margin_ratio{0.15};
};

struct AlignmentConfig
{
    bool enabled{true};
    int max_features{1500};
    int min_inliers{20};
    int analysis_width{640};
    double ransac_reproj_px{3.0};
    double max_shift_ratio{0.25};   // reject warps translating more than this share of the width
    double max_rotation_deg{15.0};
};

struct AnomalyConfig
{
    int work_width{640};
    double sensitivity{1.0};       // >1 more sensitive, <1 fewer false alarms
    double margin{1.6};            // threshold = margin * nuisance response (per pixel)
    double color_floor{7.0};       // minimum colour threshold (CIE76 deltaE)
    double structure_floor{0.14};  // minimum structural threshold (DSSIM)
    double dino_floor{0.15};       // minimum DINO threshold (cosine distance)
    int calibration_frames{0};     // extra known-good frames after the reference used for calibration
    double top_percent{1.0};       // image score = mean of top N% of map values
    // Hybrid cascade: when the classical map stays below this everywhere, the
    // fused score cannot reach 1 unless DINO exceeds 1/cascade, so DINO is skipped.
    // 0 disables the cascade.
    double hybrid_cascade{0.35};
};

struct DinoConfig
{
    std::string model_path; // empty = disabled
    std::string provider{"auto"}; // auto (coreml on Apple, else cpu) | cpu | coreml | cuda
    int short_edge{448};
    int search_radius{2}; // patch neighbourhood for positional matching; <0 = global (AnomalyDINO)
    int threads{0};       // ORT intra-op threads per session (0 = ORT default)
    bool shift_augment{true}; // half-patch shifted reference copies in the memory bank
};

struct ComponentConfig
{
    double shift_tolerance_px{0.0}; // 0 = auto (10% of the slot's short side, min 3 px)
    double match_min_ncc{0.55};
    double search_ratio{0.5};       // template search radius as share of slot size
    double changed_min_fraction{0.12};
    double min_extra_area_ratio{0.0006}; // of the frame area
    double context_pad_ratio{0.25};
};

struct EngineConfig
{
    std::string fixture_name{"fixture_auto"};
    Backend backend{Backend::Auto};
    SegmentationMode segmentation{SegmentationMode::Auto};
    BoardParams board;
    std::vector<SlotDefinition> slots; // empty = discover on the reference
    std::vector<cv::Rect> ignore_regions;
    AlignmentConfig alignment;
    AnomalyConfig anomaly;
    DinoConfig dino;
    ComponentConfig components;

    /// Loads a JSON config. Accepts the wisedo_fixture_vision_cpp slot format
    /// (`slots[].slot_id/type/bbox[x1,y1,x2,y2]`, `detection_params`) plus the
    /// sections documented in configs/example.json.
    static EngineConfig load(const std::string &path);
};

} // namespace fve
