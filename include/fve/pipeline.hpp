#pragma once

#include <fve/config.hpp>

#include <string>

namespace fve
{

struct PipelineOptions
{
    std::string input{"video.mp4"}; // video file, image file, image directory, camera index or URL
    std::string output_dir{"inspection_results"};
    std::string reference_image;    // optional: external golden image
    int reference_frame{0};         // used when reference_image is empty
    bool side_by_side{false};
    bool heatmaps{false};
    bool save_images{true};
    bool render_video{true};
    std::string render_video_path; // default <out>/annotated_results.mp4
    std::string csv_path, json_path, correct_csv, correct_json, deviated_csv, deviated_json;
    std::string codec{"mp4v"};
    bool deduplicate{false};
    bool gate_unchanged{true};    // reuse the previous result when the frame content is unchanged
    double gate_mean_diff{1.0};   // 0..255, global mean on a 320 px thumbnail
    double gate_block_diff{6.0};  // 0..255, max mean difference of any 8x8 thumbnail block
    int max_frames{-1};
    int workers{0}; // 0 = auto
    int jpeg_quality{92};
    bool quiet{false};
};

/// Runs the full inspection and writes the same artefacts as
/// wisedo_fixture_vision_cpp. Returns the process exit code.
int runPipeline(const EngineConfig &config, PipelineOptions options);

} // namespace fve
