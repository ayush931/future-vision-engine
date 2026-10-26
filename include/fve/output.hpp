#pragma once

#include <fve/alignment.hpp>
#include <fve/types.hpp>

#include <opencv2/videoio.hpp>

#include <string>
#include <vector>

namespace fve
{

// ── Rendering ─────────────────────────────────────────────────────────────

/// HUD annotation drawn on the original (unwarped) frame. Reference-coordinate
/// boxes are mapped through the inverse alignment.
///   green  = component OK            red + cross = MISSING
///   magenta + arrow = SHIFTED        orange      = CHANGED
///   amber  = EXTRA
cv::Mat renderAnnotations(const cv::Mat &frame, const FrameResult &r, const Aligner &aligner,
                          std::size_t total_slots, double fps);

/// [Reference | Inspected] dual view, same layout as wisedo_fixture_vision_cpp.
cv::Mat createSideBySideView(const cv::Mat &reference_annotated, const cv::Mat &frame_annotated,
                             int frame_index, int reference_index, bool is_correct);

/// Heat-map overlay of the normalized anomaly map (in reference coordinates).
cv::Mat renderHeatmap(const cv::Mat &reference_aligned_frame, const cv::Mat &norm_map);

// ── Reports ───────────────────────────────────────────────────────────────

std::vector<std::string> computeFrameRanges(const std::vector<int> &indices);

void exportDeviationCsv(const std::string &path, const std::vector<FrameDeviationRecord> &records);

struct ReportMeta
{
    std::string video_source;
    std::string reference_source;
    std::string backend;
    std::string segmentation;
    std::size_t total_elements{0};
    int total_frames{0};
    bool deduplicated{false};
    double elapsed_sec{0.0};
    double fps_processed{0.0};
};

void exportDeviationJson(const std::string &path, const ReportMeta &meta,
                         const std::vector<FrameDeviationRecord> &records);
void exportSubsetJson(const std::string &path, const ReportMeta &meta,
                      const std::vector<FrameDeviationRecord> &records, const std::string &status_label);

std::string jsonEscape(const std::string &s);

// ── Video writer ──────────────────────────────────────────────────────────

class PortableVideoWriter
{
public:
    static PortableVideoWriter open(const std::string &path, double fps, cv::Size size, std::string codec = "mp4v");
    void write(const cv::Mat &frame);
    void release();
    bool isOpened() const noexcept { return writer_.isOpened(); }

private:
    cv::VideoWriter writer_;
    cv::Size size_;
};

} // namespace fve
