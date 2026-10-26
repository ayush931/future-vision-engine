#include <fve/config.hpp>
#include <fve/types.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace fve
{

const char *toString(Deviation d) noexcept
{
    switch (d)
    {
    case Deviation::None:
        return "OK";
    case Deviation::Missing:
        return "MISSING";
    case Deviation::Shifted:
        return "SHIFTED";
    case Deviation::Changed:
        return "CHANGED";
    case Deviation::Extra:
        return "EXTRA";
    }
    return "UNKNOWN";
}

double SlotResult::shift() const noexcept { return std::hypot(dx, dy); }

std::size_t FrameResult::deviatedCount() const noexcept
{
    return missing_count + shifted_count + changed_count + extra_boxes.size();
}

std::vector<std::string> FrameResult::idsWith(Deviation d) const
{
    std::vector<std::string> ids;
    for (const auto &s : slots)
        if (s.deviation == d)
            ids.push_back(s.slot_id);
    return ids;
}

FrameDeviationRecord makeRecord(const FrameResult &r, double fps)
{
    FrameDeviationRecord rec;
    rec.frame_index = r.frame_index;
    rec.frame_range = std::to_string(r.frame_index);
    rec.repeat_count = 1;
    rec.timestamp_sec = r.timestamp_sec;
    rec.duration_sec = fps > 0.0 ? 1.0 / fps : 0.0;
    rec.is_correct = r.is_correct;
    rec.status = r.is_correct ? "CORRECT" : "DEVIATED";
    rec.total_elements = r.slots.size();
    rec.correct_count = r.ok_count;
    rec.deviated_count = r.deviatedCount();
    rec.missing_slots = r.idsWith(Deviation::Missing);
    rec.shifted_slots = r.idsWith(Deviation::Shifted);
    rec.changed_slots = r.idsWith(Deviation::Changed);
    for (const auto &s : r.slots)
    {
        if (s.deviation == Deviation::None)
            continue;
        rec.deviated_slots.push_back(s.slot_id);
        rec.slot_details.push_back({s.slot_id, toString(s.deviation), s.dx, s.dy, s.ncc, s.anomaly, s.board_ratio});
    }
    rec.extra_slots_count = r.extra_boxes.size();
    rec.extra_boxes = r.extra_boxes;
    rec.mean_diff = r.mean_diff;
    rec.changed_pct = r.changed_pct;
    rec.max_shift_px = r.max_shift_px;
    rec.anomaly_score = r.anomaly_score;
    rec.alignment_ok = r.alignment.ok;
    rec.alignment_shift_px = r.alignment.shift_px;
    rec.processing_ms = r.processing_ms;
    return rec;
}

namespace
{
std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

template <typename T> void readIf(const cv::FileNode &node, const char *key, T &out)
{
    const auto n = node[key];
    if (!n.empty())
        n >> out;
}

void readBool(const cv::FileNode &node, const char *key, bool &out)
{
    const auto n = node[key];
    if (n.empty())
        return;
    if (n.isString())
    {
        const std::string v = lower(static_cast<std::string>(n));
        out = (v == "true" || v == "1" || v == "yes");
    }
    else
    {
        out = static_cast<int>(n) != 0;
    }
}

cv::Rect rectFromX1Y1X2Y2(const std::vector<int> &v)
{
    if (v.size() != 4 || v[2] <= v[0] || v[3] <= v[1])
        throw std::runtime_error("bbox must be [x1, y1, x2, y2] with x2 > x1 and y2 > y1");
    return {v[0], v[1], v[2] - v[0], v[3] - v[1]};
}
} // namespace

Backend parseBackend(const std::string &s)
{
    const auto v = lower(s);
    if (v == "auto")
        return Backend::Auto;
    if (v == "classical" || v == "classic")
        return Backend::Classical;
    if (v == "dino" || v == "dinov2")
        return Backend::Dino;
    if (v == "hybrid")
        return Backend::Hybrid;
    throw std::invalid_argument("unknown backend: " + s + " (auto|classical|dino|hybrid)");
}

SegmentationMode parseSegmentation(const std::string &s)
{
    const auto v = lower(s);
    if (v == "auto")
        return SegmentationMode::Auto;
    if (v == "fixture")
        return SegmentationMode::Fixture;
    if (v == "foreground")
        return SegmentationMode::Foreground;
    if (v == "grid")
        return SegmentationMode::Grid;
    if (v == "none")
        return SegmentationMode::None;
    throw std::invalid_argument("unknown segmentation: " + s + " (auto|fixture|foreground|grid|none)");
}

const char *toString(Backend b) noexcept
{
    switch (b)
    {
    case Backend::Auto:
        return "auto";
    case Backend::Classical:
        return "classical";
    case Backend::Dino:
        return "dino";
    case Backend::Hybrid:
        return "hybrid";
    }
    return "unknown";
}

const char *toString(SegmentationMode m) noexcept
{
    switch (m)
    {
    case SegmentationMode::Auto:
        return "auto";
    case SegmentationMode::Fixture:
        return "fixture";
    case SegmentationMode::Foreground:
        return "foreground";
    case SegmentationMode::Grid:
        return "grid";
    case SegmentationMode::None:
        return "none";
    }
    return "unknown";
}

EngineConfig EngineConfig::load(const std::string &path)
{
    cv::FileStorage fs(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened())
        throw std::runtime_error("could not open config: " + path);

    EngineConfig c;
    const auto root = fs.root();
    readIf(root, "fixture_name", c.fixture_name);
    if (!root["backend"].empty())
        c.backend = parseBackend(static_cast<std::string>(root["backend"]));
    if (!root["segmentation"].empty())
        c.segmentation = parseSegmentation(static_cast<std::string>(root["segmentation"]));

    const auto params = root["detection_params"];
    if (!params.empty())
    {
        std::vector<int> lo, hi;
        readIf(params, "board_hsv_lower", lo);
        readIf(params, "board_hsv_upper", hi);
        if (lo.size() == 3 && hi.size() == 3)
        {
            c.board.hsv_lower = cv::Scalar(lo[0], lo[1], lo[2]);
            c.board.hsv_upper = cv::Scalar(hi[0], hi[1], hi[2]);
            c.board.valid = true;
        }
        readIf(params, "missing_threshold", c.board.missing_threshold);
        readIf(params, "margin_ratio", c.board.margin_ratio);
    }

    for (const auto &node : root["slots"])
    {
        SlotDefinition s;
        std::vector<int> bbox;
        readIf(node, "slot_id", s.id);
        readIf(node, "type", s.type);
        readIf(node, "bbox", bbox);
        readIf(node, "description", s.description);
        if (s.id.empty())
            throw std::runtime_error("slot without slot_id in " + path);
        s.bounds = rectFromX1Y1X2Y2(bbox);
        if (s.description.empty())
            s.description = s.id;
        c.slots.push_back(std::move(s));
    }

    for (const auto &node : root["ignore_regions"])
    {
        std::vector<int> bbox;
        node >> bbox;
        c.ignore_regions.push_back(rectFromX1Y1X2Y2(bbox));
    }

    const auto al = root["alignment"];
    if (!al.empty())
    {
        readBool(al, "enabled", c.alignment.enabled);
        readIf(al, "max_features", c.alignment.max_features);
        readIf(al, "min_inliers", c.alignment.min_inliers);
        readIf(al, "analysis_width", c.alignment.analysis_width);
        readIf(al, "ransac_reproj_px", c.alignment.ransac_reproj_px);
        readIf(al, "max_shift_ratio", c.alignment.max_shift_ratio);
        readIf(al, "max_rotation_deg", c.alignment.max_rotation_deg);
    }

    const auto an = root["anomaly"];
    if (!an.empty())
    {
        readIf(an, "work_width", c.anomaly.work_width);
        readIf(an, "sensitivity", c.anomaly.sensitivity);
        readIf(an, "margin", c.anomaly.margin);
        readIf(an, "color_floor", c.anomaly.color_floor);
        readIf(an, "structure_floor", c.anomaly.structure_floor);
        readIf(an, "dino_floor", c.anomaly.dino_floor);
        readIf(an, "calibration_frames", c.anomaly.calibration_frames);
        readIf(an, "top_percent", c.anomaly.top_percent);
        readIf(an, "hybrid_cascade", c.anomaly.hybrid_cascade);
    }

    const auto dn = root["dino"];
    if (!dn.empty())
    {
        readIf(dn, "model", c.dino.model_path);
        readIf(dn, "provider", c.dino.provider);
        readIf(dn, "short_edge", c.dino.short_edge);
        readIf(dn, "search_radius", c.dino.search_radius);
        readIf(dn, "threads", c.dino.threads);
        readBool(dn, "shift_augment", c.dino.shift_augment);
    }

    const auto cp = root["components"];
    if (!cp.empty())
    {
        readIf(cp, "shift_tolerance_px", c.components.shift_tolerance_px);
        readIf(cp, "match_min_ncc", c.components.match_min_ncc);
        readIf(cp, "search_ratio", c.components.search_ratio);
        readIf(cp, "changed_min_fraction", c.components.changed_min_fraction);
        readIf(cp, "min_extra_area_ratio", c.components.min_extra_area_ratio);
        readIf(cp, "context_pad_ratio", c.components.context_pad_ratio);
    }
    return c;
}

} // namespace fve
