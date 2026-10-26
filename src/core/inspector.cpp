#include <fve/fixture.hpp>
#include <fve/inspector.hpp>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace fve
{
namespace
{
void requireBgr(const cv::Mat &m)
{
    if (m.empty() || m.type() != CV_8UC3)
        throw std::invalid_argument("expected a non-empty 8-bit BGR frame");
}

cv::Rect scaleRect(const cv::Rect &r, double s)
{
    const int x1 = static_cast<int>(std::floor(r.x * s)), y1 = static_cast<int>(std::floor(r.y * s));
    const int x2 = static_cast<int>(std::ceil((r.x + r.width) * s)), y2 = static_cast<int>(std::ceil((r.y + r.height) * s));
    return {x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1)};
}

cv::Rect expand(const cv::Rect &r, int px) { return {r.x - px, r.y - px, r.width + 2 * px, r.height + 2 * px}; }

cv::Rect expandRatio(const cv::Rect &r, double ratio)
{
    const int px = static_cast<int>(std::lround(std::max(r.width, r.height) * ratio));
    return expand(r, px);
}

cv::Rect shrinkRatio(const cv::Rect &r, double ratio)
{
    const int mx = static_cast<int>(r.width * ratio), my = static_cast<int>(r.height * ratio);
    cv::Rect out(r.x + mx, r.y + my, r.width - 2 * mx, r.height - 2 * my);
    return (out.width > 0 && out.height > 0) ? out : r;
}

/// Sub-pixel peak offset along one axis from three samples (parabola vertex).
double parabolicOffset(double l, double c, double r)
{
    const double den = l - 2.0 * c + r;
    if (std::abs(den) < 1e-9)
        return 0.0;
    return std::clamp(0.5 * (l - r) / den, -0.5, 0.5);
}

/// Fraction of pixels >= 1 inside `roi` (respecting `valid`).
double aboveThresholdFraction(const cv::Mat &map, const cv::Mat &valid, const cv::Rect &roi)
{
    cv::Mat above;
    cv::compare(map(roi), 1.0, above, cv::CMP_GE);
    cv::bitwise_and(above, valid(roi), above);
    const int n = cv::countNonZero(valid(roi));
    return n > 0 ? static_cast<double>(cv::countNonZero(above)) / n : 0.0;
}
} // namespace

Inspector::Inspector(EngineConfig config) : config_(std::move(config)), aligner_(config_.alignment) {}

Inspector::~Inspector() = default;

Backend Inspector::backend() const noexcept { return scorer_ ? scorer_->backend() : config_.backend; }

std::string Inspector::describe() const
{
    std::ostringstream os;
    os << "backend=" << (scorer_ ? scorer_->describe() : toString(config_.backend))
       << " segmentation=" << toString(ref_.segmentation) << " components=" << ref_.slots.size()
       << " work=" << ref_.work_size.width << "x" << ref_.work_size.height
       << " alignment=" << (config_.alignment.enabled ? "on" : "off");
    return os.str();
}

void Inspector::setReference(const cv::Mat &reference_bgr, const std::vector<cv::Mat> &good_frames)
{
    requireBgr(reference_bgr);
    ref_ = ReferenceModel{};
    ref_.bgr = reference_bgr.clone();
    ref_.size = reference_bgr.size();
    ref_.work_scale = std::min(1.0, static_cast<double>(config_.anomaly.work_width) / ref_.size.width);
    ref_.work_size = cv::Size(std::max(8, static_cast<int>(std::lround(ref_.size.width * ref_.work_scale))),
                              std::max(8, static_cast<int>(std::lround(ref_.size.height * ref_.work_scale))));
    ref_.work_scale = static_cast<double>(ref_.work_size.width) / ref_.size.width;
    cv::resize(ref_.bgr, ref_.work_bgr, ref_.work_size, 0, 0, cv::INTER_AREA);
    cv::cvtColor(ref_.work_bgr, ref_.work_gray, cv::COLOR_BGR2GRAY);

    // Components: explicit config wins, otherwise discover on the reference.
    if (!config_.slots.empty())
    {
        ref_.slots = config_.slots;
        ref_.segmentation = SegmentationMode::Fixture;
        ref_.board = config_.board;
        cv::Rect board;
        if (!ref_.board.valid)
        {
            const auto [lo, hi] = boardThresholdsFromMedian(centerMedianHsv(ref_.bgr));
            ref_.board.hsv_lower = lo;
            ref_.board.hsv_upper = hi;
            ref_.board.valid = true;
        }
        cv::Mat hsv;
        cv::cvtColor(ref_.bgr, hsv, cv::COLOR_BGR2HSV);
        ref_.board_bbox = findBoardBbox(hsv, ref_.board.hsv_lower, ref_.board.hsv_upper,
                                        cv::Rect(0, 0, ref_.size.width, ref_.size.height));
    }
    else
    {
        auto d = discoverComponents(ref_.bgr, config_.segmentation, config_.board);
        ref_.slots = std::move(d.slots);
        ref_.board = d.board;
        ref_.board_bbox = d.board_bbox;
        ref_.segmentation = d.used;
    }

    const cv::Rect frame_rect(0, 0, ref_.size.width, ref_.size.height);
    std::vector<SlotDefinition> kept;
    for (auto &s : ref_.slots)
    {
        s.bounds &= frame_rect;
        if (s.bounds.width >= 2 && s.bounds.height >= 2)
            kept.push_back(s);
    }
    ref_.slots = std::move(kept);

    cv::Mat ref_hsv;
    cv::cvtColor(ref_.bgr, ref_hsv, cv::COLOR_BGR2HSV);
    ref_.slot_ref_board_ratio.clear();
    ref_.slot_search_radius_work.clear();
    for (std::size_t i = 0; i < ref_.slots.size(); ++i)
    {
        const auto &b = ref_.slots[i].bounds;
        ref_.slot_ref_board_ratio.push_back(ref_.board.valid ? boardRatio(ref_hsv, b, ref_.board) : 1.0);

        // Search radius: a share of the slot size, but below half the distance to
        // the nearest neighbour so a repeated part can never match its neighbour.
        double radius = config_.components.search_ratio * std::max(b.width, b.height);
        const cv::Point2d ci(b.x + b.width * 0.5, b.y + b.height * 0.5);
        for (std::size_t j = 0; j < ref_.slots.size(); ++j)
        {
            if (j == i)
                continue;
            const auto &o = ref_.slots[j].bounds;
            const cv::Point2d cj(o.x + o.width * 0.5, o.y + o.height * 0.5);
            radius = std::min(radius, 0.45 * std::hypot(ci.x - cj.x, ci.y - cj.y));
        }
        ref_.slot_search_radius_work.push_back(std::max(3, static_cast<int>(std::lround(radius * ref_.work_scale))));
    }

    // Inspection area: the board when one was found, minus ignore regions.
    ref_.ignore_mask_work = cv::Mat::zeros(ref_.work_size, CV_8UC1);
    const bool board_found = ref_.segmentation != SegmentationMode::None && !ref_.board_bbox.empty() &&
                             ref_.board_bbox.area() < frame_rect.area() &&
                             ref_.board_bbox.area() > 0.15 * frame_rect.area();
    if (board_found)
    {
        ref_.ignore_mask_work.setTo(255);
        const cv::Rect area = scaleRect(expandRatio(ref_.board_bbox, 0.02), ref_.work_scale) &
                              cv::Rect(0, 0, ref_.work_size.width, ref_.work_size.height);
        ref_.ignore_mask_work(area).setTo(0);
    }
    for (const auto &r : config_.ignore_regions)
    {
        const cv::Rect w = scaleRect(r, ref_.work_scale) & cv::Rect(0, 0, ref_.work_size.width, ref_.work_size.height);
        if (!w.empty())
            ref_.ignore_mask_work(w).setTo(255);
    }

    aligner_ = Aligner(config_.alignment);
    aligner_.setReference(ref_.bgr);

    std::vector<cv::Mat> good_work;
    for (const auto &g : good_frames)
    {
        if (g.size() != ref_.size)
            continue;
        const auto a = aligner_.estimate(g);
        cv::Mat w;
        cv::resize(aligner_.warp(g, a), w, ref_.work_size, 0, 0, cv::INTER_AREA);
        good_work.push_back(w);
    }
    scorer_ = std::make_unique<CalibratedScorer>(config_, config_.backend);
    {
        // The model only needs the inspected area (board), not the bench around it.
        cv::Mat inspected;
        cv::bitwise_not(ref_.ignore_mask_work, inspected);
        const cv::Rect roi = cv::boundingRect(inspected);
        scorer_->setRoi(roi.area() > 0 ? roi : cv::Rect(0, 0, ref_.work_size.width, ref_.work_size.height));
    }
    scorer_->setReference(ref_.work_bgr, good_work);
}

FrameResult Inspector::referenceResult(int frame_index, double fps) const
{
    FrameResult r;
    r.frame_index = frame_index;
    r.timestamp_sec = fps > 0.0 ? frame_index / fps : 0.0;
    r.is_reference = true;
    r.is_correct = true;
    r.alignment.warp = cv::Mat::eye(2, 3, CV_64F);
    for (const auto &s : ref_.slots)
    {
        SlotResult sr;
        sr.slot_id = s.id;
        sr.type = s.type;
        sr.bounds = s.bounds;
        sr.matched_bounds = s.bounds;
        sr.ncc = 1.0;
        sr.confidence = 1.0;
        r.slots.push_back(sr);
    }
    r.ok_count = r.slots.size();
    r.anomaly_map = cv::Mat::zeros(ref_.work_size, CV_32F);
    return r;
}

FrameResult Inspector::inspect(const cv::Mat &frame_in, int frame_index, double fps) const
{
    requireBgr(frame_in);
    if (!scorer_)
        throw std::logic_error("Inspector::setReference must be called first");
    const auto t0 = std::chrono::steady_clock::now();

    cv::Mat frame = frame_in;
    if (frame.size() != ref_.size)
        cv::resize(frame_in, frame, ref_.size, 0, 0, cv::INTER_AREA);

    FrameResult out;
    out.frame_index = frame_index;
    out.timestamp_sec = fps > 0.0 ? frame_index / fps : 0.0;

    out.alignment = aligner_.estimate(frame);
    const cv::Mat aligned = aligner_.warp(frame, out.alignment);
    cv::Mat aligned_work, aligned_gray;
    cv::resize(aligned, aligned_work, ref_.work_size, 0, 0, cv::INTER_AREA);
    cv::cvtColor(aligned_work, aligned_gray, cv::COLOR_BGR2GRAY);

    cv::Mat valid = aligner_.validMask(out.alignment, ref_.work_size);
    valid.setTo(0, ref_.ignore_mask_work);

    cv::Mat map = scorer_->normalizedMap(aligned_work);
    map.setTo(0.0f, valid == 0);

    cv::Mat gdiff;
    cv::absdiff(aligned_gray, ref_.work_gray, gdiff);
    out.mean_diff = cv::mean(gdiff, valid)[0];
    const int valid_n = std::max(1, cv::countNonZero(valid));
    out.changed_pct = 100.0 * cv::countNonZero(map >= 1.0f) / valid_n;
    out.anomaly_score = topPercentMean(map, config_.anomaly.top_percent, valid);

    analyseSlots(aligned, aligned_gray, map, valid, out);
    detectExtras(aligned, map, valid, out);

    for (const auto &s : out.slots)
    {
        switch (s.deviation)
        {
        case Deviation::None:
            ++out.ok_count;
            break;
        case Deviation::Missing:
            ++out.missing_count;
            break;
        case Deviation::Shifted:
            ++out.shifted_count;
            out.max_shift_px = std::max(out.max_shift_px, s.shift());
            break;
        case Deviation::Changed:
            ++out.changed_count;
            break;
        case Deviation::Extra:
            break;
        }
    }
    out.is_correct = out.deviatedCount() == 0;
    out.anomaly_map = map;
    out.processing_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

void Inspector::analyseSlots(const cv::Mat &aligned, const cv::Mat &gray, const cv::Mat &map, const cv::Mat &valid,
                             FrameResult &out) const
{
    if (ref_.slots.empty())
        return;
    const auto &cc = config_.components;
    const double s = ref_.work_scale;
    const cv::Rect work_rect(0, 0, ref_.work_size.width, ref_.work_size.height);

    cv::Mat hsv;
    if (ref_.board.valid)
        cv::cvtColor(aligned, hsv, cv::COLOR_BGR2HSV);

    cv::Mat cur_lab, ref_lab; // work resolution, built on first use

    for (std::size_t i = 0; i < ref_.slots.size(); ++i)
    {
        const auto &slot = ref_.slots[i];
        SlotResult sr;
        sr.slot_id = slot.id;
        sr.type = slot.type;
        sr.bounds = slot.bounds;
        sr.matched_bounds = slot.bounds;

        const cv::Rect bw = scaleRect(slot.bounds, s) & work_rect;
        const cv::Rect inner = shrinkRatio(bw, 0.10);
        cv::Mat inner_valid = valid(inner);
        if (cv::countNonZero(inner_valid) == 0)
        {
            sr.confidence = 0.0; // slot fully outside the inspected area; cannot judge
            out.slots.push_back(sr);
            continue;
        }
        cv::Mat masked;
        map(inner).copyTo(masked);
        sr.anomaly = topPercentMean(masked, 10.0, inner_valid);
        sr.anomaly_fraction = aboveThresholdFraction(map, valid, inner);

        // Template match with context padding: board edges around a uniform part
        // give the correlation surface a well-defined peak.
        const int pad = std::max(3, static_cast<int>(cc.context_pad_ratio * std::max(bw.width, bw.height)));
        const cv::Rect tpl_rect = expand(bw, pad) & work_rect;
        const int radius = ref_.slot_search_radius_work[i];
        const cv::Rect search = expand(tpl_rect, radius) & work_rect;
        const cv::Mat tpl = ref_.work_gray(tpl_rect);
        cv::Scalar tmu, tsd;
        cv::meanStdDev(tpl, tmu, tsd);
        double ncc0 = 0.0, best = 0.0;
        cv::Point zero(tpl_rect.x - search.x, tpl_rect.y - search.y), loc = zero;
        double subx = 0.0, suby = 0.0;
        if (tsd[0] >= 2.0 && search.width >= tpl_rect.width && search.height >= tpl_rect.height)
        {
            cv::Mat res;
            cv::matchTemplate(gray(search), tpl, res, cv::TM_CCOEFF_NORMED);
            cv::patchNaNs(res, -1.0);
            ncc0 = res.at<float>(zero);
            cv::minMaxLoc(res, nullptr, &best, nullptr, &loc);
            if (best - ncc0 < 0.05) // prefer "not moved" unless clearly better elsewhere
            {
                loc = zero;
                best = ncc0;
            }
            if (loc.x > 0 && loc.x < res.cols - 1)
                subx = parabolicOffset(res.at<float>(loc.y, loc.x - 1), res.at<float>(loc), res.at<float>(loc.y, loc.x + 1));
            if (loc.y > 0 && loc.y < res.rows - 1)
                suby = parabolicOffset(res.at<float>(loc.y - 1, loc.x), res.at<float>(loc), res.at<float>(loc.y + 1, loc.x));
            if (loc == zero)
                subx = suby = 0.0;
        }
        else
        {
            best = ncc0 = sr.anomaly < 1.0 ? 1.0 : 0.0; // texture-less: rely on the anomaly map
        }
        sr.ncc = best;
        sr.dx = (loc.x + subx - zero.x) / s;
        sr.dy = (loc.y + suby - zero.y) / s;
        sr.matched_bounds = slot.bounds + cv::Point(static_cast<int>(std::lround(sr.dx)), static_cast<int>(std::lround(sr.dy)));

        const double tol = cc.shift_tolerance_px > 0.0
                               ? cc.shift_tolerance_px
                               : std::max(3.0, 0.10 * std::min(slot.bounds.width, slot.bounds.height));
        const double shift = sr.shift();
        const bool matched = best >= cc.match_min_ncc;
        const bool shifted = matched && shift > tol && (sr.anomaly >= 0.75 || shift >= 2.0 * tol);
        const bool changed = sr.anomaly >= 1.0 && sr.anomaly_fraction >= cc.changed_min_fraction;

        const bool board_usable = ref_.board.valid && ref_.slot_ref_board_ratio[i] < ref_.board.missing_threshold * 0.75;
        if (board_usable)
        {
            sr.board_ratio = boardRatio(hsv, slot.bounds, ref_.board);
            if (sr.board_ratio >= ref_.board.missing_threshold)
            {
                const bool moved_there =
                    shifted && boardRatio(hsv, sr.matched_bounds, ref_.board) < ref_.board.missing_threshold;
                sr.deviation = moved_there ? Deviation::Shifted : Deviation::Missing;
            }
            else if (shifted)
                sr.deviation = Deviation::Shifted;
            else if (changed)
                sr.deviation = Deviation::Changed;
        }
        else
        {
            if (shifted)
                sr.deviation = Deviation::Shifted;
            else if (changed)
            {
                // Generic "missing": the slot now looks like the background ring
                // that surrounded it in the reference.
                const cv::Rect ring = expand(bw, pad) & work_rect;
                cv::Mat ring_mask(ref_.work_size, CV_8UC1, cv::Scalar(0));
                ring_mask(ring).setTo(255);
                ring_mask(bw).setTo(0);
                if (cur_lab.empty())
                {
                    cv::Mat cur_work;
                    cv::resize(aligned, cur_work, ref_.work_size, 0, 0, cv::INTER_AREA);
                    cv::cvtColor(cur_work, cur_lab, cv::COLOR_BGR2Lab);
                    cv::cvtColor(ref_.work_bgr, ref_lab, cv::COLOR_BGR2Lab);
                }
                const cv::Scalar ring_mean = cv::mean(ref_lab(ring), ring_mask(ring));
                cv::Scalar cur_mean, cur_sd;
                cv::meanStdDev(cur_lab(inner), cur_mean, cur_sd);
                const double de = std::sqrt(std::pow(ring_mean[0] - cur_mean[0], 2) + std::pow(ring_mean[1] - cur_mean[1], 2) +
                                            std::pow(ring_mean[2] - cur_mean[2], 2));
                sr.deviation = (de < 15.0 && cur_sd[0] < 18.0) ? Deviation::Missing : Deviation::Changed;
            }
        }
        sr.is_present = sr.deviation != Deviation::Missing;
        sr.confidence = sr.deviation == Deviation::None ? std::clamp(1.0 - 0.5 * sr.anomaly, 0.5, 1.0)
                                                        : std::clamp(0.5 + 0.25 * sr.anomaly, 0.5, 1.0);
        out.slots.push_back(sr);
    }
}

void Inspector::detectExtras(const cv::Mat &aligned, const cv::Mat &map, const cv::Mat &valid, FrameResult &out) const
{
    const double s = ref_.work_scale;
    cv::Mat bin;
    cv::compare(map, 1.0, bin, cv::CMP_GE);
    cv::bitwise_and(bin, valid, bin);
    cv::morphologyEx(bin.clone(), bin, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));
    cv::morphologyEx(bin.clone(), bin, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));

    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(bin, labels, stats, centroids, 8, CV_32S);
    const double min_area = std::max(6.0, config_.components.min_extra_area_ratio * ref_.work_size.area());

    std::vector<cv::Rect> explained;
    for (const auto &sr : out.slots)
    {
        explained.push_back(expandRatio(sr.bounds, 0.15));
        if (sr.deviation == Deviation::Shifted)
            explained.push_back(expandRatio(sr.matched_bounds, 0.15));
    }

    std::vector<cv::Rect> extras;
    for (int i = 1; i < n; ++i)
    {
        if (stats.at<int>(i, cv::CC_STAT_AREA) < min_area)
            continue;
        const cv::Rect bw(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                          stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
        const cv::Rect full = scaleRect(bw, 1.0 / s) & cv::Rect(0, 0, ref_.size.width, ref_.size.height);
        double covered = 0.0;
        for (const auto &e : explained)
            covered = std::max(covered, overlapRatio(full, e));
        if (covered < 0.5)
            extras.push_back(full);
    }

    // On a fixture board, tighten extra boxes to detected white parts.
    if (!extras.empty() && ref_.segmentation == SegmentationMode::Fixture)
    {
        const auto parts = mergeBoxesNms(detectWhiteBoxes(aligned), 0.30);
        for (auto &e : extras)
        {
            for (const auto &p : parts)
            {
                bool is_slot = false;
                for (const auto &sl : ref_.slots)
                    is_slot = is_slot || boxIoU(p, sl.bounds) >= 0.30;
                if (!is_slot && overlapRatio(p, e) >= 0.5)
                {
                    e = p;
                    break;
                }
            }
        }
    }
    out.extra_boxes = mergeBoxesNms(extras, 0.20);
}

} // namespace fve
