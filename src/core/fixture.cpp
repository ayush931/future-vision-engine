#include <fve/fixture.hpp>

#include <fve/cv_compat.hpp>

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace fve
{
namespace
{
void requireBgr(const cv::Mat &m)
{
    if (m.empty() || m.type() != CV_8UC3)
        throw std::invalid_argument("expected a non-empty 8-bit BGR image");
}

double median8u(const cv::Mat &channel)
{
    // Histogram median: O(n) and exact for 8-bit data.
    int hist[256] = {0};
    for (int y = 0; y < channel.rows; ++y)
    {
        const uchar *p = channel.ptr<uchar>(y);
        for (int x = 0; x < channel.cols; ++x)
            ++hist[p[x]];
    }
    const long half = static_cast<long>(channel.total()) / 2;
    long acc = 0;
    for (int v = 0; v < 256; ++v)
    {
        acc += hist[v];
        if (acc > half)
            return v;
    }
    return 255;
}

std::vector<SlotDefinition> toSlots(const std::vector<cv::Rect> &boxes, const char *description)
{
    std::vector<SlotDefinition> slots;
    slots.reserve(boxes.size());
    for (std::size_t i = 0; i < boxes.size(); ++i)
    {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "SLOT_%02zu", i + 1);
        slots.push_back({buf, "component", boxes[i], std::string(description) + " " + std::to_string(i + 1)});
    }
    return slots;
}

void sortRowMajor(std::vector<cv::Rect> &boxes)
{
    std::sort(boxes.begin(), boxes.end(), [](const cv::Rect &a, const cv::Rect &b) {
        const int ra = a.y / 50, rb = b.y / 50;
        return ra != rb ? ra < rb : a.x < b.x;
    });
}
} // namespace

cv::Scalar centerMedianHsv(const cv::Mat &bgr)
{
    requireBgr(bgr);
    const int x1 = static_cast<int>(bgr.cols * 0.20), x2 = static_cast<int>(bgr.cols * 0.80);
    const int y1 = static_cast<int>(bgr.rows * 0.20), y2 = static_cast<int>(bgr.rows * 0.80);
    cv::Mat hsv;
    cv::cvtColor(bgr(cv::Rect(x1, y1, std::max(1, x2 - x1), std::max(1, y2 - y1))), hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> ch;
    cv::split(hsv, ch);
    return {median8u(ch[0]), median8u(ch[1]), median8u(ch[2])};
}

std::pair<cv::Scalar, cv::Scalar> boardThresholdsFromMedian(const cv::Scalar &m)
{
    return {cv::Scalar(std::max(0.0, m[0] - 15.0), std::max(30.0, m[1] - 50.0), std::max(40.0, m[2] - 70.0)),
            cv::Scalar(std::min(180.0, m[0] + 15.0), std::min(255.0, m[1] + 60.0), std::min(255.0, m[2] + 60.0))};
}

cv::Rect findBoardBbox(const cv::Mat &hsv, const cv::Scalar &lower, const cv::Scalar &upper,
                       const cv::Rect &fallback)
{
    cv::Mat mask;
    cv::inRange(hsv, lower, upper, mask);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double best = 0.0;
    const std::vector<cv::Point> *largest = nullptr;
    for (const auto &c : contours)
    {
        const double a = cv::contourArea(c);
        if (a > best)
        {
            best = a;
            largest = &c;
        }
    }
    if (largest && best > 0.15 * hsv.cols * hsv.rows)
        return cv::boundingRect(*largest);
    return fallback;
}

double boxIoU(const cv::Rect &a, const cv::Rect &b)
{
    const double inter = (a & b).area();
    if (inter <= 0.0)
        return 0.0;
    return inter / (std::max(1, a.area()) + std::max(1, b.area()) - inter);
}

double overlapRatio(const cv::Rect &a, const cv::Rect &b)
{
    return static_cast<double>((a & b).area()) / std::max(1, a.area());
}

std::vector<cv::Rect> mergeBoxesNms(const std::vector<cv::Rect> &boxes, double iou_thresh)
{
    std::vector<cv::Rect> remaining = boxes;
    std::sort(remaining.begin(), remaining.end(), [](const cv::Rect &a, const cv::Rect &b) { return a.area() > b.area(); });
    std::vector<cv::Rect> merged;
    for (const auto &box : remaining)
    {
        bool matched = false;
        for (auto &kept : merged)
        {
            if (boxIoU(box, kept) >= iou_thresh)
            {
                kept |= box;
                matched = true;
                break;
            }
        }
        if (!matched)
            merged.push_back(box);
    }
    sortRowMajor(merged);
    return merged;
}

std::vector<cv::Rect> detectWhiteBoxes(const cv::Mat &bgr, cv::Scalar *out_median, cv::Rect *out_board)
{
    requireBgr(bgr);
    const int w = bgr.cols, h = bgr.rows;
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    const cv::Scalar med = centerMedianHsv(bgr);
    if (out_median)
        *out_median = med;

    const auto [lo, hi] = boardThresholdsFromMedian(med);
    const cv::Rect fallback(static_cast<int>(w * 0.15), static_cast<int>(h * 0.10), static_cast<int>(w * 0.70),
                            static_cast<int>(h * 0.80));
    const cv::Rect board = findBoardBbox(hsv, lo, hi, fallback);
    if (out_board)
        *out_board = board;

    cv::Mat white;
    cv::inRange(hsv, cv::Scalar(0, 0, 90), cv::Scalar(180, 60, 255), white);
    cv::Mat roi_mask = cv::Mat::zeros(white.size(), CV_8UC1);
    const int mx = std::max(4, static_cast<int>(board.width * 0.03));
    const int my = std::max(4, static_cast<int>(board.height * 0.03));
    const cv::Rect inner = cv::Rect(board.x + mx, board.y + my, std::max(1, board.width - 2 * mx),
                                    std::max(1, board.height - 2 * my)) &
                           cv::Rect(0, 0, w, h);
    if (!inner.empty())
        roi_mask(inner).setTo(255);
    cv::Mat active;
    cv::bitwise_and(white, roi_mask, active);
    cv::morphologyEx(active.clone(), active, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, {3, 3}));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(active, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    const double max_area = static_cast<double>(board.area()) * 0.08;
    std::vector<cv::Rect> boxes;
    for (const auto &c : contours)
    {
        const cv::Rect b = cv::boundingRect(c);
        const double area = b.area();
        const double aspect = static_cast<double>(b.width) / std::max(1, b.height);
        if (area < 400.0 || area > max_area || aspect < 0.20 || aspect > 5.0)
            continue;
        if (b.width > board.width * 0.45 || b.height > board.height * 0.45)
            continue;
        boxes.push_back(b);
    }
    return boxes;
}

std::vector<cv::Rect> detectForegroundComponents(const cv::Mat &bgr, cv::Rect *out_board)
{
    requireBgr(bgr);
    const int w = bgr.cols, h = bgr.rows;

    // Background colour = median of the central crop, matching the board heuristic.
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    const auto med = centerMedianHsv(bgr);
    const auto [lo, hi] = boardThresholdsFromMedian(med);
    const cv::Rect board = findBoardBbox(hsv, lo, hi, cv::Rect(0, 0, w, h));
    if (out_board)
        *out_board = board;

    cv::Mat lab;
    cv::cvtColor(bgr, lab, cv::COLOR_BGR2Lab);
    cv::Mat lab_blur;
    cv::GaussianBlur(lab, lab_blur, {5, 5}, 0);
    cv::Mat center_hsv(1, 1, CV_8UC3, cv::Scalar(med[0], med[1], med[2]));
    cv::Mat center_bgr, center_lab;
    cv::cvtColor(center_hsv, center_bgr, cv::COLOR_HSV2BGR);
    cv::cvtColor(center_bgr, center_lab, cv::COLOR_BGR2Lab);
    const cv::Vec3b bg = center_lab.at<cv::Vec3b>(0, 0);

    cv::Mat dist(h, w, CV_8UC1);
    for (int y = 0; y < h; ++y)
    {
        const cv::Vec3b *p = lab_blur.ptr<cv::Vec3b>(y);
        uchar *d = dist.ptr<uchar>(y);
        for (int x = 0; x < w; ++x)
        {
            const int dl = p[x][0] - bg[0], da = p[x][1] - bg[1], db = p[x][2] - bg[2];
            d[x] = cv::saturate_cast<uchar>(std::sqrt(static_cast<double>(dl * dl + da * da + db * db)));
        }
    }
    cv::Mat fg;
    const double otsu = cv::threshold(dist(board), fg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::threshold(dist, fg, std::max(otsu, 18.0), 255, cv::THRESH_BINARY);
    cv::Mat board_mask = cv::Mat::zeros(fg.size(), CV_8UC1);
    const int m = std::max(3, std::min(board.width, board.height) / 50);
    const cv::Rect inner = cv::Rect(board.x + m, board.y + m, board.width - 2 * m, board.height - 2 * m) &
                           cv::Rect(0, 0, w, h);
    if (!inner.empty())
        board_mask(inner).setTo(255);
    cv::bitwise_and(fg, board_mask, fg);
    cv::morphologyEx(fg.clone(), fg, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, {5, 5}));
    cv::morphologyEx(fg.clone(), fg, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, {7, 7}));

    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(fg, labels, stats, centroids, 8, CV_32S);
    const double min_area = std::max(200.0, 0.0004 * w * h);
    const double max_area = 0.25 * board.area();
    std::vector<cv::Rect> boxes;
    for (int i = 1; i < n; ++i)
    {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area < min_area || area > max_area)
            continue;
        boxes.emplace_back(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                           stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT));
    }
    return mergeBoxesNms(boxes, 0.30);
}

std::vector<SlotDefinition> gridSlots(cv::Size size, std::size_t slot_count)
{
    if (size.width < 2 || size.height < 2 || slot_count == 0)
        throw std::invalid_argument("frame size and slot count must be positive");
    std::vector<SlotDefinition> slots;
    const int bx = static_cast<int>(size.width * 0.15), by = static_cast<int>(size.height * 0.10);
    const int bw = static_cast<int>(size.width * 0.70), bh = static_cast<int>(size.height * 0.80);
    const int x_min = bx + 6, y_min = by + 6;
    const int step_x = std::max(10, (bw - 12) / 5), step_y = std::max(10, (bh - 12) / 5);
    for (int r = 0; r < 5 && slots.size() < slot_count; ++r)
        for (int c = 0; c < 5 && slots.size() < slot_count; ++c)
        {
            const int x1 = x_min + c * step_x + 4, y1 = y_min + r * step_y + 4;
            const int x2 = std::min(size.width - 2, x1 + step_x - 8), y2 = std::min(size.height - 2, y1 + step_y - 8);
            char id[32];
            std::snprintf(id, sizeof(id), "SLOT_%02zu", slots.size() + 1);
            slots.push_back({id, "grid_roi", cv::Rect(x1, y1, std::max(2, x2 - x1), std::max(2, y2 - y1)),
                             std::string("Grid slot ") + id});
        }
    return slots;
}

DiscoveryResult discoverComponents(const cv::Mat &ref, SegmentationMode mode, const BoardParams &base)
{
    requireBgr(ref);
    DiscoveryResult out;
    out.board = base;

    if (mode == SegmentationMode::None)
    {
        out.used = SegmentationMode::None;
        out.board_bbox = cv::Rect(0, 0, ref.cols, ref.rows);
        return out;
    }

    if (mode == SegmentationMode::Auto || mode == SegmentationMode::Fixture)
    {
        cv::Scalar med;
        cv::Rect board;
        const auto merged = mergeBoxesNms(detectWhiteBoxes(ref, &med, &board), 0.30);
        out.board_bbox = board;
        if (!merged.empty() || mode == SegmentationMode::Fixture)
        {
            const auto [lo, hi] = boardThresholdsFromMedian(med);
            if (!base.valid)
            {
                out.board.hsv_lower = lo;
                out.board.hsv_upper = hi;
            }
            out.board.valid = true;
            out.slots = toSlots(merged, "Auto-detected component");
            out.used = SegmentationMode::Fixture;
            if (!out.slots.empty())
                return out;
        }
    }

    if (mode == SegmentationMode::Auto || mode == SegmentationMode::Foreground)
    {
        cv::Rect board;
        const auto boxes = detectForegroundComponents(ref, &board);
        out.board_bbox = board;
        if (!boxes.empty() || mode == SegmentationMode::Foreground)
        {
            out.slots = toSlots(boxes, "Foreground component");
            out.used = SegmentationMode::Foreground;
            if (!out.slots.empty())
                return out;
        }
    }

    out.slots = gridSlots(ref.size());
    out.used = SegmentationMode::Grid;
    if (out.board_bbox.empty())
        out.board_bbox = cv::Rect(0, 0, ref.cols, ref.rows);
    return out;
}

double boardRatio(const cv::Mat &hsv, const cv::Rect &roi_in, const BoardParams &board)
{
    const cv::Rect frame(0, 0, hsv.cols, hsv.rows);
    cv::Rect bounds = roi_in & frame;
    if (bounds.empty())
        return 1.0;
    const int mx = std::max(2, static_cast<int>(bounds.width * board.margin_ratio));
    const int my = std::max(2, static_cast<int>(bounds.height * board.margin_ratio));
    cv::Rect roi(bounds.x + mx, bounds.y + my, bounds.width - 2 * mx, bounds.height - 2 * my);
    if (roi.width <= 0 || roi.height <= 0)
        roi = bounds;
    cv::Mat mask;
    cv::inRange(hsv(roi), board.hsv_lower, board.hsv_upper, mask);
    return static_cast<double>(cv::countNonZero(mask)) / static_cast<double>(roi.area());
}

} // namespace fve
