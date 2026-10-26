#pragma once

// Deterministic synthetic fixture scenes with known ground truth, shared by the
// unit tests and the fve_synth video generator.

#include <opencv2/imgproc.hpp>
#if __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>
#endif

#include <string>
#include <vector>

namespace fve::synth
{

struct Scene
{
    cv::Size size{1280, 720};
    int rows{4}, cols{5};
    std::vector<cv::Rect> parts; // reference part positions, row-major
};

inline Scene makeScene()
{
    Scene s;
    const int bx = 260, by = 120, pitch_x = 160, pitch_y = 130, w = 70, h = 56;
    for (int r = 0; r < s.rows; ++r)
        for (int c = 0; c < s.cols; ++c)
            s.parts.emplace_back(bx + 60 + c * pitch_x, by + 50 + r * pitch_y, w, h);
    return s;
}

struct Edit
{
    int remove{-1};        // index of part to remove
    int shift{-1};         // index of part to shift
    cv::Point shift_by{0, 0};
    int damage{-1};        // index of part to mark with a dark scratch
    bool extra{false};     // add a part between slots
    double gain{1.0};      // global lighting gain
    cv::Point2f camera{0, 0}; // global translation (camera shake)
    double camera_rot{0.0};
    int noise_seed{0};
};

inline cv::Mat render(const Scene &s, const Edit &e)
{
    cv::Mat img(s.size, CV_8UC3, cv::Scalar(60, 60, 60)); // dark bench
    // Tan board with subtle texture so registration has features.
    const cv::Rect board(260, 120, 820, 540);
    cv::rectangle(img, board, cv::Scalar(120, 170, 200), cv::FILLED);
    cv::RNG tex(1234);
    for (int i = 0; i < 400; ++i)
    {
        const cv::Point p(board.x + tex.uniform(5, board.width - 5), board.y + tex.uniform(5, board.height - 5));
        cv::circle(img, p, tex.uniform(1, 3), cv::Scalar(105, 155, 185), cv::FILLED);
    }
    cv::rectangle(img, board, cv::Scalar(70, 110, 140), 3);
    for (int i = 0; i < 6; ++i)
        cv::circle(img, {board.x + 20 + i * 150, board.y + 18}, 6, cv::Scalar(40, 40, 40), cv::FILLED);

    for (int i = 0; i < static_cast<int>(s.parts.size()); ++i)
    {
        if (i == e.remove)
            continue;
        cv::Rect r = s.parts[i];
        if (i == e.shift)
            r += e.shift_by;
        cv::rectangle(img, r, cv::Scalar(235, 235, 232), cv::FILLED);
        cv::rectangle(img, r, cv::Scalar(150, 150, 150), 1);
        cv::circle(img, {r.x + r.width / 2, r.y + r.height / 2}, 8, cv::Scalar(90, 90, 90), 2);
        if (i == e.damage)
            cv::line(img, {r.x + 8, r.y + 8}, {r.x + r.width - 10, r.y + r.height - 12}, cv::Scalar(40, 40, 60), 6);
    }
    if (e.extra)
    {
        // Between the first two parts of the second row.
        const cv::Rect a = s.parts[s.cols], b = s.parts[s.cols + 1];
        const cv::Rect x((a.x + a.width + b.x) / 2 - 22, a.y + 4, 44, 44);
        cv::rectangle(img, x, cv::Scalar(235, 235, 232), cv::FILLED);
    }

    if (e.gain != 1.0)
        img.convertTo(img, -1, e.gain, 0);
    if (e.camera.x != 0 || e.camera.y != 0 || e.camera_rot != 0)
    {
        cv::Mat m = cv::getRotationMatrix2D({s.size.width / 2.f, s.size.height / 2.f}, e.camera_rot, 1.0);
        m.at<double>(0, 2) += e.camera.x;
        m.at<double>(1, 2) += e.camera.y;
        cv::warpAffine(img, img, m, s.size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    }
    cv::RNG rng(0xC0FFEE + e.noise_seed);
    cv::Mat noise(s.size, CV_16SC3);
    rng.fill(noise, cv::RNG::NORMAL, cv::Scalar::all(0.0), cv::Scalar::all(2.5));
    cv::Mat img16;
    img.convertTo(img16, CV_16SC3);
    img16 += noise;
    img16.convertTo(img, CV_8UC3);
    return img;
}

/// Labelled timeline used by fve_synth and the end-to-end test.
struct Segment
{
    std::string label;
    int frames;
    Edit edit;
    bool expect_correct;
};

inline std::vector<Segment> timeline()
{
    std::vector<Segment> t;
    t.push_back({"baseline", 30, {}, true});
    Edit e;
    e.remove = 6;
    t.push_back({"missing SLOT_07", 20, e, false});
    e = {};
    e.shift = 11;
    e.shift_by = {18, 6};
    t.push_back({"shifted SLOT_12 by (18,6)px", 20, e, false});
    e = {};
    e.extra = true;
    t.push_back({"extra part", 20, e, false});
    e = {};
    e.camera = {5.0f, -4.0f};
    e.camera_rot = 0.6;
    t.push_back({"camera shake (should be correct)", 20, e, true});
    e = {};
    e.damage = 2;
    t.push_back({"damaged SLOT_03", 20, e, false});
    e = {};
    e.gain = 0.88;
    t.push_back({"lighting -12% (should be correct)", 20, e, true});
    t.push_back({"baseline again", 20, {}, true});
    return t;
}

} // namespace fve::synth
