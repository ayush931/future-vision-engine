#include <fve/alignment.hpp>

#include <opencv2/features2d.hpp>

#include <algorithm>
#include <cmath>

namespace fve
{
namespace
{
cv::Mat toSmallGray(const cv::Mat &bgr, double scale)
{
    cv::Mat gray, small;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    if (std::abs(scale - 1.0) < 1e-9)
        return gray;
    cv::resize(gray, small, cv::Size(), scale, scale, cv::INTER_AREA);
    return small;
}

cv::Mat identityWarp() { return cv::Mat::eye(2, 3, CV_64F); }
} // namespace

Aligner::Aligner(AlignmentConfig config) : config_(config) {}

void Aligner::setReference(const cv::Mat &reference_bgr)
{
    ref_size_ = reference_bgr.size();
    scale_ = std::min(1.0, static_cast<double>(config_.analysis_width) / std::max(1, reference_bgr.cols));
    ref_small_ = toSmallGray(reference_bgr, scale_);
    cv::GaussianBlur(ref_small_, ref_blur_, {5, 5}, 0);
    ref_kp_.clear();
    ref_desc_.release();
    if (!config_.enabled)
        return;
    auto orb = cv::ORB::create(config_.max_features);
    orb->detectAndCompute(ref_small_, cv::noArray(), ref_kp_, ref_desc_);
}

AlignmentResult Aligner::estimate(const cv::Mat &frame_bgr) const
{
    AlignmentResult r;
    r.warp = identityWarp();
    if (!config_.enabled || ref_desc_.empty() || frame_bgr.size() != ref_size_)
    {
        r.ok = config_.enabled ? frame_bgr.size() == ref_size_ : true;
        return r;
    }

    const cv::Mat cur = toSmallGray(frame_bgr, scale_);
    cv::Mat cur_blur;
    cv::GaussianBlur(cur, cur_blur, {5, 5}, 0);

    // Fast path: static camera and nearly identical content need no registration.
    // Compared on blurred images so sensor noise alone never triggers RANSAC.
    cv::Mat diff;
    cv::absdiff(cur_blur, ref_blur_, diff);
    const double identity_err = cv::mean(diff)[0];
    if (identity_err < 1.5)
        return r;

    std::vector<cv::KeyPoint> kp;
    cv::Mat desc;
    auto orb = cv::ORB::create(config_.max_features);
    orb->detectAndCompute(cur, cv::noArray(), kp, desc);
    if (desc.rows < config_.min_inliers)
    {
        r.ok = false;
        return r;
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn;
    matcher.knnMatch(desc, ref_desc_, knn, 2);
    std::vector<cv::Point2f> src, dst;
    for (const auto &m : knn)
    {
        if (m.size() == 2 && m[0].distance < 0.75f * m[1].distance)
        {
            src.push_back(kp[m[0].queryIdx].pt);
            dst.push_back(ref_kp_[m[0].trainIdx].pt);
        }
    }
    if (static_cast<int>(src.size()) < config_.min_inliers)
    {
        r.ok = false;
        return r;
    }

    std::vector<uchar> inliers;
    cv::Mat m = cv::estimateAffinePartial2D(src, dst, inliers, cv::RANSAC, config_.ransac_reproj_px, 3000, 0.995);
    r.inliers = static_cast<int>(std::count(inliers.begin(), inliers.end(), 1));
    if (m.empty() || r.inliers < config_.min_inliers)
    {
        r.ok = false;
        return r;
    }

    m.convertTo(m, CV_64F);

    // Verify: RANSAC is randomised and a local change (a removed part) can bias
    // it. Only accept a warp that clearly lowers the photometric residual.
    {
        cv::Mat warped, mask, d;
        cv::warpAffine(cur_blur, warped, m, ref_blur_.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::warpAffine(cv::Mat(cur_blur.size(), CV_8UC1, cv::Scalar(255)), mask, m, ref_blur_.size(),
                       cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
        cv::Mat eroded;
        cv::erode(mask, eroded, cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));
        mask = eroded;
        cv::absdiff(warped, ref_blur_, d);
        const double warped_err = cv::mean(d, mask)[0];
        const double identity_in_mask = cv::mean(diff, mask)[0];
        if (!(warped_err < 0.85 * identity_in_mask))
            return r; // identity is as good: keep it (ok = true)
    }

    // Lift the small-image transform to full resolution: only translation scales.
    m.at<double>(0, 2) /= scale_;
    m.at<double>(1, 2) /= scale_;

    const double a = m.at<double>(0, 0), b = m.at<double>(1, 0);
    const double sc = std::hypot(a, b);
    r.rotation_deg = std::atan2(b, a) * 180.0 / CV_PI;
    r.shift_px = std::hypot(m.at<double>(0, 2), m.at<double>(1, 2));

    if (std::abs(r.rotation_deg) > config_.max_rotation_deg || std::abs(sc - 1.0) > 0.15 ||
        r.shift_px > config_.max_shift_ratio * ref_size_.width)
    {
        r.ok = false; // implausible; keep identity and report
        r.shift_px = 0.0;
        r.rotation_deg = 0.0;
        return r;
    }

    // Sub-0.3 px motion is below interpolation noise; skip the warp.
    const double corner_motion =
        std::max({std::abs(m.at<double>(0, 2)), std::abs(m.at<double>(1, 2)),
                  std::abs(a - 1.0) * ref_size_.width, std::abs(b) * ref_size_.height});
    if (corner_motion < 0.5)
        return r;

    r.warp = m;
    r.applied = true;
    return r;
}

cv::Mat Aligner::warp(const cv::Mat &frame, const AlignmentResult &a, int interpolation) const
{
    if (!a.applied)
        return frame;
    cv::Mat out;
    const cv::Size size = ref_size_.area() > 0 ? ref_size_ : frame.size();
    cv::warpAffine(frame, out, a.warp, size, interpolation, cv::BORDER_REPLICATE);
    return out;
}

cv::Mat Aligner::validMask(const AlignmentResult &a, cv::Size size) const
{
    cv::Mat mask(size, CV_8UC1, cv::Scalar(255));
    if (!a.applied)
        return mask;
    cv::Mat m = a.warp.clone();
    const double sx = static_cast<double>(size.width) / ref_size_.width;
    const double sy = static_cast<double>(size.height) / ref_size_.height;
    // warp expressed at the target scale: S * M * S^-1
    m.at<double>(0, 1) *= sx / sy;
    m.at<double>(1, 0) *= sy / sx;
    m.at<double>(0, 2) *= sx;
    m.at<double>(1, 2) *= sy;
    cv::Mat src(size, CV_8UC1, cv::Scalar(255)), out;
    cv::warpAffine(src, out, m, size, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
    cv::Mat eroded;
    cv::erode(out, eroded, cv::getStructuringElement(cv::MORPH_RECT, {5, 5}));
    return eroded;
}

cv::Point2f Aligner::toFrame(cv::Point2f p, const AlignmentResult &a) const
{
    if (!a.applied)
        return p;
    cv::Mat inv;
    cv::invertAffineTransform(a.warp, inv);
    return {static_cast<float>(inv.at<double>(0, 0) * p.x + inv.at<double>(0, 1) * p.y + inv.at<double>(0, 2)),
            static_cast<float>(inv.at<double>(1, 0) * p.x + inv.at<double>(1, 1) * p.y + inv.at<double>(1, 2))};
}

std::vector<cv::Point> Aligner::toFrame(const cv::Rect &r, const AlignmentResult &a) const
{
    const cv::Point2f corners[4] = {{static_cast<float>(r.x), static_cast<float>(r.y)},
                                    {static_cast<float>(r.x + r.width), static_cast<float>(r.y)},
                                    {static_cast<float>(r.x + r.width), static_cast<float>(r.y + r.height)},
                                    {static_cast<float>(r.x), static_cast<float>(r.y + r.height)}};
    std::vector<cv::Point> out;
    for (const auto &c : corners)
    {
        const auto p = toFrame(c, a);
        out.emplace_back(cvRound(p.x), cvRound(p.y));
    }
    return out;
}

} // namespace fve
