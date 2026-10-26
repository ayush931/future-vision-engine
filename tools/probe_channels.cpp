// Developer probe: prints raw and normalised per-channel anomaly statistics for
// a region of one frame, to tune fusion and floors.
//   fve_probe <video> <frame> <x> <y> <w> <h> [model]
#include <fve/anomaly.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
int main(int argc, char **argv)
{
    if (argc < 7) { std::cerr << "usage: fve_probe <video> <frame> <x> <y> <w> <h> [model]\n"; return 2; }
    cv::VideoCapture cap(argv[1]);
    cv::Mat ref, f, cur;
    cap.read(ref);
    for (int i = 1; i <= std::stoi(argv[2]) && cap.read(f); ++i) cur = f;
    const double s = 640.0 / ref.cols;
    cv::Mat rw, cw;
    cv::resize(ref, rw, {}, s, s, cv::INTER_AREA);
    cv::resize(cur, cw, rw.size(), 0, 0, cv::INTER_AREA);
    cv::Rect roi(int(std::stoi(argv[3]) * s), int(std::stoi(argv[4]) * s), int(std::stoi(argv[5]) * s), int(std::stoi(argv[6]) * s));
    std::vector<std::unique_ptr<fve::AnomalyScorer>> sc;
    sc.push_back(std::make_unique<fve::ClassicalScorer>());
    if (argc > 7) { fve::DinoConfig d; d.model_path = argv[7]; sc.push_back(std::make_unique<fve::DinoScorer>(d)); }
    if (argc > 7)
    {
        for (auto b : {fve::Backend::Classical, fve::Backend::Dino, fve::Backend::Hybrid})
        {
            fve::EngineConfig cfg;
            cfg.dino.model_path = argv[7];
            fve::CalibratedScorer cs(cfg, b);
            cs.setReference(rw);
            const cv::Mat m = cs.normalizedMap(cw);
            double mn, mx;
            cv::minMaxLoc(m(roi), &mn, &mx);
            std::cout << "normalized " << fve::toString(b) << " roi mean " << cv::mean(m(roi))[0] << " max " << mx
                      << " top10% " << fve::topPercentMean(m(roi).clone(), 10.0) << "\n";
        }
    }
    for (auto &x : sc)
    {
        x->setReference(rw);
        for (const auto &c : x->score(cw))
        {
            double mn, mx; cv::minMaxLoc(c.map(roi), &mn, &mx);
            std::cout << c.name << " roi mean " << cv::mean(c.map(roi))[0] << " max " << mx
                      << " | frame top1% " << fve::topPercentMean(c.map, 1.0) << " median-ish mean " << cv::mean(c.map)[0] << "\n";
        }
    }
}
