// Generates a synthetic fixture video with a ground-truth timeline, for
// end-to-end checks when no real footage is at hand:
//   fve_synth <out.mp4> [ground_truth.json]

#include "../tests/synthetic.hpp"

#include <opencv2/videoio.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "usage: fve_synth <out.mp4> [ground_truth.json]\n";
        return 2;
    }
    using namespace fve::synth;
    const Scene scene = makeScene();
    const auto parent = std::filesystem::path(argv[1]).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    cv::VideoWriter w(argv[1], cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 25.0, scene.size, true);
    if (!w.isOpened())
    {
        std::cerr << "could not open video writer\n";
        return 1;
    }
    std::ofstream gt;
    if (argc > 2)
        gt.open(argv[2]);
    if (gt)
        gt << "{\n  \"segments\": [\n";
    int idx = 0;
    const auto t = timeline();
    for (std::size_t s = 0; s < t.size(); ++s)
    {
        if (gt)
            gt << "    {\"label\": \"" << t[s].label << "\", \"start\": " << idx << ", \"end\": " << idx + t[s].frames - 1
               << ", \"expect_correct\": " << (t[s].expect_correct ? "true" : "false") << "}"
               << (s + 1 < t.size() ? "," : "") << "\n";
        for (int f = 0; f < t[s].frames; ++f, ++idx)
        {
            Edit e = t[s].edit;
            e.noise_seed = idx;
            w.write(render(scene, e));
        }
    }
    if (gt)
        gt << "  ]\n}\n";
    std::cout << "wrote " << idx << " frames to " << argv[1] << '\n';
    return 0;
}
