// End-to-end check: compares fve_inspect's frame_deviations.json against the
// ground-truth timeline written by fve_synth. Frames in the first two frames of
// each segment are allowed to lag (video codecs blend at cuts).
//   fve_e2e_check <frame_deviations.json> <synthetic_truth.json>

#include <opencv2/core.hpp>

#include <iostream>
#include <map>
#include <string>

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: fve_e2e_check <frame_deviations.json> <truth.json>\n";
        return 2;
    }
    cv::FileStorage report(argv[1], cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    cv::FileStorage truth(argv[2], cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!report.isOpened() || !truth.isOpened())
    {
        std::cerr << "BLOCKED: could not open report or truth file\n";
        return 2;
    }

    std::map<int, bool> correct;
    for (const auto &f : report["frames"])
    {
        std::string status;
        f["status"] >> status;
        correct[static_cast<int>(f["frame_index"])] = status == "CORRECT";
    }
    if (correct.empty())
    {
        std::cerr << "FAIL: report has no frames\n";
        return 1;
    }

    int checked = 0, wrong = 0;
    for (const auto &seg : truth["segments"])
    {
        std::string label;
        seg["label"] >> label;
        const int start = static_cast<int>(seg["start"]), end = static_cast<int>(seg["end"]);
        const bool expect = static_cast<int>(seg["expect_correct"]) != 0;
        int seg_wrong = 0;
        for (int i = start + 2; i <= end; ++i)
        {
            const auto it = correct.find(i);
            ++checked;
            if (it == correct.end() || it->second != expect)
                ++seg_wrong;
        }
        wrong += seg_wrong;
        std::cout << (seg_wrong ? "  [FAIL] " : "  [PASS] ") << label << " (frames " << start << "-" << end
                  << ", expect " << (expect ? "CORRECT" : "DEVIATED") << ", wrong " << seg_wrong << ")\n";
    }
    std::cout << (wrong ? "FAILED" : "ALL SEGMENTS PASSED") << ": " << checked - wrong << "/" << checked
              << " frames match ground truth\n";
    return wrong ? 1 : 0;
}
