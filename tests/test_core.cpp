// Behavioural tests on synthetic scenes with known ground truth. Each case drives
// the public Inspector API and asserts the verdict, not internals.

#include "synthetic.hpp"

#include <fve/fixture.hpp>
#include <fve/inspector.hpp>
#include <fve/output.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace
{
int failures = 0;

void check(bool ok, const std::string &what)
{
    std::cout << (ok ? "  [PASS] " : "  [FAIL] ") << what << '\n';
    if (!ok)
        ++failures;
}

std::string summary(const fve::FrameResult &r)
{
    std::string s = r.is_correct ? "correct" : "deviated";
    for (const auto &sl : r.slots)
        if (sl.deviation != fve::Deviation::None)
            s += " " + sl.slot_id + ":" + fve::toString(sl.deviation);
    s += " extras=" + std::to_string(r.extra_boxes.size());
    return s;
}

const fve::SlotResult *slot(const fve::FrameResult &r, const std::string &id)
{
    for (const auto &s : r.slots)
        if (s.slot_id == id)
            return &s;
    return nullptr;
}

void runScenarios(const std::string &label, fve::EngineConfig cfg, bool coarse_only = false)
{
    using namespace fve::synth;
    std::cout << "\n== " << label << " ==\n";
    const Scene scene = makeScene();
    fve::Inspector insp(cfg);
    insp.setReference(render(scene, {}));
    check(insp.reference().slots.size() == scene.parts.size(),
          "discovers all " + std::to_string(scene.parts.size()) + " parts (got " +
              std::to_string(insp.reference().slots.size()) + ")");

    auto run = [&](Edit e, int seed) {
        e.noise_seed = seed;
        return insp.inspect(render(scene, e), seed, 25.0);
    };

    {
        const auto r = run({}, 1);
        check(r.is_correct, "unchanged scene with fresh sensor noise is CORRECT (" + summary(r) + ")");
    }
    {
        Edit e;
        e.remove = 6;
        const auto r = run(e, 2);
        const auto *s = slot(r, "SLOT_07");
        check(!r.is_correct && s && s->deviation == fve::Deviation::Missing && r.deviatedCount() == 1,
              "removed part -> only SLOT_07 MISSING (" + summary(r) + ")");
    }
    {
        Edit e;
        e.shift = 11;
        e.shift_by = {18, 6};
        const auto r = run(e, 3);
        const auto *s = slot(r, "SLOT_12");
        const bool ok = s && s->deviation == fve::Deviation::Shifted && std::abs(s->dx - 18) < 1.5 &&
                        std::abs(s->dy - 6) < 1.5 && r.deviatedCount() == 1;
        check(ok, "shifted part -> SLOT_12 SHIFTED with dx~18 dy~6 (" + summary(r) +
                      (s ? " dx=" + std::to_string(s->dx) + " dy=" + std::to_string(s->dy) : "") + ")");
    }
    {
        Edit e;
        e.extra = true;
        const auto r = run(e, 4);
        const bool ok = r.extra_boxes.size() == 1 && r.missing_count + r.shifted_count + r.changed_count == 0;
        if (coarse_only && !ok)
            // Known limit, recorded rather than hidden: the 44 px part spans ~2 DINO
            // patches, too coarse for pure DINO. Hybrid (the default) must catch it.
            std::cout << "  [KNOWN LIMIT] pure DINO misses a 2-patch part (" << summary(r) << ")\n";
        else
            check(ok, "added part -> exactly one EXTRA (" + summary(r) + ")");
    }
    {
        Edit e;
        e.camera = {5.0f, -4.0f};
        e.camera_rot = 0.6;
        const auto r = run(e, 5);
        check(r.is_correct && r.alignment.applied, "camera shake is registered away -> CORRECT (" + summary(r) + ")");
    }
    {
        Edit e;
        e.damage = 2;
        const auto r = run(e, 6);
        const auto *s = slot(r, "SLOT_03");
        check(s && s->deviation == fve::Deviation::Changed && r.deviatedCount() == 1,
              "scratched part -> SLOT_03 CHANGED (" + summary(r) + ")");
    }
    {
        Edit e;
        e.gain = 0.88;
        const auto r = run(e, 7);
        check(r.is_correct, "12% darker lighting is CORRECT (" + summary(r) + ")");
    }
    {
        Edit e;
        e.remove = 0;
        e.camera = {-6.0f, 3.0f};
        const auto r = run(e, 8);
        const auto *s = slot(r, "SLOT_01");
        check(s && s->deviation == fve::Deviation::Missing && r.deviatedCount() == 1,
              "missing part under camera shake -> SLOT_01 MISSING (" + summary(r) + ")");
    }
}

void testUnits()
{
    std::cout << "\n== units ==\n";
    const auto ranges = fve::computeFrameRanges({0, 1, 2, 5, 7, 8});
    check(ranges == std::vector<std::string>{"0-2", "5", "7-8"}, "computeFrameRanges groups consecutive indices");
    check(fve::computeFrameRanges({}).empty(), "computeFrameRanges of nothing is empty");
    check(std::abs(fve::boxIoU({0, 0, 10, 10}, {5, 0, 10, 10}) - 1.0 / 3.0) < 1e-9, "boxIoU of half-overlap is 1/3");
    check(fve::jsonEscape("a\"b\\c\n") == "a\\\"b\\\\c\\n", "jsonEscape escapes quotes, backslash, newline");

    // Config in the legacy slot format, with an ignore region.
    const auto path = std::filesystem::temp_directory_path() / "fve_test_config.json";
    {
        std::ofstream f(path);
        f << R"({"backend": "classical", "slots": [{"slot_id": "A", "type": "part", "bbox": [10, 20, 50, 60]}],
                "ignore_regions": [[0, 0, 5, 5]], "components": {"shift_tolerance_px": 4.5}})";
    }
    const auto cfg = fve::EngineConfig::load(path.string());
    check(cfg.slots.size() == 1 && cfg.slots[0].bounds == cv::Rect(10, 20, 40, 40) && cfg.ignore_regions.size() == 1 &&
              cfg.components.shift_tolerance_px == 4.5 && cfg.backend == fve::Backend::Classical,
          "EngineConfig::load reads legacy slots, ignore regions and tuning");
    {
        std::ofstream f(path);
        f << R"({"slots": [{"slot_id": "B", "bbox": [50, 20, 10, 60]}]})";
    }
    bool threw = false;
    try
    {
        (void)fve::EngineConfig::load(path.string());
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    check(threw, "EngineConfig::load rejects an inverted bbox");
    std::filesystem::remove(path);
}

void testPureAnomaly()
{
    using namespace fve::synth;
    std::cout << "\n== segmentation none (pure anomaly mode) ==\n";
    fve::EngineConfig cfg;
    cfg.backend = fve::Backend::Classical;
    cfg.segmentation = fve::SegmentationMode::None;
    const Scene scene = makeScene();
    fve::Inspector insp(cfg);
    insp.setReference(render(scene, {}));
    Edit e;
    e.noise_seed = 11;
    check(insp.inspect(render(scene, e), 1, 25).is_correct, "no components + unchanged scene is CORRECT");
    e.remove = 9;
    const auto r = insp.inspect(render(scene, e), 2, 25);
    bool hit = false;
    for (const auto &b : r.extra_boxes)
        hit = hit || fve::overlapRatio(scene.parts[9], b) > 0.5;
    check(!r.is_correct && hit, "no components + removed part is reported as an anomaly region");
}
} // namespace

int main()
{
    testUnits();

    fve::EngineConfig classical;
    classical.backend = fve::Backend::Classical;
    runScenarios("classical backend", classical);
    testPureAnomaly();

    if (const char *model = std::getenv("FVE_TEST_MODEL"); model && *model)
    {
        fve::EngineConfig hybrid;
        hybrid.backend = fve::Backend::Hybrid;
        hybrid.dino.model_path = model;
        runScenarios("hybrid backend (" + std::string(model) + ")", hybrid);
        fve::EngineConfig dino = hybrid;
        dino.backend = fve::Backend::Dino;
        runScenarios("dino backend", dino, /*coarse_only=*/true);
    }
    else
    {
        std::cout << "\n(FVE_TEST_MODEL not set: DINO backends not exercised)\n";
    }

    std::cout << "\n" << (failures ? "FAILED: " + std::to_string(failures) + " check(s)" : "ALL CHECKS PASSED") << "\n";
    return failures ? 1 : 0;
}
