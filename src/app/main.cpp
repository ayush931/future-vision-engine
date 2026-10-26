#include <fve/config.hpp>
#include <fve/pipeline.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

void usage()
{
    std::cout <<
        R"(Usage: fve_inspect [options]

Input / reference
  --video, --input <path>    Video file, image, image directory, camera index or stream URL (default video.mp4)
  --reference-frame <N>      Use frame N of the input as the golden reference (default 0)
  --reference-image <path>   Use an external golden image instead
  --config <path>            JSON configuration (slots, ignore regions, tuning; see configs/example.json)

Engine
  --backend <name>           auto | classical | dino | hybrid (default auto: hybrid if a model is given)
  --model <path.onnx>        DINOv2 ONNX model (default: models/dinov2_ml.onnx if present)
  --provider <name>          auto | cpu | coreml | cuda (default auto: CoreML on Apple, else CPU)
  --dino-size <px>           DINO input short edge, multiple of 14 recommended (default 448)
  --search-radius <n>        DINO positional search radius in patches, -1 = global (default 2)
  --segmentation <mode>      auto | fixture | foreground | grid | none (default auto)
  --sensitivity <x>          >1 more sensitive, <1 fewer false alarms (default 1.0)
  --calibration-frames <N>   Also calibrate on N known-good frames after the reference (default 0)
  --shift-tolerance <px>     Allowed component displacement in px (default auto: 10% of part size, min 3)
  --work-width <px>          Analysis resolution width (default 640)
  --no-align                 Disable camera-shake / fixture registration
  --workers <N>              Parallel inspection workers (default: half the CPU cores, max 6)

Output (same layout as wisedo_fixture_vision_cpp)
  --output-dir <path>        Output directory (default inspection_results)
  --side-by-side             Also export [Reference | Inspected] images and video
  --heatmap                  Also export anomaly heat-maps for deviated frames
  --render-video, --output <path>   Annotated video path (default <out>/annotated_results.mp4)
  --no-video                 Skip annotated video output
  --no-images                Skip per-frame JPEG output
  --csv / --json / --correct-csv / --correct-json / --deviated-csv / --deviated-json <path>
  --codec <fourcc>           mp4v | avc1 | MJPG ... (default mp4v)
  --jpeg-quality <q>         JPEG quality (default 92)

Speed
  --deduplicate              Collapse consecutive identical states into one record with a frame range
  --include-duplicates       Keep one record per frame (default)
  --no-gate                  Inspect every frame even when content is unchanged
  --diff-threshold <x>       Change gate: mean thumbnail difference (0-255) that counts as changed (default 1.0)
  --max-frames <N>           Stop after N frames
  --quiet                    Only print the summary
  --help
)";
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        fve::EngineConfig config;
        fve::PipelineOptions opt;

        // The config file is applied first so CLI flags override it.
        for (int i = 1; i + 1 < argc; ++i)
            if (std::string(argv[i]) == "--config")
                config = fve::EngineConfig::load(argv[i + 1]);

        for (int i = 1; i < argc; ++i)
        {
            const std::string f = argv[i];
            auto val = [&]() -> std::string {
                if (i + 1 >= argc)
                    throw std::invalid_argument("missing value for " + f);
                return argv[++i];
            };
            if (f == "--video" || f == "--input")
                opt.input = val();
            else if (f == "--reference-frame")
                opt.reference_frame = std::stoi(val());
            else if (f == "--reference-image")
                opt.reference_image = val();
            else if (f == "--config")
                (void)val();
            else if (f == "--backend")
                config.backend = fve::parseBackend(val());
            else if (f == "--model")
                config.dino.model_path = val();
            else if (f == "--provider")
                config.dino.provider = val();
            else if (f == "--dino-size")
                config.dino.short_edge = std::stoi(val());
            else if (f == "--search-radius")
                config.dino.search_radius = std::stoi(val());
            else if (f == "--segmentation")
                config.segmentation = fve::parseSegmentation(val());
            else if (f == "--sensitivity")
                config.anomaly.sensitivity = std::stod(val());
            else if (f == "--calibration-frames")
                config.anomaly.calibration_frames = std::stoi(val());
            else if (f == "--shift-tolerance")
                config.components.shift_tolerance_px = std::stod(val());
            else if (f == "--work-width")
                config.anomaly.work_width = std::stoi(val());
            else if (f == "--no-align")
                config.alignment.enabled = false;
            else if (f == "--workers")
                opt.workers = std::stoi(val());
            else if (f == "--output-dir")
                opt.output_dir = val();
            else if (f == "--side-by-side")
                opt.side_by_side = true;
            else if (f == "--heatmap")
                opt.heatmaps = true;
            else if (f == "--render-video" || f == "--output")
                opt.render_video_path = val();
            else if (f == "--no-video")
                opt.render_video = false;
            else if (f == "--no-images")
                opt.save_images = false;
            else if (f == "--csv")
                opt.csv_path = val();
            else if (f == "--json")
                opt.json_path = val();
            else if (f == "--correct-csv")
                opt.correct_csv = val();
            else if (f == "--correct-json")
                opt.correct_json = val();
            else if (f == "--deviated-csv")
                opt.deviated_csv = val();
            else if (f == "--deviated-json")
                opt.deviated_json = val();
            else if (f == "--codec")
                opt.codec = val();
            else if (f == "--jpeg-quality")
                opt.jpeg_quality = std::stoi(val());
            else if (f == "--deduplicate")
                opt.deduplicate = true;
            else if (f == "--include-duplicates")
                opt.deduplicate = false;
            else if (f == "--no-gate")
                opt.gate_unchanged = false;
            else if (f == "--diff-threshold" || f == "--diff-thresh")
                opt.gate_mean_diff = std::stod(val());
            else if (f == "--max-frames")
                opt.max_frames = std::stoi(val());
            else if (f == "--quiet")
                opt.quiet = true;
            else if (f == "--help" || f == "-h")
            {
                usage();
                return 0;
            }
            else
                throw std::invalid_argument("unknown argument: " + f + " (see --help)");
        }

        // Zero-config accuracy: pick up the prepared model next to the working
        // directory or the executable unless a backend without it was requested.
        if (config.dino.model_path.empty() &&
            (config.backend == fve::Backend::Auto || config.backend == fve::Backend::Hybrid ||
             config.backend == fve::Backend::Dino))
        {
            const auto exe_dir = std::filesystem::weakly_canonical(std::filesystem::path(argv[0])).parent_path();
            for (const auto &p : {std::filesystem::path("models/dinov2_ml.onnx"), exe_dir / "../models/dinov2_ml.onnx",
                                  exe_dir / "models/dinov2_ml.onnx"})
                if (std::filesystem::exists(p))
                {
                    config.dino.model_path = p.string();
                    break;
                }
        }
        if (!config.dino.model_path.empty() && !std::filesystem::exists(config.dino.model_path))
            throw std::runtime_error("model not found: " + config.dino.model_path);
        return fve::runPipeline(config, opt);
    }
    catch (const std::exception &e)
    {
        std::cerr << "fve_inspect: " << e.what() << '\n';
        return 2;
    }
}
