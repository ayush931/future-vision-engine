cmake_minimum_required(VERSION 3.22)
project(future_vision_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(FVE_WITH_ONNX "Build the DINOv2 backend with ONNX Runtime (auto-disabled if not found)" ON)
option(FVE_NATIVE "Optimise for the build machine's CPU" ON)

# OpenCV 4.x or 5.x: core imgproc imgcodecs videoio features(2d) + calib3d/geometry
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV ${OpenCV_VERSION}")
# Link only the modules we use (OpenCV 5 renamed features2d/calib3d), so a
# runtime image needs a handful of libraries instead of the whole of OpenCV.
if(OpenCV_VERSION_MAJOR GREATER_EQUAL 5)
    set(FVE_CV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_features opencv_geometry)
else()
    set(FVE_CV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_features2d opencv_calib3d)
endif()

set(FVE_ONNX_FOUND OFF)
if(FVE_WITH_ONNX)
    # Explicit paths win (release tarballs ship an incomplete CMake config).
    if(NOT (ORT_INCLUDE AND ORT_LIB))
        find_package(onnxruntime CONFIG QUIET)
    endif()
    if(onnxruntime_FOUND)
        set(FVE_ONNX_TARGET onnxruntime::onnxruntime)
        set(FVE_ONNX_FOUND ON)
    else()
        find_path(ORT_INCLUDE onnxruntime_cxx_api.h PATH_SUFFIXES onnxruntime onnxruntime/core/session)
        find_library(ORT_LIB onnxruntime)
        if(ORT_INCLUDE AND ORT_LIB)
            add_library(fve_onnxruntime INTERFACE)
            target_include_directories(fve_onnxruntime INTERFACE ${ORT_INCLUDE})
            target_link_libraries(fve_onnxruntime INTERFACE ${ORT_LIB})
            set(FVE_ONNX_TARGET fve_onnxruntime)
            set(FVE_ONNX_FOUND ON)
        endif()
    endif()
endif()
message(STATUS "ONNX Runtime (DINOv2 backend): ${FVE_ONNX_FOUND}")

add_library(fve_core STATIC
    src/core/types_config.cpp
    src/core/fixture.cpp
    src/core/alignment.cpp
    src/core/anomaly.cpp
    src/core/dino.cpp
    src/core/inspector.cpp
    src/core/output.cpp
    src/core/pipeline.cpp
)
target_include_directories(fve_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(fve_core PUBLIC ${FVE_CV_LIBS})
find_package(Threads REQUIRED)
target_link_libraries(fve_core PUBLIC Threads::Threads)
if(FVE_ONNX_FOUND)
    target_compile_definitions(fve_core PRIVATE FVE_WITH_ONNX=1)
    target_link_libraries(fve_core PRIVATE ${FVE_ONNX_TARGET})
endif()

set(FVE_WARN -Wall -Wextra -Wpedantic)
set(FVE_OPT)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    list(APPEND FVE_OPT -O3)
    if(FVE_NATIVE)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag(-mcpu=native HAS_MCPU_NATIVE)
        check_cxx_compiler_flag(-march=native HAS_MARCH_NATIVE)
        if(HAS_MCPU_NATIVE)
            list(APPEND FVE_OPT -mcpu=native)
        elseif(HAS_MARCH_NATIVE)
            list(APPEND FVE_OPT -march=native)
        endif()
    endif()
endif()
target_compile_options(fve_core PRIVATE ${FVE_WARN} ${FVE_OPT})

add_executable(fve_inspect src/app/main.cpp)
target_link_libraries(fve_inspect PRIVATE fve_core)
target_compile_options(fve_inspect PRIVATE ${FVE_WARN} ${FVE_OPT})

add_executable(fve_synth tools/synth.cpp)
target_include_directories(fve_synth PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(fve_synth PRIVATE opencv_core opencv_imgproc opencv_videoio)

include(CTest)
if(BUILD_TESTING)
    add_executable(fve_core_test tests/test_core.cpp)
    target_link_libraries(fve_core_test PRIVATE fve_core)
    target_compile_options(fve_core_test PRIVATE ${FVE_WARN} -UNDEBUG)
    add_test(NAME fve_core_test COMMAND fve_core_test)

    # End-to-end: synthetic video with known truth -> CLI -> per-frame verdict check.
    add_executable(fve_e2e_check tests/e2e_check.cpp)
    target_include_directories(fve_e2e_check PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(fve_e2e_check PRIVATE opencv_core)
    set(E2E_DIR ${CMAKE_CURRENT_BINARY_DIR}/e2e)
    add_test(NAME e2e_synth COMMAND fve_synth ${E2E_DIR}/synthetic.mp4 ${E2E_DIR}/truth.json)
    add_test(NAME e2e_classical_run COMMAND fve_inspect --video ${E2E_DIR}/synthetic.mp4 --backend classical
             --output-dir ${E2E_DIR}/classical --quiet --no-images --no-video)
    add_test(NAME e2e_classical_check COMMAND fve_e2e_check ${E2E_DIR}/classical/frame_deviations.json ${E2E_DIR}/truth.json)
    set_tests_properties(e2e_synth PROPERTIES FIXTURES_SETUP synth_video)
    set_tests_properties(e2e_classical_run PROPERTIES FIXTURES_REQUIRED synth_video FIXTURES_SETUP classical_report)
    set_tests_properties(e2e_classical_check PROPERTIES FIXTURES_REQUIRED classical_report)
    if(FVE_ONNX_FOUND AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/dinov2_ml.onnx)
        add_test(NAME e2e_hybrid_run COMMAND fve_inspect --video ${E2E_DIR}/synthetic.mp4 --backend hybrid
                 --model ${CMAKE_CURRENT_SOURCE_DIR}/models/dinov2_ml.onnx --output-dir ${E2E_DIR}/hybrid --quiet --no-images --no-video)
        add_test(NAME e2e_hybrid_check COMMAND fve_e2e_check ${E2E_DIR}/hybrid/frame_deviations.json ${E2E_DIR}/truth.json)
        set_tests_properties(e2e_hybrid_run PROPERTIES FIXTURES_REQUIRED synth_video FIXTURES_SETUP hybrid_report)
        set_tests_properties(e2e_hybrid_check PROPERTIES FIXTURES_REQUIRED hybrid_report)
        set_tests_properties(fve_core_test PROPERTIES ENVIRONMENT FVE_TEST_MODEL=${CMAKE_CURRENT_SOURCE_DIR}/models/dinov2_ml.onnx)
    endif()
endif()

add_executable(fve_probe tools/probe_channels.cpp)
target_link_libraries(fve_probe PRIVATE fve_core)
