#pragma once

// OpenCV 4 / 5 compatibility: OpenCV 5 moved contour, rotation and robust
// estimation helpers into the `geometry` module.
#include <opencv2/imgproc.hpp>
#if __has_include(<opencv2/geometry.hpp>)
#include <opencv2/geometry.hpp>
#else
#include <opencv2/calib3d.hpp>
#endif
