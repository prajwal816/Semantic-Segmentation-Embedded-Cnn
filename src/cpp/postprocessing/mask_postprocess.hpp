#pragma once

#include <opencv2/opencv.hpp>

namespace seg {

struct PostprocessConfig {
  int morph_kernel_size = 5;
  int morph_iterations = 1;
  int min_contour_area = 80;
};

void refine_semantic_mask(const cv::Mat& class_mask_u8, cv::Mat& refined_u8, const PostprocessConfig& cfg);

void render_overlay(const cv::Mat& bgr, const cv::Mat& refined_u8, cv::Mat& vis_bgr, int num_classes);

}  // namespace seg
