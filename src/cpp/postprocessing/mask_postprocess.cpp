#include "mask_postprocess.hpp"

#include <vector>

namespace seg {

void refine_semantic_mask(const cv::Mat& class_mask_u8, cv::Mat& refined_u8, const PostprocessConfig& cfg) {
  CV_Assert(class_mask_u8.type() == CV_8UC1);
  const int k = std::max(3, cfg.morph_kernel_size | 1);
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
  cv::Mat tmp;
  cv::morphologyEx(class_mask_u8, tmp, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), cfg.morph_iterations);
  cv::morphologyEx(tmp, refined_u8, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), cfg.morph_iterations);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(refined_u8.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  cv::Mat mask = cv::Mat::zeros(refined_u8.size(), CV_8UC1);
  for (const auto& c : contours) {
    const double a = cv::contourArea(c);
    if (a >= cfg.min_contour_area) {
      cv::drawContours(mask, std::vector<std::vector<cv::Point>>{c}, 0, cv::Scalar(255), cv::FILLED);
    }
  }
  refined_u8.setTo(0, mask == 0);
}

void render_overlay(const cv::Mat& bgr, const cv::Mat& refined_u8, cv::Mat& vis_bgr, int num_classes) {
  bgr.copyTo(vis_bgr);
  static const cv::Vec3b palette[] = {
      {50, 50, 200},
      {50, 200, 50},
      {200, 150, 50},
      {200, 50, 150},
      {50, 200, 200},
      {180, 180, 180},
  };
  for (int y = 0; y < refined_u8.rows; ++y) {
    const auto* pm = refined_u8.ptr<uint8_t>(y);
    auto* pv = vis_bgr.ptr<cv::Vec3b>(y);
    for (int x = 0; x < refined_u8.cols; ++x) {
      const int id = pm[x];
      if (id <= 0 || id >= num_classes) {
        continue;
      }
      const auto& col = palette[id % (sizeof(palette) / sizeof(palette[0]))];
      pv[x][0] = static_cast<uint8_t>(0.55f * pv[x][0] + 0.45f * col[0]);
      pv[x][1] = static_cast<uint8_t>(0.55f * pv[x][1] + 0.45f * col[1]);
      pv[x][2] = static_cast<uint8_t>(0.55f * pv[x][2] + 0.45f * col[2]);
    }
  }
}

}  // namespace seg
