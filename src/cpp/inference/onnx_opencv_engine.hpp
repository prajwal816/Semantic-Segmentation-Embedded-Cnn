#pragma once

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>

namespace seg {

struct InferenceConfig {
  std::string model_path;
  int input_w = 256;
  int input_h = 256;
  int num_classes = 5;
  bool prefer_cuda = true;
  bool int8_simulation = false;
};

class OnnxOpenCVEngine {
 public:
  explicit OnnxOpenCVEngine(InferenceConfig cfg);

  void preprocess(const cv::Mat& bgr, cv::Mat& blob) const;
  void forward(const cv::Mat& blob, cv::Mat& class_mask_u8) const;

 private:
  InferenceConfig cfg_;
  cv::dnn::Net net_;
};

}  // namespace seg
