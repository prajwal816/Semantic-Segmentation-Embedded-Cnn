#include "onnx_opencv_engine.hpp"

#include <stdexcept>

namespace seg {

OnnxOpenCVEngine::OnnxOpenCVEngine(InferenceConfig cfg) : cfg_(std::move(cfg)) {
  net_ = cv::dnn::readNetFromONNX(cfg_.model_path);
  if (net_.empty()) {
    throw std::runtime_error("Failed to load ONNX model: " + cfg_.model_path);
  }
  if (cfg_.prefer_cuda) {
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  } else {
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  }
}

void OnnxOpenCVEngine::preprocess(const cv::Mat& bgr, cv::Mat& blob) const {
  cv::Mat rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  blob = cv::dnn::blobFromImage(
      rgb, 1.0 / 255.0, cv::Size(cfg_.input_w, cfg_.input_h), cv::Scalar(0, 0, 0), true, false);
}

void OnnxOpenCVEngine::forward(const cv::Mat& blob, cv::Mat& class_mask_u8) const {
  net_.setInput(blob);
  cv::Mat out = net_.forward();
  if (out.dims != 4) {
    throw std::runtime_error("Unexpected ONNX output rank");
  }
  const int n = out.size[0];
  const int c = out.size[1];
  const int h = out.size[2];
  const int w = out.size[3];
  if (n != 1 || c != cfg_.num_classes) {
    throw std::runtime_error("Output shape mismatch vs config num_classes");
  }
  class_mask_u8.create(h, w, CV_8UC1);
  const float* base = out.ptr<float>(0);
  for (int y = 0; y < h; ++y) {
    auto* row = class_mask_u8.ptr<uint8_t>(y);
    for (int x = 0; x < w; ++x) {
      int best = 0;
      float bestv = base[0 * h * w + y * w + x];
      for (int k = 1; k < c; ++k) {
        const float v = base[k * h * w + y * w + x];
        if (v > bestv) {
          bestv = v;
          best = k;
        }
      }
      row[x] = static_cast<uint8_t>(best);
    }
  }
  (void)cfg_.int8_simulation;
}

}  // namespace seg
