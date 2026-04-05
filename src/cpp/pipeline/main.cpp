#include <nlohmann/json.hpp>

#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "inference/logger.hpp"
#include "inference/onnx_opencv_engine.hpp"
#include "postprocessing/mask_postprocess.hpp"
#include "cuda_utils/system_metrics.hpp"

using json = nlohmann::json;

int main(int argc, char** argv) {
  const std::string cfg_path = argc > 1 ? argv[1] : "configs/pipeline_cpp.json";
  std::ifstream cf(cfg_path);
  if (!cf) {
    std::cerr << "Cannot open config: " << cfg_path << std::endl;
    return 2;
  }
  json j;
  cf >> j;

  seg::InferenceConfig icfg;
  icfg.model_path = j.value("model_path", std::string("models/onnx/unet_scene_seg.onnx"));
  icfg.input_w = j.value("input_width", 256);
  icfg.input_h = j.value("input_height", 256);
  icfg.num_classes = j.value("num_classes", 5);
  icfg.prefer_cuda = j.value("prefer_cuda", true);
  icfg.int8_simulation = j.value("use_int8_simulation", false);

  seg::PostprocessConfig pcfg;
  pcfg.morph_kernel_size = j.value("morph_kernel_size", 5);
  pcfg.morph_iterations = j.value("morph_iterations", 1);
  pcfg.min_contour_area = j.value("min_contour_area", 80);

  const std::string log_file = j.value("log_file", std::string("benchmarks/runtime_cpp.log"));
  const int cam_index = j.value("camera_index", 0);
  const bool use_file = j.value("use_video_file", false);
  const std::string video_path = j.value("video_path", std::string("data/sample_video.mp4"));
  const bool show = j.value("show_window", true);
  const int profile_ms = j.value("memory_profile_interval_ms", 500);

  seg::Logger log(log_file);
  log.info("Loading ONNX: " + icfg.model_path);
  if (icfg.int8_simulation) {
    log.warn("INT8 simulation flag set: engine still runs FP32 ONNX; TensorRT INT8 is a separate build step.");
  }

  seg::OnnxOpenCVEngine engine(icfg);

  cv::VideoCapture cap;
  if (use_file) {
    cap.open(video_path);
  } else {
    cap.open(cam_index);
  }
  if (!cap.isOpened()) {
    log.warn("Camera/video open failed — generating synthetic 640x480 feed for demo.");
    // Synthetic feed keeps CI/demo usable without hardware CSI.
  }

  std::vector<double> t_pre;
  std::vector<double> t_inf;
  std::vector<double> t_post;
  t_pre.reserve(300);
  t_inf.reserve(300);
  t_post.reserve(300);

  cv::Mat frame;
  int frame_idx = 0;
  auto last_profile = std::chrono::steady_clock::now();

  while (true) {
    if (cap.isOpened()) {
      cap >> frame;
    }
    if (frame.empty()) {
      frame = cv::Mat(480, 640, CV_8UC3, cv::Scalar(40, 40, 40));
      cv::circle(frame, cv::Point(320 + (frame_idx % 80), 240), 90, cv::Scalar(180, 90, 60), -1);
      cv::putText(frame, "synthetic CSI", cv::Point(30, 40), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                  cv::Scalar(220, 220, 220), 2);
    }

    cv::Mat blob;
    auto t0 = std::chrono::steady_clock::now();
    engine.preprocess(frame, blob);
    auto t1 = std::chrono::steady_clock::now();
    cv::Mat mask;
    engine.forward(blob, mask);
    auto t2 = std::chrono::steady_clock::now();
    cv::Mat refined;
    seg::refine_semantic_mask(mask, refined, pcfg);
    cv::Mat vis;
    seg::render_overlay(frame, refined, vis, icfg.num_classes);
    auto t3 = std::chrono::steady_clock::now();

    t_pre.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    t_inf.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
    t_post.push_back(std::chrono::duration<double, std::milli>(t3 - t2).count());

    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_profile).count() >= profile_ms) {
      last_profile = now;
      const auto rss = seg::current_rss_kb();
      const double sum =
          std::accumulate(t_pre.begin(), t_pre.end(), 0.0) + std::accumulate(t_inf.begin(), t_inf.end(), 0.0) +
          std::accumulate(t_post.begin(), t_post.end(), 0.0);
      const double fps = t_pre.empty() ? 0.0 : (1000.0 * static_cast<double>(t_pre.size()) / sum);
      log.info("rss_kb=" + std::to_string(rss) + " est_fps=" + std::to_string(fps) + " " +
               seg::gpu_utilization_line());
    }

    if (show) {
      cv::imshow("segmentation", vis);
      const int key = cv::waitKey(1);
      if (key == 27 || key == 'q') {
        break;
      }
    }

    frame_idx++;
    if (!show && frame_idx >= 200) {
      break;
    }
  }

  auto pct = [](std::vector<double>& v, double p) {
    if (v.empty()) {
      return 0.0;
    }
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
  };
  log.info(std::string("latency_p50_ms preprocess=") + std::to_string(pct(t_pre)) +
           " infer=" + std::to_string(pct(t_inf)) + " post=" + std::to_string(pct(t_post)));

  return 0;
}
