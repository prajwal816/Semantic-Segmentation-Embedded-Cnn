cmake_minimum_required(VERSION 3.16)
project(semantic_seg_edge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn imgcodecs videoio highgui)

include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3)
FetchContent_MakeAvailable(nlohmann_json)

add_executable(
  seg_edge_pipeline
  src/cpp/pipeline/main.cpp
  src/cpp/inference/onnx_opencv_engine.cpp
  src/cpp/postprocessing/mask_postprocess.cpp
  src/cpp/cuda_utils/system_metrics.cpp)

target_include_directories(seg_edge_pipeline PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp)
target_link_libraries(seg_edge_pipeline PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

if(WIN32)
  target_link_libraries(seg_edge_pipeline PRIVATE psapi)
endif()
