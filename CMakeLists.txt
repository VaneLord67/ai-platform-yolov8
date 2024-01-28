cmake_minimum_required(VERSION 3.12)
PROJECT(app_yolo LANGUAGES C CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)

# set
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

include_directories(
    /home/vanelord/bysj/opencv/include
    /home/vanelord/bysj/opencv/world_build
    /home/vanelord/bysj/opencv/modules/calib3d/include
    /home/vanelord/bysj/opencv/modules/core/include
    /home/vanelord/bysj/opencv/modules/dnn/include
    /home/vanelord/bysj/opencv/modules/features2d/include
    /home/vanelord/bysj/opencv/modules/flann/include
    /home/vanelord/bysj/opencv/modules/gapi/include
    /home/vanelord/bysj/opencv/modules/highgui/include
    /home/vanelord/bysj/opencv/modules/imgcodecs/include
    /home/vanelord/bysj/opencv/modules/imgproc/include
    /home/vanelord/bysj/opencv/modules/java/include
    /home/vanelord/bysj/opencv/modules/js/include
    /home/vanelord/bysj/opencv/modules/ml/include
    /home/vanelord/bysj/opencv/modules/objc/include
    /home/vanelord/bysj/opencv/modules/objdetect/include
    /home/vanelord/bysj/opencv/modules/photo/include
    /home/vanelord/bysj/opencv/modules/python/include
    /home/vanelord/bysj/opencv/modules/stitching/include
    /home/vanelord/bysj/opencv/modules/ts/include
    /home/vanelord/bysj/opencv/modules/video/include
    /home/vanelord/bysj/opencv/modules/videoio/include
    /home/vanelord/bysj/opencv/modules/world/include
)

include_directories(
    /home/vanelord/bysj/cpp_redis/includes
    /home/vanelord/bysj/cpp_redis/tacopie/includes
    /home/vanelord/bysj/spdlog/include
    /home/vanelord/bysj/cpp_ai_utils/cpp_ai_utils
)

include_directories(
    /home/vanelord/anaconda3/envs/ai-platform/lib/python3.10/site-packages/pybind11/include
    /home/vanelord/anaconda3/envs/ai-platform/include/python3.10
    /home/vanelord/anaconda3/envs/ai-platform/include
)

include_directories(
    /home/vanelord/bysj/TensorRT-8.6.1.6/include
    /home/vanelord/bysj/TensorRT-8.6.1.6/samples/common
)

include_directories(
    /home/vanelord/bysj/ByteTrack-cpp/include
    /home/vanelord/bysj/Eigen3/include
    /home/vanelord/bysj/Eigen3/include/eigen3;
)

# 指定库路径
link_directories(
    /home/vanelord/bysj/opencv/world_build/lib
    /home/vanelord/bysj/cpp_redis/build/lib
    /home/vanelord/bysj/cpp_ai_utils/cpp_ai_utils/build
    /home/vanelord/anaconda3/envs/ai-platform/lib
    /home/vanelord/TensorRT-8.6.1.6/lib
)


# 添加你的库的源文件
add_library(app_yolo SHARED
    app_yolov8.cpp
    yolov8.cpp
    logger.cpp
    utils.cpp
    yolo.cpp
)

# 链接库
target_link_libraries(app_yolo
    cpp_ai_utils
    opencv_world
    cpp_redis
    tacopie
    python3
    python3.10
    bytetrack
    cudart
    cudadevrt
    nvinfer
    nvinfer_plugin
    nvonnxparser
    nvparsers
)
