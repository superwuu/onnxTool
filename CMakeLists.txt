cmake_minimum_required(VERSION 3.5.0)

project(onnxTool)

# windows需要设置opencv路径
set (OpenCV_DIR "opencv_path")  #opencv roots
find_package(OpenCV REQUIRED)

set(onnxruntime_DIR "onnxruntime_path")

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${onnxruntime_DIR}/include")

set(cmake_cxx_standard 14)
set(cmake_cxx_standard_required on)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(onnxTool SHARED onnx.h onnx.cpp code/yolov8.h code/yolov10.h code/adaface.h code/face_detector.h)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

if(WIN32)
    file(GLOB LIBS "${onnxruntime_DIR}/lib/*.lib")
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${LIBS} ${OpenCV_LIBS})
endif(WIN32)