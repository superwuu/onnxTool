cmake_minimum_required(VERSION 3.5.0)

project(onnxTool)

set (OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/depend/opencv-4.7.0")
set(OnnxRuntime_DIR "${CMAKE_CURRENT_SOURCE_DIR}/depend/onnxruntime-1.20.1")

include_directories("${OpenCV_DIR}/include")
include_directories("${OnnxRuntime_DIR}/include")

set(cmake_cxx_standard 14)
set(cmake_cxx_standard_required on)

set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install")

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS on)

file(GLOB ONNX_SOURCES
    "onnx.cpp"
    "code/*.h"
)
add_library(onnxTool SHARED ${ONNX_SOURCES} onnx.h)

install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/onnx.h"
        DESTINATION include)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/code/"
        DESTINATION include
        FILES_MATCHING PATTERN "*.h")
install(TARGETS onnxTool
        LIBRARY DESTINATION lib)

# 搜索编译库
set(LIB_DIRS
    ${OpenCV_DIR}/lib
    ${OnnxRuntime_DIR}/lib
)
set(ALL_LIBS "")

if(WIN32)
    foreach(DIR IN LISTS LIB_DIRS)
        # 收集当前目录中的所有 .so 文件
        file(GLOB SO_FILES "${DIR}/*.lib")
        # 将找到的文件添加到 ALL_LIBS 列表中
        list(APPEND ALL_LIBS ${SO_FILES})
    endforeach()
endif(WIN32)

if(UNIX)
    foreach(DIR IN LISTS LIB_DIRS)
        # 收集当前目录中的所有 .so 文件
        file(GLOB SO_FILES "${DIR}/*.so")
        # 将找到的文件添加到 ALL_LIBS 列表中
        list(APPEND ALL_LIBS ${SO_FILES})
    endforeach()
endif(UNIX)

target_link_libraries(${PROJECT_NAME} PRIVATE ${ALL_LIBS})