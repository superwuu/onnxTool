cmake_minimum_required(VERSION 3.5.0)

project(onnxDemo)

# windows需要设置opencv路径
set (OpenCV_DIR "opencv_path")  #opencv roots
find_package(OpenCV REQUIRED)
set(ONNXTOOL_DIR "onnxTool_path")

add_executable(${PROJECE_NAME} main.cpp)

set(cmake_cxx_standard 14)
set(cmake_cxx_standard_required on)

include_directories(${OpenCV_INCLUDE_DIRS})
target_include_directories(onnxDemo PRIVATE "${ONNXTOOL_DIR}/include")

TARGET_LINK_LIBRARIES(onnxDemo PRIVATE ${OpenCV_LIBS})

if(WIN32)
    file(GLOB LIBS "${ONNXTOOL_DIR}/lib/*.lib")
    TARGET_LINK_LIBRARIES(${PROJECE_NAME} PRIVATE ${LIBS})
endif(WIN32)

if(UNIX)
    file(GLOB LIBS "${ONNXTOOL_DIR}/lib/*.so")
    TARGET_LINK_LIBRARIES(${PROJECE_NAME} PRIVATE ${LIBS})
endif(UNIX)

if (MSVC)
	# 添加编译选项, 设置 输入代码的编码格式为 utf-8
	add_compile_options("/source-charset:utf-8")
endif()