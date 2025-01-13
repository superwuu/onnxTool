cmake_minimum_required(VERSION 3.5.0)

project(onnxDemo)

set(OpenCV_DIR "/media/sdd/wqz/onnxTool/onnxTool/depend/opencv-4.7.0")  # Change to your directory
set(ONNXTOOL_DIR "/media/sdd/wqz/onnxTool/onnxTool/install")    # Change to your directory
set(ONNXRUNTIME_DIR "/media/sdd/wqz/onnxTool/onnxTool/depend/onnxruntime-1.20.1")   # Change to your directory

add_executable(onnxDemo main.cpp)

set(cmake_cxx_standard 14)
set(cmake_cxx_standard_required on)

target_include_directories(onnxDemo PRIVATE
    "${OpenCV_DIR}/include"
    "${ONNXTOOL_DIR}/include"
    "${ONNXRUNTIME_DIR}/include")

set(LIB_DIRS
    ${OpenCV_DIR}/lib
    ${ONNXTOOL_DIR}/lib
    ${ONNXRUNTIME_DIR}/lib
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

target_link_libraries(onnxDemo PRIVATE ${ALL_LIBS})

if (MSVC)
	# 添加编译选项, 设置 输入代码的编码格式为 utf-8
	add_compile_options("/source-charset:utf-8")
endif()