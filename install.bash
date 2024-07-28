#!/bin/bash

# 创建必要的目录
mkdir -p build/include
mkdir -p build/lib

# 复制所有的 .h 文件
cp ./*.h build/include/
cp ./depend/*.h build/include/

# 复制所有的 .so 文件
cp ./depend/*.so* build/lib/
cp ./build/*.so build/lib/

echo "Files have been copied successfully."