# onnxTool：深度学习onnxRuntime部署工具

介绍

## 工具库获取方法

> [!IMPORTANT]
>
> 出于工程考虑，需要自己提供OpenCV库，需要支持dnn模块的版本，参考版本为 **OpenCV 4.4.0**
>
> [**OpenCV安装示例（linux && windows）**](https://blog.csdn.net/KRISNAT/article/details/122154491)

### 1.从源码编译

##### 代码结构

​	结构结构

<details open>
<summary>Linux</summary>
`Linux` 系统命令如下`（cmake＋make）`

```bash
# 创建build空间
mkdir build && cd build
# cmake构建
cmake ..
# make编译
make
# 构建库文件
cd .. && bash install.bash
```

</details>

<details open>
<summary>Windows</summary>
`Windows` 系统命令如下`（cmake＋msbuild）`

```bash
# 创建build空间
mkdir build && cd build
# cmake构建
cmake ..
# make编译
make
# 构建库文件
cd .. && install.bat
```
</details>

执行完成后，在build目录下出现`include`和`lib`两个目录，即onnxTool工具库
### 2.调用库直接运行

xxx

## 用法

用法用法

## 已支持算法

算法算法

## 未来计划

计划计划

## 参考

参考参考
