#pragma once

#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace Otool
{

	struct Info
	{
		Info(cv::Rect rect, float confidence, int classId) : _rect(rect), _confidence(confidence), _classId(classId) {}
		Info(std::vector<float> output) : _output(output) {}
		cv::Rect _rect;
		float _confidence;
		int _classId;
		std::vector<float> _output;
	};

	class OnnxTool
	{
	protected:
		OnnxTool();
		~OnnxTool();

	public:
		// onnxruntime模型运行相关
		bool ReadModel(const std::string &modelPath);
		void OnnxBatchRun(const std::vector<cv::Mat> &srcImages, std::vector<std::vector<Info>> &resInfo);
		void SavePicture(const cv::Mat &image, const std::vector<Info> &resInfo, const std::string &savePath);

	public:
		// 设置相关变量的接口
		void SetBatchSize(int num);
		void SetOrigSize(int width, int height);
		void SetThresholdConfidence(float confidence);
		void SetThresholdNMS(float nms);
		void SetDectorVersion(int version);

		void SetColor();
		void SetClasses(const std::string &fileName);
		int GetObjNum();

	public:
		// 预处理，输入为batchSize图像vector，输出为[Batch, Channel, Height, Width]
		virtual void Preprocess(const std::vector<cv::Mat> &inputImages, cv::Mat &blobImage);
		// 后处理，输入为batchSize份内存，分别为不同的图像的结果，输出为batchsize个vector<Info>结构的数组
		void Postprocess_all(std::vector<float *> &outputVector, std::vector<std::vector<Info>> &resInfo, const int batch_index);
		virtual void Postprocess(float *output, std::vector<Info> &resInfo, const int level_index, const int batch_index);

		// Preprocessing
		// 从原图尺寸到模型尺寸的转换函数，在Preprocessing中调用
		void Letterbox(cv::Mat &src, const cv::Size &size);
		void Letterbox_lr(cv::Mat &src, const cv::Size &size);

	public:
		// 重置批次图像设置
		void Reset();

	protected:
		// 计算vector所有元素的乘积
		int64_t VectorProduct(const std::vector<int64_t> &vec)
		{
			return std::accumulate(vec.begin(), vec.end(), 1, std::multiplies<int64_t>());
		}

	protected:
		// onnxruntime环境相关变量
		Ort::Env _env;
		Ort::SessionOptions _sessionOptions;
		Ort::Session *_session;
		Ort::MemoryInfo _memoryInfo;

		// onnx模型相关变量
		std::vector<std::string> _inputName;				  // e.g. "image"
		std::vector<std::string> _outputName;				  // e.g. "output"
		std::vector<int64_t> _inputTensorShape;				  // e.g. [1,3,640,640]
		std::vector<std::vector<int64_t>> _outputTensorShape; // e.g. [[1,3,640,640], [1,3,320,320]] 多输出头

		bool _isInputUDFsize = false;
		std::vector<bool> _isOutputUDFsize;

		// 输入输出头的数量
		size_t _inputHeadNum;
		size_t _outputHeadNum;

		// 输入模型推理的尺寸
		int _modelWidth;
		int _modelHeight;

		// 图像原始尺寸，batchSize张图像
		std::vector<int> _origWidth;
		std::vector<int> _origHeight;

		// 图像预处理填充，batchSize张图像（如需要）
		std::vector<int> _padWidth;
		std::vector<int> _padHeight;

		// 图像预处理后缩放尺寸，batchSize张图像（如需要）
		std::vector<int> _preWidth;
		std::vector<int> _preHeight;

		// 目标检测相关变量
		int _batchSize = 1;
		int detectorVersion = 8;
		float threshold_confidence = 0.4;
		float threshold_nms = 0.6;

		std::vector<cv::Scalar> _color = {};
		std::vector<std::string> _className = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
											   "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
											   "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
											   "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
											   "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
											   "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
											   "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
											   "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
											   "hair drier", "toothbrush"};
	};

};