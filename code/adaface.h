#pragma once
#include "onnx.h"

class Adaface :public Otool::OnnxTool {
public:
	Adaface(std::vector<cv::Mat> inputImages, std::string modelPath)
		:_inputSrcImages(inputImages)
	{
		SetBatchSize(_inputSrcImages.size());
		ReadModel(modelPath);
	}

	void Detection() {
		_res.clear();
		OnnxBatchRun(_inputSrcImages, _res);
	}

	void SavePic(int idx, std::string save_path) {
		SavePicture(_inputSrcImages[idx], _res[idx], save_path);
	}

	std::vector<std::vector<Otool::Info>> GetResult() {
		return _res;
	}

	// 重写预处理，输入为batchSize图像vector，输出为[Batch, Channel, Height, Width]
	void Preprocess(const std::vector<cv::Mat>& inputImages, cv::Mat& blobImage) {
		//std::cout << "----Using adaface Preprocessing!----" << std::endl;
		std::vector<cv::Mat> middleImages;
		for (size_t i = 0; i < inputImages.size(); ++i) {
			cv::Mat img_m;
			inputImages[i].convertTo(img_m, CV_32F);
			middleImages.push_back(img_m);
		}
		blobImage = cv::dnn::blobFromImages(middleImages, 1, cv::Size(_modelWidth, _modelHeight), cv::Scalar(0, 0, 0), false);
		blobImage /= 255.;
		blobImage -= 0.5;
		blobImage /= 0.5;
	}

	// 重写后处理，直接返回结果向量
	void Postprocess(float* output, std::vector<Otool::Info>& resInfo, const int level_index, const int batch_index) {
		//std::cout << "----Using adaface Postprocess!----" << std::endl;
		std::vector<float> res;
		int num_ = _outputTensorShape[level_index][1];
		for (size_t i = 0; i < num_; ++i) {
			res.push_back(output[i]);
		}
		resInfo.push_back(Otool::Info(res));
	}
private:
	std::vector<cv::Mat> _inputSrcImages;
	std::vector<std::vector<Otool::Info>> _res;
};