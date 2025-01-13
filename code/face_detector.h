#pragma once
#include "onnx.h"

class FaceDetector :public Otool::OnnxTool {
public:
	FaceDetector(std::vector<cv::Mat> inputImages, std::string modelPath)
		:_inputSrcImages(inputImages)
	{
		SetBatchSize(_inputSrcImages.size());
		ReadModel(modelPath);
	}

	FaceDetector(std::string modelPath)
	{
		ReadModel(modelPath);
	}

    void SetBatchImgs(std::vector<cv::Mat> inputImages){
        _inputSrcImages = inputImages;
        Reset();
    }

	void Detection() {
		_res.clear();
        if(_inputSrcImages.size()==0){
            std::cout<<"lack of srcImgs"<<std::endl;
            return;
        }
		SetBatchSize(_inputSrcImages.size());
		OnnxBatchRun(_inputSrcImages, _res);
	}

	void SavePic(int idx, std::string save_path) {
		SavePicture(_inputSrcImages[idx], _res[idx], save_path);
	}

	std::vector<std::vector<Otool::Info>> GetResult() {
		return _res;
	}

	float sigmoid_x(float x)
	{
		return static_cast<float>(1.f / (1.f + exp(-x)));
	}

	void softmax_(const float* x, float* y, int length)
	{
		float sum = 0;
		int i = 0;
		for (i = 0; i < length; i++)
		{
			y[i] = exp(x[i]);
			sum += y[i];
		}
		for (i = 0; i < length; i++)
		{
			y[i] /= sum;
		}
	}

	void Postprocess(float* output, std::vector<Otool::Info>& resInfo, int level_index, int batch_index) {
		//std::cout << "----Using adaface Postprocess!----" << std::endl;
		int _iorigW = _origWidth[batch_index], _iorigH = _origHeight[batch_index];
		int _ipadW = _padWidth[batch_index], _ipadH = _padHeight[batch_index];
		int _ipreW = _preWidth[batch_index], _ipreH = _preHeight[batch_index];

		std::vector<cv::Rect> _rects_nms;
		std::vector<float> _confidence_nms;
		std::vector<int> _class;

		std::vector<std::int64_t> _levelTensorShape = _outputTensorShape[level_index];

		const int _height = _levelTensorShape[2];
		const int _width = _levelTensorShape[3];

		const int stride = (int)ceil((float)_modelHeight / _height);
		const int area = _height * _width;
		const int reg_max = 16;
		float* ptr_cls = output + area * reg_max * 4;
		float* ptr_kp = output + area * (reg_max * 4 + 1);

		for (size_t i = 0; i < _height; ++i) {
			for (size_t j = 0; j < _width; ++j) {
				const int idx = i * _width + j;
				float conf = ptr_cls[0 * area + idx];

				float box_prob = sigmoid_x(conf);

				if (box_prob > threshold_confidence) {
					float pred_ltrb[4];
					float* dfl_value = new float[reg_max];
					float* dfl_softmax = new float[reg_max];
					for (int k = 0; k < 4; k++)
					{
						for (int n = 0; n < reg_max; n++)
						{
							dfl_value[n] = output[(k * reg_max + n) * area + idx];
						}
						softmax_(dfl_value, dfl_softmax, reg_max);

						float dis = 0.f;
						for (int n = 0; n < reg_max; n++)
						{
							dis += n * dfl_softmax[n];
						}
						pred_ltrb[k] = dis * stride;
					}
					float cx = (j + 0.5f) * stride;
					float cy = (i + 0.5f) * stride;
					float xmin = std::max((cx - pred_ltrb[0] - _ipadW) * _iorigW / _ipreW, 0.f);  ///还原回到原图
					float ymin = std::max((cy - pred_ltrb[1] - _ipadH) * _iorigH / _ipreH, 0.f);
					float xmax = std::min((cx + pred_ltrb[2] - _ipadW) * _iorigW / _ipreW, float(_iorigW - 1));
					float ymax = std::min((cy + pred_ltrb[3] - _ipadH) * _iorigH / _ipreH, float(_iorigH - 1));
					_rects_nms.push_back(cv::Rect(int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)));
					_confidence_nms.push_back(box_prob);
					_class.push_back(0);
				}
			}
		}
		std::vector<int> _idx_nms;
		cv::dnn::NMSBoxes(_rects_nms, _confidence_nms, 0, threshold_nms, _idx_nms);
		for (auto& i : _idx_nms) {
			resInfo.push_back(Otool::Info(_rects_nms[i], _confidence_nms[i], _class[i]));
		}
	}
private:
	std::vector<cv::Mat> _inputSrcImages;
	std::vector<std::vector<Otool::Info>> _res;
};