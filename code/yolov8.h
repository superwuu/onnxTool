#pragma once
#include "onnx.h"

class Yolov8 : public Otool::OnnxTool {
public:
	Yolov8(std::vector<cv::Mat> inputImages, std::string modelPath, float thresholdconfidence = 0.4)
		:_inputSrcImages(inputImages)
	{
		SetBatchSize(_inputSrcImages.size());
		SetThresholdConfidence(thresholdconfidence);
		SetDectorVersion(8);
		ReadModel(modelPath);
	}

	Yolov8(std::string modelPath, float thresholdconfidence = 0.4)
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

    void Postprocess(float* output, std::vector<Otool::Info>& resInfo, const int level_index, const int batch_index) {
        // yolov8返回结果格式为[batchsize,classnum+4,cnt]
        // classnum+4中前四个为框的x,y,w,h，后classnum个为属于每个框的置信度
        int _iorigW = _origWidth[batch_index], iorigH = _origHeight[batch_index];
        int _ipadW = _padWidth[batch_index], _ipadH = _padHeight[batch_index];
        int _ipreW = _preWidth[batch_index], _ipreH = _preHeight[batch_index];

        std::vector<cv::Rect> _rects_nms;
        std::vector<float> _confidence_nms;
        std::vector<int> _class;
        cv::Mat all_scores = cv::Mat(cv::Size(_outputTensorShape[level_index][2], _outputTensorShape[level_index][1]), CV_32F, output).t();
        float* pdata = (float*)all_scores.data;
        for (size_t r = 0; r < all_scores.rows; ++r) {
            cv::Mat scores(cv::Size(OnnxTool::GetObjNum(), 1), CV_32F, pdata + 4);
            cv::Point max_loc;
            double max_confidence;
            cv::minMaxLoc(scores, 0, &max_confidence, 0, &max_loc);
            if (max_confidence > threshold_confidence) {
                float x = (pdata[0] - _ipadW) * _iorigW / _ipreW;
                float y = (pdata[1] - _ipadH) * iorigH / _ipreH;
                float w = pdata[2] * _iorigW / _ipreW;
                float h = pdata[3] * iorigH / _ipreH;
                int left = int(x - 0.5 * w);
                int top = int(y - 0.5 * h);
                int right = int(x + 0.5 * w);
                int bottom = int(y + 0.5 * h);
                cv::Rect _rect(left, top, w, h);
                _rects_nms.push_back(cv::Rect(left, top, w, h));
                _confidence_nms.push_back(max_confidence);
                _class.push_back(max_loc.x);
            }
            pdata += _outputTensorShape[level_index][1];
        }
        std::vector<int> _idx_nms;
        cv::dnn::NMSBoxes(_rects_nms, _confidence_nms, 0, threshold_nms, _idx_nms);
        for (auto& index : _idx_nms) {
            resInfo.push_back(Otool::Info(_rects_nms[index], _confidence_nms[index], _class[index]));
        }
    }
private:
	std::vector<cv::Mat> _inputSrcImages;
	std::vector<std::vector<Otool::Info>> _res;
};