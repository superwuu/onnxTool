#pragma once
#include "onnx.h"

class Yolov10 : public Otool::OnnxTool {
public:
    Yolov10(std::vector<cv::Mat> inputImages, std::string modelPath, float thresholdconfidence)
        :_inputSrcImages(inputImages)
    {
        SetBatchSize(_inputSrcImages.size());
        SetThresholdConfidence(thresholdconfidence);
        SetDectorVersion(10);
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

    void Postprocess(float* output, std::vector<Otool::Info>& resInfo, int index) {
        // yolov10返回结果格式为[batchsize,300,6]
        // 300为设置的返回框数量
        // 6为{left, top, right ,bottom, confidence, class]
        // 前四个均为在input尺寸下的具体坐标值
        int num_detections = _outputTensorShape[1];
        int _iorigW = _origWidth[index], iorigH = _origHeight[index];
        int _ipadW = _padWidth[index], _ipadH = _padHeight[index];
        int _ipreW = _preWidth[index], _ipreH = _preHeight[index];

        for (size_t j = 0; j < num_detections; ++j) {
            float confidence = output[j * 6 + 4];
            if (confidence > threshold_confidence) {
                float left = output[j * 6 + 0];
                float top = output[j * 6 + 1];
                float right = output[j * 6 + 2];
                float bottom = output[j * 6 + 3];
                int classId = static_cast<int>(output[j * 6 + 5]);
                int x = static_cast<int>((left - _ipadW) * _iorigW / _ipreW);
                int y = static_cast<int>((top - _ipadH) * iorigH / _ipreH);
                int width = static_cast<int>((right - left) * _iorigW / _ipreW);
                int height = static_cast<int>((bottom - top) * iorigH / _ipreH);
                cv::Rect _rect(x, y, width, height);
                resInfo.push_back(Otool::Info(_rect, confidence, classId));
            }
        }
    }

private:
    std::vector<cv::Mat> _inputSrcImages;
    std::vector<std::vector<Otool::Info>> _res;
};