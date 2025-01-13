#pragma once
#include "onnx.h"

class Yolov10 : public Otool::OnnxTool
{
public:
    Yolov10(std::vector<cv::Mat> inputImages, std::string modelPath, float thresholdconfidence)
        : _inputSrcImages(inputImages)
    {
        SetBatchSize(_inputSrcImages.size());
        SetThresholdConfidence(thresholdconfidence);
        SetDectorVersion(10);
        ReadModel(modelPath);
    }

    Yolov10(std::string modelPath, float thresholdconfidence = 0.4)
    {
        ReadModel(modelPath);
    }

    void SetBatchImgs(std::vector<cv::Mat> inputImages)
    {
        _inputSrcImages = inputImages;
        Reset();
    }

    void Detection()
    {
        _res.clear();
        if (_inputSrcImages.size() == 0)
        {
            std::cout << "lack of srcImgs" << std::endl;
            return;
        }
        SetBatchSize(_inputSrcImages.size());
        OnnxBatchRun(_inputSrcImages, _res);
    }

    void SavePic(int idx, std::string save_path)
    {
        SavePicture(_inputSrcImages[idx], _res[idx], save_path);
    }

    std::vector<std::vector<Otool::Info>> GetResult()
    {
        return _res;
    }

    void Postprocess(float *output, std::vector<Otool::Info> &resInfo, const int level_index, const int batch_index)
    {
        // yolov10返回结果格式为[batchsize,300,6]
        // 300为设置的返回框数量
        // 6为{left, top, right ,bottom, confidence, class]
        // 前四个均为在input尺寸下的具体坐标值
        int num_detections = _outputTensorShape[level_index][1];
        int _iorigW = _origWidth[batch_index], iorigH = _origHeight[batch_index];
        int _ipadW = _padWidth[batch_index], _ipadH = _padHeight[batch_index];
        int _ipreW = _preWidth[batch_index], _ipreH = _preHeight[batch_index];

        for (size_t j = 0; j < num_detections; ++j)
        {
            float confidence = output[j * 6 + 4];
            if (confidence > threshold_confidence)
            {
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