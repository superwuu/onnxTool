#include "onnx.h"

Otool::OnnxTool::OnnxTool() :_memoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput)) {
    _env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "g_env");
    _sessionOptions = Ort::SessionOptions();
    _session = nullptr;
    std::cout << "[OnnxTool is constructing...]" << std::endl;
}

Otool::OnnxTool::~OnnxTool() {
    if (_session != nullptr) { delete _session; }
}

// protected
bool Otool::OnnxTool::ReadModel(const std::string& modelPath) {
    try {
        // 设置图优化级别为最高级别
        _sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // 读取模型，创建session
#ifdef _WIN32
		std::wstring _wmodelPath(modelPath.begin(), modelPath.end());
        _session = new Ort::Session(_env, _wmodelPath.c_str(), _sessionOptions);
#else
		_session = new Ort::Session(_env, modelPath.c_str(), _sessionOptions);
#endif

        // 获取模型输入输出信息
        size_t _numInputNodes = _session->GetInputCount();
        size_t _numOutputNodes = _session->GetOutputCount();

        Ort::AllocatorWithDefaultOptions _allocator;
        // 获取输入节点的名称和维度信息
        _inputName.push_back(std::string(""));
        auto _inputNameAlloc = _session->GetInputNameAllocated(0, _allocator);
        _inputName[0].append(_inputNameAlloc.get());
        Ort::TypeInfo _inputTypeInfo = _session->GetInputTypeInfo(0);
        auto _inputTensorInfo = _inputTypeInfo.GetTensorTypeAndShapeInfo();
        _inputTensorShape = _inputTensorInfo.GetShape();

        // 获取输出节点的名称和维度信息
        _outputName.push_back(std::string(""));
        auto _outputNameAlloc = _session->GetOutputNameAllocated(0, _allocator);
        _outputName[0].append(_outputNameAlloc.get());
        Ort::TypeInfo _outputTypeInfo = _session->GetOutputTypeInfo(0);
        auto _outputTensorInfo = _outputTypeInfo.GetTensorTypeAndShapeInfo();
        _outputTensorShape = _outputTensorInfo.GetShape();

        // 从输入节点维度中获取输入图像的高度和宽度
        _modelHeight = _inputTensorShape[2];
        _modelWidth = _inputTensorShape[3];

        // 如果是多batch推理，则设置batchsize
        if (_inputTensorShape[0] == -1) {
            _inputTensorShape[0] = _batchSize;
        }
        if (_outputTensorShape[0] == -1) {
            _outputTensorShape[0] = _batchSize;
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cout << "[----readModel error----] " << e.what() << std::endl;
        if (_session != nullptr) { delete _session; }
        std::exit(7758258);
    }
    catch (const std::string& _msg) {
        std::cout << "[----error message----] " << _msg.c_str() << std::endl;
    }
    return true;
}

void Otool::OnnxTool::OnnxBatchRun(const std::vector<cv::Mat>& srcImages, std::vector<std::vector<Info>>& resInfo) {
    try {
        cv::Mat blobImage;
        // 预处理
        Preprocessing(srcImages, blobImage);

        // 元素长度
        int64_t _inputTensorLength = VectorProduct(_inputTensorShape);

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(_memoryInfo, (float*)blobImage.data, _inputTensorLength, _inputTensorShape.data(), _inputTensorShape.size()));

        // 推理
        const std::array<const char*, 1> _inputName_run = { _inputName[0].c_str() };
        const std::array<const char*, 1> _outputName_run = { _outputName[0].c_str() };
        std::vector<Ort::Value> _ortOutputs = _session->Run(Ort::RunOptions{ nullptr }, _inputName_run.data(), input_tensors.data(), _inputName_run.size(), _outputName_run.data(), _outputName_run.size());

        // all_data是结果的内存起点
        float* all_data = _ortOutputs[0].GetTensorMutableData<float>();
        // 内存上的分界
        size_t count = _ortOutputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        // vector存放多batch结果
        std::vector<float*> outputData;
        for (size_t i = 0; i < _batchSize; ++i) {
            outputData.push_back(all_data + count / _batchSize * i);
        }
        // 后处理
        Postprocessing(outputData, resInfo);
    }
    catch (const std::string& _msg) {
        std::cout << "[----error message----] " << _msg.c_str() << std::endl;
        std::exit(7758258);
    }
    catch (const std::exception& e) {
        std::cout << "[----OnnxBatchRun error----] " << e.what() << std::endl;
        if (_session != nullptr) { delete _session; }
        std::exit(7758258);
    }
}

void Otool::OnnxTool::Preprocessing(const std::vector<cv::Mat>& inputImages, cv::Mat& blobImage) {
    try {
        std::cout << "----Using default Preprocessing!----" << std::endl;
        std::vector<cv::Mat> middleImages;
        for (size_t i = 0; i < inputImages.size(); ++i) {
            cv::Mat img_m;
            inputImages[i].convertTo(img_m, CV_32F);   // 转float
            Letterbox(img_m, cv::Size(_modelWidth, _modelHeight));   // 填充并resize
            middleImages.push_back(img_m);
        }
        blobImage = cv::dnn::blobFromImages(middleImages, 1. / 255., cv::Size(_modelWidth, _modelHeight), cv::Scalar(0, 0, 0), false);
    }
    catch (const std::string& _msg) {
        std::cout << "[----error message----] " << _msg.c_str() << std::endl;
        std::exit(7758258);
    }
    catch (const std::exception& e) {
        std::cout << "[----Preprocessing error----] " << e.what() << std::endl;
        if (_session != nullptr) { delete _session; }
        std::exit(7758258);
    }
}

void Otool::OnnxTool::Postprocessing(const std::vector<float*>& outputVector, std::vector<std::vector<Info>>& resInfo) {
    try {
        std::cout << "----Using default Postprocessing! [detector] [yolov" + std::to_string(detectorVersion) + "] ----" << std::endl;
        // 遍历所有batchSize的图像
        for (size_t i = 0; i < outputVector.size(); ++i) {
            std::vector<Info> tmp_info;
            if (detectorVersion == 10) {
                detect_yolov10(outputVector[i], tmp_info, i);
                resInfo.push_back(tmp_info);
            }
            else if (detectorVersion == 8) {
                detect_yolov8(outputVector[i], tmp_info, i);
                resInfo.push_back(tmp_info);
            }
        }
    }
    catch (const std::exception& e) {
        std::cout << "[----PostProcessing error----] " << e.what() << std::endl;
        if (_session != nullptr) { delete _session; }
        std::exit(7758258);
    }
}

void Otool::OnnxTool::SavePicture(const cv::Mat& image, const std::vector<Info>& resInfo, const std::string& savePath) {
    try {
        if (OnnxTool::_color.empty()) {
            OnnxTool::SetColor();
        }
        for (auto& info : resInfo) {
            cv::Scalar color = OnnxTool::_color[info._classId];
            cv::rectangle(image, info._rect, color, 2);
            int x = info._rect.x;
            int y = info._rect.y - 5;
            // cv::putText(image, "class: " + OnnxTool::_className[info._classId] + " : " + std::to_string(info._confidence).substr(0, 4), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::putText(image, OnnxTool::_className[info._classId] + " " + std::to_string(info._confidence).substr(0, 4), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
        cv::imwrite(savePath, image);
    }
    catch (const std::exception& e) {
        std::cout << "[----SavePicture error----] " << e.what() << std::endl;
        if (_session != nullptr) { delete _session; }
        std::exit(7758258);
    }
}

void Otool::OnnxTool::Letterbox(cv::Mat& src, const cv::Size& size) {
    try {
        int origW = src.cols;
        int origH = src.rows;
        int tarW = size.width;
        int tarH = size.height;

        SetOrigSize(origW, origH);

        float ratio = std::min(float(tarH) / origH, float(tarW) / origW);

        int _spreWidth = round(origW * ratio);
        int _spreHeight = round(origH * ratio);

        _preWidth.push_back(_spreWidth);
        _preHeight.push_back(_spreHeight);

        int padW = (tarW - _spreWidth) / 2;
        int padH = (tarH - _spreHeight) / 2;

        cv::resize(src, src, cv::Size(_spreWidth, _spreHeight));

        int top = int(round(padH - 0.1));
        int bottom = int(round(padH + 0.1));
        int left = int(round(padW - 0.1));
        int right = int(round(padW + 0.1));

        _padWidth.push_back(left);
        _padHeight.push_back(top);

        cv::copyMakeBorder(src, src, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
    }
    catch (const std::exception& e) {
        std::cout << "[----Letterbox error----] " << e.what() << std::endl;
    }
}

void Otool::OnnxTool::detect_yolov10(const float* output, std::vector<Info>& resInfo, int index) {
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
            resInfo.push_back(Info(_rect, confidence, classId));
        }
    }
}

void Otool::OnnxTool::detect_yolov8(float* output, std::vector<Info>& resInfo, int index) {
    // yolov8返回结果格式为[batchsize,classnum+4,cnt]
    // classnum+4中前四个为框的x,y,w,h，后classnum个为属于每个框的置信度
    int _iorigW = _origWidth[index], iorigH = _origHeight[index];
    int _ipadW = _padWidth[index], _ipadH = _padHeight[index];
    int _ipreW = _preWidth[index], _ipreH = _preHeight[index];

    std::vector<cv::Rect> _rects_nms;
    std::vector<float> _confidence_nms;
    std::vector<int> _class;
    cv::Mat all_scores = cv::Mat(cv::Size(_outputTensorShape[2], _outputTensorShape[1]), CV_32F, output).t();
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
        pdata += _outputTensorShape[1];
    }
    std::vector<int> _idx_nms;
    cv::dnn::NMSBoxes(_rects_nms, _confidence_nms, 0, threshold_nms, _idx_nms);
    for (auto& index : _idx_nms) {
        resInfo.push_back(Info(_rects_nms[index], _confidence_nms[index], _class[index]));
    }
}




// protected
void Otool::OnnxTool::SetBatchSize(int num) {
    _batchSize = num;
}

void Otool::OnnxTool::SetOrigSize(int width, int height) {
    _origWidth.push_back(width);
    _origHeight.push_back(height);
}

void Otool::OnnxTool::SetThresholdConfidence(float confidence) {
    threshold_confidence = confidence;
}

void Otool::OnnxTool::SetThresholdNMS(float nms) {
    threshold_nms = nms;
}

void Otool::OnnxTool::SetDectorVersion(int version) {
    detectorVersion = version;
}

// Static
int Otool::OnnxTool::GetObjNum() {
    return _className.size();
}

void Otool::OnnxTool::SetColor() {
    srand(0);
    for (size_t i = 0; i < OnnxTool::GetObjNum(); ++i) {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        OnnxTool::_color.push_back(cv::Scalar(b, g, r));
    }
}

void Otool::OnnxTool::SetClasses(const std::string& fileName) {
    try {
        std::ifstream f(fileName);
        OnnxTool::_className.clear();
        std::string line;
        while (std::getline(f, line)) {
            OnnxTool::_className.push_back(line);
        }
        f.close();
    }
    catch (const std::exception& e) {
        std::cout << "[SetClasses] error! :" << e.what() << std::endl;
    }
}