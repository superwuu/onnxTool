#include "onnx.h"

Otool::OnnxTool::OnnxTool() : _memoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput))
{
    _env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "g_env");
    _sessionOptions = Ort::SessionOptions();
    _session = nullptr;
    std::cout << "[OnnxTool is constructing...]" << std::endl;
}

Otool::OnnxTool::~OnnxTool()
{
    if (_session != nullptr)
    {
        delete _session;
    }
}

// protected
bool Otool::OnnxTool::ReadModel(const std::string &modelPath)
{
    try
    {
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

        _inputHeadNum = _numInputNodes;
        _outputHeadNum = _numOutputNodes;

        Ort::AllocatorWithDefaultOptions _allocator;
        // 获取输入节点的名称和维度信息
        _inputName.push_back(std::string(""));
        auto _inputNameAlloc = _session->GetInputNameAllocated(0, _allocator);
        _inputName[0].append(_inputNameAlloc.get());
        Ort::TypeInfo _inputTypeInfo = _session->GetInputTypeInfo(0);
        auto _inputTensorInfo = _inputTypeInfo.GetTensorTypeAndShapeInfo();
        _inputTensorShape = _inputTensorInfo.GetShape();

        // 获取输出节点的名称和维度信息
        for (size_t i = 0; i < _numOutputNodes; ++i)
        {
            _outputName.push_back(std::string(""));
            auto _outputNameAlloc = _session->GetOutputNameAllocated(i, _allocator);
            _outputName[i].append(_outputNameAlloc.get());
            Ort::TypeInfo _outputTypeInfo = _session->GetOutputTypeInfo(i);
            auto _outputTensorInfo = _outputTypeInfo.GetTensorTypeAndShapeInfo();
            std::vector<std::int64_t> shape_one = _outputTensorInfo.GetShape();
            _outputTensorShape.push_back(shape_one);
            if (_outputTensorShape[i][0] == -1)
            {
                _isOutputUDFsize.push_back(true);
                _outputTensorShape[i][0] = _batchSize;
            }
            else
            {
                _isOutputUDFsize.push_back(false);
            }
        }

        // 从输入节点维度中获取输入图像的高度和宽度
        _modelHeight = _inputTensorShape[2];
        _modelWidth = _inputTensorShape[3];

        // 如果是多batch推理，则设置batchsize
        if (_inputTensorShape[0] == -1)
        {
            _isInputUDFsize = true;
            _inputTensorShape[0] = _batchSize;
        }

        return true;
    }
    catch (const std::exception &e)
    {
        std::cout << "[----readModel error----] " << e.what() << std::endl;
        if (_session != nullptr)
        {
            delete _session;
        }
        std::exit(7758258);
    }
    catch (const std::string &_msg)
    {
        std::cout << "[----error message----] " << _msg.c_str() << std::endl;
    }
    return true;
}

void Otool::OnnxTool::OnnxBatchRun(const std::vector<cv::Mat> &srcImages, std::vector<std::vector<Info>> &resInfo)
{
    try
    {
        cv::Mat blobImage;
        // 预处理
        Preprocess(srcImages, blobImage);

        // 元素长度
        int64_t _inputTensorLength = VectorProduct(_inputTensorShape);

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(_memoryInfo, (float *)blobImage.data, _inputTensorLength, _inputTensorShape.data(), _inputTensorShape.size()));

        // 推理
        const std::array<const char *, 1> _inputName_run = {_inputName[0].c_str()};
        // const std::array<const char*, 3> _outputName_run = { _outputName[0].c_str(),_outputName[1].c_str(),_outputName[2].c_str() };
        //  const std::array<const char*, 1> _outputName_run = { _outputName[0].c_str() };
        std::vector<const char *> _OutputName_vec;
        for (size_t i = 0; i < _outputName.size(); ++i)
        {
            _OutputName_vec.push_back(_outputName[i].c_str());
        }
        std::vector<Ort::Value> _ortOutputs = _session->Run(Ort::RunOptions{nullptr}, _inputName_run.data(), input_tensors.data(), _inputName_run.size(), _OutputName_vec.data(), _OutputName_vec.size());

        std::vector<size_t> level_count;
        for (size_t i = 0; i < _outputHeadNum; ++i)
        {
            level_count.push_back(_ortOutputs[i].GetTensorTypeAndShapeInfo().GetElementCount() / _batchSize);
        }

        // 对batch中每张图像
        for (size_t i = 0; i < _batchSize; ++i)
        {
            // 每张图片一个vec,存储不同输出头的内存起点
            std::vector<float *> batch_data_vec;
            for (size_t j = 0; j < _outputHeadNum; ++j)
            {
                batch_data_vec.push_back(_ortOutputs[j].GetTensorMutableData<float>() + i * level_count[j]);
            }

            Postprocess_all(batch_data_vec, resInfo, i);
        }
    }
    catch (const std::string &_msg)
    {
        std::cout << "[----error message----] " << _msg.c_str() << std::endl;
        std::exit(7758258);
    }
    catch (const std::exception &e)
    {
        std::cout << "[----OnnxBatchRun error----] " << e.what() << std::endl;
        if (_session != nullptr)
        {
            delete _session;
        }
        std::exit(7758258);
    }
}

void Otool::OnnxTool::Preprocess(const std::vector<cv::Mat> &inputImages, cv::Mat &blobImage)
{
    try
    {
        std::cout << "----Using default Preprocessing!----" << std::endl;
        std::vector<cv::Mat> middleImages;
        for (size_t i = 0; i < inputImages.size(); ++i)
        {
            cv::Mat img_m;
            inputImages[i].convertTo(img_m, CV_32F);               // 转float
            Letterbox(img_m, cv::Size(_modelWidth, _modelHeight)); // 填充并resize
            middleImages.push_back(img_m);
        }
        blobImage = cv::dnn::blobFromImages(middleImages, 1. / 255., cv::Size(_modelWidth, _modelHeight), cv::Scalar(0, 0, 0), false);
    }
    catch (const std::string &_msg)
    {
        std::cout << "[----error message----] " << _msg.c_str() << std::endl;
        std::exit(7758258);
    }
    catch (const std::exception &e)
    {
        std::cout << "[----Preprocessing error----] " << e.what() << std::endl;
        if (_session != nullptr)
        {
            delete _session;
        }
        std::exit(7758258);
    }
}

void Otool::OnnxTool::Postprocess_all(std::vector<float *> &outputVector, std::vector<std::vector<Info>> &resInfo, const int batch_index)
{
    try
    {
        std::cout << "----Postprocessing! [detector] [yolov" + std::to_string(detectorVersion) + "] ----" << std::endl;

        std::vector<Info> one_info; // 一张图像中的信息
        for (size_t i = 0; i < outputVector.size(); ++i)
        {
            Postprocess(outputVector[i], one_info, i, batch_index);
        }
        resInfo.push_back(one_info);
    }
    catch (const std::exception &e)
    {
        std::cout << "[----PostProcessing error----] " << e.what() << std::endl;
        if (_session != nullptr)
        {
            delete _session;
        }
        std::exit(7758258);
    }
}

void Otool::OnnxTool::Postprocess(float *output, std::vector<Info> &resInfo, const int level_index, const int batch_index)
{
    std::cout << "----[error] no postprocess function! Please check your class! ----" << std::endl;
}

void Otool::OnnxTool::SavePicture(const cv::Mat &image, const std::vector<Info> &resInfo, const std::string &savePath)
{
    try
    {
        if (OnnxTool::_color.empty())
        {
            OnnxTool::SetColor();
        }
        for (auto &info : resInfo)
        {
            cv::Scalar color = OnnxTool::_color[info._classId];
            cv::rectangle(image, info._rect, color, 2);
            int x = info._rect.x;
            int y = info._rect.y - 5;
            // cv::putText(image, "class: " + OnnxTool::_className[info._classId] + " : " + std::to_string(info._confidence).substr(0, 4), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            cv::putText(image, OnnxTool::_className[info._classId] + " " + std::to_string(info._confidence).substr(0, 4), cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
        cv::imwrite(savePath, image);
    }
    catch (const std::exception &e)
    {
        std::cout << "[----SavePicture error----] " << e.what() << std::endl;
        if (_session != nullptr)
        {
            delete _session;
        }
        std::exit(7758258);
    }
}

void Otool::OnnxTool::Letterbox(cv::Mat &src, const cv::Size &size)
{
    try
    {
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
    catch (const std::exception &e)
    {
        std::cout << "[----Letterbox error----] " << e.what() << std::endl;
    }
}

void Otool::OnnxTool::Letterbox_lr(cv::Mat &src, const cv::Size &size)
{
    try
    {
        int origW = src.cols;
        int origH = src.rows;
        int tarW = size.width;
        int tarH = size.height;

        SetOrigSize(origW, origH);

        float ratio = std::min(tarH / (origH * 1.0), tarW / (origW * 1.0));

        int _spreWidth = round(origW * ratio);
        int _spreHeight = round(origH * ratio);

        _preWidth.push_back(_spreWidth);
        _preHeight.push_back(_spreHeight);

        int padW = (tarW - _spreWidth);
        int padH = (tarH - _spreHeight);

        cv::resize(src, src, cv::Size(_spreWidth, _spreHeight));

        int bottom = int(round(padH + 0.1));
        int right = int(round(padW + 0.1));

        _padWidth.push_back(right);
        _padHeight.push_back(bottom);

        cv::copyMakeBorder(src, src, 0, bottom, 0, right, 0, cv::Scalar(114, 114, 114));
    }
    catch (const std::exception &e)
    {
        std::cout << "[----Letterbox_LR error----] " << e.what() << std::endl;
    }
}

void Otool::OnnxTool::Reset()
{
    _origWidth.clear();
    _origHeight.clear();
    _padWidth.clear();
    _padHeight.clear();
    _preWidth.clear();
    _preHeight.clear();
}

// protected
void Otool::OnnxTool::SetBatchSize(int num)
{
    _batchSize = num;
    if (_isInputUDFsize)
    {
        _inputTensorShape[0] = num;
    }
    for (size_t i = 0; i < _outputTensorShape.size(); ++i)
    {
        if (_isOutputUDFsize[i] == true)
        {
            _outputTensorShape[i][0] = num;
        }
    }
}

void Otool::OnnxTool::SetOrigSize(int width, int height)
{
    _origWidth.push_back(width);
    _origHeight.push_back(height);
}

void Otool::OnnxTool::SetThresholdConfidence(float confidence)
{
    threshold_confidence = confidence;
}

void Otool::OnnxTool::SetThresholdNMS(float nms)
{
    threshold_nms = nms;
}

void Otool::OnnxTool::SetDectorVersion(int version)
{
    detectorVersion = version;
}

// Static
int Otool::OnnxTool::GetObjNum()
{
    return _className.size();
}

void Otool::OnnxTool::SetColor()
{
    srand(0);
    for (size_t i = 0; i < OnnxTool::GetObjNum(); ++i)
    {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        OnnxTool::_color.push_back(cv::Scalar(b, g, r));
    }
}

void Otool::OnnxTool::SetClasses(const std::string &fileName)
{
    try
    {
        std::ifstream f(fileName);
        OnnxTool::_className.clear();
        std::string line;
        while (std::getline(f, line))
        {
            OnnxTool::_className.push_back(line);
        }
        f.close();
    }
    catch (const std::exception &e)
    {
        std::cout << "[SetClasses] error! :" << e.what() << std::endl;
    }
}