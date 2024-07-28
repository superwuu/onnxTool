#include "onnx.h"

using namespace std;

class My : public Otool::OnnxTool {
public:
	std::vector<cv::Mat> _inputSrcImages;
	std::vector<std::vector<Otool::Info>> _resInfo;
	std::string modelPath = "./model/yolov8n.onnx";
private:
};

int main() {
	cv::Mat src_image = cv::imread("bus.jpg");

	My mytest;
	mytest._inputSrcImages.push_back(src_image);

	mytest.SetBatchSize(mytest._inputSrcImages.size());
	mytest.SetThresholdConfidence(0.1);
	mytest.ReadModel(mytest.modelPath);
	mytest.SetDectorVersion(8);

	mytest.OnnxBatchRun(mytest._inputSrcImages, mytest._resInfo);

	mytest.SavePicture(mytest._inputSrcImages[0], mytest._resInfo[0], "res.jpg");

	return 0;
}