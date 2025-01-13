#include "face_detector.h"

using namespace std;

int main() {
	cv::Mat img1 = cv::imread("images/img1.jpeg");
	cv::Mat img2 = cv::imread("images/img2.jpeg");
	cv::Mat img3 = cv::imread("images/img3.jpeg");

	std::vector<cv::Mat> faceImages;
	std::vector<cv::Mat> adaImages;

	faceImages = { img1,img2,img3 };

	FaceDetector facedetector("model/face-detector.onnx");
	facedetector.SetBatchImgs(faceImages);

	facedetector.Detection();
	std::vector<std::vector<Otool::Info>> batch_infos = facedetector.GetResult();
	for (size_t i = 0; i < batch_infos.size();++i) {
		std::cout<<batch_infos[i].size()<<std::endl;
		for (auto& info : batch_infos[i]) {
			if (info._confidence < 0.7) {
				continue;
			}
			cv::Mat mid = faceImages[i](info._rect);
			adaImages.push_back(mid);
		}
	}

	return 0;
}