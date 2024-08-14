#include "face_detector.h"
#include "adaface.h"
#include <Eigen/Dense>

using namespace std;

int main() {
	cv::Mat img1 = cv::imread("images/img1.jpeg");
	cv::Mat img2 = cv::imread("images/img2.jpeg");
	cv::Mat img3 = cv::imread("images/img3.jpeg");

	std::vector<cv::Mat> faceImages;
	std::vector<cv::Mat> adaImages;

	faceImages = { img1,img2,img3 };

	FaceDetector facedetector(faceImages, "model/face-detector.onnx");

	facedetector.Detection();
	std::vector<std::vector<Otool::Info>> batch_infos = facedetector.GetResult();
	for (size_t i = 0; i < batch_infos.size();++i) {
		for (auto& info : batch_infos[i]) {
			if (info._confidence < 0.7) {
				continue;
			}
			cv::Mat mid = faceImages[i](info._rect);
			adaImages.push_back(mid);
		}
	}

	Adaface adaface(adaImages, "model/adaface_ir101_ms1mv2.onnx");
	adaface.Detection();
	std::vector<std::vector<Otool::Info>> info = adaface.GetResult();

	std::vector<std::vector<float>> features;
	for (size_t i = 0; i < info.size(); ++i) {
		features.push_back(info[i][0]._output);
	}

	Eigen::MatrixXf featureMatrix(features.size(), features[0].size());
	for (size_t i = 0; i < features.size(); ++i) {
		for (size_t j = 0; j < features[i].size(); ++j) {
			featureMatrix(i, j) = features[i][j];
		}
	}
	Eigen::MatrixXf normalizedFeatures = featureMatrix.rowwise().normalized();
	Eigen::MatrixXf similarityScores = normalizedFeatures * normalizedFeatures.transpose();

	std::cout << "Similarity Scores:\n" << similarityScores << std::endl;

	return 0;
}