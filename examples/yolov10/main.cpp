#include "onnx.h"

using namespace std;

class My : public Otool::OnnxTool {
public:
	std::vector<cv::Mat> _inputSrcImages;
	std::vector<std::vector<Otool::Info>> _resInfo;
	std::string modelPath = "./model/yolov10n.onnx";
private:
};

bool getImgPath(int argc,char** argv,string& inputName,string& outputName){
	if(argc==3){
		inputName=argv[1];
		outputName=argv[2];
		cout<<"inputName:"<<inputName<<" outputName:"<<outputName<<endl;
	}
	else if(argc==1){
		inputName="bus.jpg";
		outputName="res.jpg";
		cout<<"[default] inputName:bus.jpg outputName:res.jpg"<<endl;
	}
	else if(argc==2){
		inputName=argv[1];
		outputName="res.jpg";
		cout<<"inputName:"<<inputName<<" outputName:res.jpg"<<endl;
	}
	else{
		cout<<"[error] usage: ./onnxDemo [inputName] [outputName]"<<endl;
		return false;
	}
	return true;
}

int main(int argc,char **argv) {
	string inputName;
	string outputName;
	
	if(!getImgPath(argc,argv,inputName,outputName)){
		return 0;
	}

	cv::Mat src_image = cv::imread(inputName);

	My mytest;
	mytest._inputSrcImages.push_back(src_image);

	mytest.SetBatchSize(mytest._inputSrcImages.size());
	mytest.SetThresholdConfidence(0.1);
	mytest.ReadModel(mytest.modelPath);
	mytest.SetDectorVersion(10);

	mytest.OnnxBatchRun(mytest._inputSrcImages, mytest._resInfo);

	mytest.SavePicture(mytest._inputSrcImages[0], mytest._resInfo[0], outputName);

	return 0;
}