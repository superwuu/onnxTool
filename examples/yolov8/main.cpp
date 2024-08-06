#include "onnx.h"
#include "yolov8.h"

using namespace std;

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

	Yolov8 yolov8({ src_image }, "model/yolov8n.onnx", 0.4);
	yolov8.Detection();
	yolov8.SavePic(0,outputName);

	return 0;
}