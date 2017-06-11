#include <iostream>
#include <string>
#include <filesystem>
#include<windows.h>

#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

using namespace std::tr2::sys;


std::string cascade_Path = "C:\\Users\\NoteBook\\Dropbox\\Univer\\opencv_test\\haar.xml";
//string im_path = "F:\\���� ����� 2\\������ croped";

int binCount = 8;

Vec4i getLongestLine(Mat nomer)
{
	Mat  dst, color_dst;
	Canny(nomer,dst,15, 100 );
	cvtColor( nomer, nomer, COLOR_GRAY2BGR );

	vector<Vec4i> lines;
	HoughLinesP( dst, lines, 1, CV_PI/180, 80, 30, 10 );
	for( size_t i = 0; i < lines.size(); i++ )
	{
		line( nomer, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 2, 8 );
	}

	if(lines.empty())
		return NULL;
	int i_max=0;
	float max = hypot(lines[0][0] - lines[0][2], lines[0][1] - lines[0][3]);
	for( size_t i = 1; i < lines.size(); i++ )
	{
		float cur = hypot(lines[i][0] - lines[i][2], lines[i][1] - lines[i][3]);
		if(cur > max)
		{
			max = cur;
			i_max = i;
		}
	}

	return lines[i_max];


}



int main() {

	int noalign =0;
	cv::CascadeClassifier plateCascade;
	plateCascade.load(cascade_Path);



	path pathToPhotos = "C:\\data\\train\\21099 croped";//"F:\\Priora\\Bad croped croped";//"C:\\curs\\Test"; //"C:\\CarModel\\����";// "C:\\curs\\Test"; 
	string cur, next;
	for(directory_iterator it(pathToPhotos); it != directory_iterator(); ++it)
	{
		String imagename = it->path().string();
		cur = pathToPhotos.string() + "/" + it->path().string();
		std::cout << cur << "\n";

		cv::Mat src = cv::imread(cur, cv::IMREAD_COLOR); // ������������� �����������
		cv::Mat gray;
		cvtColor(src, gray, cv::COLOR_BGR2GRAY); // ������� � �����-�����

		/*cv::imshow("mat", gray);
		cv::waitKey();*/

		int margin = gray.rows / 8;


		std::vector<cv::Rect> symbols;
		plateCascade.detectMultiScale(gray, symbols); // ����� � ������� �������
		for(auto& p : symbols)
		{
			std::cout <<"X: " << p.x << " Y: " << p.y << " Width: " << p.width << " Height: " << p.height << std::endl;

			cv::Rect morda = cv::Rect(p.x, p.y, p.width, p.height);
			//rectangle(src, morda, cv::Scalar(0,0,255), 2);	

		

		}


		Mat roi = gray;

		if(!symbols.empty())			//���� ��� ������
		{
			Mat nomer = gray( Range(symbols[0].y, symbols[0].y + symbols[0].height), Range(symbols[0].x, symbols[0].x + symbols[0].width) ); 

			//--------------------------Hough ����� ����� �� ������. ������ ����� �������, ��������� �� ����������� �� ���

			Vec4i line = getLongestLine(nomer);

			if(line != Vec4i::zeros())
			{

				float x0 = line[0];
				float y0 = line[1];
				float x1 = line[2];
				float y1 = line[3];

				//������� �������� ������ �������� ������ �������� �������

				Point2f center((x0 + x1) / 2 + symbols[0].x, (y0 + y1) / 2 + symbols[0].y);
				Mat rotMat = getRotationMatrix2D(center, atan((y1-y0)/(x1-x0)) * 180 / 3.1416, 1);
				warpAffine( src, src, rotMat, src.size() );

				//rectangle(src, // nomer, cv::Scalar(0,0,255), 2);	
				Point pt1(symbols[0].x,symbols[0].y);
				Point pt2(symbols[0].x + symbols[0].width,symbols[0].y + symbols[0].height);
				rectangle(src,pt1,pt2,(0,0,0),-3);

				//int lenNumber = symbols[0].
				//cv::Rect morda = cv::Rect(center.x - symbols[0].width * 0.9, center.y - symbols[0].height * 1.5, symbols[0].width * 2.4, symbols[0].height * 3);
				//rectangle(src, morda, cv::Scalar(0,0,255), 2);	

				y0 = center.y - symbols[0].height * 1.5;
				y1 = center.y + symbols[0].height * 1.5;
				x0 = center.x - symbols[0].width * 0.9;
				x1 = center.x + symbols[0].width * 1.5;

				x0 = std::max(x0, (float)0);
				x1 = std::max(x1, (float)0);
				y0 = std::max(y0, (float)0);
				y1 = std::max(y1, (float)0);

				x0 = std::min(x0, (float)src.cols);
				x1 = std::min(x1, (float)src.cols);
				y0 = std::min(y0, (float)src.rows);
				y1 = std::min(y1, (float)src.rows);

				roi = src(Range(y0, y1), Range(x0, x1));
				
			}
			else
			{
				//symbols[0].x
				cout<<"no align"<<endl;
				noalign++;
 
				float y0 = symbols[0].y - symbols[0].height * 1.5;
				float y1 = symbols[0].y + symbols[0].height * 1.5;
				float x0 = symbols[0].x - symbols[0].width * 0.9;
				float x1 = symbols[0].x + symbols[0].width * 1.9;

				x0 = std::max(x0, (float)0);
				x1 = std::max(x1, (float)0);
				y0 = std::max(y0, (float)0);
				y1 = std::max(y1, (float)0);

				x0 = std::min(x0, (float)src.cols);
				x1 = std::min(x1, (float)src.cols);
				y0 = std::min(y0, (float)src.rows);
				y1 = std::min(y1, (float)src.rows);
				roi = src(Range(y0, y1), Range(x0, x1));
			}

			cvtColor( roi, roi, COLOR_BGR2GRAY );
			//roi = src(Range(center.y - symbols[0].height * 1.5, center.y + symbols[0].height * 1.5), Range(center.x - symbols[0].width * 0.9, center.x + symbols[0].width * 1.5));

			/*cv::imshow("mat", roi);
			cv::waitKey();*/
		}

		//adaptiveThreshold(roi,roi,100,ADAPTIVE_THRESH_MEAN_C ,THRESH_BINARY,3,1);
		
		GaussianBlur( roi, roi, Size(3,3) , 1.6 ); 
		//adaptiveThreshold(roi, roi, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 0);
		Canny(roi, roi, 10, 60 );


		int erosion_size = 2;
		int erosion_type = MORPH_ELLIPSE;
		Mat element2 = getStructuringElement( erosion_type,
			Size( 2*erosion_size + 1, 2*erosion_size+1 ),
			Point( erosion_size, erosion_size ) );
		dilate(roi,roi,element2);

		cv::imwrite("C:\\data3\\" + it->path().string(),roi);

		/*cv::imshow("mat", roi);
		cv::waitKey();*/
		cout<<endl<< noalign<<endl;
	}

	return 0;
}