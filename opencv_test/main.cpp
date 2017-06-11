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
//string im_path = "F:\\фото тачек 2\\гранта croped";

int binCount = 8;

int main() {

	//std::cout << "Hello!" << std::endl;


	cv::CascadeClassifier plateCascade;
	plateCascade.load(cascade_Path);



	path pathToPhotos = "F:\\Priora\\Bad croped croped";//"C:\\curs\\Test"; //"C:\\CarModel\\поло";// "C:\\curs\\Test"; 
	string cur, next;
	for(directory_iterator it(pathToPhotos); it != directory_iterator(); ++it)
	{
		cur = pathToPhotos.string() + "/" + it->path().string();
		std::cout << cur << "\n";

		cv::Mat src = cv::imread(cur, cv::IMREAD_COLOR); // Инициализация изображения
		cv::Mat gray;
		cvtColor(src, gray, cv::COLOR_BGR2GRAY); // Перевод в чёрно-белое

		cv::imshow("mat", gray);
		cv::waitKey();

		int margin = gray.rows / 8;


		std::vector<cv::Rect> symbols;
		plateCascade.detectMultiScale(gray, symbols); // Поиск с помощью каскада
		for(auto& p : symbols)
		{
			//double ratio  = 1.0 *p.width / gray.rows;
			//std::cout<<ratio<<endl;


			//if(p.y > margin && p.y < gray.rows - 1.5 * margin)
			//{

			//cv::Point symbolBegin	= cv::Point(p.x, p.y);
			//cv::Point symbolEnd		= cv::Point(p.x+p.width, p.y+p.height);

			std::cout <<"X: " << p.x << " Y: " << p.y << " Width: " << p.width << " Height: " << p.height << std::endl;

			//cv::rectangle(src,p, cv::Scalar(0,255,0),2);
			//rectangle(src, symbolBegin, symbolEnd, cv::Scalar(0,255,0), 2);	

			//morda
			cv::Rect morda = cv::Rect(p.x - p.width, p.y - p.height, p.width * 3, p.height * 3);
			//	rectangle(src, morda, cv::Scalar(0,0,255), 2);	
			//}

		}


		Mat nomer = gray( Range(symbols[0].y, symbols[0].y + symbols[0].height), Range(symbols[0].x, symbols[0].x + symbols[0].width) ); 
		//--------------------------Hough


		cv::imshow("mat", nomer);
		cv::waitKey();
		Mat  dst, color_dst;
		Canny(nomer,dst,15, 100 );
		cvtColor( nomer, nomer, COLOR_GRAY2BGR );


		//cv::imshow("mat", dst);
		//cv::waitKey();
		// нахождение линий


		vector<Vec4i> lines;
		HoughLinesP( dst, lines, 1, CV_PI/180, 80, 30, 10 );
		for( size_t i = 0; i < lines.size(); i++ )
		{
			line( nomer, Point(lines[i][0], lines[i][1]), Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 2, 8 );
		}


		/*cv::imshow("mat", nomer);
		cv::waitKey();*/


		//-------------------------------------------------------Выравнивание
		//Надо найти самый длинный, 


		if(lines.empty())
			throw;
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


		float x0 = lines[i_max][0];
		float y0 = lines[i_max][1];
		float x1 = lines[i_max][2];
		float y1 = lines[i_max][3];

		//матрица вращения вокруг середины самого длинного отрезка
		//Point2f center((x0 + x1) / 2, (y0 + y1) / 2);
		Point2f center((x0 + x1) / 2 + symbols[0].x, (y0 + y1) / 2 + symbols[0].y);

		Mat rotMat = getRotationMatrix2D(center, atan((y1-y0)/(x1-x0)) * 180 / 3.1416, 1);

		

		warpAffine( src, src, rotMat, src.size() );


	/*	cv::imshow("mat", src);
		cv::waitKey();*/


		//int lenNumber = symbols[0].
		cv::Rect morda = cv::Rect(center.x - symbols[0].width * 0.9, center.y - symbols[0].height * 1.5, symbols[0].width * 2.4, symbols[0].height * 3);
		rectangle(src, morda, cv::Scalar(0,0,255), 2);	

		Mat roi = src(Range(center.y - symbols[0].height * 1.5, center.y + symbols[0].height * 1.5), Range(center.x - symbols[0].width * 0.9, center.x + symbols[0].width * 1.5));

		cv::imshow("mat", roi);
		cv::waitKey();

		//adaptiveThreshold(roi,roi,100,ADAPTIVE_THRESH_MEAN_C ,THRESH_BINARY,3,1);
		cvtColor( roi, roi, COLOR_BGR2GRAY );
		GaussianBlur( roi, roi, Size(3,3) , 1.6 ); 
		//adaptiveThreshold(roi, roi, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 0);
		Canny(roi, roi, 10, 60 );

		/*int erosion_size = 1;
		int erosion_type = MORPH_ELLIPSE;
		Mat element = getStructuringElement( erosion_type,
			Size( 2*erosion_size + 1, 2*erosion_size+1 ),
			Point( erosion_size, erosion_size ) );

		erode(roi,roi,element);*/
		int erosion_size = 2;
		int erosion_type = MORPH_ELLIPSE;
		Mat element2 = getStructuringElement( erosion_type,
			Size( 2*erosion_size + 1, 2*erosion_size+1 ),
			Point( erosion_size, erosion_size ) );
		dilate(roi,roi,element2);
		
		cv::imshow("mat", roi);
		cv::waitKey();
	}

	return 0;
}