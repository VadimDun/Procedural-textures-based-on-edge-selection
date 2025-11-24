#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;


void getContours(Mat imgDil, Mat img);

int main() {
	//////////////////////////
	////  выделить контуры 
	/////////////////////////
	//Mat image = imread("Resources/brick_wall.jpg");
 //   Mat image = imread("Resources/brick_wall_2.jpg");
 //   if (image.empty()) {
 //       std::cout << "Could not open or find the image" << endl;
 //       return -1;
 //   }

 //   Mat gray, edges;

 //   // Конвертация в оттенки серого
 //   cvtColor(image, gray, COLOR_BGR2GRAY);

 //   // Применение детектора границ Canny
	//Canny(gray, edges, 130, 300);

 //   // Поиск контуров
 //   vector<vector<Point>> contours;
 //   vector<Vec4i> hierarchy;
 //   findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

 //   // Отрисовка контуров
 //   Mat result = image.clone();
 //   drawContours(result, contours, -1, Scalar(0, 255, 0), 2); // -1 все контуры

 //   imshow("Original", image);
 //   imshow("Edges", edges);
 //   imshow("Contours", result);


	//////////////////////////
	////  выделить контуры 2
	/////////////////////////

	//string path = "Resources/brick_wall.jpg";
	string path = "Resources/stone_wall.jpg";
	//string path = "Resources/stone_wall2.jpg";
	//string path = "Resources/lava.jpg";
	//string path = "Resources/tree_bark.jpg";

	Mat img = imread(path);
	Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

	cvtColor(img, imgGray, COLOR_BGR2GRAY);
	GaussianBlur(imgGray, imgBlur, Size(7, 7), 5, 0);
	Canny(imgBlur, imgCanny, 25, 75);

	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(imgCanny, imgDil, kernel);
	erode(imgDil, imgErode, kernel);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imgCanny, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	Mat result = img.clone();
	drawContours(result, contours, -1, Scalar(0, 255, 0), 2); // -1 все контуры

	imshow("Image", img);
	imshow("Image Gray", imgGray);
	imshow("Image Blur", imgBlur);
	imshow("Image Canny", imgCanny);
	imshow("Image Dilation", imgDil);
	imshow("Image Erode", imgErode);
	imshow("Contours", result);



	//////////////////////////
	////  выделить контуры 3 с ограничивающей рамкой(с проверкой площадью)
	/////////////////////////
	//string path = "Resources/brick_wall.jpg";
	////string path = "Resources/brick_wall_2.jpg";
	//Mat img = imread(path);
	//Mat imgGray, imgBlur, imgCanny, imgDil, imgErode;

	//// Preprocessing
	//cvtColor(img, imgGray, COLOR_BGR2GRAY);
	//GaussianBlur(imgGray, imgBlur, Size(3, 3), 3, 0);
	//Canny(imgBlur, imgCanny, 25, 75);
	//Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate(imgCanny, imgDil, kernel);

	//getContours(imgDil, img);

	//imshow("Image", img);
	//imshow("Image Gray", imgGray);
	//imshow("Image Blur", imgBlur);
	//imshow("Image Canny", imgCanny);
	//imshow("Image Dil", imgDil);


	waitKey(0);
	return 0;
}

void getContours(Mat imgDil, Mat img) {

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	findContours(imgDil, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	//drawContours(img, contours, -1, Scalar(255, 0, 255), 2);

	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());

	for (int i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		cout << area << endl;
		string objectType;

		if (area > 1000)
		{
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
			cout << conPoly[i].size() << endl;
			boundRect[i] = boundingRect(conPoly[i]);

			int objCor = (int)conPoly[i].size();

			drawContours(img, conPoly, i, Scalar(255, 0, 255), 2);
			rectangle(img, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 5);
			putText(img, objectType, { boundRect[i].x,boundRect[i].y - 5 }, FONT_HERSHEY_PLAIN, 1, Scalar(0, 69, 255), 2);
		}
	}
}