//*****************************************************************************//
//    Программа сегментации людей на изображении на основе HOG-детектора	   //
//					и фильтра grabCut из библиотки OpenCV					   //
//*****************************************************************************//

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <vector>
#include <string.h>
#include <ctime>
#include <iostream>

using namespace std;
using namespace cv;

const size_t Factor = 2;							// Коэффициент преобразования изображения
const string imagePath = "Resources/person_01.bmp"; // Путь к обрабатываемому изображению

// Процедура нахождения прямоугольника выделения
Rect boundingRect(Mat& img);

// Процедура сегментации найденного объекта (вариант 1)
void segmentationObject(const Mat& image, Mat& mask);

// Процедура сегментации найденного объекта (вариант 2, дополнительно передается рамка выделения)
void segmentationObject(const Mat& image, Mat& mask, const Rect& r);

// Процедура подготовки изображения
void ImagePreparation(const Mat& inputImage, Mat& outputImage, const size_t Factor);

// Процедура финализации изображения
void ImageFinalization(Mat& inputImage, const size_t Factor);

int main()
{
	Mat image = imread(imagePath);
	clock_t tStart = clock();
	Mat preparedImage;
	ImagePreparation(image, preparedImage, Factor);

	// Нахождение выделения
	Mat mask;
	Rect r = boundingRect(preparedImage);
	// Печать времени нахождения выделения 
	cout << "Bounding time: " << (clock() - tStart) / (double)CLOCKS_PER_SEC << endl;
	
	// Нахождение маски
	tStart = clock();
	segmentationObject(preparedImage, mask, r);
	ImageFinalization(mask, Factor);
	// Печать времени нахождения маски
	cout << "Segmentation time: " << (clock() - tStart) / (double)CLOCKS_PER_SEC << endl;

	r.height *= Factor;
	r.width *= Factor;
	r.x *= Factor;
	r.y *= Factor;
	rectangle(image, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);

	// Вывод изображения
	imshow("image", image);
	// Вывод маски
	imshow("mask", mask);

	waitKey(0);
	return 0;
}

Rect boundingRect(Mat& img)
{
	HOGDescriptor hog;
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
	vector<Rect> found;
	hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
	Rect r;
	if (found.size() > 0)
		r = found[0];
	for (size_t i = 0; i < found.size(); i++)
	{
		if ((r.height*r.width) < (found[i].height*found[i].width))
			r = found[i];
	}

	r.x += cvRound(r.width*0.1);
	r.width = cvRound(r.width*0.8);
	r.y += cvRound(r.height*0.09);
	r.height = cvRound(r.height*0.8);
	return r;
}

void segmentationObject(const Mat& image, Mat& mask)
{
	Mat bgModel, fgModel;
	Rect r = boundingRect(image);
	if (!r.empty())
	{
		// Сегментация фильтром GrabCut
		grabCut(image, mask, r, bgModel, fgModel, 1, GC_INIT_WITH_RECT);
		compare(mask, GC_PR_FGD, mask, CMP_EQ);
	}
	else
		mask = Mat(image.size(), CV_8UC3, Scalar(0, 0, 0));
}

void segmentationObject(const Mat& image, Mat& mask, const Rect& r)
{
	Mat bgModel, fgModel;
	if (!r.empty())
	{
		// Сегментация фильтром GrabCut
		grabCut(image, mask, r, bgModel, fgModel, 1, GC_INIT_WITH_RECT);
		compare(mask, GC_PR_FGD, mask, CMP_EQ);
	}
	else
		mask = Mat(image.size(), CV_8UC3, Scalar(0, 0, 0));
}

void ImagePreparation(const Mat& inputImage, Mat& outputImage, const size_t Factor)
{
	Size size(inputImage.cols / Factor, inputImage.rows / Factor);
	resize(inputImage, outputImage, size);
}

void ImageFinalization(Mat& inputImage, const size_t Factor)
{
	Size size(inputImage.cols * Factor, inputImage.rows * Factor);
	resize(inputImage, inputImage, size);
}