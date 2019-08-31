#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main(void)
{
    std::cout << "Hello World" << std::endl;
    string path = "./image/demo.jpg";
    cv::Mat src_img = cv::imread(path);
    cv::imshow("src_img", src_img);
    waitKey(5000);

    return 0;
}