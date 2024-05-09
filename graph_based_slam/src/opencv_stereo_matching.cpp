// C++ standard headers
#include <iostream>
#include <memory>
#include <chrono>

// #include <detector.hpp>
// #include "cxxopts.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/gapi/stereo.hpp>

// ROS headers
#include <rclcpp/rclcpp.hpp>

// Global variables
int input_argc;
const char **input_argv;

const float SCORE_THRESHOLD = 0.7; // score of cone detection

/*
 * Camera calibration must be performed on the single left and right cameras first.
 * Then perform stereo calibration by keeping fixed the two camera matrices, and find the extrinsics.
 */
// camera intrinsics
float fx_left = 8.3445384486091120e+02;
float fy_left = 8.3319331763415039e+02;
float cx_left = 5.3688116338358907e+02;
float cy_left = 3.9247515377754166e+02;

float fx_right = 8.4282791458461202e+02;
float fy_right = 8.4185466189136139e+02;
float cx_right = 5.4914075573301375e+02;
float cy_right = 3.8964607572973455e+02;

int numDisparities = 8; // how many pixels to slide the window over
int blockSize = 5;
// int preFilterType = 1;
// int preFilterSize = 1;
// int preFilterCap = 31;
int minDisparity = 0;      // the offset from the x-position of the left pixel at which to begin searching
int textureThreshold = 10; // filters out areas that don't have enough texture
int uniquenessRatio = 15;  // if the best matching disparity is not sufficiently better than every other disparity in the search range, the pixel is filtered out
int speckleRange = 0;      // how close in value disparities must be to be considered part of the same blob
int speckleWindowSize = 0; // number of pixels below which a disparity blob is dismissed as "speckle"
// int disp12MaxDiff = -1;

cv::Ptr<cv::StereoBM> block_matcher = cv::StereoBM::create();
cv::Mat left, right, left_gray, right_gray;

class YoloDetector : public rclcpp::Node
{
public:
    // constructor
    YoloDetector(const rclcpp::NodeOptions &options) : Node("yolo_detector", options)
    {
        std::vector<cv::String> fn_left, fn_right;
        std::vector<cv::Mat> imgs_left, imgs_right;
        cv::glob("/media/navid/FiLix/imgs/rectified_images/left/*.png", fn_left, false);
        cv::glob("/media/navid/FiLix/imgs/rectified_images/right/*.png", fn_right, false);
        //cv::glob("/home/alessandra/Scrivania/imgs/rect_left/*.png", fn_left, false);
        //cv::glob("/home/alessandra/Scrivania/imgs/rect_right/*.png", fn_right, false);

        for (size_t i = 0; i < 10; i++)
        {
            imgs_left.push_back(cv::imread(fn_left[i])); // NB: imread reads in BGR color space
            imgs_right.push_back(cv::imread(fn_right[i]));
        }

        left = imgs_left[0];
        right = imgs_right[0];

        // 3 - perform stereo matching to find cones coords
        while (true)
        {
            //***************************************************************
            // cv::Mat only_cones_img = cv::Mat::zeros(640, 640, CV_32FC3);

            //***************************************************************
            cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);

            cv::namedWindow("disparity", cv::WINDOW_NORMAL);
            cv::createTrackbar("numDisparities /16", "disparity", &numDisparities, 500, on_trackbar1);
            cv::createTrackbar("blockSize odd 3/11", "disparity", &blockSize, 50, on_trackbar2);

            // normally, no need to adjust these! (OpenCV docs)
            // cv::createTrackbar("preFilterType", "disparity", &preFilterType, 1, on_trackbar3);
            // cv::createTrackbar("preFilterSize", "disparity", &preFilterSize, 25, on_trackbar4);
            // cv::createTrackbar("preFilterCap", "disparity", &preFilterCap, 62, on_trackbar5);
            cv::createTrackbar("textureThreshold", "disparity", &textureThreshold, 100, on_trackbar6);
            cv::createTrackbar("uniquenessRatio", "disparity", &uniquenessRatio, 100, on_trackbar7);
            cv::createTrackbar("speckleRange", "disparity", &speckleRange, 500, on_trackbar8);
            cv::createTrackbar("speckleWindowSize", "disparity", &speckleWindowSize, 500, on_trackbar9);
            // cv::createTrackbar("disp12MaxDiff", "disparity", &disp12MaxDiff, 25, on_trackbar10);
            cv::createTrackbar("minDisparity", "disparity", &minDisparity, 50, on_trackbar11);

            cv::Mat disp, disparity;

            block_matcher->compute(left_gray, right_gray, disp);

            // NOTE: Code returns a 16bit signed single channel image,
            // CV_16S containing a disparity map scaled by 16. Hence it
            // is essential to convert it to CV_32F and scale it down 16 times.

            // Converting disparity values to CV_32F from CV_16S
            disp.convertTo(disparity, CV_32F, 1.0);

            // Scaling down the disparity values and normalizing them
            disparity = (disparity / 16.0f - (float)minDisparity) / ((float)numDisparities);
            std::cout<<"DISPARITY: "<< disparity << std::endl;

            // cv::namedWindow("disparity", cv::WINDOW_NORMAL);
            // cv::resizeWindow("disparity", 600, 600);
            cv::imshow("disparity", disparity);

            if (cv::waitKey(10) == 27)
            {
                cv::imwrite("/media/navid/FiLix/imgs/disparity_m.tiff", disparity);
                cv::imwrite("/media/navid/FiLix/imgs/disparity_16m.tiff", disp);
                cv::destroyAllWindows();
                break;
            }
        }
    }

private:
    static void on_trackbar1(int, void *)
    {
        block_matcher->setNumDisparities(numDisparities * 16);
        // numDisparities = numDisparities * 16;
    }

    static void on_trackbar2(int, void *)
    {
        block_matcher->setBlockSize(blockSize * 2 + 5);
        // blockSize = blockSize * 2 + 5;
    }

    // static void on_trackbar3(int, void *)
    // {
    //     block_matcher->setPreFilterType(preFilterType);
    // }

    // static void on_trackbar4(int, void *)
    // {
    //     block_matcher->setPreFilterSize(preFilterSize * 2 + 5);
    //     // preFilterSize = preFilterSize*2+5;
    // }

    // static void on_trackbar5(int, void *)
    // {
    //     block_matcher->setPreFilterCap(preFilterCap);
    // }

    static void on_trackbar6(int, void *)
    {
        block_matcher->setTextureThreshold(textureThreshold);
    }

    static void on_trackbar7(int, void *)
    {
        block_matcher->setUniquenessRatio(uniquenessRatio);
    }

    static void on_trackbar8(int, void *)
    {
        block_matcher->setSpeckleRange(speckleRange);
    }

    static void on_trackbar9(int, void *)
    {
        block_matcher->setSpeckleWindowSize(speckleWindowSize * 2);
        // speckleWindowSize = speckleWindowSize * 2;
    }

    // static void on_trackbar10(int, void *)
    // {
    //     block_matcher->setDisp12MaxDiff(disp12MaxDiff);
    // }

    static void on_trackbar11(int, void *)
    {
        int m_value = minDisparity - 25;
        //std::cout << m_value << "\n";
        block_matcher->setMinDisparity(m_value);
    }
};

int main(int argc, const char *argv[])
{

    // input_argc = argc;
    // input_argv = argv;

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    rclcpp::Node::SharedPtr node_ptr = std::make_shared<YoloDetector>(options);

    rclcpp::spin(node_ptr);

    rclcpp::shutdown();
    return 0;
}