// C++ standard headers
#include <iostream>
#include <fstream>
#include <memory>
#include <chrono>
#include <random>

#include <detector.hpp>
#include "cxxopts.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/gapi/stereo.hpp>

#include <torch/torch.h>

// ROS headers
#include <rclcpp/rclcpp.hpp>
#include <bumblebee2_ros_driver/msg/stereo_image.hpp>

// Global variables
int input_argc;
const char **input_argv;

const float SCORE_THRESHOLD = 0.7; // score of cone detection
const int DISP_THRESHOLD = 40;
const int EPIPOLAR_TOLERANCE = 4;


// ***************** RESCALED RECTIFIED CAMERA MODELS ******************
// left rectified model
float fx_left = 1609.856176264433;
float fy_left = 1609.856176264433;
float cx_left = 1201.695770263672;
float cy_left = 812.000862121582;
// right rectified model
float fx_right = 1609.856176264433;
float fy_right = 1609.856176264433;
float cx_right = 1225.234039306641;
float cy_right = 812.000862121582;
// rectified baseline
float rect_baseline = 23.538269043;

std::vector<float> left_dist_coeffs{-3.7152596406995730e-01, 2.4024546380721631e-01, 8.0797409468957426e-04,
                                    -1.5833606305018795e-04, -1.2147409531133743e-01, 0., 0., 0.};
std::vector<float> right_dist_coeffs{-3.6759469062780470e-01, 2.2688413072653429e-01, 4.3216084268341573e-04,
                                     5.5310709486644440e-05, -1.0353544521627576e-01, 0., 0., 0.};
std::vector<float> zero_dist_coeffs{0.0, 0.0, 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0};

cv::Mat tvec{-0.1199111738126831, 0.0002864946717650346, 0.0008028565063186949};
cv::Mat zero_tvec{0.0, 0.0, 0.0};

class YoloDetector : public rclcpp::Node
{

public:
    // variables
    int img_number = 0;

    // constructor
    YoloDetector(const rclcpp::NodeOptions &options) : Node("yolo_detector", options)
    {
        cv::namedWindow("left", cv::WINDOW_NORMAL);
        cv::namedWindow("right", cv::WINDOW_NORMAL);
        cv::resizeWindow("left", 1024, 768);
        cv::resizeWindow("right", 1024, 768);

        ModelSetup();

        performDetection();
    }
    virtual ~YoloDetector(){};

private:
    // variables
    rclcpp::Subscription<bumblebee2_ros_driver::msg::StereoImage>::SharedPtr img_sub;
    std::vector<std::string> class_names;
    std::string weights;
    float conf_thres;
    float iou_thres;
    torch::DeviceType device_type;
    std::string labels_path = "/home/navid/eufs_ws/slam_module/src/graph_based_slam/DeepNetPorting/coco_labels/fsoco.names";
    std::string left_path = "/media/navid/FiLix/imgs/raw_images/left/";
    std::string right_path = "/media/navid/FiLix/imgs/raw_images/right/";

    std::string left_rect_path = "/media/navid/FiLix/imgs/rectified_images/left/";
    std::string right_rect_path = "/media/navid/FiLix/imgs/rectified_images/right/";

    std::string left_yolo_path = "/media/navid/FiLix/imgs/yolo_images/left/";
    std::string right_yolo_path = "/media/navid/FiLix/imgs/yolo_images/right/";

    // methods

    /**
     * TODO:
     */
    int performDetection()
    {
        // read images from a folder
        // perform cones detection with yolo
        // perform stereo matching to find x,y coords
        //      '---> convert disparity into m and get positions in camera system
        // write the result to a txt file

        //************************** LOAD RAW IMAGES (only for rescaling) *******************
        // std::string raw_left_path = "/home/alessandra/Documenti/thesis/cameras_experiments/scena_mix/raw_images/left/";
        // std::string raw_right_path = "/home/alessandra/Documenti/thesis/cameras_experiments/scena_mix/raw_images/right/";
        //**************************************************************

        // rectified rescaled images
        std::string res_left_path = "/media/navid/FiLix/imgs/rectified_images/left/";
        std::string res_right_path = "/media/navid/FiLix/imgs/rectified_images/right/";
        // rescaled yolo output folders
        std::string res_left_yolo_path = "/media/navid/FiLix/imgs/yolo_images/left/";
        std::string res_right_yolo_path = "/media/navid/FiLix/imgs/yolo_images/right/";

        // 1 - load saved images from folders
        std::vector<cv::String> fn_left, fn_right;
        std::vector<cv::Mat> imgs_left, imgs_right;
        cv::glob(res_left_yolo_path + "*.png", fn_left, false); // rectified images
        cv::glob(res_right_yolo_path + "*.png", fn_right, false);
        for (size_t i = 0; i < fn_left.size(); i++)
        {
            imgs_left.push_back(cv::imread(fn_left[i])); // NB: imread reads in BGR color space
            imgs_right.push_back(cv::imread(fn_right[i]));
        }

        // 2 - load saved YOLO detections from txt files
        std::vector<cv::String> fn_txt_left, fn_txt_right;
        std::vector<cv::Mat> txt_left, txt_right;
        cv::glob(res_left_yolo_path + "*.txt", fn_txt_left, false);
        cv::glob(res_right_yolo_path + "*.txt", fn_txt_right, false);

        // 3 - stereo matching for every image
        for (size_t i = 0; i < imgs_left.size(); i++)
        {
            // current pair of images
            cv::Mat left_image = imgs_left[i];
            cv::Mat right_image = imgs_right[i];

            std::vector<std::vector<Detection>> result_left, result_right; // detections in every image
            std::vector<Detection> det_vec_l, det_vec_r;

            // read detection of left img
            std::ifstream l_infile(fn_txt_left[i]);
            std::string l_line;
            std::cout << "... reading left detections ...\n";
            while (getline(l_infile, l_line))
            {
                Detection det_line;
                std::stringstream line_stream(l_line);
                line_stream >> det_line.bbox.x >> det_line.bbox.y >> det_line.bbox.width >> det_line.bbox.height >> det_line.score >> det_line.class_idx;
                det_vec_l.push_back(det_line);
            }
            result_left.push_back(det_vec_l);

            l_infile.close();

            // read detection of right img
            std::ifstream r_infile(fn_txt_right[i]);
            std::string r_line;
            std::cout << "... reading right detections ...\n";
            while (getline(r_infile, r_line))
            {
                Detection det_line;
                std::stringstream line_stream(r_line);
                line_stream >> det_line.bbox.x >> det_line.bbox.y >> det_line.bbox.width >> det_line.bbox.height >> det_line.score >> det_line.class_idx;
                det_vec_r.push_back(det_line);
            }
            result_right.push_back(det_vec_r);

            r_infile.close();

            std::cout << result_left[0].size() << "\n";
            std::cout << result_right[0].size() << "\n";

            std::vector<cv::Point> centers_l, centers_r;

            // process the result
            // search along the epipolar line for the corresponding bb in the other image

            std::vector<cv::Point3f> sparse_pts_to_reproj;
            std::vector<cv::Point3f> sparse_pts_to_reproj_test; // try to reproject right centers in red

            for (auto l_det : result_left[0])
            {
                auto lbb_tl = l_det.bbox.tl(); // retrieve tl corner of left bb
                for (auto r_det : result_right[0])
                {
                    auto rbb_tl = r_det.bbox.tl();
                    /*
                     * To match bboxes in left and right images:
                     *  - must be in a neighborhood of 2 pixels along y (instead of only precisely on epipolar line)
                     *  - must have same class (cone color)
                     *  - must have at least a certain confidence score
                     *  - must be at least 10 px height (to cut too far away cones)
                     *
                     * Then, compute depth both as:
                     *      --> (focal * baseline) / disparity
                     *      --> 1 / disparity
                     */
                    if (abs(rbb_tl.y - lbb_tl.y) <= EPIPOLAR_TOLERANCE && r_det.class_idx == l_det.class_idx && l_det.score > SCORE_THRESHOLD && l_det.bbox.height > 10)
                    {
                        // // compute the center of the bbox, to average the variance of the rectangle
                        cv::Point l_center(lbb_tl.x + (l_det.bbox.width / 2.0), lbb_tl.y + l_det.bbox.height / 2.0);
                        cv::Point r_center(rbb_tl.x + (r_det.bbox.width / 2.0), rbb_tl.y + r_det.bbox.height / 2.0);

                        // if (abs(lbb_tl.x - rbb_tl.x) < DISP_THRESHOLD)
                        if (abs(l_center.x - r_center.x) < DISP_THRESHOLD)
                        {
                            centers_l.push_back(l_center);
                            centers_r.push_back(r_center);

                            // float disparity = lbb_tl.x - rbb_tl.x;
                            float disparity = l_center.x - r_center.x;
                            if (disparity == 0)
                                disparity = 0.01;
                            float depth1 = (fy_right * rect_baseline) / disparity;
                            float depth2 = (fy_left * rect_baseline) / disparity;

                            std::cout << "Found BB correspondence\n";
                            std::cout << "dimension is " << l_det.bbox.height << "\n";
                            std::cout << "color is " << l_det.class_idx << "\n";
                            std::cout << "left tl corner is " << lbb_tl.x << " " << lbb_tl.y << "\n";
                            std::cout << "right tl corner is " << rbb_tl.x << " " << rbb_tl.y << "\n";
                            std::cout << "left center is " << l_center.x << " " << l_center.y << "\n";
                            std::cout << "right center is " << r_center.x << " " << r_center.y << "\n";
                            std::cout << "disparity = " << disparity << " --> depth = " << depth1 << " or " << depth2 << "\n\n\n";

                            // 2D img point to be reprojected in 3D
                            // start from left centers
                            sparse_pts_to_reproj.push_back(cv::Point3f(l_center.x, l_center.y, disparity));
                            // start from right centers
                            sparse_pts_to_reproj_test.push_back(cv::Point3f(r_center.x, r_center.y, -disparity));
                        }
                    }
                }
            }

            centers_l.clear();
            centers_r.clear();


            // ******************* RESCALED Q MATRICES ****************
            cv::Mat q_matrix = cv::Mat::zeros(4, 4, CV_32F);
            q_matrix.at<float>(0, 0) = 1.0;
            q_matrix.at<float>(1, 1) = 1.0;
            q_matrix.at<float>(0, 3) = -1201.695770263672;
            q_matrix.at<float>(1, 3) = -812.000862121582;
            q_matrix.at<float>(2, 3) = 1609.856176264433;
            q_matrix.at<float>(3, 2) = 8.33929566858118;
            q_matrix.at<float>(3, 3) = 196.2925850759278;

            cv::Mat inv_q_matrix = cv::Mat::zeros(4, 4, CV_32F);
            inv_q_matrix.at<float>(0, 0) = 1.0;
            inv_q_matrix.at<float>(1, 1) = 1.0;
            inv_q_matrix.at<float>(0, 3) = -1225.234039306641;
            inv_q_matrix.at<float>(1, 3) = -812.000862121582;
            inv_q_matrix.at<float>(2, 3) = 1609.856176264433;
            inv_q_matrix.at<float>(3, 2) = -8.33929566858118;
            inv_q_matrix.at<float>(3, 3) = 196.2925850759278;
            //*********************************************************

            // get 3D points from left centers
            std::vector<cv::Point3f> real_3D_pts(sparse_pts_to_reproj.size());
            cv::perspectiveTransform(sparse_pts_to_reproj, real_3D_pts, q_matrix);

            // same for right centers
            std::vector<cv::Point3f> test_real_3D_pts(sparse_pts_to_reproj_test.size());
            cv::perspectiveTransform(sparse_pts_to_reproj_test, test_real_3D_pts, inv_q_matrix);

            std::cout << "++++++++++++++++ IMAGE " << i << " ++++++++++++++++++\n";
            for (int i = 0; i < real_3D_pts.size(); i++)
            {
                std::cout << "3D coordinates of cone " << i << " (from left centers): \n";
                std::cout << real_3D_pts[i] << "\n";

                std::cout << "3D coordinates of cone " << i << " (from right centers): \n";
                std::cout << test_real_3D_pts[i] << "\n";
            }

            // reproject back to (rectified) image plane
            // take the 3D point, draw a rectangle around it and reproject the corners into the image plane
            //  --> if I obtain something similar to the BB, then it is correct
            // or simply reproject the point and see if it corresponds to the center of the BB

            // cv::Mat left_cam_mat = GetCameraMat(fx_left, fy_left, cx_left, cy_left);
            // cv::Mat right_cam_mat = GetCameraMat(fx_right, fy_right, cx_right, cy_right);

            // std::vector<cv::Point2f> left_img_points, right_img_points, test_left_img_points, test_right_img_points;
            // cv::Mat rvec, inv_rvec;

            // // convert rotation matrix to r_vec
            // cv::Rodrigues(GetRotMat(), rvec);

            // // same for right centers
            // cv::Rodrigues(GetRotMat().inv(), inv_rvec);

            // // project into the left image plane
            // // no transform because we are in left camera frame
            // cv::projectPoints(real_3D_pts, zero_tvec, zero_tvec, left_cam_mat, zero_dist_coeffs, left_img_points);
            // cv::projectPoints(real_3D_pts, rvec, tvec, right_cam_mat, zero_dist_coeffs, right_img_points);

            // // same for right centers
            // cv::projectPoints(test_real_3D_pts, inv_rvec, -tvec, left_cam_mat, zero_dist_coeffs, test_left_img_points);
            // cv::projectPoints(test_real_3D_pts, zero_tvec, zero_tvec, right_cam_mat, zero_dist_coeffs, test_right_img_points);

            // // draw the projected points to compare with the BB centers
            // for (int j = 0; j < left_img_points.size(); j++)
            // {
            //     std::cout << "GREEN: reprojection into left image (from left centers) ** 2x focal ** " << left_img_points[j].x << " " << left_img_points[j].y << "\n";
            //     std::cout << "GREEN: reprojection into right image (from left centers) ** 2x focal ** " << right_img_points[j].x << " " << right_img_points[j].y << "\n";
            //     cv::circle(left_image, left_img_points[j], 1, cv::Scalar(0, 255, 0), 4);
            //     cv::circle(right_image, right_img_points[j], 1, cv::Scalar(0, 255, 0), 4);

            //     std::cout << "RED: reprojection into left image (from right centers) ** 2x focal ** " << test_left_img_points[j].x << " " << test_left_img_points[j].y << "\n";
            //     std::cout << "RED: reprojection into right image (from right centers) ** 2x focal ** " << test_right_img_points[j].x << " " << test_right_img_points[j].y << "\n\n";
            //     cv::circle(left_image, test_left_img_points[j], 1, cv::Scalar(0, 0, 255), 4);
            //     cv::circle(right_image, test_right_img_points[j], 1, cv::Scalar(0, 0, 255), 4);
            // }

            // // *** 3D POINTS WITH ORIGINAL IMAGE SCALE ***
            // // reproject into the scaled images also the 3D points computed with the original images
            // // to see if the error of the reprojection between left and right images is smaller

            // // ++++++++++++++++ IMAGE 0 ++++++++++++++++++
            // // CONE (left centers) 0:
            // // [-0.223043, 1.28451, 3.]
            // // CONE (right centers) 0:
            // // [-0.342957, 1.2748, 3.90842]

            // // CONE (left centers) 1:
            // // [0.766825, 1.28451, 3.90842]
            // // CONE (right centers) 1:
            // // [0.646911, 1.2748, 3.90842]

            // // CONE (left centers) 2:
            // // [-0.213428, 1.30288, 6.14708]
            // // CONE (right centers) 2:
            // // [-0.333342, 1.30288, 6.14708]

            // // CONE (left centers) 3:
            // // [0.760786, 1.2321, 5.77927]
            // // CONE (right centers) 3:
            // // [0.640872, 1.22492, 5.77927]

            // std::vector<cv::Point3f> original_3D_pts_left;
            // original_3D_pts_left.push_back(cv::Point3f(-0.223043, 1.28451, 3.));
            // original_3D_pts_left.push_back(cv::Point3f(0.766825, 1.28451, 3.90842));
            // original_3D_pts_left.push_back(cv::Point3f(-0.223043, 1.28451, 3.));
            // original_3D_pts_left.push_back(cv::Point3f(0.760786, 1.2321, 5.77927));

            // std::vector<cv::Point3f> original_3D_pts_right;
            // original_3D_pts_right.push_back(cv::Point3f(-0.342957, 1.2748, 3.90842));
            // original_3D_pts_right.push_back(cv::Point3f(0.646911, 1.2748, 3.90842));
            // original_3D_pts_right.push_back(cv::Point3f(-0.333342, 1.30288, 6.14708));
            // original_3D_pts_right.push_back(cv::Point3f(0.640872, 1.22492, 5.77927));

            // std::vector<cv::Point2f> orig_left_img_points, orig_right_img_points, orig_test_left_img_points, orig_test_right_img_points;

            // // project into the left image plane
            // cv::projectPoints(original_3D_pts_left, zero_tvec, zero_tvec, left_cam_mat, zero_dist_coeffs, orig_left_img_points);
            // cv::projectPoints(original_3D_pts_left, rvec, tvec, right_cam_mat, zero_dist_coeffs, orig_right_img_points);

            // // same for right centers
            // cv::projectPoints(original_3D_pts_right, inv_rvec, -tvec, left_cam_mat, zero_dist_coeffs, orig_test_left_img_points);
            // cv::projectPoints(original_3D_pts_right, zero_tvec, zero_tvec, right_cam_mat, zero_dist_coeffs, orig_test_right_img_points);

            // // draw the projected points to compare with the BB centers
            // for (int j = 0; j < orig_left_img_points.size(); j++)
            // {
            //     std::cout << "BLUE: reprojection into left image (from left centers) ** orig focal ** \n"
            //               << orig_left_img_points[j].x << " " << orig_left_img_points[j].y << "\n";
            //     std::cout << "BLUE: reprojection into right image (from left centers) ** orig focal ** \n"
            //               << orig_right_img_points[j].x << " " << orig_right_img_points[j].y << "\n";
            //     cv::circle(left_image, orig_left_img_points[j], 1, cv::Scalar(255, 0, 0), 4);
            //     cv::circle(right_image, orig_right_img_points[j], 1, cv::Scalar(255, 0, 0), 4);

            //     std::cout << "FUCHSIA: reprojection into left image (from right centers) ** orig focal ** \n"
            //               << orig_test_left_img_points[j].x << " " << orig_test_left_img_points[j].y << "\n";
            //     std::cout << "FUCHSIA: reprojection into right image (from right centers) ** orig focal ** \n"
            //               << orig_test_right_img_points[j].x << " " << orig_test_right_img_points[j].y << "\n\n";
            //     cv::circle(left_image, orig_test_left_img_points[j], 1, cv::Scalar(255, 0, 255), 4);
            //     cv::circle(right_image, orig_test_right_img_points[j], 1, cv::Scalar(255, 0, 255), 4);
            // }

            // // *******************************************

            // cv::namedWindow("match", cv::WINDOW_NORMAL);
            // cv::Mat match_img;
            // cv::hconcat(left_image, right_image, match_img);

            // resize(match_img, match_img, cv::Size(0, 0), 0.5, 0.5, cv::INTER_CUBIC); // double scale

            // cv::imshow("match", match_img);
            // cv::imwrite("/home/alessandra/Scrivania/match.png", match_img);
            // cv::waitKey();

            // reproject to raw image planes
            // load raw images from folders
            // std::vector<cv::String> raw_fn_left, raw_fn_right;
            // std::vector<cv::Mat> raw_imgs_left, raw_imgs_right;
            // cv::glob(left_path + "*.png", raw_fn_left, false); // rectified images
            // cv::glob(right_path + "*.png", raw_fn_right, false);
            // for (size_t i = 0; i < raw_fn_left.size(); i++)
            // {
            //     raw_imgs_left.push_back(cv::imread(raw_fn_left[i])); // NB: imread reads in BGR color space
            //     raw_imgs_right.push_back(cv::imread(raw_fn_right[i]));
            // }
        }
    }

   /*
     * Parse input parameters, load labels and weight for the model and set confidence and IOU thresholds.
     */
    void ModelSetup()
    {
        cxxopts::Options parser(input_argv[0], "A LibTorch implementation of the yolov5");
        // TODO: add other args
        parser.allow_unrecognised_options().add_options()("weights", "cone_detection.torchscript.pt path", cxxopts::value<std::string>())("source", "source", cxxopts::value<std::string>())("conf-thres", "object confidence threshold", cxxopts::value<float>()->default_value("0.4"))("iou-thres", "IOU threshold for NMS", cxxopts::value<float>()->default_value("0.5"))("gpu", "Enable cuda device or cpu", cxxopts::value<bool>()->default_value("false"))("view-img", "display results", cxxopts::value<bool>()->default_value("false"))("h,help", "Print usage");

        cxxopts::ParseResult opt = parser.parse(input_argc, input_argv);

        if (opt.count("help"))
        {
            std::cout << parser.help() << std::endl;
            exit(0);
        }

        // check if gpu flag is set
        bool is_gpu = opt["gpu"].as<bool>();

        // set device type - CPU/GPU
        if (torch::cuda::is_available())
        {
            device_type = torch::kCUDA;
            std::cout << "Cuda is available" << std::endl;
        }
        else
        {
            device_type = torch::kCPU;
            std::cout << "Cuda is not available" << std::endl;
        }

        // load class names from dataset for visualization
        class_names = YoloDetector::LoadNames(labels_path);
        if (class_names.empty())
        {
            std::cout << "Class names is empty. Error starting the program!\n";
            return;
        }

        // load network weights
        weights = opt["weights"].as<std::string>();

        // set up threshold
        conf_thres = opt["conf-thres"].as<float>();
        iou_thres = opt["iou-thres"].as<float>();
        std::cout<<"Setting up model finished \n";
    }

    /**
     * Load the labels of the classes used by the model.
     * @param path The path to the labels' file.
     * @return A vector containing the labels.
     */
    std::vector<std::string> LoadNames(const std::string &path)
    {
        std::vector<std::string> class_names;
        std::ifstream infile(path);
        if (infile.is_open())
        {
            std::string line;
            while (getline(infile, line))
            {
                class_names.emplace_back(line);
            }
            infile.close();
        }
        else
        {
            std::cerr << "Error loading the class names!\n";
        }
        return class_names;
    }

    /**
     * @param fx X-axis focal length of the camera.
     * @param fy Y-axis focal length of the camera.
     * @param cx X coordinate of the principal point.
     * @param cy Y coordinate of the principal point.
     *
     * @return The camera matrix associated with the given calibration parameters.
     */
    cv::Mat GetCameraMat(float fx, float fy, float cx, float cy)
    {
        cv::Mat camera_mat = cv::Mat::zeros(3, 3, CV_32F);
        camera_mat.at<float>(0, 0) = fx;
        camera_mat.at<float>(1, 1) = fy;
        camera_mat.at<float>(2, 2) = 1.0;
        camera_mat.at<float>(0, 2) = cx;
        camera_mat.at<float>(1, 2) = cy;
        return camera_mat;
    }

    /**
     * @return The rotation matrix resulting from stereo camera calibration.
     */
    cv::Mat GetRotMat()
    {
        // rotation matrix and translation vector from stereo calib

        cv::Mat rot_mat = cv::Mat::zeros(3, 3, CV_32F);
        rot_mat.at<float>(0, 0) = 0.9999832200986878;
        rot_mat.at<float>(0, 1) = 0.002073707143006042;
        rot_mat.at<float>(0, 2) = 0.005409182909121895;
        rot_mat.at<float>(1, 0) = -0.002074285730544136;
        rot_mat.at<float>(1, 1) = 0.99999784353051;
        rot_mat.at<float>(1, 2) = 0.0001013559947295586;
        rot_mat.at<float>(2, 0) = -0.005408961061733729;
        rot_mat.at<float>(2, 1) = -0.0001125744849082649;
        rot_mat.at<float>(2, 2) = 0.9999853651265193;

        return rot_mat;
    }

    /**
     * TODO:
     */
    void Demo(std::string direction,
              cv::Mat &img,
              const std::vector<std::vector<Detection>> &detections,
              const std::vector<std::string> &class_names,
              bool label = true)
    {

        int i = 0;
        if (!detections.empty())
        {
            for (const auto &detection : detections[0])
            {
                i++;
                const auto &box = detection.bbox;
                float score = detection.score;
                int class_idx = detection.class_idx;

                // in this way, draw all the detection boxes, independently from the confidence score
                cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

                if (label)
                {
                    std::stringstream ss;
                    ss << std::fixed << std::setprecision(2) << score;
                    std::string s = class_names[class_idx] + " " + ss.str();

                    auto font_face = cv::FONT_HERSHEY_DUPLEX;
                    auto font_scale = 1.0;
                    int thickness = 1;
                    int baseline = 0;
                    auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);

                    // draw bounding box only if the confidence is high enough
                    // if (std::stof("" + ss.str()) > SCORE_THRESHOLD)
                    // {
                    //     cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);
                    //     // cv::rectangle(img,
                    //     //               cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                    //     //               cv::Point(box.tl().x + s_size.width, box.tl().y),
                    //     //               cv::Scalar(0, 0, 255), -1);
                    //     // cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                    //     //             font_face, font_scale, cv::Scalar(255, 255, 255), thickness);
                    // }
                }
            }
        }

        // cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
        if (direction == "left")
        {
            cv::imshow("left", img);
        }
        else
        {
            cv::imshow("right", img);
        }
    }
};

int main(int argc, const char *argv[])
{

    input_argc = argc;
    input_argv = argv;

    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    rclcpp::Node::SharedPtr node_ptr = std::make_shared<YoloDetector>(options);

    rclcpp::spin(node_ptr);

    rclcpp::shutdown();
    return 0;
}