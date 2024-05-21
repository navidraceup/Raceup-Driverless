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

#include <eufs_msgs/msg/cone_array_with_covariance.hpp>
#include <eufs_msgs/msg/cone_with_covariance.hpp>

// TF2 headers
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/exceptions.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/transform_datatypes.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

// Global variables
int input_argc;
const char **input_argv;

const float SCORE_THRESHOLD = 0.7; // score of cone detection
const int DISP_THRESHOLD = 8;
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
float rect_baseline = 119.9111738126831; //new value

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
        //subscriber to images published by bumblebee stereo camera
        img_sub = this->create_subscription<bumblebee2_ros_driver::msg::StereoImage>("/stereo_image", 10, std::bind(&YoloDetector::img_callback, this, std::placeholders::_1));
        ModelSetup();

        auto detector = Detector(weights, device_type);
        // run once to warm up
        std::cout << "Run once on empty image" << std::endl;
        auto temp_img = cv::Mat::zeros(640, 640, CV_32FC3);
        auto w1 = detector.Run(temp_img, 1.0f, 1.0f);

        //publish 3d position of the cones
        cone_pub = this->create_publisher<eufs_msgs::msg::ConeArrayWithCovariance>("/cone_pose", 1);
        //graph_viz_pub = this->create_publisher<visualization_msgs::msg::Marker>("graph_viz", 1);
        
    }
    virtual ~YoloDetector(){};

private:
    // variables
    rclcpp::Subscription<bumblebee2_ros_driver::msg::StereoImage>::SharedPtr img_sub;
    rclcpp::Publisher<eufs_msgs::msg::ConeArrayWithCovariance>::SharedPtr cone_pub;

    std::vector<std::string> class_names;
    std::string weights;
    float conf_thres;
    float iou_thres;
    torch::DeviceType device_type;

    std::string labels_path = "/home/navid/eufs_ws/slam_module/src/graph_based_slam/DeepNetPorting/coco_labels/fsoco.names";
    std::string left_yolo_path = "/home/navid/eufs_ws/slam_module/src/graph_based_slam/imgs/yolo_images/left/";
    std::string right_yolo_path = "/home/navid/eufs_ws/slam_module/src/graph_based_slam/imgs/yolo_images/right/";

    int performDetection(cv::Mat_<uchar> left_img, cv::Mat_<uchar> right_img)
    {
        std::string res_left_yolo_path = "/media/navid/FiLix/imgs2/yolo_images/left/";
        std::string res_right_yolo_path = "/media/navid/FiLix/imgs2/yolo_images/right/";

        //************************************************************** TODO: ***************************************************************
        //run YOLO for cones detection
        const int fps = 20;

        auto detector = Detector(weights, device_type);

        // result has a number of layers --> consider result[0] to have a vector of detections
        // each element of this vector, a Detection, contains: cv::Rect bbox, float score, int class_index
        auto result_left = detector.Run(left_img, conf_thres, iou_thres); // NB: it converts to RGB
        auto result_right = detector.Run(right_img, conf_thres, iou_thres);
    }

    /**
     * Called whenever a new image is produced by the stereo camera. It is a custom ROS message.
     * TODO: use this when using in real time, now work directly on saved images!
     */
    void img_callback(const bumblebee2_ros_driver::msg::StereoImage::SharedPtr img_msg){
    
        // std::cout << "img_callback reached!\n";
        

        cv::Mat_<uchar> left_img(img_msg->height, img_msg->width, img_msg->left_data.data());
        cv::Mat_<uchar> right_img(img_msg->height, img_msg->width, img_msg->right_data.data());

        cv::Mat_<cv::Vec3b> left_img_rgb(left_img.rows, left_img.cols),
            right_img_rgb(right_img.rows, right_img.cols);

        // convert to RGB images
        cv::cvtColor(left_img, left_img_rgb, cv::COLOR_BayerGR2RGB);
        cv::cvtColor(right_img, right_img_rgb, cv::COLOR_BayerGR2RGB);
        
        std::string res_left_yolo_path = "/media/navid/FiLix/imgs2/yolo_images/left/";
        std::string res_right_yolo_path = "/media/navid/FiLix/imgs2/yolo_images/right/";

        auto detector = Detector(weights, device_type);

        // result has a number of layers --> consider result[0] to have a vector of detections
        // each element of this vector, a Detection, contains: cv::Rect bbox, float score, int class_index
        auto result_left = detector.Run(left_img_rgb, conf_thres, iou_thres); // NB: it converts to RGB
        auto result_right = detector.Run(right_img_rgb, conf_thres, iou_thres);


        //stereo matching
        std::vector<Detection> det_vec_l, det_vec_r;
        std::vector<cv::Point> centers_l, centers_r;

        // search along the epipolar line for the corresponding bb in the other image
        std::vector<cv::Point3f> sparse_pts_to_reproj;
        std::vector<cv::Point3f> sparse_pts_to_reproj_blue; //index 0
        std::vector<cv::Point3f> sparse_pts_to_reproj_yellow; //index 1
        std::vector<cv::Point3f> sparse_pts_to_reproj_big_orange; //index 3
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
                   yolo_cones_detection_offline.cpp *  - must be at least 10 px height (to cut too far away cones)
                    *
                    * Then, compute depth both as:
                    *      --> (focal * baseline) / disparity
                    *      --> 1 / disparity
                */
                if (abs(rbb_tl.y - lbb_tl.y) <= EPIPOLAR_TOLERANCE && l_det.score > SCORE_THRESHOLD && l_det.bbox.height > 10)// r_det.class_idx == l_det.class_idx && 
                {
                    // // compute the center of the bbox, to average the variance of the rectangle
                    cv::Point l_center(lbb_tl.x + (l_det.bbox.width / 2.0), lbb_tl.y + l_det.bbox.height / 2.0);
                    cv::Point r_center(rbb_tl.x + (r_det.bbox.width / 2.0), rbb_tl.y + r_det.bbox.height / 2.0);

                    // if (abs(lbb_tl.x - rbb_tl.x) < DISP_THRESHOLD)
                    if (abs(l_center.x - r_center.x) < DISP_THRESHOLD) //&& 
                    {
                        centers_l.push_back(l_center);
                        centers_r.push_back(r_center);

                        // float disparity = lbb_tl.x - rbb_tl.x;
                        float disparity = l_center.x - r_center.x;
                        if (disparity == 0)
                            disparity = 0.01;
                        float depth1 = (fy_right * rect_baseline) / disparity;
                        float depth2 = (fy_left * rect_baseline) / disparity;

                        // std::cout << "Found BB correspondence\n";
                        // std::cout << "dimension is " << l_det.bbox.height << "\n";
                        std::cout << "class"<<l_det.class_idx << "\n";
                        // std::cout << "left tl corner is " << lbb_tl.x << " " << lbb_tl.y << "\n";
                        // std::cout << "right tl corner is " << rbb_tl.x << " " << rbb_tl.y << "\n";
                        // std::cout << "left center is " << l_center.x << " " << l_center.y << "\n";
                        // std::cout << "right center is " << r_center.x << " " << r_center.y << "\n";
                        // std::cout << "disparity = " << disparity << " --> depth = " << depth1 << " or " << depth2 << "\n\n\n";

                        // 2D img point to be reprojected in 3D
                        // start from left centers
                        if(l_det.class_idx == 0)
                            sparse_pts_to_reproj_blue.push_back(cv::Point3f(l_center.x, l_center.y, disparity));
                        if(l_det.class_idx == 1)
                            sparse_pts_to_reproj_yellow.push_back(cv::Point3f(l_center.x, l_center.y, disparity));
                        if(l_det.class_idx == 3)
                            sparse_pts_to_reproj_big_orange.push_back(cv::Point3f(l_center.x, l_center.y, disparity));
                        else
                            sparse_pts_to_reproj_blue.push_back(cv::Point3f(l_center.x, l_center.y, disparity));
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
        std::vector<cv::Point3f> real_3D_pts_blue(sparse_pts_to_reproj_blue.size());
        if(sparse_pts_to_reproj_blue.size() != 0)
            cv::perspectiveTransform(sparse_pts_to_reproj_blue, real_3D_pts_blue, q_matrix);

        std::vector<cv::Point3f> real_3D_pts_yellow(sparse_pts_to_reproj_yellow.size());
        if(sparse_pts_to_reproj_yellow.size() != 0)
            cv::perspectiveTransform(sparse_pts_to_reproj_yellow, real_3D_pts_yellow, q_matrix);

        std::vector<cv::Point3f> real_3D_pts_big_orange(sparse_pts_to_reproj_big_orange.size());
        if(sparse_pts_to_reproj_big_orange.size() != 0)
            cv::perspectiveTransform(sparse_pts_to_reproj_big_orange, real_3D_pts_big_orange, q_matrix);


        // same for right centers
        std::vector<cv::Point3f> test_real_3D_pts(sparse_pts_to_reproj_test.size());
        cv::perspectiveTransform(sparse_pts_to_reproj_test, test_real_3D_pts, inv_q_matrix);

        //publish cones in camera frame: x: direction of car, y: to right z: up
        eufs_msgs::msg::ConeWithCovariance cone_pose_msg;
        eufs_msgs::msg::ConeArrayWithCovariance cone_pose_array_msg;
        auto now_time = this->get_clock()->now();
        for (int i = 0; i < real_3D_pts_blue.size(); i++)
        {
            cone_pose_msg.point.x = real_3D_pts_blue[i].z;
            cone_pose_msg.point.y = real_3D_pts_blue[i].x;
            cone_pose_msg.point.z = real_3D_pts_blue[i].y;
            cone_pose_array_msg.blue_cones.push_back(cone_pose_msg);
        }

        for (int i = 0; i < real_3D_pts_yellow.size(); i++)
        {
            cone_pose_msg.point.x = real_3D_pts_yellow[i].z;
            cone_pose_msg.point.y = real_3D_pts_yellow[i].x;
            cone_pose_msg.point.z = real_3D_pts_yellow[i].y;
            cone_pose_array_msg.yellow_cones.push_back(cone_pose_msg);
        }

        for (int i = 0; i < real_3D_pts_big_orange.size(); i++)
        {
            cone_pose_msg.point.x = real_3D_pts_big_orange[i].z;
            cone_pose_msg.point.y = real_3D_pts_big_orange[i].x;
            cone_pose_msg.point.z = real_3D_pts_big_orange[i].y;
            cone_pose_array_msg.big_orange_cones.push_back(cone_pose_msg);
        }


        cone_pose_array_msg.header.stamp = now_time;
        cone_pose_array_msg.header.frame_id = "base_footprint";
        cone_pub->publish(cone_pose_array_msg);
            
        // reproject
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
        //     // std::cout << "GREEN: reprojection into left image (from left centers) ** 2x focal ** " << left_img_points[j].x << " " << left_img_points[j].y << "\n";
        //     // std::cout << "GREEN: reprojection into right image (from left centers) ** 2x focal ** " << right_img_points[j].x << " " << right_img_points[j].y << "\n";
        //     cv::circle(left_img_rgb, left_img_points[j], 1, cv::Scalar(0, 255, 0), 4);
        //     cv::circle(right_img_rgb, right_img_points[j], 1, cv::Scalar(0, 255, 0), 4);

        //     // std::cout << "RED: reprojection into left image (from right centers) ** 2x focal ** " << test_left_img_points[j].x << " " << test_left_img_points[j].y << "\n";
        //     // std::cout << "RED: reprojection into right image (from right centers) ** 2x focal ** " << test_right_img_points[j].x << " " << test_right_img_points[j].y << "\n\n";
        //     cv::circle(left_img_rgb, test_left_img_points[j], 1, cv::Scalar(0, 0, 255), 4);
        //     cv::circle(right_img_rgb, test_right_img_points[j], 1, cv::Scalar(0, 0, 255), 4);
        // }

        // cv::namedWindow("match", cv::WINDOW_NORMAL);
        // cv::Mat match_img;
        // cv::hconcat(left_img_rgb, right_img_rgb, match_img);

        // resize(match_img, match_img, cv::Size(0, 0), 0.5, 0.5, cv::INTER_CUBIC); // double scale

        // cv::imshow("match", match_img);
        // cv::waitKey();
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
     * Load the labels of the classes used by the model.depth
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

// source install/setup.bash
// export LD_LIBRARY_PATH=/home/navid/eufs_ws/lib/libtorch/lib:$LD_LIBRARY_PATH
// ros2 run graph_based_slam yolo_detector --weights /home/navid/eufs_ws/slam_module/src/graph_based_slam/DeepNetPorting/models/cone_detect.torchscript.pt  
