// C++ standard headers
#include <iostream>
#include <memory>
#include <chrono>
#include <fstream>

// ROS headers
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <rclcpp/rclcpp.hpp>

// calibrate drone camera (TODO: see if it is correct to use cv_ext, or if it is better to do it from scratch)
// undistort the image (see cv::undistortRectify(), or similar functions, or cv_ext if already done)
// the result should be with same K as the input camera, but with D = cv::Mat() (zero matrix, no distortion)
// exploit the real coords of the 4 corner cones --> cv::findHomography()
// in this way, get H to go from ground-plane (parking) to image-plane
// to go from image-plane (pixel) to ground-plane (meters), consider H^(-1)
// NB: remember that H is defined up to a scale factor, so we need to find it
// once we have the real ground-truth coordinates, insert the map into the simulator

/**
 *TODO: read directly from file the camera matrices
 * bool PinholeCameraModel::readFromOpenCVFile(const std::string &filename)
{
  string yml_filename = generateYAMLFilename(filename);

  cv::FileStorage fs(yml_filename, cv::FileStorage::READ);

  if (!fs.isOpened())
    return false;

  cv::Size image_size;
  cv::Mat intrinsic_matrix, dist_coeffs;

  fs["width"] >> image_size.width;
  fs["height"] >> image_size.height;

  fs["K"] >> intrinsic_matrix;
  fs["D"] >> dist_coeffs;

  fs.release();

  setFromKD(intrinsic_matrix, image_size.width, image_size.height, dist_coeffs);

  return true;
}
*/

using std::placeholders::_1;

rclcpp::Node::SharedPtr node_ptr;

// camera model
float fx = 3.0256833980685710e+03;
float fy = 3.0237458629910070e+03;
float cx = 1.9698561912453497e+03;
float cy = 1.4805654836483563e+03;

// dist coeffs
std::vector<float> dist_coeffs{1.7957662522214438e-02, -7.2931360682731400e-02,
                               -3.1276484529960856e-04, -1.1775132636181523e-03,
                               8.8087136549679007e-02, 0., 0., 0.};

class GroundTruthCones : public rclcpp::Node
{

public:
    // constructor
    GroundTruthCones(const rclcpp::NodeOptions &options) : Node("gt_cones", options)
    {
        cv::Mat cam_matrix = getCameraMat(fx, fy, cx, cy);
        cv::Mat original_img = cv::imread("/media/navid/FiLix/imgs/track_drone2_calib/ground_truth_track.JPG");
        cv::Mat undist_image = getUndistortedImage(original_img, cam_matrix, dist_coeffs);
        cv::imwrite("/media/navid/FiLix/imgs/track_drone2_calib/undistorted_track.JPG", undist_image);

        std::vector<cv::Point2f> ground_pts, img_points;
        ground_pts.push_back(cv::Point2f(0, 0));          // bottom-right cone (gazebo)
        ground_pts.push_back(cv::Point2f(59.90, 0));      // top-right cone
        ground_pts.push_back(cv::Point2f(59.90, 64.035)); // top-left cone
        ground_pts.push_back(cv::Point2f(0, 64.035));     // bottom-left cone

        img_points.push_back(cv::Point2f(3022, 2625)); // bottom-right cone (gazebo)
        img_points.push_back(cv::Point2f(3077, 164));  // top-right cone
        img_points.push_back(cv::Point2f(494, 64));    // top-left cone
        img_points.push_back(cv::Point2f(292, 2534));  // bottom-left cone

        cv::Mat pix_to_m = getPixelToMetersTransform(ground_pts, img_points);
        std::string file = "/media/navid/FiLix/imgs/gt_cones/BlueCones.txt";

        getRealCoords(file, pix_to_m);
    }

    virtual ~GroundTruthCones(){};

private:
    /**
     * Retrieve the real world coordinates of the specified cones.
     * Data is read from a txt file in which each row contains pixel's coordinates, comma-separated.
     *
     * @param filename The txt file containing pixel coordinates.
     * @param transf_matrix The homography matrix to convert from pixels to meters.
     */
    void getRealCoords(std::string filename, cv::Mat transf_matrix)
    {
        std::vector<cv::Point2f> cones_pixels = readPixelFromTxt(filename);
        std::vector<cv::Point2f> cones_meters;
        // apply the inverse homography to find real cones' coords (in m)
        cv::perspectiveTransform(cones_pixels, cones_meters, transf_matrix);
        for (int i = 0; i < cones_meters.size(); i++)
        {
            if (abs(cones_meters[i].x) < 0.001)
                cones_meters[i].x = 0;
            if (abs(cones_meters[i].y) < 0.001)
                cones_meters[i].y = 0;
        }
        for (auto cone : cones_meters)
        {
            std::cout << cone << std::endl;
        }
        // save to a txt file
        saveMetersToTxt("/media/navid/FiLix/imgs/gt_cones/gt_BlueCones.txt", cones_meters);
    }

    /**
     * Read a txt file containing points' coordinates in pixel.
     * Each line contains two comma-separated coordinates.
     *
     * @param filename The txt file to read to get cones' coords in pixels.
     *
     * @return A vector of 2D points with the given coordinates.
     */
    std::vector<cv::Point2f> readPixelFromTxt(std::string filename)
    {
        std::vector<cv::Point2f> cones_pixels;
        std::ifstream infile(filename);

        if (infile.is_open())
        {
            std::string line;
            std::vector<int> cone;
            while (getline(infile, line))
            {
                std::stringstream ss(line);
                std::string tmp_coord;

                while (getline(ss, tmp_coord, ','))
                {
                    std::cout << tmp_coord << std::endl;
                    cone.push_back(stoi(tmp_coord));
                }
                cones_pixels.push_back(cv::Point2f(cone[0], cone[1]));
                cone.clear();
            }

            infile.close();
        }
        else
        {
            std::cerr << "Error reading pixel coords' file!\n";
        }

        return cones_pixels;
    }

    /**
     * Save a txt file containing the ground truth coordinates of cones in meters.
     *
     * @param filename The name of the file to save.
     */
    void saveMetersToTxt(std::string filename, std::vector<cv::Point2f> cones)
    {
        std::ofstream out_file(filename);

        for (auto cone : cones)
        {
            out_file << cone.x << "," << cone.y << "\n";
        }
        out_file.close();
    }

    /**
     * Find the transformation between the image plane (drone picture) and the ground plane (parking lot)
     *
     * @param ground_pts The set of points in the ground plane.
     * @param img_pts The set of points in the image plane.
     */
    cv::Mat getPixelToMetersTransform(cv::InputArray ground_pts, cv::InputArray img_pts)
    {
        // transformation from ground to image (NB: up to a scale factor)
        cv::Mat h_matrix = cv::findHomography(ground_pts, img_pts, 0);

        // transformation image to ground
        cv::Mat h_inv = h_matrix.inv();

        return h_inv;
    }

    /**
     * Transform an image, accounting for lens distorsion.
     */
    cv::Mat getUndistortedImage(cv::Mat img, cv::Mat cam_mat, std::vector<float> dist_coeffs)
    {
        cv::Mat undistorted_image;

        cv::undistort(img, undistorted_image, cam_mat, dist_coeffs);

        cv::namedWindow("distorted image", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("undistorted image", cv::WINDOW_AUTOSIZE);

        cv::imshow("distorted image", img);
        cv::imshow("undistorted image", undistorted_image);
        cv::waitKey();

        return undistorted_image;
    }

    /**
     * @param fx X-axis focal length of the camera.
     * @param fy Y-axis focal length of the camera.
     * @param cx X coordinate of the principal point.
     * @param cy Y coordinate of the principal point.
     *
     * @return The camera matrix associated with the given calibration parameters.
     */
    cv::Mat getCameraMat(float fx, float fy, float cx, float cy)
    {
        cv::Mat camera_mat = cv::Mat::zeros(3, 3, CV_32F);
        camera_mat.at<float>(0, 0) = fx;
        camera_mat.at<float>(1, 1) = fy;
        camera_mat.at<float>(2, 2) = 1.0;
        camera_mat.at<float>(0, 2) = cx;
        camera_mat.at<float>(1, 2) = cy;
        return camera_mat;
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    node_ptr = std::make_shared<GroundTruthCones>(options);
    rclcpp::spin(node_ptr);
    rclcpp::shutdown();
    return 0;
}
