/**
 * This class performs the deskew operation on LiDAR point cloud.
 * This is necessary when the sensor is moving fast during the acquisition, since the difference in pose between the first and the
 * last scans in huge.
 * So, the entire point cloud must be transformed using an average reference frame, which is the one of the middle scan.
 *
 * This code has been adapted from LIOSAM package (TODO: add link).
 */

// C++ standard headers
#include <iostream>
#include <mutex>
#include <thread>
#include <memory>
#include <deque>

// OpenCV
#include <opencv2/opencv.hpp>

// ROS headers
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>

// PCL headers
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/impl/filter.hpp>

// TF2 headers
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

struct VelodynePoint
{
    PCL_ADD_POINT4D // this adds x, y, z (it's a macro) ?
        PCL_ADD_INTENSITY;
    int ring;
    float time;

    // if you define a structure having members of fixed-size vectorizable Eigen types, you must ensure that calling operator new on it allocates properly aligned buffers
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(VelodynePoint, (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(int, ring, ring)(float, time, time))
using POINTXYZIRT = VelodynePoint;

const int queue_len = 2000;

class DeskewLidar : public rclcpp::Node
{
public:
    // variables
    int n_scan = 16;
    int horiz_scan = 512; // TODO: check this with velodyne vlp16
    float *imu_roll_init, *imu_pitch_init, *imu_yaw_init;

    // constructor
    DeskewLidar(const rclcpp::NodeOptions &options) : Node("deskew_node", options)
    {
        cb_group_lidar = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_imu = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        cb_group_odom = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        auto lidar_options = rclcpp::SubscriptionOptions();
        lidar_options.callback_group = cb_group_lidar;
        auto imu_options = rclcpp::SubscriptionOptions();
        imu_options.callback_group = cb_group_imu;
        auto odom_options = rclcpp::SubscriptionOptions();
        odom_options.callback_group = cb_group_odom;

        imu_sub = create_subscription<sensor_msgs::msg::Imu>("/imu", 10, std::bind(&DeskewLidar::imu_callback, this, std::placeholders::_1), imu_options);
        odom_sub = create_subscription<nav_msgs::msg::Odometry>("/ground_truth/odom", 10, std::bind(&DeskewLidar::odom_callback, this, std::placeholders::_1), odom_options);
        laser_cloud_sub = create_subscription<sensor_msgs::msg::PointCloud2>("/velodyne_points", 10, std::bind(&DeskewLidar::lidar_callback, this, std::placeholders::_1), lidar_options);

        deskewed_cloud_pub = create_publisher<sensor_msgs::msg::PointCloud2>("deskewed_cloud", 1);

        allocate_memory();
        reset_params();
    }

    /**
     * TODO:
     */
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr imu_msg)
    {
        sensor_msgs::msg::Imu this_imu = imu_converter(*imu_msg);

        std::lock_guard<std::mutex> lock1(imu_lock);
        imu_queue.push_back(this_imu);

        // std::cout<<"IMU callback done!\n";
    }

    /**
     * TODO:
     */
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr odom_msg)
    {
        std::lock_guard<std::mutex> lock2(odom_lock);
        odom_queue.push_back(*odom_msg);
        // std::cout<<"odom callback done!\n";
    }

    /**
     * TODO:
     */
    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr lidar_msg)
    {
        if (!cache_pointcloud(lidar_msg))
            return;

        if (!deskew_info())
            return;

        projectPtcloud();

        cloudExtraction();

        publishClouds();

        reset_params();
        std::cout << "LiDAR callback done!\n";
    }

private:
    std::mutex imu_lock;
    std::mutex odom_lock;

    // TODO: check types here
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr laser_cloud_sub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr laser_cloud_pub;
    rclcpp::CallbackGroup::SharedPtr cb_group_lidar;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr deskewed_cloud_pub;
    // TODO: check if pubLaserCloudInfo is necessary (it's a LIOSAM specific msg)
    std::deque<sensor_msgs::msg::PointCloud2> cloud_queue;
    sensor_msgs::msg::PointCloud2 current_cloud_msg;

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub;
    rclcpp::CallbackGroup::SharedPtr cb_group_imu;
    std::deque<sensor_msgs::msg::Imu> imu_queue;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::CallbackGroup::SharedPtr cb_group_odom;
    std::deque<nav_msgs::msg::Odometry> odom_queue;

    double *imu_time = new double[queue_len];
    double *imu_rot_x = new double[queue_len];
    double *imu_rot_y = new double[queue_len];
    double *imu_rot_z = new double[queue_len];

    int imu_ptr_curr;
    bool first_point;
    Eigen::Affine3f trans_start_inverse;

    pcl::PointCloud<POINTXYZIRT>::Ptr laser_cloud_in;
    pcl::PointCloud<pcl::PointXYZI>::Ptr full_cloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr extracted_cloud; // TODO: change with deskewed cloud

    int deskew_flag;
    cv::Mat rangeMat;

    bool odom_deskew_flag;
    float odom_increm_x;
    float odom_increm_y;
    float odom_increm_z;

    double time_scan_curr;
    double time_scan_end;
    std_msgs::msg::Header cloud_header;

    /**
     * TODO:
     */
    void allocate_memory()
    {
        laser_cloud_in.reset(new pcl::PointCloud<POINTXYZIRT>());
        full_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
        extracted_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());

        full_cloud->points.resize(n_scan * horiz_scan);

        reset_params();
    }

    /**
     * TODO:
     */
    void reset_params()
    {
        laser_cloud_in->clear();
        extracted_cloud->clear();
        rangeMat = cv::Mat(n_scan, horiz_scan, CV_32F, cv::Scalar::all(FLT_MAX));

        imu_ptr_curr = 0;
        first_point = true;
        odom_deskew_flag = false;

        for (int i = 0; i < queue_len; i++)
        {
            imu_time[i] = 0;
            imu_rot_x[i] = 0;
            imu_rot_y[i] = 0;
            imu_rot_z[i] = 0;
        }
    }

    /**
     * TODO:
     */
    sensor_msgs::msg::Imu imu_converter(const sensor_msgs::msg::Imu &imu_in)
    {
        sensor_msgs::msg::Imu imu_out = imu_in;

        double ida[] = {1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0};
        std::vector<double> id(ida, std::end(ida));
        std::vector<double> extRotV = id; // extrinsic rotation
        std::vector<double> extRPYV = id; // extrinsic RPY

        double zea[] = {0.0, 0.0, 0.0};
        std::vector<double> ze(zea, std::end(zea));
        std::vector<double> extTransV = ze; // extrinsic translation
        Eigen::Matrix3d extRot = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRotV.data(), 3, 3);
        Eigen::Matrix3d extRPY = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extRPYV.data(), 3, 3);
        Eigen::Vector3d extTrans = Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(extTransV.data(), 3, 1);
        Eigen::Quaterniond extQRPY = Eigen::Quaterniond(extRPY);

        // rotate acceleration
        Eigen::Vector3d acc(imu_in.linear_acceleration.x, imu_in.linear_acceleration.y, imu_in.linear_acceleration.z);
        acc = extRot * acc;
        imu_out.linear_acceleration.x = acc.x();
        imu_out.linear_acceleration.y = acc.y();
        imu_out.linear_acceleration.z = acc.z();

        // rotate gyroscope
        Eigen::Vector3d gyr(imu_in.angular_velocity.x, imu_in.angular_velocity.y, imu_in.angular_velocity.z);
        gyr = extRot * gyr;
        imu_out.angular_velocity.x = gyr.x();
        imu_out.angular_velocity.y = gyr.y();
        imu_out.angular_velocity.z = gyr.z();

        // rotate roll pitch yaw
        Eigen::Quaterniond q_from(imu_in.orientation.w, imu_in.orientation.x, imu_in.orientation.y, imu_in.orientation.z);
        Eigen::Quaterniond q_final = q_from * extQRPY;
        imu_out.orientation.x = q_final.x();
        imu_out.orientation.y = q_final.y();
        imu_out.orientation.z = q_final.z();
        imu_out.orientation.w = q_final.w();

        if (sqrt(q_final.x() * q_final.x() + q_final.y() * q_final.y() + q_final.z() * q_final.z() + q_final.w() * q_final.w()) < 0.1)
        {
            RCLCPP_ERROR(get_logger(), "Invalid quaternion, please use a 9-axis IMU!");
            rclcpp::shutdown();
        }

        return imu_out;
    }

    /**
     * TODO:
     */
    bool cache_pointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr &laser_cloud_msg)
    {
        cloud_queue.push_back(*laser_cloud_msg);
        if (cloud_queue.size() <= 2)
            return false;

        // convert the pointcloud
        current_cloud_msg = std::move(cloud_queue.front()); // "move" changes an expression from being an lvalue to being an xvalue
        cloud_queue.pop_front();
        pcl::moveFromROSMsg(current_cloud_msg, *laser_cloud_in);

        // remove NaN points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laser_cloud_in, *laser_cloud_in, indices);

        // get timestamp
        cloud_header = current_cloud_msg.header;
        time_scan_curr = rclcpp::Time(cloud_header.stamp).seconds();
        time_scan_end = time_scan_curr + laser_cloud_in->points.back().time;

        // check dense
        if (laser_cloud_in->is_dense == false)
        {
            RCLCPP_ERROR(get_logger(), "Point cloud is not in dense format, please remove NaN points first!");
            rclcpp::shutdown();
        }

        // check ring channel
        static int ring_flag = 0;
        if (ring_flag == 0)
        {
            ring_flag = -1;
            for (int i = 0; i < (int)current_cloud_msg.fields.size(); ++i)
            {
                if (current_cloud_msg.fields[i].name == "ring")
                {
                    ring_flag = 1;
                    break;
                }
            }
            if (ring_flag == -1)
            {
                RCLCPP_ERROR(get_logger(), "Point cloud ring channel not available, please configure your point cloud data!");
                rclcpp::shutdown();
            }
        }

        // check point time
        if (deskew_flag == 0)
        {
            deskew_flag = -1;
            for (auto &field : current_cloud_msg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskew_flag = 1;
                    break;
                }
            }
            if (deskew_flag == -1)
                RCLCPP_WARN(get_logger(), "Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    /**
     * TODO:
     */
    bool deskew_info()
    {
        std::lock_guard<std::mutex> lock1(imu_lock);
        std::lock_guard<std::mutex> lock2(odom_lock);

        // make sure IMU data available for the scan
        if (imu_queue.empty() || rclcpp::Time(imu_queue.front().header.stamp).seconds() > time_scan_curr || rclcpp::Time(imu_queue.back().header.stamp).seconds() < time_scan_end)
        {
            RCLCPP_INFO(get_logger(), "Waiting for IMU data ...");
            return false;
        }

        imu_deskew_info();

        odom_deskew_info();

        return true;
    }

    /**
     * TODO:
     */
    void imu_deskew_info()
    {

        while (!imu_queue.empty())
        {
            if (rclcpp::Time(imu_queue.front().header.stamp).seconds() < time_scan_curr - 0.01)
                imu_queue.pop_front();
            else
                break;
        }
        if (imu_queue.empty())
            return;
        imu_ptr_curr = 0;
        for (int i = 0; i < (int)imu_queue.size(); ++i)
        {

            sensor_msgs::msg::Imu this_imu_msg = imu_queue[i];
            double curr_imu_time = rclcpp::Time(this_imu_msg.header.stamp).seconds();

            // get RPY for this scan TODO: check if this is needed
            // if (curr_imu_time <= time_scan_curr)
            // imuRPY2rosRPY(&this_imu_msg, imu_roll_init, imu_pitch_init, imu_yaw_init); // TODO: check if this is necessary

            if (curr_imu_time > time_scan_end + 0.01)
                break;

            if (imu_ptr_curr == 0)
            {
                imu_rot_x[0] = 0;
                imu_rot_y[0] = 0;
                imu_rot_z[0] = 0;
                imu_time[0] = curr_imu_time;
                ++imu_ptr_curr;
                continue;
            }

            // get angular vel
            double angular_x, angular_y, angular_z;
            imuAng2rosAng(&this_imu_msg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            double time_diff = curr_imu_time - imu_time[imu_ptr_curr - 1];
            imu_rot_x[imu_ptr_curr] = imu_rot_x[imu_ptr_curr - 1] + angular_x * time_diff;
            imu_rot_y[imu_ptr_curr] = imu_rot_y[imu_ptr_curr - 1] + angular_y * time_diff;
            imu_rot_z[imu_ptr_curr] = imu_rot_z[imu_ptr_curr - 1] + angular_z * time_diff;
            imu_time[imu_ptr_curr] = curr_imu_time;
            ++imu_ptr_curr;
        }
        --imu_ptr_curr;

        if (imu_ptr_curr <= 0)
            return;
    }

    /**
     * TODO:
     */
    void odom_deskew_info()
    {
        while (!odom_queue.empty())
        {
            if (rclcpp::Time(odom_queue.front().header.stamp).seconds() < time_scan_curr - 0.01)
                odom_queue.pop_front();
            else
                break;
        }

        if (odom_queue.empty())
            return;

        if (rclcpp::Time(odom_queue.front().header.stamp).seconds() > time_scan_curr)
            return;

        // get odom at beginning of scan
        nav_msgs::msg::Odometry start_odom_msg;
        for (int i = 0; i < (int)odom_queue.size(); ++i)
        {
            start_odom_msg = odom_queue[i];

            if (rclcpp::Time(start_odom_msg.header.stamp).seconds() < time_scan_curr)
                continue;
            else
                break;
        }
        tf2::Quaternion orientation;
        tf2::fromMsg(start_odom_msg.pose.pose.orientation, orientation);
        double roll, pitch, yaw;
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // get odom at end of scan
        odom_deskew_flag = false;
        if (rclcpp::Time(odom_queue.back().header.stamp).seconds() < time_scan_end)
            return;

        nav_msgs::msg::Odometry end_odom_msg;
        for (int i = 0; i < (int)odom_queue.size(); ++i)
        {
            end_odom_msg = odom_queue[i];

            if (rclcpp::Time(end_odom_msg.header.stamp).seconds() < time_scan_end)
                continue;
            else
                break;
        }

        if (int(round(start_odom_msg.pose.covariance[0])) != int(round(end_odom_msg.pose.covariance[0])))
            return;

        Eigen::Affine3f trans_begin = pcl::getTransformation(start_odom_msg.pose.pose.position.x, start_odom_msg.pose.pose.position.y, start_odom_msg.pose.pose.position.z, roll, pitch, yaw);
        tf2::fromMsg(end_odom_msg.pose.pose.orientation, orientation);
        tf2::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f trans_end = pcl::getTransformation(end_odom_msg.pose.pose.position.x, end_odom_msg.pose.pose.position.y, end_odom_msg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f trans_b_t = trans_begin.inverse() * trans_end;
        float roll_incr, pitch_incr, yaw_incr;
        pcl::getTranslationAndEulerAngles(trans_b_t, odom_increm_x, odom_increm_y, odom_increm_z, roll_incr, pitch_incr, yaw_incr);

        odom_deskew_flag = true;
    }

    /**
     * TODO:
     */
    void projectPtcloud()
    {
        int cloud_size = laser_cloud_in->points.size();

        // range img projection
        for (int i = 0; i < cloud_size; i++)
        {
            pcl::PointXYZI this_pt;
            this_pt.x = laser_cloud_in->points[i].x;
            this_pt.y = laser_cloud_in->points[i].y;
            this_pt.z = laser_cloud_in->points[i].z;
            this_pt.intensity = laser_cloud_in->points[i].intensity;

            float range = sqrt(pow(this_pt.x, 2) + pow(this_pt.y, 2) + pow(this_pt.z, 2));
            // lidar_min_range = 5.5 TODO:
            // lidar_max_range = 1000.0
            if (range < 5.5 || range > 1000.0)
                continue;

            int row_idx = laser_cloud_in->points[i].ring;
            if (row_idx < 0 || row_idx >= n_scan)
                continue;

            // downsamplre_rate = 1 TODO:
            if (row_idx % 1 != 0)
                continue;

            float horiz_angle = atan2(this_pt.x, this_pt.y) * 180 / M_PI;
            static float ang_res_x = 360.0 / float(horiz_scan);
            int column_idx = -round((horiz_angle - 90.0) / ang_res_x) + horiz_scan / 2;
            if (column_idx >= horiz_scan)
                column_idx -= horiz_scan;

            if (column_idx < 0 || column_idx >= horiz_scan)
                continue;

            if (rangeMat.at<float>(row_idx, column_idx) != FLT_MAX)
                continue;

            this_pt = deskewPoint(&this_pt, laser_cloud_in->points[i].time);

            rangeMat.at<float>(row_idx, column_idx) = range;

            int index = column_idx + row_idx * horiz_scan;
            full_cloud->points[index] = this_pt;
        }
    }

    /**
     * TODO:
     */
    pcl::PointXYZI deskewPoint(pcl::PointXYZI *point, double rel_time)
    {
        if (deskew_flag == -1)
            return *point;

        double pt_time = time_scan_curr + rel_time;
        float rot_x_cur, rot_y_cur, rot_z_cur;
        findRotation(pt_time, &rot_x_cur, &rot_y_cur, &rot_z_cur);

        float pos_x_cur, pos_y_cur, pos_z_cur;
        findPosition(rel_time, &pos_x_cur, &pos_y_cur, &pos_z_cur);

        if (first_point == true)
        {
            trans_start_inverse = (pcl::getTransformation(pos_x_cur, pos_y_cur, pos_z_cur, rot_x_cur, rot_y_cur, rot_z_cur)).inverse();
            first_point = false;

            // transform pts to start
            Eigen::Affine3f trans_final = pcl::getTransformation(pos_x_cur, pos_y_cur, pos_z_cur, rot_x_cur, rot_y_cur, rot_z_cur);
            Eigen::Affine3f trans_b_t = trans_start_inverse * trans_final;
            pcl::PointXYZI new_pt;

            new_pt.x = trans_b_t(0, 0) * point->x + trans_b_t(0, 1) * point->y + trans_b_t(0, 2) * point->z + trans_b_t(0, 3);
            new_pt.y = trans_b_t(1, 0) * point->x + trans_b_t(1, 1) * point->y + trans_b_t(1, 2) * point->z + trans_b_t(1, 3);
            new_pt.z = trans_b_t(2, 0) * point->x + trans_b_t(2, 1) * point->y + trans_b_t(2, 2) * point->z + trans_b_t(2, 3);
            new_pt.intensity = point->intensity;

            return new_pt;
        }
    }

    /**
     * TODO:
     */
    void cloudExtraction()
    {
        // extract segmented cloud
        for (int i = 0; i < n_scan; ++i)
        {
            for (int j = 0; j < horiz_scan; ++j)
            {
                if (rangeMat.at<float>(i, j) != FLT_MAX)
                {
                    // save extracted cloud
                    extracted_cloud->push_back(full_cloud->points[j + i * horiz_scan]);
                }
            }
        }
    }

    /**
     * TODO:
     */
    void publishClouds()
    {
        sensor_msgs::msg::PointCloud2 tmp_cloud;
        pcl::toROSMsg(*extracted_cloud, tmp_cloud);
        tmp_cloud.header.stamp = cloud_header.stamp;
        tmp_cloud.header.frame_id = "velodyne"; // lidar_frame is velodyne TODO: check this
        if (deskewed_cloud_pub->get_subscription_count() != 0)
            deskewed_cloud_pub->publish(tmp_cloud);
    }

    /**
     * TODO:
     */
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0;
        *rotYCur = 0;
        *rotZCur = 0;

        int imuPointerFront = 0;
        while (imuPointerFront < imu_ptr_curr)
        {
            if (pointTime < imu_time[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        if (pointTime > imu_time[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imu_rot_x[imuPointerFront];
            *rotYCur = imu_rot_y[imuPointerFront];
            *rotZCur = imu_rot_z[imuPointerFront];
        }
        else
        {
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imu_time[imuPointerBack]) / (imu_time[imuPointerFront] - imu_time[imuPointerBack]);
            double ratioBack = (imu_time[imuPointerFront] - pointTime) / (imu_time[imuPointerFront] - imu_time[imuPointerBack]);
            *rotXCur = imu_rot_x[imuPointerFront] * ratioFront + imu_rot_x[imuPointerBack] * ratioBack;
            *rotYCur = imu_rot_y[imuPointerFront] * ratioFront + imu_rot_y[imuPointerBack] * ratioBack;
            *rotZCur = imu_rot_z[imuPointerFront] * ratioFront + imu_rot_z[imuPointerBack] * ratioBack;
        }
    }

    /**
     * TODO:
     */
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0;
        *posYCur = 0;
        *posZCur = 0;

        if (odom_deskew_flag == false)
            return;

        float ratio = relTime / (time_scan_end - time_scan_curr);

        *posXCur = ratio * odom_increm_x;
        *posYCur = ratio * odom_increm_y;
        *posZCur = ratio * odom_increm_z;
    }

    /**
     * TODO:
     */
    void imuRPY2rosRPY(sensor_msgs::msg::Imu *thisImuMsg, float *rosRoll, float *rosPitch, float *rosYaw)
    {
        double imuRoll, imuPitch, imuYaw;
        tf2::Quaternion orientation;

        tf2::fromMsg(thisImuMsg->orientation, orientation);

        tf2::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);

        *rosRoll = imuRoll;
        *rosPitch = imuPitch;
        *rosYaw = imuYaw;
    }

    /**
     * TODO:
     */
    void imuAng2rosAng(sensor_msgs::msg::Imu *thisImuMsg, double *angular_x, double *angular_y, double *angular_z)
    {
        *angular_x = thisImuMsg->angular_velocity.x;
        *angular_y = thisImuMsg->angular_velocity.y;
        *angular_z = thisImuMsg->angular_velocity.z;
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    rclcpp::spin(std::make_shared<DeskewLidar>(options));
    rclcpp::shutdown();
    return 0;
}