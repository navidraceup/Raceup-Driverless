#define BOOST_BIND_NO_PLACEHOLDERS

// C++ standard headers
#include <memory>
#include <iostream>

// ROS headers
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

// PCL headers
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>

// from registration tutorial
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

// TF2 headers
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/exceptions.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/transform_datatypes.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>

using std::placeholders::_1;

/**
 * This node performs a basic traffic cone detection from LiDAR point cloud data.
 *
 * It has to do the following operations:
 *  - obtain and accumulate raw LiDAR data
 *  - trim the field of view
 *  - segment ground plane and potential obstacles
 *  - cluster potential obstacles, based on expected num. of points
 *  - return the cone position
 * It exploits the fact that the cone dimensions are known.
 *
 * This perception node is based on the work done by Steven Lee at MUR Motorsports, university of Melbourne.
 *
 */

class LidarDetection : public rclcpp::Node
{

public:
    // variables
    std::vector<sensor_msgs::msg::PointCloud2> ptcloud_accumulator;

    // constructor
    LidarDetection(const rclcpp::NodeOptions &options) : Node("lidar_detection_node", options)
    {
        // get data of LiDAR perception
        lidar_sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10, std::bind(&LidarDetection::lidar_callback, this, _1));

        // get odometry data
        odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ground_truth/odom", 10, std::bind(&LidarDetection::odom_callback, this, _1));

        // initialize tf2 components
        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
    }

private:
    // member variables
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer;

    // for pointcloud registration/visualization
    pcl::PointCloud<pcl::PointXYZ>::Ptr output = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::visualization::PCLVisualizer *p;
    int vp_1, vp_2; // its left and right viewports
    Eigen::Matrix4f final_transform = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f GlobalTransform = Eigen::Matrix4f::Identity(), pairTransform;
    int iter = 0;

    // methods

    /**
     * callback to process LiDAR data
     */
    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Processing LiDAR data ...");

        // data frame is velodyne
        std::string data_frame = msg->header.frame_id;
        std::string robot_frame = "base_footprint";
        std::string map_frame = "map";

        geometry_msgs::msg::TransformStamped t_vel_robot, t_robot_map;

        // transform velodyne data into robot frame
        try
        {
            t_vel_robot = tf_buffer->lookupTransform(data_frame, robot_frame, tf2::TimePointZero);
            RCLCPP_INFO(this->get_logger(), "SUCCESSFULL TRANSFORM from %s to %s",
                        t_vel_robot.header.frame_id.c_str(), t_vel_robot.child_frame_id.c_str());
            std::cout << "t_vel_robot translation x is " << t_vel_robot.transform.translation.x << "\n";
            std::cout << "t_vel_robot translation y is " << t_vel_robot.transform.translation.y << "\n";
            std::cout << "t_vel_robot translation z is " << t_vel_robot.transform.translation.z << "\n";
        }
        catch (const tf2::TransformException &exception)
        {
            RCLCPP_INFO(this->get_logger(), "Could not transform %s to %s: %s",
                        robot_frame.c_str(), data_frame.c_str(), exception.what());
            return;
        }
        // transform robot frame into map frame
        try
        {
            t_robot_map = tf_buffer->lookupTransform(robot_frame, map_frame, tf2::TimePointZero);
            RCLCPP_INFO(this->get_logger(), "SUCCESSFULL TRANSFORM from %s to %s",
                        t_robot_map.header.frame_id.c_str(), t_robot_map.child_frame_id.c_str());
            std::cout << "t_robot_map translation x is " << t_robot_map.transform.translation.x << "\n";
            std::cout << "t_robot_map translation y is " << t_robot_map.transform.translation.y << "\n";
            std::cout << "t_robot_map translation z is " << t_robot_map.transform.translation.z << "\n";
        }
        catch (const tf2::TransformException &exception)
        {
            RCLCPP_INFO(this->get_logger(), "Could not transform %s to %s: %s",
                        map_frame.c_str(), robot_frame.c_str(), exception.what());
            return;
        }

        // transform pointcloud into map frame
        // so --> cloud_map = t_robot_map * t_vel_robot * cloud_vel
        sensor_msgs::msg::PointCloud2 original_cloud, transformed_cloud;
        original_cloud = *msg;

        //*************************************** VISUALIZE PTCLOUD BEFORE TRANSFORM
        // pcl::PointCloud<pcl::PointXYZ>::Ptr ptcloud_before = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        // pcl::fromROSMsg(original_cloud, *ptcloud_before);
        // pcl::visualization::CloudViewer viewer_before("Original pointcloud");
        // viewer_before.showCloud(ptcloud_before);
        // while (!viewer_before.wasStopped())
        // {
        // }
        //********************************************
        tf2::doTransform(original_cloud, transformed_cloud, t_vel_robot);
        tf2::doTransform(transformed_cloud, transformed_cloud, t_robot_map);

        //*************************************** VISUALIZE PTCLOUD AFTER TRANSFORM
        // pcl::PointCloud<pcl::PointXYZ>::Ptr ptcloud_after = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        // pcl::fromROSMsg(transformed_cloud, *ptcloud_after);
        // pcl::visualization::CloudViewer viewer_after("Transformed pointcloud");
        // viewer_after.showCloud(ptcloud_after);
        // while (!viewer_after.wasStopped())
        // {
        // }
        //********************************************

        // pcl::transformPointCloud()

        // accumulate pointclouds already in map frame
        ptcloud_accumulator.push_back(transformed_cloud);

        if (ptcloud_accumulator.size() == 5)
        {
            // call a method to process the clouds
            process_ptcloud(ptcloud_accumulator);
            ptcloud_accumulator.clear();
        }
    }

    /**
     * callback to process wheel odometry data
     */
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        // RCLCPP_INFO(this->get_logger(), "RECEIVED DATA FROM ODOMETRY");

        geometry_msgs::msg::TransformStamped t_odom;

        // Read message content and create tf (noisy if not ground truth odom)
        t_odom.header.stamp = this->get_clock()->now();
        t_odom.header.frame_id = "map";
        t_odom.child_frame_id = "base_footprint";

        // get x and y translation coordinates from the message (z=0)
        t_odom.transform.translation.x = msg->pose.pose.position.x;
        t_odom.transform.translation.y = msg->pose.pose.position.y;
        t_odom.transform.translation.z = msg->pose.pose.position.z;

        t_odom.transform.rotation.x = msg->pose.pose.orientation.x;
        t_odom.transform.rotation.y = msg->pose.pose.orientation.y;
        t_odom.transform.rotation.z = msg->pose.pose.orientation.z;
        t_odom.transform.rotation.w = msg->pose.pose.orientation.w;

        // Send the transformation
        tf_broadcaster->sendTransform(t_odom);
    }

    /**
     * process the pointcloud to get a denser one and retrieve cone shape
     */
    void process_ptcloud(std::vector<sensor_msgs::msg::PointCloud2> accumulator)
    {
        iter++;
        // Create a PCLVisualizer object
        p = new pcl::visualization::PCLVisualizer("Pairwise Incremental Registration");
        p->createViewPort(0.0, 0, 0.5, 1.0, vp_1);
        p->createViewPort(0.5, 0, 1.0, 1.0, vp_2);
        // pcl::PointCloud<pcl::PointXYZ>::Ptr dense_ptcloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);

        for (int i = 1; i < accumulator.size(); i++)
        {
            // convert point cloud msg to pcl::PointCloud
            //  pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            //  pcl::fromROSMsg(accumulator[i], *pcl_cloud);
            //  *dense_ptcloud += *pcl_cloud;

            // register pairs of pointclouds
            register_ptclouds(accumulator[i - 1], accumulator[i]);
            // transform current pair into the global transform
            pcl::transformPointCloud(*output, *result, GlobalTransform);
            // update the global transform
            GlobalTransform *= final_transform;
        }
        std::stringstream ss;
        ss << iter << ".pcd";
        pcl::io::savePCDFile(ss.str(), *result, true);
        /*pcl::visualization::CloudViewer viewer("Dense pointcloud viewer");
        viewer.showCloud(result);
        while (!viewer.wasStopped())
        {
        }*/
        // dense_ptcloud->clear();
    }

    /**
     * register a source and a target pointcloud
     */
    void register_ptclouds(sensor_msgs::msg::PointCloud2 src, sensor_msgs::msg::PointCloud2 trg)
    {
        // convert to pcl pointclouds
        pcl::PointCloud<pcl::PointXYZ>::Ptr src_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::fromROSMsg(src, *src_cloud);
        pcl::PointCloud<pcl::PointXYZ>::Ptr trg_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::fromROSMsg(trg, *trg_cloud);

        // don't downsample because there's too low number of points

        // compute normals
        pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> normal_estimation;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointNormal>::Ptr src_cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
        pcl::PointCloud<pcl::PointNormal>::Ptr trg_cloud_normals(new pcl::PointCloud<pcl::PointNormal>);

        normal_estimation.setSearchMethod(tree);
        normal_estimation.setRadiusSearch(0.03);

        normal_estimation.setInputCloud(src_cloud); // source cloud
        normal_estimation.compute(*src_cloud_normals);
        pcl::copyPointCloud(*src_cloud, *src_cloud_normals);

        normal_estimation.setInputCloud(trg_cloud); // target cloud
        normal_estimation.compute(*trg_cloud_normals);
        pcl::copyPointCloud(*trg_cloud, *trg_cloud_normals);

        // Align
        pcl::IterativeClosestPointNonLinear<pcl::PointNormal, pcl::PointNormal> reg;
        reg.setTransformationEpsilon(1e-6);

        reg.setMaxCorrespondenceDistance(0.03); // max distance between two correspondences (src<->tgt) = 3cm
        reg.setInputSource(src_cloud_normals);
        reg.setInputTarget(trg_cloud_normals);

        // Run the same optimization in a loop and visualize the results
        Eigen::Matrix4f Ti = Eigen::Matrix4f::Identity(), prev, targetToSource;
        pcl::PointCloud<pcl::PointNormal>::Ptr reg_result = src_cloud_normals;
        reg.setMaximumIterations(2);
        for (int i = 0; i < 5; ++i)
        {
            PCL_INFO("Iteration Nr. %d.\n", i);

            // save cloud for visualization purpose
            src_cloud_normals = reg_result;

            // Estimate
            reg.setInputSource(src_cloud_normals);
            reg.align(*reg_result);

            // accumulate transformation between each Iteration
            Ti = reg.getFinalTransformation() * Ti;

            // if the difference between this transformation and the previous one
            // is smaller than the threshold, refine the process by reducing
            // the maximal correspondence distance
            if (std::abs((reg.getLastIncrementalTransformation() - prev).sum()) < reg.getTransformationEpsilon())
                reg.setMaxCorrespondenceDistance(reg.getMaxCorrespondenceDistance() - 0.001);

            prev = reg.getLastIncrementalTransformation();
        }

        // Get the transformation from target to source
        targetToSource = Ti.inverse();

        // Transform target back in source frame
        pcl::transformPointCloud(*trg_cloud, *output, targetToSource);

        p->removePointCloud("source");
        p->removePointCloud("target");

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_tgt_h(output, 0, 255, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_src_h(src_cloud, 255, 0, 0);
        p->addPointCloud(output, cloud_tgt_h, "target", vp_2);
        p->addPointCloud(src_cloud, cloud_src_h, "source", vp_2);

        PCL_INFO("Press q to continue the registration.\n");
        // p->spin();

        p->removePointCloud("source");
        p->removePointCloud("target");

        // add the source to the transformed target
        *output += *src_cloud;

        final_transform = targetToSource;
        // pcl::copyPointCloud(*trg_cloud, *trg_cloud_normals);
        // return trg_cloud;
    }
};
int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    options.use_intra_process_comms(true);
    rclcpp::spin(std::make_shared<LidarDetection>(options));
    rclcpp::shutdown();
    return 0;
}