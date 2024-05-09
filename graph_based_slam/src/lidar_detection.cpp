#define BOOST_BIND_NO_PLACEHOLDERS

// C++ standard headers
#include <memory>
#include <iostream>
#include <queue>
#include <chrono>
#include <inttypes.h>

// ROS headers
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// Project headers
#include <eufs_msgs/msg/cone_array_with_covariance.hpp>
#include <eufs_msgs/msg/car_state.hpp>

// PCL headers
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/conversions.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/filter.h>

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

// time synchronization headers
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

using std::placeholders::_1;

rclcpp::Node::SharedPtr node_ptr;
static double SMALL_CONE_HEIGHT = 0.325;
static double SMALL_CONE_WIDTH = 0.228;
static double BIG_CONE_HEIGHT = 0.505;
static double BIG_CONE_WIDTH = 0.285;

struct ClusterParams
{
    ClusterParams() : cluster_tol(0.1), cluster_min(10), cluster_max(250), reconst_radius(0.2),
                      marker_x(0.35), marker_y(0.35), marker_z(0.7), marker_alpha(1.0),
                      marker_r(0.0), marker_g(1.0), marker_b(0.0) {}

    // cluster tolerance
    double cluster_tol;
    // cluster min n. of points
    int cluster_min;
    // cluster max n. of points
    int cluster_max;
    // radius of cluster reconstruction
    double reconst_radius; //

    double marker_x;     // marker scale x
    double marker_y;     // marker scale y
    double marker_z;     // marker scale z
    double marker_alpha; // marker alpha
    double marker_r;     // marker red
    double marker_g;     // marker green
    double marker_b;     // marker blue
};

/**
 * This node performs a basic traffic cone detection from LiDAR point cloud data.
 *
 * It has to do the following operations:
 *  - obtain and accumulate raw LiDAR data
 *  (- trim the field of view)
 *  - segment ground plane and potential obstacles
 *  - cluster potential obstacles, based on expected number of points
 *  - return the cone position
 * It exploits the fact that the cone dimensions are known.
 *
 * This perception node is based on the work done by Steven Lee at MUR Motorsports, University of Melbourne.
 *
 */

class LidarDetection : public rclcpp::Node
{

public:
    // variables
    std::queue<sensor_msgs::msg::PointCloud2> ptcloud_queue;

    // constructor
    LidarDetection(const rclcpp::NodeOptions &options) : Node("lidar_detection_node", options)
    {
        // param to use sim_time
        this->set_parameter(rclcpp::Parameter("use_sim_time", true));
        std::cout << this->get_parameter("use_sim_time") << "\n";

        // initialize tf2 components
        tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

        // remove this if not needed for visualization
        dense_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("dense_cloud", 10);
        crop_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("cropped_cloud", 10);
        cones_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("cones_cloud", 10);
        ground_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("ground_cloud", 10);
        clust_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("cluster_result", 10);
        markers_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("cluster_markers", 10);

        sync_init();
    }

private:
    // member variables
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer;
    geometry_msgs::msg::TransformStamped prev_t_odom;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr dense_pub, crop_pub, cones_pub, ground_pub, clust_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr markers_pub;

    ClusterParams cluster_params; // params for euclidean clustering

    geometry_msgs::msg::TransformStamped t_vel_robot;
    pcl::PointCloud<pcl::PointXYZ> total_map;
    std::vector<pcl::PointXYZ> all_cones_pos;
    // std::set<pcl::PointXYZ> all_cones_pos; // find in O(logn), while O(n) for vector
    int general_id = 0; // to have unique markers ids, otherwise they repeat at every callback
    int process_count;

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sync_lid_sub;
    message_filters::Subscriber<eufs_msgs::msg::CarState> sync_odom_sub;
    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, eufs_msgs::msg::CarState> approx_policy;
    // std::shared_ptr<message_filters::Synchronizer<approx_policy>> syncApprox;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::msg::PointCloud2, eufs_msgs::msg::CarState> exact_policy;
    std::shared_ptr<message_filters::Synchronizer<exact_policy>> syncExact;

    // save last odom transform
    geometry_msgs::msg::TransformStamped last_odom_transf;

    // methods

    /**
     * @brief Initialize synchronization-related stuff (subscriptions, policy, etc..)
     */
    void sync_init()
    {
        rmw_qos_profile_t custom_qos_profile = rmw_qos_profile_default;

        custom_qos_profile.depth = 5;
        custom_qos_profile.reliability = rmw_qos_reliability_policy_t::RMW_QOS_POLICY_RELIABILITY_RELIABLE;
        custom_qos_profile.history = rmw_qos_history_policy_t::RMW_QOS_POLICY_HISTORY_KEEP_LAST;
        custom_qos_profile.durability = rmw_qos_durability_policy_t::RMW_QOS_POLICY_DURABILITY_VOLATILE;

        sync_lid_sub.subscribe(this, "/velodyne_points", custom_qos_profile);
        // sync_odom_sub.subscribe(this, "/ground_truth/odom", custom_qos_profile);
        sync_odom_sub.subscribe(this, "/odometry_integration/car_state", custom_qos_profile);

        // queue_size = how many msgs in the queue to check for a candidate to match ???
        // syncApprox = std::make_shared<message_filters::Synchronizer<approx_policy>>(approx_policy(5), sync_lid_sub, sync_odom_sub);
        // syncApprox->registerCallback(&LidarDetection::sync_callback, this);

        syncExact = std::make_shared<message_filters::Synchronizer<exact_policy>>(exact_policy(50), sync_lid_sub, sync_odom_sub);
        syncExact->registerCallback(&LidarDetection::sync_callback, this);
    }

    /**
     * Synchronized callback to process LiDAR and odometry data
     * @param lidar_msg Message containing LiDAR sensor data.
     * @param odom_msg Message containing odometry sensor data.
     */
    void sync_callback(sensor_msgs::msg::PointCloud2::SharedPtr lidar_msg, eufs_msgs::msg::CarState::SharedPtr odom_msg)
    {
        RCLCPP_INFO(this->get_logger(), "RECEIVED DATA FROM ODOMETRY");
        RCLCPP_INFO(this->get_logger(), " odometry time %ds and %dns", odom_msg->header.stamp.sec, odom_msg->header.stamp.nanosec);
        RCLCPP_INFO(this->get_logger(), " odometry frame_id is %s", odom_msg->header.frame_id.c_str()); // odom_data is in map frame
        RCLCPP_INFO(this->get_logger(), "RECEIVED DATA FROM LIDAR");
        RCLCPP_INFO(this->get_logger(), " lidar frame_id is %s", lidar_msg->header.frame_id.c_str()); // lidar is in velodyne frame
        RCLCPP_INFO(this->get_logger(), " lidar time %ds and %dns", lidar_msg->header.stamp.sec, lidar_msg->header.stamp.nanosec);

        // ----------------------------------- ODOM PROCESSING --------------------------------------------------------
        geometry_msgs::msg::TransformStamped t_odom;

        // Read message content and create tf (noisy if not ground truth odom)
        t_odom.header.stamp = rclcpp::Node::now(); // TODO: check if this time is correct, or if sim time prolem is here
        t_odom.header.frame_id = "map";
        t_odom.child_frame_id = "base_footprint";

        // get x and y translation coordinates from the message (z=0)
        t_odom.transform.translation.x = odom_msg->pose.pose.position.x;
        t_odom.transform.translation.y = odom_msg->pose.pose.position.y;
        t_odom.transform.translation.z = odom_msg->pose.pose.position.z;

        t_odom.transform.rotation.x = odom_msg->pose.pose.orientation.x;
        t_odom.transform.rotation.y = odom_msg->pose.pose.orientation.y;
        t_odom.transform.rotation.z = odom_msg->pose.pose.orientation.z;
        t_odom.transform.rotation.w = odom_msg->pose.pose.orientation.w;

        last_odom_transf = t_odom;

        // Send the transformation
        tf_broadcaster->sendTransform(t_odom);

        // ---------------------------------------- LIDAR PROCESSING ----------------------------------------------
        // data frame is velodyne
        std::string data_frame = lidar_msg->header.frame_id;
        std::string robot_frame = "base_footprint";
        std::string map_frame = "map";

        // geometry_msgs::msg::TransformStamped t_robot_map;

        // transform velodyne data into robot frame
        try
        {
            t_vel_robot = tf_buffer->lookupTransform(robot_frame, data_frame, tf2::TimePointZero);
            RCLCPP_INFO(this->get_logger(), "SUCCESSFULL TRANSFORM from %s to %s",
                        t_vel_robot.header.frame_id.c_str(), t_vel_robot.child_frame_id.c_str());
        }
        catch (const tf2::TransformException &exception)
        {
            RCLCPP_INFO(this->get_logger(), "Could not transform %s to %s: %s",
                        robot_frame.c_str(), data_frame.c_str(), exception.what());
            return;
        }

        // transform pointcloud into map frame
        // so --> cloud_map = t_robot_map * t_vel_robot * cloud_vel
        sensor_msgs::msg::PointCloud2 original_cloud, transformed_cloud, final_cloud;
        original_cloud = *lidar_msg;
        tf2::doTransform(original_cloud, transformed_cloud, t_vel_robot);
        tf2::doTransform(transformed_cloud, final_cloud, last_odom_transf);

        // every 5 PointClouds, process them (aggregation, segmentation, clustering, markers visualization)
        if (ptcloud_queue.size() == 5)
        {
            std::queue<sensor_msgs::msg::PointCloud2> tmp_queue = ptcloud_queue;
            process_ptcloud(tmp_queue);
            ptcloud_queue.pop();
        }
        ptcloud_queue.push(final_cloud);
    }

    /**
     * Process the PointCloud to extract cones
     * @param cloud_queue A queue containing a fixed number of consecutive PointCloud.
     */
    void process_ptcloud(std::queue<sensor_msgs::msg::PointCloud2> cloud_queue)
    {
        std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
        process_count++;

        sensor_msgs::msg::PointCloud2 aggregated_ptcloud, cropped_ptcloud, cluster_cloud;
        std::vector<sensor_msgs::msg::PointCloud2> segmented_ptcloud;

        // step 1: aggregate pointcloud
        RCLCPP_INFO(this->get_logger(), "----------------------------------- AGGREGATION...");
        aggregated_ptcloud = ptcloud_aggregation(cloud_queue);
        RCLCPP_INFO(this->get_logger(), " Aggregated cloud has size %d", aggregated_ptcloud.width);

        // step2: field of view trimming (box filter)
        RCLCPP_INFO(this->get_logger(), "----------------------------------- CROPPING...");
        cropped_ptcloud = fov_cropping(aggregated_ptcloud, last_odom_transf);
        RCLCPP_INFO(this->get_logger(), " cropped cloud has size %d", cropped_ptcloud.width);

        // step 3 : segment pointcloud
        RCLCPP_INFO(this->get_logger(), "----------------------------------- SEGMENTATION...");
        segmented_ptcloud = ptcloud_segmentation(cropped_ptcloud);
        RCLCPP_INFO(this->get_logger(), " cones cloud has size %d", segmented_ptcloud[0].width);
        RCLCPP_INFO(this->get_logger(), " ground cloud has size %d", segmented_ptcloud[1].width);

        // step 4 : cluster pointcloud
        RCLCPP_INFO(this->get_logger(), "----------------------------------- CLUSTERING...");
        cluster_cloud = ptcloud_clustering(segmented_ptcloud[0], segmented_ptcloud[1]);
        RCLCPP_INFO(this->get_logger(), " clustered cloud has size %d", cluster_cloud.width);

        std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
        std::cout << "==================================\n\n\nPROCESSING TIME IS (s) " << std::chrono::duration_cast<std::chrono::microseconds>(s - e).count() / 1000000.0 << "\n\n\n==================================\n\n\n\n\n\n";
    }

    /**
     * @brief Aggregate a fixed number of PointCloud to get a denser one.
     * @param queue The queue containing the PointCloud to be processed.
     * @return The final dense PointCloud, as PointCloud2 sensor_msgs.
     */
    sensor_msgs::msg::PointCloud2 ptcloud_aggregation(std::queue<sensor_msgs::msg::PointCloud2> queue)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr dense_ptcloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        while (!queue.empty())
        {
            sensor_msgs::msg::PointCloud2 &top_element = queue.front();
            pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
            pcl::fromROSMsg(top_element, *pcl_cloud);
            *dense_ptcloud += *pcl_cloud;
            queue.pop();
            total_map += *pcl_cloud;
        }
        sensor_msgs::msg::PointCloud2 dense_msg_cloud;
        pcl::toROSMsg(*dense_ptcloud, dense_msg_cloud);
        dense_msg_cloud.header.set__frame_id("map");
        std::cout << " Processing iteration num. " << process_count << " \n";
        dense_pub->publish(dense_msg_cloud);

        // total map
        pcl::io::savePCDFile("map_cloud.pcd", total_map);

        return dense_msg_cloud;
    }

    /**
     * Apply a box filter to a PointCloud to trim the field of view.
     * @param ptcloud The input PointCloud to be cropped.
     * @return The cropped PointCloud.
     */
    sensor_msgs::msg::PointCloud2 fov_cropping(const sensor_msgs::msg::PointCloud2 input_ptcloud, geometry_msgs::msg::TransformStamped transf)
    {
        pcl::PCLPointCloud2 *cloud = new pcl::PCLPointCloud2;
        pcl::PCLPointCloud2ConstPtr cloudPtr(cloud);
        pcl::PCLPointCloud2 cropped_cloud;

        pcl_conversions::toPCL(input_ptcloud, *cloud);

        // min is bottom right corner of the box
        double minX = -2.0;
        double minY = -6.0;
        double minZ = -3.0;
        // max is top left corner of the box
        double maxX = 15.0;
        double maxY = 6;
        double maxZ = 3.0;

        Eigen::Matrix4f total_transf = (transf_to_3Dmatrix(transf) * transf_to_3Dmatrix(t_vel_robot)).cast<float>();

        // apply filter
        pcl::CropBox<pcl::PCLPointCloud2> box_filter;
        box_filter.setMin(Eigen::Vector4f(minX, minY, minZ, 1.0));
        box_filter.setMax(Eigen::Vector4f(maxX, maxY, maxZ, 1.0));
        box_filter.setRotation(Eigen::Vector3f(total_transf.block<3, 3>(0, 0).eulerAngles(0, 1, 2)));
        box_filter.setTranslation(Eigen::Vector3f(total_transf.block<3, 1>(0, 3)));
        box_filter.setInputCloud(cloudPtr);
        box_filter.filter(cropped_cloud);

        sensor_msgs::msg::PointCloud2 output_cloud;
        pcl_conversions::fromPCL(cropped_cloud, output_cloud);

        // publish the cropped cloud (for visualization)
        output_cloud.header.set__frame_id("map");
        crop_pub->publish(output_cloud);
        pcl::io::savePCDFile("crop.pcd", cropped_cloud);
        RCLCPP_INFO(this->get_logger(), "CROPPING FINISHED!");

        return output_cloud;
    }

    /**
     * Segment a PointCloud to separate ground plane from cones.
     * @param input_cloud The PointCloud to be segmented.
     * @return The segmented PointCloud.
     */
    std::vector<sensor_msgs::msg::PointCloud2> ptcloud_segmentation(sensor_msgs::msg::PointCloud2 input_cloud)
    {
        pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients());
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
        pcl::SACSegmentation<pcl::PointXYZ> segmenter;

        segmenter.setOptimizeCoefficients(true);
        segmenter.setModelType(pcl::SACMODEL_PLANE);
        segmenter.setMaxIterations(1000);
        segmenter.setDistanceThreshold(0.01); // how close a point must be to the model to be considered an inlier

        // from PointCloud2 msg to PointCloud<PointXYZ>
        pcl::PointCloud<pcl::PointXYZ>::Ptr cropped_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(input_cloud, *cropped_cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr ransac_filt_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        segmenter.setInputCloud(cropped_cloud);
        segmenter.segment(*inliers, *coeffs); // inliers = point indices that support the model

        if (inliers->indices.size() == 0)
        {
            RCLCPP_INFO(this->get_logger(), "Segmentation failed. Planar model can't be estimated for this ptcloud!\n");
        }
        pcl::ExtractIndices<pcl::PointXYZ> extract_indices(true);
        extract_indices.setInputCloud(cropped_cloud);

        // get the cones (obstacles) points
        extract_indices.setIndices(inliers);
        extract_indices.setNegative(true); // this means to consider the cones (not the plane points)
        extract_indices.filter(*ransac_filt_cloud);

        // get the ground points
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_filt_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        auto ground_indices = extract_indices.getRemovedIndices(); // ground
        extract_indices.setIndices(ground_indices);
        extract_indices.setNegative(false);
        extract_indices.filter(*ground_filt_cloud);

        // print plane coeffs ( ax + by + cz + d = 0 )
        float a = coeffs->values[0];
        float b = coeffs->values[1];
        float c = coeffs->values[2];
        float d = coeffs->values[3];
        RCLCPP_INFO(this->get_logger(), "Plane coeffs are a=%f, b=%f, c=%f, d=%f \n", a, b, c, d);

        // publish cones and ground topics
        sensor_msgs::msg::PointCloud2 segmented_cloud;
        pcl::toROSMsg(*ransac_filt_cloud, segmented_cloud);
        sensor_msgs::msg::PointCloud2 track_cloud;
        pcl::toROSMsg(*ground_filt_cloud, track_cloud);

        segmented_cloud.header.set__frame_id("map");
        cones_pub->publish(segmented_cloud);
        pcl::io::savePCDFile("cones.pcd", segmented_cloud);

        track_cloud.header.set__frame_id("map");
        ground_pub->publish(track_cloud);
        pcl::io::savePCDFile("ground.pcd", track_cloud);

        RCLCPP_INFO(this->get_logger(), "SEGMENTATION FINISHED!");

        std::vector<sensor_msgs::msg::PointCloud2> cones_ground_vec;
        cones_ground_vec.push_back(segmented_cloud);
        cones_ground_vec.push_back(track_cloud);

        return cones_ground_vec;
    }

    /**
     * Cluster a PointCloud to group all the segmented non-ground planes.
     * @param cones_cloud The PointCloud resulting from the previous segmentation step, containing only segmented cones.
     * @param cropped_cloud The PointCloud before segmentation. Used to get ground points.
     * @return TODO:
     */
    sensor_msgs::msg::PointCloud2 ptcloud_clustering(sensor_msgs::msg::PointCloud2 cones_cloud, sensor_msgs::msg::PointCloud2 ground_cloud)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cones = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::fromROSMsg(cones_cloud, *cones);
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::fromROSMsg(ground_cloud, *ground);

        // conditional filter to remove points at the origin (points to keep must be >0 or <0)
        pcl::ConditionOr<pcl::PointXYZ>::Ptr range_condition(new pcl::ConditionOr<pcl::PointXYZ>());
        range_condition->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::GT, 0.0)));
        range_condition->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::GT, 0.0)));
        range_condition->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::GT, 0.0)));

        range_condition->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::LT, 0.0)));
        range_condition->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::LT, 0.0)));
        range_condition->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::LT, 0.0)));
        pcl::ConditionalRemoval<pcl::PointXYZ> cond_removal;
        cond_removal.setCondition(range_condition);
        cond_removal.setInputCloud(cones);
        cond_removal.setKeepOrganized(false);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cones_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        cond_removal.filter(*cones_filtered);

        // kd-tree
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kd_tree(new pcl::search::KdTree<pcl::PointXYZ>);
        kd_tree->setInputCloud(cones_filtered);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> eucl_cluster;

        // too small --> an obj is seen as multiple clusters
        // too high --> multiple objs seen as same cluster
        eucl_cluster.setClusterTolerance(cluster_params.cluster_tol);
        eucl_cluster.setMinClusterSize(cluster_params.cluster_min);
        eucl_cluster.setMaxClusterSize(cluster_params.cluster_max);
        eucl_cluster.setSearchMethod(kd_tree);
        eucl_cluster.setInputCloud(cones_filtered);
        eucl_cluster.extract(cluster_indices);
        pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cones(new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<pcl::PointXYZ> cones_markers;

        // for all the found clusters
        for (auto iter = cluster_indices.begin(); iter != cluster_indices.end(); iter++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cone_cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const int &index : iter->indices)
                cone_cluster->points.push_back(cones_filtered->points[index]);
            cone_cluster->header.frame_id = "map";
            cone_cluster->width = cone_cluster->points.size();
            cone_cluster->height = 1;
            cone_cluster->is_dense = true;

            // extract centroids of clusters
            pcl::PointXYZ centroid;
            pcl::computeCentroid(*cone_cluster, centroid);

            // compare
            double d = sqrt(pow(centroid.x, 2) + pow(centroid.y, 2) + pow(centroid.z, 2));
            std::cout << "dist " << d << "\n";
            std::cout << "cone cluster size: " << cone_cluster->size() << "\n";

            // distinguish between different cones dimensions
            double expected_pts_small = num_exp_points(centroid, "SMALL_CONE");
            std::cout << "small cone expected pts = " << expected_pts_small << "\n";
            double expected_pts_big = num_exp_points(centroid, "BIG_CONE");
            std::cout << "big cone expected pts = " << expected_pts_big << "\n";
            int expected_pts;

            if (abs((int)cone_cluster->size() - expected_pts_small) < abs((int)cone_cluster->size() - expected_pts_big))
            {
                expected_pts = expected_pts_small;
                std::cout << "small\n";
            }
            else
            {
                expected_pts = expected_pts_big;
                std::cout << "big\n";
            }

            if (cone_cluster->size() < 0.5 * expected_pts)
            {
                std::cout << " TOO SMALL NUM OF POINTS: skipping processing! \n\n";
                continue;
            } // skip processing if not enough points

            // check if this cone has been already seen
            // if not, insert the current centroid into the global vector of cones
            bool found = false;
            for (int s = 0; s < all_cones_pos.size(); s++)
            {
                if (euclid_distance(all_cones_pos[s], centroid) < 0.7)
                    found = true;
            }
            if (!found)
            {
                all_cones_pos.push_back(centroid);
                cones_markers.push_back(centroid);
            }

            std::cout << "total cones dimension = " << all_cones_pos.size() << "\n\n";

            // cylindrical reconstruction from ground points
            pcl::ConditionAnd<pcl::PointXYZ>::Ptr cylinder_cond(new pcl::ConditionAnd<pcl::PointXYZ>());
            Eigen::Matrix3f cylinder_matrix;
            cylinder_matrix(0, 0) = 1.0;
            cylinder_matrix(1, 1) = 1.0;

            Eigen::Vector3f cylinder_position(-centroid.x, -centroid.y, 0);

            double radius = cluster_params.reconst_radius;
            float cylinder_scalar = -(pow(radius, 2)) + pow(centroid.x, 2) + pow(centroid.y, 2);

            pcl::TfQuadraticXYZComparison<pcl::PointXYZ>::Ptr cylinder_comp(new pcl::TfQuadraticXYZComparison<pcl::PointXYZ>(
                pcl::ComparisonOps::LE, cylinder_matrix, cylinder_position, cylinder_scalar));
            cylinder_cond->addComparison(cylinder_comp);

            pcl::PointCloud<pcl::PointXYZ> recovered;
            cond_removal.setCondition(cylinder_cond);
            cond_removal.setInputCloud(ground);
            cond_removal.setKeepOrganized(false);
            cond_removal.filter(recovered);

            // combine all the cones clusters for visualization
            *clustered_cones += *cone_cluster + recovered;
        }

        // prepare marker array
        visualization_msgs::msg::MarkerArray markers_msg;
        markers_msg.markers.resize(cones_markers.size());
        for (int i = 0; i < cones_markers.size(); i++)
        {
            general_id++;
            set_marker_properties(&markers_msg.markers[i], cones_markers[i], general_id, ground->header.frame_id);
        }
        sensor_msgs::msg::PointCloud2 cluster_output;
        pcl::toROSMsg(*clustered_cones, cluster_output);
        cluster_output.header.frame_id = ground->header.frame_id;
        cluster_output.header.stamp = rclcpp::Node::now();
        pcl::io::savePCDFile("cluster.pcd", cluster_output);

        markers_pub->publish(markers_msg);
        clust_pub->publish(cluster_output);

        return cluster_output;
    }

    /**
     * @brief Construct transformation matrix from tf2 transform.
     * @param t A transformation as translation and rotation components.
     * @return A 2D transformation matrix representing the given transformation.
     */
    Eigen::Matrix4d transf_to_3Dmatrix(geometry_msgs::msg::TransformStamped t)
    {
        Eigen::Quaterniond quat;
        quat.x() = t.transform.rotation.x;
        quat.y() = t.transform.rotation.y;
        quat.z() = t.transform.rotation.z;
        quat.w() = t.transform.rotation.w;

        auto rot_mat = quat.toRotationMatrix();
        Eigen::Vector3d translation;
        translation.x() = t.transform.translation.x;
        translation.y() = t.transform.translation.y;
        translation.z() = t.transform.translation.z;

        Eigen::Matrix4d t_matrix;
        t_matrix.setIdentity();
        t_matrix.block<3, 3>(0, 0) = rot_mat;
        t_matrix.block<3, 1>(0, 3) = translation;

        return t_matrix;
    }

    /**
     * Computes the number of expected points in each cone, according to its known dimensions.
     * @param central_point The centre of the detected cone object.
     * @param cone_type Small or big, depending on the type of the cone.
     * @return The number of points expected to be inside this object.
     */
    double num_exp_points(pcl::PointXYZ central_point, std::string cone_type)
    {
        double dist = sqrt(pow(central_point.x, 2) + pow(central_point.y, 2) + pow(central_point.z, 2)); // distance to the cluster

        static double cone_height;
        static double cone_width;

        if (cone_type == "SMALL_CONE")
        {
            cone_height = SMALL_CONE_HEIGHT;
            cone_width = SMALL_CONE_WIDTH;
        }
        else
        {
            cone_height = BIG_CONE_HEIGHT;
            cone_width = BIG_CONE_WIDTH;
        }

        static double vertical_res = 2 * (10 * M_PI / 180) / 16;
        // try different values for resolution, to match small/big cones dimension
        static double horizontal_res = 2 * M_PI / 1800; // horiz. res is 0.1°-0.4° --> 3600 with 0.1°, 1800, 1200, or 900 with 0.4°

        // multiply by 5 because 5 pointclouds are aggregated TODO: try to think about a better method!
        double exp_pts = 3 * (0.5 * cone_height / (2 * dist * tan(vertical_res / 2)) * cone_width / (2 * dist * tan(horizontal_res / 2)));

        return exp_pts;
    }
    /**
     * TODO::
     */
    void set_marker_properties(visualization_msgs::msg::Marker *marker, pcl::PointXYZ center, int id, std::string frame_id)
    {
        marker->header.frame_id = frame_id;
        marker->header.stamp = rclcpp::Node::now();
        marker->ns = "cone_det_";
        marker->id = id;
        marker->type = visualization_msgs::msg::Marker::CYLINDER;
        marker->action = visualization_msgs::msg::Marker::MODIFY;

        marker->pose.position.x = center.x;
        marker->pose.position.y = center.y;
        marker->pose.position.z = center.z;

        marker->pose.orientation.x = 0.0;
        marker->pose.orientation.y = 0.0;
        marker->pose.orientation.z = 0.0;
        marker->pose.orientation.w = 1.0;

        marker->scale.x = cluster_params.marker_x;
        marker->scale.y = cluster_params.marker_y;
        marker->scale.z = cluster_params.marker_z;

        marker->color.a = cluster_params.marker_alpha;
        marker->color.r = cluster_params.marker_r;
        marker->color.g = cluster_params.marker_g;
        marker->color.b = cluster_params.marker_b;

        marker->lifetime = rclcpp::Duration(0); // last forever
    }

    /**
     * @param a The first point.
     * @param b The second point.
     * @return The euclidean distance between two PCL points.
     */
    double euclid_distance(pcl::PointXYZ a, pcl::PointXYZ b)
    {
        return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    rclcpp::executors::MultiThreadedExecutor executor;
    node_ptr = std::make_shared<LidarDetection>(options);
    executor.add_node(node_ptr);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}