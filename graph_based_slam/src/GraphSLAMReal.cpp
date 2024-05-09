#define BOOST_BIND_NO_PLACEHOLDERS

// Project headers
#include "graph_based_slam/GraphHandling.h"
#include "graph_based_slam/ConeData.h"
#include "graph_based_slam/PoseData.h"
#include <eufs_msgs/msg/cone_array_with_covariance.hpp>
#include <eufs_msgs/msg/wheel_speeds_stamped.hpp>
#include <eufs_msgs/msg/car_state.hpp>
#include <graph_based_slam/msg/graph_edge.hpp>
#include <graph_based_slam/msg/graph_node.hpp>
#include <graph_based_slam/msg/pose_graph.hpp>
#include <graph_based_slam/car_params.hpp>
#include <graph_based_slam/car_state.hpp>
#include <graph_based_slam/ks_kinematics.hpp>
#include <graph_based_slam/utilities.hpp>

//opencv headers
#include <opencv2/opencv.hpp>
#include <opencv2/gapi/stereo.hpp>

// C++ standard headers
#include <memory>
#include <iostream>
#include <chrono>
#include <iterator>
#include <random>
#include <deque>
#include <stack>
#include <mutex>

// ROS headers
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <message_filters/subscriber.h>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <ackermann_msgs/msg/ackermann_drive.h>

// Eigen headers
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
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

// time sync headers
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

using std::placeholders::_1;
using namespace std::chrono_literals;

rclcpp::Node::SharedPtr slam_node_ptr;

// global variables to define cone colors/classes
std::string BLUE_CONE = "blue";
std::string YELLOW_CONE = "yellow";
std::string ORANGE_CONE = "orange";

// global variables to distinguish element in the graph
std::string LANDMARK_TYPE = "landmark";
std::string ROBOT_TYPE = "robot";

// sentinel value for identification of the first node
const int FIRST_NODE_ID = -999;


// Structures to handle simultaneous data acquisition and graph optimization
// store all info about a landmark node that has been created, but not added to the graph yet
struct ConeToAdd
{
  g2o::VertexPointXY *cone_node;
  std_msgs::msg::Header data_header;
  geometry_msgs::msg::Point bf_coords;
  geometry_msgs::msg::Point map_coords;
  geometry_msgs::msg::PointStamped ideal_coords;
  std::string color;
  int unique_id;
  g2o::VertexSE2 *curr_car_pose;
};

// store all info about an observation edge that has been created, but not added to the graph yet
struct ObsEdgeToAdd
{
  geometry_msgs::msg::Point rel_obs; // observation coords in relative frame
  g2o::VertexPointXY *cone_vertex;   // cone vertex in the graph
  g2o::VertexSE2 *pose_vertex;       // current position of the car
};

// store all info about a robot pose node that has been created, but not added to the graph yet
struct PoseToAdd
{
  g2o::VertexSE2 *pose_node;
  std_msgs::msg::Header data_header;
  Eigen::Isometry2d real_pose;
  geometry_msgs::msg::Transform ideal_pose;
  double rot_angle;
  g2o::VertexSE2 *prev_pose;
};

class GraphSLAMReal : public rclcpp::Node
{

public:
    // variables
    int msg_count = 0;
    bool doing_optim = false;
    int total_cones = 0;
    int added_pose_vertices = 0;
    bool init_cones = false;
    int optim_c = 0;
    int cone_unique_ID = 0;


    // thresholds for data association
    double DIST_THRESHOLD = 2.5; // use 2.5 if not using ideal data association, 0.8 for ideal
    // time for data quantizatin
    double TIME_THRESHOLD = 1.0;
    // threshold for optimization
    double OPTIM_THRESHOLD = 10.0;

    //Only perform loop closure once
    bool LOOP_CLOSED = false;


    //graph visualization
    std::vector<graph_based_slam::msg::GraphNode> all_nodes_msg;
    std::vector<graph_based_slam::msg::GraphEdge> all_edges_msg;

    Eigen::Vector2d position_noise; // odometry noise, along x and y
    Eigen::Vector2d cone_noise;     // cone position noise, along x and y

    // Variables to handle the modification of the graph while doing optimization
    // Nodes and edges created while optimizing the graph, will be added later, when finished
    std::vector<ConeToAdd> pending_cones;        // hold info about pending cone nodes
    std::vector<ObsEdgeToAdd> pending_obs_edges; // hold info about pending observation edges
    std::vector<PoseToAdd> pending_poses;        // hold info about pending pose nodes

    // variables to handle data quantization over time
    std::chrono::steady_clock::time_point cones_timer, odom_timer, wheel_timer, optim_timer;

    g2o::VertexSE2 *first_fixed_pose;
    
    std::chrono::steady_clock::time_point _init_time = std::chrono::steady_clock::now();

    // constructor
    GraphSLAMReal(const rclcpp::NodeOptions &options) : Node("graph_slam_real", options)
    {
        graph_handler_ptr.reset(new GraphHandling());

        rmw_qos_profile_t custom_qos_profile = rmw_qos_profile_default;
        custom_qos_profile.depth = 1;
        custom_qos_profile.reliability = rmw_qos_reliability_policy_t::RMW_QOS_POLICY_RELIABILITY_RELIABLE;
        custom_qos_profile.history = rmw_qos_history_policy_t::RMW_QOS_POLICY_HISTORY_KEEP_LAST;
        custom_qos_profile.durability = rmw_qos_durability_policy_t::RMW_QOS_POLICY_DURABILITY_VOLATILE;
        // declare and acquire `target_frame` parameter
        target_frame = this->declare_parameter<std::string>("target_frame", "base_footprint");
        

        //subscriber to odometry
        cones_sub.subscribe(this, "/cone_pose", custom_qos_profile);

        //subscriber to cones in world frame
        odom_sub.subscribe(this, "/car_real_state", custom_qos_profile);
        
        //synchronize odometry and cone positions
        syncApprox = std::make_shared<message_filters::Synchronizer<approx_policy>>(approx_policy(1), cones_sub, odom_sub);
        syncApprox->registerCallback(&GraphSLAMReal::sync_callback, this);

        // add the first fixed node [0 0 0], that will not be optimized
        Eigen::Isometry2d first_pose = Eigen::Isometry2d::Identity();
        first_pose.translation() = Eigen::Vector2d(0, 0);
        first_pose.linear() = Eigen::Rotation2Dd(0).toRotationMatrix();
        g2o::VertexSE2 *initial_pose = graph_handler_ptr->addPoseVertex(graph_handler_ptr->createPoseVertex(first_pose));

        // message for new vertex visualization
        all_nodes_msg.push_back(nodeMessage(ROBOT_TYPE, static_cast<int>(initial_pose->id()), rclcpp::Node::now(), first_pose.translation()[0], first_pose.translation()[1], "", 0.0));

        rclcpp::Time zero_stamp;
        geometry_msgs::msg::Transform zero_transform;
        zero_transform.translation.x = first_pose.translation().x();
        zero_transform.translation.y = first_pose.translation().y();
        // unit quaternion, no rotation
        zero_transform.rotation.x = 0;
        zero_transform.rotation.y = 0;
        zero_transform.rotation.z = 0;
        zero_transform.rotation.w = 1;

        // noise params
        // TODO: tune these values
        position_noise[0] = 0.00000001;
        position_noise[1] = 0.00000001;
        position_noise[2] = 0.00000001; // heading noise: this should be zero, but it creates problems with the inverse
        cone_noise[0] = 0.00000001;
        cone_noise[1] = 0.00000001;

        // add pose data
        addPoseData(initial_pose, zero_transform, zero_stamp, FIRST_NODE_ID);

        initial_pose->setFixed(true);
        first_fixed_pose = initial_pose;

        timer_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        timer_optim = this->create_wall_timer(300.5s, std::bind(&GraphSLAMReal::timer_callback, this), timer_cb_group); // TODO: handle callback group
        
        // initialize publisher for the pose graph
        graph_pub = this->create_publisher<graph_based_slam::msg::PoseGraph>("pose_graph", 1);
    }
    virtual ~GraphSLAMReal(){};

private:
    // variables
    std::unique_ptr<GraphHandling> graph_handler_ptr;
    message_filters::Subscriber<eufs_msgs::msg::ConeArrayWithCovariance> cones_sub;
    message_filters::Subscriber<eufs_msgs::msg::CarState> odom_sub;
    typedef message_filters::sync_policies::ApproximateTime<eufs_msgs::msg::ConeArrayWithCovariance, eufs_msgs::msg::CarState> approx_policy;
    std::shared_ptr<message_filters::Synchronizer<approx_policy>> syncApprox;
    std::string target_frame;
    rclcpp::Publisher<graph_based_slam::msg::PoseGraph>::SharedPtr graph_pub;
    rclcpp::TimerBase::SharedPtr timer_optim;
    rclcpp::CallbackGroup::SharedPtr timer_cb_group;

    // cone metrics
    int num_cones_mapped, num_cones_gt;              // compare the number of detected cones with the real
    double true_pos, true_neg, false_pos, false_neg; // metrics about cone association
    int total_assoc_test;                            // num of times data association has been checked
    bool gt_cones_done = false;


    /**
     * @brief Generate a random noise value, following Gaussian distribution.
     * @param mean The mean of the desired Gaussian distribution.
     * @param stddev The standard deviation of the desired Gaussian distribution.
     *
     * @return The value of noise to add to data.
     */
    void sync_callback(eufs_msgs::msg::ConeArrayWithCovariance::SharedPtr cones_msg, eufs_msgs::msg::CarState::SharedPtr odom_msg)
    {
        msg_count++;

        // save ground truth odom
        nav_msgs::msg::Odometry ideal_odom;
        ideal_odom.header = odom_msg->header;
        ideal_odom.child_frame_id = odom_msg->child_frame_id;
        ideal_odom.pose = odom_msg->pose;
        ideal_odom.twist = odom_msg->twist;

        // save ground truth cones
        eufs_msgs::msg::ConeArrayWithCovariance ideal_cones;
        ideal_cones.header = cones_msg->header;
        ideal_cones.big_orange_cones = cones_msg->big_orange_cones;
        ideal_cones.orange_cones = cones_msg->orange_cones;
        ideal_cones.blue_cones = cones_msg->blue_cones;
        ideal_cones.yellow_cones = cones_msg->yellow_cones;
        ideal_cones.unknown_color_cones = cones_msg->unknown_color_cones;

        // quantization of incoming data
        std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();

        auto now_time = this->get_clock()->now();

        // retrieve the ideal odometry transformation, without noise
        geometry_msgs::msg::TransformStamped ideal_t_robot_map;
        ideal_t_robot_map.header.stamp = now_time;
        ideal_t_robot_map.header.frame_id = "map";
        ideal_t_robot_map.child_frame_id = "base_footprint";

        ideal_t_robot_map.transform.translation.x = ideal_odom.pose.pose.position.x;
        ideal_t_robot_map.transform.translation.y = ideal_odom.pose.pose.position.y;
        ideal_t_robot_map.transform.translation.z = ideal_odom.pose.pose.position.z;

        ideal_t_robot_map.transform.rotation.x = ideal_odom.pose.pose.orientation.x;
        ideal_t_robot_map.transform.rotation.y = ideal_odom.pose.pose.orientation.y;
        ideal_t_robot_map.transform.rotation.z = ideal_odom.pose.pose.orientation.z;
        ideal_t_robot_map.transform.rotation.w = ideal_odom.pose.pose.orientation.w;

        g2o::VertexSE2 *prev_pose = graph_handler_ptr->getLastPoseNode();

        msg_count = 0;
        // ------------------------------------ ODOMETRY DATA PROCESSING ------------------------------------
        odom_timer = now;

        // ideal rotation angle
        Eigen::Quaterniond ideal_quat;
        ideal_quat.w() = ideal_t_robot_map.transform.rotation.w;
        ideal_quat.x() = ideal_t_robot_map.transform.rotation.x;
        ideal_quat.y() = ideal_t_robot_map.transform.rotation.y;
        ideal_quat.z() = ideal_t_robot_map.transform.rotation.z;
        double ideal_rotation_angle = ideal_quat.toRotationMatrix().eulerAngles(0, 1, 2)[2]; // yaw, around z


        // ideal pose
        Eigen::Isometry2d ideal_pose = Eigen::Isometry2d::Identity();
        ideal_pose.translation() = Eigen::Vector2d(ideal_t_robot_map.transform.translation.x, ideal_t_robot_map.transform.translation.y);
        ideal_pose.linear() = Eigen::Rotation2Dd(ideal_rotation_angle).toRotationMatrix();

        g2o::VertexSE2 *new_node = graph_handler_ptr->createPoseVertex(ideal_pose);

        if (!doing_optim)
        {
            g2o::VertexSE2 *curr_node = graph_handler_ptr->addPoseVertex(new_node);
            added_pose_vertices++;

            // message for new vertex visualization
            all_nodes_msg.push_back(nodeMessage(ROBOT_TYPE, static_cast<int>(curr_node->id()), cones_msg->header.stamp, ideal_pose.translation()[0], ideal_pose.translation()[1], "", ideal_rotation_angle));

            // add pose data
            addPoseData(curr_node, ideal_t_robot_map.transform, cones_msg->header.stamp);

            // add odometry edge between consecutive poses
            auto prev_mat = prev_pose->estimate();
            auto curr_mat = curr_node->estimate();

            // rel_pose = inverse(prev_pose) * curr_pose
            Eigen::Isometry2d rel_pose = prev_mat.toIsometry().inverse() * curr_mat.toIsometry();

            Eigen::Matrix3d cov_mat;
            cov_mat.fill(0.0);
            cov_mat(0, 0) = pow(position_noise[0], 2);
            cov_mat(1, 1) = pow(position_noise[1], 2);
            cov_mat(2, 2) = pow(position_noise[2], 2);
            Eigen::Matrix3d info_mat = cov_mat.inverse();
            g2o::EdgeSE2 *new_edge = graph_handler_ptr->createPoseEdge(prev_pose, curr_node, rel_pose, info_mat);
            graph_handler_ptr->addPoseEdge(new_edge);
            all_edges_msg.push_back(edgeMessage(ROBOT_TYPE, static_cast<int>(new_edge->vertices()[0]->id()), static_cast<int>(new_edge->vertices()[1]->id())));
        }
        else
        {
            PoseToAdd pending_pose;
            pending_pose.pose_node = new_node;
            pending_pose.data_header = odom_msg->header;
            pending_pose.real_pose = ideal_pose;
            pending_pose.rot_angle = ideal_rotation_angle;
            pending_pose.ideal_pose = ideal_t_robot_map.transform;
            pending_pose.prev_pose = prev_pose;
            pending_poses.push_back(pending_pose);
        }

            //************************************************** LOCAL OPTIMIZATION ***********************************************
            // // perform local optimization every 10 new nodes
            // if (added_pose_vertices >= 10 && !doing_optim)
            // {
            //     doing_optim = true;

            //     // restore the counter
            //     added_pose_vertices = 0;

            //     // save the portion of the graph not fixed (the one to optimize)
            //     g2o::HyperGraph::EdgeSet edges_to_optim;
            //     auto pre_opt_edges = graph_handler_ptr->getEdges();
            //     for (g2o::HyperGraph::EdgeSet::iterator it = pre_opt_edges.begin(); it != pre_opt_edges.end(); ++it)
            //     {
            //         g2o::OptimizableGraph::Edge *edge = (g2o::OptimizableGraph::Edge *)(*it);
            //         NodeData *vert_i_data = dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(edge->vertices()[0])->userData());
            //         NodeData *vert_j_data = dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(edge->vertices()[1])->userData());
            //         bool vert_i_fixed, vert_j_fixed;
            //         // check if vertex_i is fixed
            //         if (vert_i_data->getNodeType() == ROBOT_TYPE)
            //         {
            //             g2o::VertexSE2 *vert_i = dynamic_cast<g2o::VertexSE2 *>(edge->vertices()[0]);
            //             vert_i_fixed = vert_i->fixed();
            //         }
            //         else
            //         {
            //             g2o::VertexPointXY *vert_i = dynamic_cast<g2o::VertexPointXY *>(edge->vertices()[0]);
            //             vert_i_fixed = vert_i->fixed();
            //         }
            //         // check if vertex_j is fixed
            //         if (vert_j_data->getNodeType() == ROBOT_TYPE)
            //         {
            //             g2o::VertexSE2 *vert_j = dynamic_cast<g2o::VertexSE2 *>(edge->vertices()[1]);
            //             vert_j_fixed = vert_j->fixed();
            //         }
            //         else
            //         {
            //             g2o::VertexPointXY *vert_j = dynamic_cast<g2o::VertexPointXY *>(edge->vertices()[1]);
            //             vert_j_fixed = vert_j->fixed();
            //         }

            //         // if at least one of the two vertices is not fixed, optimize this edge
            //         if ((!vert_i_fixed) || (!vert_j_fixed))
            //         {
            //             edges_to_optim.insert(edge);
            //         }
            //     }

            //     // perform local optimization
            //     graph_handler_ptr->localOptimization(1000, edges_to_optim);

            //     // fix all the optimized vertices
            //     auto opt_vertices = graph_handler_ptr->getNodes();
            //     for (int i = 0; i < opt_vertices.size(); i++)
            //     {
            //         auto vert = opt_vertices[i];
            //         NodeData *vert_data = dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(vert)->userData());
            //         if (vert_data->getNodeType() == ROBOT_TYPE)
            //         {
            //             g2o::VertexSE2 *pose_vert = dynamic_cast<g2o::VertexSE2 *>(vert);
            //             pose_vert->setFixed(true);
            //         }
            //         else
            //         {
            //             g2o::VertexPointXY *cone_vert = dynamic_cast<g2o::VertexPointXY *>(vert);
            //             cone_vert->setFixed(true);
            //         }
            //     }

            //     // release the optimization flag
            //     doing_optim = false;
            // }

        //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        // ------------------------------------ CONES DATA PROCESSING -----------------------------------

        cones_timer = now;

        // NB: TF2 frames are: source is the frame of the data, while target is the frame to transform to
        std::string to_frame = "map";                  //(called target frame in tf2 params)
        std::string from_frame = target_frame.c_str(); //(called source frame in tf2 params)  //base_footprint


        /** To process data of perceived cones:
         * 1. if first scan, add all of them to the graph by doing initialization
         * 2. otherwise, check for data association
         * 3. if no association, add a new node, otherwise add only an edge
         */

        //************************************************************** INITIAL SCAN **************************************************************

        // add initial data about first scan (initialize the graph)
        if (!init_cones)
        {
            // update the time of the last processed cone callback
            cones_timer = now; 

            // retrieve the node corresponding to the current car pose
            g2o::VertexSE2 *curr_car_pose = graph_handler_ptr->getLastPoseNode();

            // process big orange cones
            for (int i = 0; i < static_cast<int>(cones_msg->big_orange_cones.size()); i++)
            {
                cone_unique_ID++;

                // transform to map frame
                geometry_msgs::msg::PointStamped cone_map_frame = coneTransform(cones_msg->header, cones_msg->big_orange_cones[i].point, ideal_t_robot_map);

                geometry_msgs::msg::PointStamped ideal_cone_map_frame = coneTransform(ideal_cones.header, ideal_cones.big_orange_cones[i].point, ideal_t_robot_map);

                g2o::VertexPointXY *cone_node = graph_handler_ptr->createLandmarkVertex(Eigen::Vector2d(cone_map_frame.point.x, cone_map_frame.point.y));

                if (!doing_optim)
                {
                    // if not doing optimization, modify the graph and add the new elements

                    g2o::VertexPointXY *added_vertex = graph_handler_ptr->addLandmarkVertex(cone_node);
                    addConeData(added_vertex, cone_unique_ID, ORANGE_CONE, cones_msg->header.stamp, cone_map_frame);

                    // message for new vertex visualization
                    all_nodes_msg.push_back(nodeMessage(LANDMARK_TYPE, static_cast<int>(added_vertex->id()), cones_msg->header.stamp, cone_map_frame.point.x, cone_map_frame.point.y, ORANGE_CONE));

                    total_cones++;

                    // add an edge between the current pose and the observed cones (pose-landmark edge)
                    g2o::EdgeSE2PointXY *added_edge = add_cone_edge(CameraToRobotTransformer(cones_msg->big_orange_cones[i].point), added_vertex, curr_car_pose);
                }
                else
                {
                    // otherwise, store them and process them later

                    ConeToAdd pending_cone;
                    pending_cone.cone_node = cone_node;
                    pending_cone.data_header = cones_msg->header;
                    pending_cone.bf_coords = cones_msg->big_orange_cones[i].point;
                    pending_cone.map_coords = cone_map_frame.point;
                    pending_cone.ideal_coords = ideal_cone_map_frame;
                    pending_cone.color = ORANGE_CONE;
                    pending_cone.unique_id = cone_unique_ID;
                    pending_cone.curr_car_pose = curr_car_pose;
                    pending_cones.push_back(pending_cone);
                }
            }
        
        

            // process blue cones
            for (int i = 0; i < static_cast<int>(cones_msg->blue_cones.size()); i++)
            {
                // RCLCPP_INFO(this->get_logger(), "Adding blue cone with unique ID = %d", cone_unique_ID);
                cone_unique_ID++;

                // transform to map frame
                geometry_msgs::msg::PointStamped cone_map_frame = coneTransform(cones_msg->header, cones_msg->blue_cones[i].point, ideal_t_robot_map);

                geometry_msgs::msg::PointStamped ideal_cone_map_frame = coneTransform(ideal_cones.header, ideal_cones.blue_cones[i].point, ideal_t_robot_map);

                g2o::VertexPointXY *cone_node = graph_handler_ptr->createLandmarkVertex(Eigen::Vector2d(cone_map_frame.point.x, cone_map_frame.point.y));

                if (!doing_optim)
                {
                    g2o::VertexPointXY *added_vertex = graph_handler_ptr->addLandmarkVertex(cone_node);
                    total_cones++;

                    // message for new vertex visualization
                    all_nodes_msg.push_back(nodeMessage(LANDMARK_TYPE, static_cast<int>(added_vertex->id()), cones_msg->header.stamp, cone_map_frame.point.x, cone_map_frame.point.y, BLUE_CONE));

                    addConeData(added_vertex, cone_unique_ID, BLUE_CONE, cones_msg->header.stamp, cone_map_frame);

                    // add an edge between the current pose and the observed cones (pose-landmark edge)
                    g2o::EdgeSE2PointXY *added_edge = add_cone_edge(CameraToRobotTransformer(cones_msg->blue_cones[i].point), added_vertex, curr_car_pose);
                }
                else
                {
                    // TODO: handle the graph modification while doing optimization
                    ConeToAdd pending_cone;
                    pending_cone.cone_node = cone_node;
                    pending_cone.data_header = cones_msg->header;
                    pending_cone.bf_coords = cones_msg->blue_cones[i].point;
                    pending_cone.map_coords = cone_map_frame.point;
                    pending_cone.ideal_coords = ideal_cone_map_frame;
                    pending_cone.color = BLUE_CONE;
                    pending_cone.unique_id = cone_unique_ID;
                    pending_cone.curr_car_pose = curr_car_pose;
                    pending_cones.push_back(pending_cone);
                }
            }

            // process yellow cones
            for (int i = 0; i < static_cast<int>(cones_msg->yellow_cones.size()); i++)
            {
                // RCLCPP_INFO(this->get_logger(), "Adding yellow cone with unique ID = %d", cone_unique_ID);
                cone_unique_ID++;

                // transform to map frame
                geometry_msgs::msg::PointStamped cone_map_frame = coneTransform(cones_msg->header, cones_msg->yellow_cones[i].point, ideal_t_robot_map);

                g2o::VertexPointXY *cone_node = graph_handler_ptr->createLandmarkVertex(Eigen::Vector2d(cone_map_frame.point.x, cone_map_frame.point.y));

                geometry_msgs::msg::PointStamped ideal_cone_map_frame = coneTransform(ideal_cones.header, ideal_cones.yellow_cones[i].point, ideal_t_robot_map);

                if (!doing_optim)
                {
                    g2o::VertexPointXY *added_vertex = graph_handler_ptr->addLandmarkVertex(cone_node);
                    total_cones++;

                    // message for new vertex visualization
                    all_nodes_msg.push_back(nodeMessage(LANDMARK_TYPE, static_cast<int>(added_vertex->id()), cones_msg->header.stamp, cone_map_frame.point.x, cone_map_frame.point.y, YELLOW_CONE));

                    addConeData(added_vertex, cone_unique_ID, YELLOW_CONE, cones_msg->header.stamp, cone_map_frame);

                    // add an edge between the current pose and the observed cones (pose-landmark edge)
                    g2o::EdgeSE2PointXY *added_edge = add_cone_edge(CameraToRobotTransformer(cones_msg->yellow_cones[i].point), added_vertex, curr_car_pose);
                }
                else
                {
                    ConeToAdd pending_cone;
                    pending_cone.cone_node = cone_node;
                    pending_cone.data_header = cones_msg->header;
                    pending_cone.bf_coords = cones_msg->yellow_cones[i].point;
                    pending_cone.map_coords = cone_map_frame.point;
                    pending_cone.ideal_coords = ideal_cone_map_frame;
                    pending_cone.color = YELLOW_CONE;
                    pending_cone.unique_id = cone_unique_ID;
                    pending_cone.curr_car_pose = curr_car_pose;
                    pending_cones.push_back(pending_cone);
                }
            }

            init_cones = true;
            int num_cones = graph_handler_ptr->num_vertices();
            int num_edges = graph_handler_ptr->num_edges();
        }

        //***************************************************** CONES OBSERVATIONS PROCESSING ******************************************************

        // Every time a cone observation is received, perform data association.
        // If it does not find any correspondence with the landmark nodes already in the graph, process the new cone and add it (node and edges).
        // If some correspondences are found, then just add the corresponding edges between the current pose and the observed cones.

        // retrieve the node corresponding to the current car pose
        g2o::VertexSE2 *last_pose = graph_handler_ptr->getLastPoseNode();
        auto curr_pose_node_id = last_pose->id();

        //---------------------------------------------------- LOOP CLOSURE CHECK START------------------------------------------------------------------
        // NB: add loop closure edges when orange cones are observed --> add edge between current pose and first one
        // g2o::VertexSE2 *curr_car_pose = graph_handler_ptr->getLastPoseNode();

        //timer condition to prevent loop closure on the first time the cone is visited
        auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(now - _init_time);
        if (cones_msg->big_orange_cones.size() >= 2 && !LOOP_CLOSED) //&& time_diff.count() > 90.0 && !LOOP_CLOSED
        {
            // LOOP CLOSURE DETECTED!
            auto prev_mat = first_fixed_pose->estimate();
            auto curr_mat = last_pose->estimate();

            //  currently, rel_pose = inverse(prev_pose) * curr_pose
            Eigen::Isometry2d rel_pose = prev_mat.toIsometry().inverse() * curr_mat.toIsometry();

            Eigen::Matrix3d cov_mat;
            cov_mat.fill(0.0);
            cov_mat(0, 0) = pow(position_noise[0], 2);
            cov_mat(1, 1) = pow(position_noise[1], 2);
            cov_mat(2, 2) = pow(position_noise[2], 2);
            Eigen::Matrix3d info_mat = cov_mat.inverse();

            // add loop closure edge
            g2o::EdgeSE2 *new_edge = graph_handler_ptr->createPoseEdge(first_fixed_pose, last_pose, rel_pose.inverse(), info_mat);
            std::cout << "first fixed point: " << first_fixed_pose << "  last pose: " << last_pose << std::endl;
            graph_handler_ptr->addPoseEdge(new_edge);
            //LOOP_CLOSED = true;
            timer_callback();

            // construct the message for the new edge
            all_edges_msg.push_back(edgeMessage(ROBOT_TYPE, static_cast<int>(new_edge->vertices()[0]->id()), static_cast<int>(new_edge->vertices()[1]->id())));
        }


        //---------------------------------------------------- LOOP CLOSURE CHECK END------------------------------------------------------------------

        // big orange cones
        for (int i = 0; i < cones_msg->big_orange_cones.size(); i++)
        {
            // transform to map frame 
            geometry_msgs::msg::PointStamped cone_map_frame = coneTransform(cones_msg->header, cones_msg->big_orange_cones[i].point, ideal_t_robot_map);

            geometry_msgs::msg::PointStamped ideal_cone_map_frame = coneTransform(ideal_cones.header, ideal_cones.big_orange_cones[i].point, ideal_t_robot_map);
            
            // test association
            g2o::VertexPointXY *assoc_found = perform_data_assoc(ideal_cone_map_frame.point, cone_map_frame.point, curr_pose_node_id);
            // g2o::VertexPointXY *assoc_found = perform_data_assoc(cone_map_frame.point);
            if (assoc_found != nullptr)
            {
            std::cout << "orange_association[" << dynamic_cast<g2o::VertexPointXY *>(assoc_found)->estimate().x()
                      << "," << dynamic_cast<g2o::VertexPointXY *>(assoc_found)->estimate().y() << "]\n";

            }

            if (assoc_found == nullptr)
            {
                // new cone observed, add the node
                cone_unique_ID++;
                g2o::VertexPointXY *cone_node = graph_handler_ptr->createLandmarkVertex(Eigen::Vector2d(cone_map_frame.point.x, cone_map_frame.point.y));

                if (!doing_optim)
                {
                    g2o::VertexPointXY *added_vertex = graph_handler_ptr->addLandmarkVertex(cone_node);
                    total_cones++;

                    // message for new vertex visualization
                    all_nodes_msg.push_back(nodeMessage(LANDMARK_TYPE, static_cast<int>(added_vertex->id()), cones_msg->header.stamp, cone_map_frame.point.x, cone_map_frame.point.y, ORANGE_CONE));

                    addConeData(added_vertex, cone_unique_ID, ORANGE_CONE, cones_msg->header.stamp, cone_map_frame);
                    // add an edge between the current pose and the observed cones (pose-landmark edge)
                    g2o::EdgeSE2PointXY *added_edge = add_cone_edge(CameraToRobotTransformer(cones_msg->big_orange_cones[i].point), added_vertex, last_pose);
                }
                else
                {
                    ConeToAdd pending_cone;
                    pending_cone.cone_node = cone_node;
                    pending_cone.data_header = cones_msg->header;
                    pending_cone.bf_coords = cones_msg->big_orange_cones[i].point;
                    pending_cone.map_coords = cone_map_frame.point;
                    pending_cone.ideal_coords = ideal_cone_map_frame;
                    pending_cone.color = ORANGE_CONE;
                    pending_cone.unique_id = cone_unique_ID;
                    pending_cone.curr_car_pose = last_pose;
                    pending_cones.push_back(pending_cone);
                }
            }
            else // otherwise add only an edge between the current pose and the previously observed node
            {
                // add an edge between the current pose and associated cone
                if (!doing_optim)
                {
                    // compute the coordinates of the associateed vertex in base footprint --> apply inverse transformation
                    // it was cones_msg->big_orange_cones[i].point
                    g2o::EdgeSE2PointXY *added_edge = add_cone_edge(CameraToRobotTransformer(cones_msg->big_orange_cones[i].point), assoc_found, last_pose);
                }
                else
                {
                    ObsEdgeToAdd pending_edge;
                    pending_edge.rel_obs = cone_map_frame.point;
                    pending_edge.cone_vertex = assoc_found;
                    pending_edge.pose_vertex = last_pose;

                    pending_obs_edges.push_back(pending_edge);
                }
            }
        }

        //  blue cones
        for (int i = 0; i < cones_msg->blue_cones.size(); i++)
        {
            // transform to map frame
            geometry_msgs::msg::PointStamped cone_map_frame = coneTransform(cones_msg->header, cones_msg->blue_cones[i].point, ideal_t_robot_map);
            // transform to map frame the ideal observation (ground truth)
            geometry_msgs::msg::PointStamped ideal_cone_map_frame = coneTransform(ideal_cones.header, ideal_cones.blue_cones[i].point, ideal_t_robot_map);

            // test association
            g2o::VertexPointXY *assoc_found = perform_data_assoc(ideal_cone_map_frame.point, cone_map_frame.point, curr_pose_node_id);
            // g2o::VertexPointXY *assoc_found = perform_data_assoc(cone_map_frame.point);

            if (assoc_found != nullptr)
            {
            // std::cout << "blue_association[" << dynamic_cast<g2o::VertexPointXY *>(assoc_found)->estimate().x()
            //           << "," << dynamic_cast<g2o::VertexPointXY *>(assoc_found)->estimate().y() << "]\n";
            }

            if (assoc_found == nullptr)
            {
                // new cone observed, add the node
                cone_unique_ID++;
                g2o::VertexPointXY *cone_node = graph_handler_ptr->createLandmarkVertex(Eigen::Vector2d(cone_map_frame.point.x, cone_map_frame.point.y));

                if (!doing_optim)
                {
                    g2o::VertexPointXY *added_vertex = graph_handler_ptr->addLandmarkVertex(cone_node);
                    total_cones++;

                    // message for new vertex visualization
                    all_nodes_msg.push_back(nodeMessage(LANDMARK_TYPE, static_cast<int>(added_vertex->id()), cones_msg->header.stamp, cone_map_frame.point.x, cone_map_frame.point.y, BLUE_CONE));

                    addConeData(added_vertex, cone_unique_ID, BLUE_CONE, cones_msg->header.stamp, cone_map_frame);

                    // add an edge between the current pose and the observed cones (pose-landmark edge)
                    g2o::EdgeSE2PointXY *added_edge = add_cone_edge(CameraToRobotTransformer(cones_msg->blue_cones[i].point), added_vertex, last_pose);
                }
                else
                {
                    // handle the graph modification while doing optimization
                    ConeToAdd pending_cone;
                    pending_cone.cone_node = cone_node;
                    pending_cone.data_header = cones_msg->header;
                    pending_cone.bf_coords = cones_msg->blue_cones[i].point;
                    pending_cone.map_coords = cone_map_frame.point;
                    pending_cone.ideal_coords = ideal_cone_map_frame;
                    pending_cone.color = BLUE_CONE;
                    pending_cone.unique_id = cone_unique_ID;
                    pending_cone.curr_car_pose = last_pose;
                    pending_cones.push_back(pending_cone);
                }
            }
            else // otherwise add only an edge between the current pose and the previously observed node
            {
                // add an edge between the current pose and associated cone
                if (!doing_optim)
                {
                    // compute the coordinates of the associateed vertex in base footprint --> apply inverse transformation
                    g2o::EdgeSE2PointXY *added_edge = add_cone_edge(CameraToRobotTransformer(cones_msg->blue_cones[i].point), assoc_found, last_pose);
                }
                else
                {

                    ObsEdgeToAdd pending_edge;
                    pending_edge.rel_obs = cone_map_frame.point;
                    pending_edge.cone_vertex = assoc_found;
                    pending_edge.pose_vertex = last_pose;

                    pending_obs_edges.push_back(pending_edge);
                }
            }
        }

        //  yellow cones
        for (int i = 0; i < cones_msg->yellow_cones.size(); i++)
        {
            // transform to map frame the real observation
            geometry_msgs::msg::PointStamped cone_map_frame = coneTransform(cones_msg->header, cones_msg->yellow_cones[i].point, ideal_t_robot_map);

            // transform to map frame the ideal observation (ground truth)
            geometry_msgs::msg::PointStamped ideal_cone_map_frame = coneTransform(ideal_cones.header, ideal_cones.yellow_cones[i].point, ideal_t_robot_map);

            // test association
            g2o::VertexPointXY *assoc_found = perform_data_assoc(ideal_cone_map_frame.point, cone_map_frame.point, curr_pose_node_id);
            // g2o::VertexPointXY *assoc_found = perform_data_assoc(cone_map_frame.point);

            if (assoc_found == nullptr) // if this is a new cone, add vertex and corresponding edges
            {
                // new cone observed, add the node
                cone_unique_ID++;
                g2o::VertexPointXY *cone_node = graph_handler_ptr->createLandmarkVertex(Eigen::Vector2d(cone_map_frame.point.x, cone_map_frame.point.y));

                if (!doing_optim)
                {
                    g2o::VertexPointXY *added_vertex = graph_handler_ptr->addLandmarkVertex(cone_node);
                    total_cones++;

                    // message for new vertex visualization
                    all_nodes_msg.push_back(nodeMessage(LANDMARK_TYPE, static_cast<int>(added_vertex->id()), cones_msg->header.stamp, cone_map_frame.point.x, cone_map_frame.point.y, YELLOW_CONE));

                    addConeData(added_vertex, cone_unique_ID, YELLOW_CONE, cones_msg->header.stamp, cone_map_frame);

                    // add an edge between the current pose and the observed cones (pose-landmark edge)
                    g2o::EdgeSE2PointXY *added_edge = add_cone_edge(CameraToRobotTransformer(cones_msg->yellow_cones[i].point), added_vertex, last_pose);
                }
                else
                {
                    ConeToAdd pending_cone;
                    pending_cone.cone_node = cone_node;
                    pending_cone.data_header = cones_msg->header;
                    pending_cone.bf_coords = cones_msg->yellow_cones[i].point;
                    pending_cone.map_coords = cone_map_frame.point;
                    pending_cone.ideal_coords = ideal_cone_map_frame;
                    pending_cone.color = YELLOW_CONE;
                    pending_cone.unique_id = cone_unique_ID;
                    pending_cone.curr_car_pose = last_pose;
                    pending_cones.push_back(pending_cone);
                }
            }
            else // otherwise add only an edge between the current pose and the previously observed node
            {
                // add an edge between the current pose and associated cone

                if (!doing_optim)
                {
                    // compute the coordinates of the associateed vertex in base footprint --> apply inverse transformation
                    g2o::EdgeSE2PointXY *added_edge = add_cone_edge(CameraToRobotTransformer(cones_msg->yellow_cones[i].point), assoc_found, last_pose);
                }
                else
                {
                    ObsEdgeToAdd pending_edge;
                    pending_edge.rel_obs = cone_map_frame.point;
                    pending_edge.cone_vertex = assoc_found;
                    pending_edge.pose_vertex = last_pose;
                    pending_obs_edges.push_back(pending_edge);
                }
            }
        }
        
        // publish pose graph msg for visualization
        graph_based_slam::msg::PoseGraph pose_graph_msg;
        pose_graph_msg.header.frame_id = "map";
        pose_graph_msg.header.stamp = rclcpp::Node::now();
        pose_graph_msg.graph_edges = all_edges_msg;
        pose_graph_msg.graph_nodes = all_nodes_msg;
        graph_pub->publish(pose_graph_msg);

    }
    
    /**
     * Adds an edge between a pose vertex and the observed cone.
     * @param cone_data The cone observation, relative to the robot (base_footprint frame).
     * @param cone_vertex The cone vertex already inserted in the graph.
     * @param pose_vertex The pose vertex in the graph, representing the current position.
     * @return A pointer to the added edge.
     */
    g2o::EdgeSE2PointXY *add_cone_edge(geometry_msgs::msg::Point cone_data, g2o::VertexPointXY *cone_vertex, g2o::VertexSE2 *pose_vertex)
    {
        Eigen::Vector2d rel_observation(cone_data.x, cone_data.y); // cone in base footprint

        Eigen::Matrix2d cov_mat;
        cov_mat.fill(0.0);
        cov_mat(0, 0) = pow(cone_noise[0], 2);
        cov_mat(1, 1) = pow(cone_noise[1], 2);
        Eigen::Matrix2d cone_info_mat = cov_mat.inverse();
        g2o::EdgeSE2PointXY *new_edge = graph_handler_ptr->createLandmarkEdge(pose_vertex, cone_vertex, rel_observation, cone_info_mat);
        g2o::EdgeSE2PointXY *added_edge = graph_handler_ptr->addLandmarkEdge(new_edge);

        // construct the message for the new edge
        all_edges_msg.push_back(edgeMessage(LANDMARK_TYPE, static_cast<int>(new_edge->vertices()[0]->id()), static_cast<int>(new_edge->vertices()[1]->id())));

        return added_edge;
    }

    float computeEuclideanConeError()
    {
        float total_error;
        int n_landmark_nodes = 0;

        // retrieve nodes of the graph
        auto graph_vertices = graph_handler_ptr->getNodes();
        for (int i = 0; i < graph_vertices.size(); i++)
        {
            auto tmp_vertex = graph_vertices[i];

            // consider only cones nodes
            NodeData *node_data = dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(tmp_vertex)->userData());
            if (node_data->getNodeType() == LANDMARK_TYPE)
            {
                n_landmark_nodes++;
                // compute euclidean distance between ideal and estimated positions
                g2o::VertexPointXY *tmp = dynamic_cast<g2o::VertexPointXY *>(tmp_vertex);
                ConeData *cone_data = dynamic_cast<ConeData *>(node_data);

                double x_ideal = cone_data->getIdealCoords().point.x;
                double y_ideal = cone_data->getIdealCoords().point.y;

                double x_estimated = tmp->estimate().x();
                double y_estimated = tmp->estimate().y();

                double euclid_dist = sqrt(pow(x_estimated - x_ideal, 2) + pow(y_estimated - y_ideal, 2));
                std::cout << "current euclidean error = " << euclid_dist << "\n";
                std::cout << "current values = " << x_ideal << "  " << y_ideal << "  "<< x_estimated<< "  " << y_estimated<< "\n";

                total_error += euclid_dist;
            }
        }
        std::cout << "num_cones_nodes = " << n_landmark_nodes << "\n";
        std::cout << "total error = " << total_error << "\n";
        // compute the mean error for a cone
        //return total_error / n_landmark_nodes;
        return 10;
    }

    /**
     * Process a pending cone node, meaning a node that has been created but not added to the graph yet.
     * Actually, add a new vertex into the graph and publish it for visualization.
     * @param cone_to_add Custom struct that holds all info about the pending cone node.
    */
    void process_pending_cone(ConeToAdd cone_to_add)
    {
        g2o::VertexPointXY *added_vertex = graph_handler_ptr->addLandmarkVertex(cone_to_add.cone_node);
        all_nodes_msg.push_back(nodeMessage(LANDMARK_TYPE, static_cast<int>(added_vertex->id()), cone_to_add.data_header.stamp, cone_to_add.map_coords.x, cone_to_add.map_coords.y, cone_to_add.color));
        addConeData(added_vertex, cone_to_add.unique_id, cone_to_add.color, cone_to_add.data_header.stamp, cone_to_add.ideal_coords);
        total_cones++;
        std::cout << "Processing pending cones: " << total_cones;
        // add an edge between the current pose and the observed cones (pose-landmark edge)
        g2o::EdgeSE2PointXY *added_edge = add_cone_edge(cone_to_add.bf_coords, added_vertex, cone_to_add.curr_car_pose);
    }

    /**
     * Process a pending pose node, meaning a node that has been created but not added to the graph yet.
     * Actually, add a new vertex into the graph and publish it for visualization.
     * @param pose_to_add Custom struct that holds all info about the pending car pose node.
    */
    void process_pending_pose(PoseToAdd pose_to_add)
    {
        g2o::VertexSE2 *curr_node = graph_handler_ptr->addPoseVertex(pose_to_add.pose_node);
        added_pose_vertices++;

        // message for new vertex visualization
        all_nodes_msg.push_back(nodeMessage(ROBOT_TYPE, static_cast<int>(curr_node->id()), pose_to_add.data_header.stamp, pose_to_add.real_pose.translation()[0], pose_to_add.real_pose.translation()[1], "", pose_to_add.rot_angle));

        // add pose data
        addPoseData(curr_node, pose_to_add.ideal_pose, pose_to_add.data_header.stamp);

        // add odometry edge between consecutive poses
        auto prev_mat = pose_to_add.prev_pose->estimate();
        auto curr_mat = curr_node->estimate();

        // rel_pose = inverse(prev_pose) * curr_pose
        Eigen::Isometry2d rel_pose = prev_mat.toIsometry().inverse() * curr_mat.toIsometry();

        Eigen::Matrix3d cov_mat;
        cov_mat.fill(0.0);
        cov_mat(0, 0) = pow(position_noise[0], 2);
        cov_mat(1, 1) = pow(position_noise[1], 2);
        cov_mat(2, 2) = pow(position_noise[2], 2);
        Eigen::Matrix3d info_mat = cov_mat.inverse();

        g2o::EdgeSE2 *new_edge = graph_handler_ptr->createPoseEdge(pose_to_add.prev_pose, curr_node, rel_pose, info_mat);
        graph_handler_ptr->addPoseEdge(new_edge);

        // construct the message for the new edge
        all_edges_msg.push_back(edgeMessage(ROBOT_TYPE, static_cast<int>(new_edge->vertices()[0]->id()), static_cast<int>(new_edge->vertices()[1]->id())));
    }


    /**
     * Simple data association to check if an observation corresponds to an already seen landmark.
     * It computes the euclidean distance between the given observation and the landmark nodes already in the graph,
     * and it decides according to a threshold.
     *
     * @param obs_to_test The given observation in map frame, to be tested for data association.
     * TODO:
     * @return The associated node, if a correspondence is found, null pointer otherwise (meaning new cone observed).
     */
    g2o::VertexPointXY *perform_data_assoc(geometry_msgs::msg::Point ideal_obs_to_test, geometry_msgs::msg::Point est_obs_to_test, int curr_pose_node_id)
    {

        // increase the counter of the performed association tests
        total_assoc_test++;

        // distringuish the two cases
        double min_dist_NN_ideal = 10000.0;
        double min_dist_NN_est = 10000.0;
        double min_MAHAL_NN = 10000.0;
        g2o::VertexPointXY *nn_cone_ideal;
        g2o::VertexPointXY *nn_cone_est;
        g2o::VertexPointXY *nn_cone_mahal;

        //  coords of the observation to test
        // use the ideal coordinates for data association, then use the real one for the edge
        double x_obs_ideal = ideal_obs_to_test.x;
        double y_obs_ideal = ideal_obs_to_test.y;
        

        double x_obs_est = est_obs_to_test.x;
        double y_obs_est = est_obs_to_test.y;

        auto graph_vertices = graph_handler_ptr->getNodes();

        auto graph = graph_handler_ptr->getGraph();
        
        
        for (int i = 0; i < graph_vertices.size(); i++)
        {
            auto tmp_vertex = graph_vertices[i];
            auto curr_land_node_id = tmp_vertex->id();

            // filter out only nodes of the correct type (cones/landmarks)
            NodeData *node_data = dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(tmp_vertex)->userData());
            if (node_data->getNodeType() == LANDMARK_TYPE)
            {
                // use ideal coords for association
                ConeData *cone_data = dynamic_cast<ConeData *>(node_data);

                double x_vertex_ideal = cone_data->getIdealCoords().point.x;
                double y_vertex_ideal = cone_data->getIdealCoords().point.y;

                double x_vertex_est = dynamic_cast<g2o::VertexPointXY *>(tmp_vertex)->estimate().x();
                double y_vertex_est = dynamic_cast<g2o::VertexPointXY *>(tmp_vertex)->estimate().y();

                // compute euclidean distance
                double euclid_dist_ideal = sqrt(pow(x_obs_ideal - x_vertex_ideal, 2) + pow(y_obs_ideal - y_vertex_ideal, 2));
                double euclid_dist_est = sqrt(pow(x_obs_est - x_vertex_est, 2) + pow(y_obs_est - y_vertex_est, 2));

                if (euclid_dist_ideal < min_dist_NN_ideal)
                {
                    min_dist_NN_ideal = euclid_dist_ideal;
                    nn_cone_ideal = dynamic_cast<g2o::VertexPointXY *>(tmp_vertex);
                }

                if (euclid_dist_est < min_dist_NN_est)
                {
                    min_dist_NN_est = euclid_dist_est;
                    nn_cone_est = dynamic_cast<g2o::VertexPointXY *>(tmp_vertex);
                }
            }
        }


        //association found
        if ((min_dist_NN_ideal < DIST_THRESHOLD))
        {
            // retrieve the ID of the cone associated both with ideal data and with estimated data
            ConeData *assoc_cone_data_ideal = dynamic_cast<ConeData *>(dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(nn_cone_ideal)->userData()));
            ConeData *assoc_cone_data_est = dynamic_cast<ConeData *>(dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(nn_cone_est)->userData()));

            // // TRUE POSITIVE (same IDs)
            // if (assoc_cone_data_est->getUniqueID() == assoc_cone_data_ideal->getUniqueID())
            // {
            //     true_pos++;
            // }
            // else // TRUE NEGATIVE (different IDs)
            // {
            //     false_pos++;
            // }
            return nn_cone_ideal;
            //return nn_cone_est;
        }
        else 
            return nullptr;

    }


    void timer_callback()
    {
        RCLCPP_INFO(this->get_logger(), "**************** Entering optimization callback! ******************** ");

        // float error_cones_before_optim = computeEuclideanConeError();
        // RCLCPP_INFO(this->get_logger(), "Cones error before optimization: %f", error_cones_before_optim);

        doing_optim = true;
        optim_timer = std::chrono::steady_clock::now();

        // used to find transformation between poses before and after optimization
        auto old_vertices = graph_handler_ptr->getNodes();
        std::vector<geometry_msgs::msg::Transform> position_transformations;

        // very high n. iterations, it stops when the error does not decrease anymore
        graph_handler_ptr->globalOptimization(50);

        optim_c++;
        std::cout << "finished graph optimization \n";

        // float error_cones_after_optim = computeEuclideanConeError();
        // RCLCPP_INFO(this->get_logger(), "Cones error after optimization: %f", error_cones_after_optim);

        // now that optim is finished, process the pending elements
        // pending cone nodes
        for (int i = 0; i < pending_cones.size(); i++)
        {
            process_pending_cone(pending_cones[i]);
        }
        // pending observation edges
        for (int i = 0; i < pending_obs_edges.size(); i++)
        {
            g2o::EdgeSE2PointXY *added_edge = add_cone_edge(pending_obs_edges[i].rel_obs, pending_obs_edges[i].cone_vertex, pending_obs_edges[i].pose_vertex);
        }

        for (int i = 0; i < pending_poses.size(); i++)
        {
            process_pending_pose(pending_poses[i]);
        }

        pending_cones.clear();
        pending_obs_edges.clear();
        pending_poses.clear();
        // pending_odom_edges.clear();
        auto optimized_vertices = graph_handler_ptr->getNodes();

        all_nodes_msg.clear();
        all_edges_msg.clear();

        // publish pose graph msg for visualization
        for (int i = 0; i < optimized_vertices.size(); i++)
        {
            auto vert = optimized_vertices[i];
            auto vert_old = old_vertices[i];

            NodeData *vert_data = dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(vert)->userData());
            

            double x_vert, y_vert;
            double x_vert_old, y_vert_old; 

            if (vert_data->getNodeType() == LANDMARK_TYPE)
            {
                x_vert = dynamic_cast<g2o::VertexPointXY *>(vert)->estimate().x();
                y_vert = dynamic_cast<g2o::VertexPointXY *>(vert)->estimate().y();

                x_vert_old = dynamic_cast<g2o::VertexPointXY *>(vert_old)->estimate().x();
                y_vert_old = dynamic_cast<g2o::VertexPointXY *>(vert_old)->estimate().y();

                geometry_msgs::msg::Transform transformation;
                transformation.translation.x = x_vert - x_vert_old;
                transformation.translation.y = y_vert - y_vert_old;

                position_transformations.push_back(transformation);
                //std::cout << "transformation cone: x" << transformation.translation.x << " y: " << transformation.translation.y << std::endl;

                all_nodes_msg.push_back(nodeMessage(LANDMARK_TYPE, static_cast<int>(vert->id()), vert_data->getTimestamp(), x_vert, y_vert, dynamic_cast<ConeData *>(vert_data)->getColor()));

                num_cones_mapped++;
            }
            else
            {
                x_vert = dynamic_cast<g2o::VertexSE2 *>(vert)->estimate().translation()[0];
                y_vert = dynamic_cast<g2o::VertexSE2 *>(vert)->estimate().translation()[1];

                x_vert_old = dynamic_cast<g2o::VertexSE2 *>(vert_old)->estimate().translation()[0];
                y_vert_old = dynamic_cast<g2o::VertexSE2 *>(vert_old)->estimate().translation()[1];

                geometry_msgs::msg::Transform transformation;
                transformation.translation.x = x_vert - x_vert_old;
                transformation.translation.y = y_vert - y_vert_old;

                position_transformations.push_back(transformation);
                std::cout << "transformation localization: x" << transformation.translation.x << " y: " << transformation.translation.y << std::endl;

                all_nodes_msg.push_back(nodeMessage(ROBOT_TYPE, static_cast<int>(vert->id()), vert_data->getTimestamp(), x_vert, y_vert));
            }
        }

        auto optimized_edges = graph_handler_ptr->getEdges();
        for (auto &edge : optimized_edges)
        {
            g2o::OptimizableGraph::Edge *tmp_edge = dynamic_cast<g2o::OptimizableGraph::Edge *>(edge);

            // TODO: handle edge type
            std::string type;
            try
            {
                if (dynamic_cast<g2o::EdgeSE2 *>(tmp_edge) != nullptr)
                type = ROBOT_TYPE;
            }
            catch (const std::bad_cast &exception)
            {
                type = LANDMARK_TYPE;
            }

            all_edges_msg.push_back(edgeMessage(type, static_cast<int>(tmp_edge->vertices()[0]->id()), static_cast<int>(tmp_edge->vertices()[1]->id())));
        }

        graph_based_slam::msg::PoseGraph graph_msg;
        graph_msg.header.frame_id = "map";
        graph_msg.header.stamp = rclcpp::Node::now();
        graph_msg.graph_edges = all_edges_msg;
        graph_msg.graph_nodes = all_nodes_msg;
        graph_pub->publish(graph_msg);

        doing_optim = false;
    }

    /**
     * @brief Transform cone positions from camera frame to map frame
     * @param cones_msg cone positions in camera frame
     *
     * @return cone positions in map frame
     */
    geometry_msgs::msg::PointStamped coneTransform(std_msgs::msg::Header msg_header, geometry_msgs::msg::Point cone_coords, geometry_msgs::msg::TransformStamped transformation)
    {
        cv::Mat pointInCameraFrameMat = (cv::Mat_<double>(3, 1) << cone_coords.x, cone_coords.y, cone_coords.z);
        geometry_msgs::msg::PointStamped pointInCarFrame = CameraToRobotTransformer(pointInCameraFrameMat);
        geometry_msgs::msg::PointStamped cone_map;
        pointInCarFrame.header.frame_id = msg_header.frame_id;
        pointInCarFrame.header.stamp = msg_header.stamp;
        tf2::doTransform(pointInCarFrame, cone_map, transformation);

        return cone_map;
    }

    /**
     * Transform coordinates of a cone into car frame.
     * @param pointInCameraFrameMat cone position in the camera frame.
     * @return Coordinates of the considered cone in car frame.
     */
    geometry_msgs::msg::Point CameraToRobotTransformer(geometry_msgs::msg::Point cone_coords)
    {
        cv::Mat pointInCameraFrameMat = (cv::Mat_<double>(3, 1) << cone_coords.x, cone_coords.y, cone_coords.z);
        geometry_msgs::msg::Point pointInCarFrame = CameraToRobotTransformer(pointInCameraFrameMat).point;

        return pointInCarFrame;
    }


    /**
     * Transform coordinates of a cone into car frame.
     * @param pointInCameraFrameMat cone position in the camera frame.
     * @return Coordinates of the considered cone in car frame.
     */
    geometry_msgs::msg::PointStamped CameraToRobotTransformer(cv::Mat pointInCameraFrameMat)
    {
        //transformation from camera frame to robot frame
        cv::Mat R = (cv::Mat_<double>(3, 3) <<  0.9318794, -0.3153596, -0.1793018,
        0.3596048,  0.8681628,  0.3420202,
        0.0478038, -0.3831993,  0.9224278 );  // Rotation matrix //experimental number
        cv::Mat T = (cv::Mat_<double>(3, 1) << -6.3437, 2.560, 0.9245);  // Translation vector //changed to positive //best values so far -5.7437, 2.560, -0.9245
        cv::Mat pointInCarFrameMat = (pointInCameraFrameMat)+T;
        geometry_msgs::msg::PointStamped pointInCarFrame;
        pointInCarFrame.point.x = 0;
        pointInCarFrame.point.y = static_cast<float>(pointInCarFrameMat.at<double>(1, 0));
        pointInCarFrame.point.z = static_cast<float>(pointInCarFrameMat.at<double>(2, 0));

        return pointInCarFrame;
    }

    /**
     * Create a custom edge message to be published to visualize the graph.
     * @param type The category of the edge (pose-pose or pose-cone).
     * @param id_from The id of the vertex from which the edge originates.
     * @param id_to The id of the vertex to which the edge ends.
     */
    graph_based_slam::msg::GraphEdge edgeMessage(std::string type, int id_from, int id_to)
    {
        graph_based_slam::msg::GraphEdge new_edge_msg;
        new_edge_msg.type = type;
        new_edge_msg.vertex_i = id_from;
        new_edge_msg.vertex_j = id_to;
        return new_edge_msg;
    }

    /**
     * Transform coordinates of a cone into map frame (global reference frame).
     * @param msg_header Header of the received message, containing reference frame and timestamp of data.
     * @param cone_coords Coordinates of a cone in base footprint frame (robot frame).
     * @param robot_map_tf Rigid body transformation between the data reference frame and the target one.
     * @return Coordinates of the considered cone in map reference frame.
     */
    geometry_msgs::msg::PointStamped RobotToWorldTransformer(std_msgs::msg::Header msg_header, geometry_msgs::msg::Point cone_coords, geometry_msgs::msg::TransformStamped transformation)
    {
        geometry_msgs::msg::PointStamped cone_bf, cone_map;
        cone_bf.header.frame_id = msg_header.frame_id;
        cone_bf.header.stamp = msg_header.stamp;
        cone_bf.point = cone_coords;
        tf2::doTransform(cone_bf, cone_map, transformation);

        return cone_map;
    }

    /**
     * Add custom user data to a landmark node.
     * @param node The node to which data has to be added.
     * @param id The unique id to assign.
     * @param color The color of the cone.
     * @param timestamp The timestamp when data has been acquired.
     * @param ideal_coords The ground truth (ideal, not noisy) coordinates of this cone.
     */
    void addConeData(g2o::VertexPointXY *node, int id, std::string color, rclcpp::Time timestamp, geometry_msgs::msg::PointStamped ideal_coords)
    {
        ConeData *cone_data = new ConeData;
        cone_data->setNodeType(LANDMARK_TYPE);
        cone_data->setUniqueID(id);
        cone_data->setColor(color);
        cone_data->setTimestamp(timestamp);
        cone_data->setIdealCoords(ideal_coords);
        node->setUserData(cone_data);
    }

     /**
     * Add custom user data to a pose node.
     * @param node The node to which data has to be added.
     * @param id (Optional) The unique id to assign. Mainly used as sentinel value for the first node.
     * //TODO:finish docs
     */
    void addPoseData(g2o::VertexSE2 *node, geometry_msgs::msg::Transform ideal_coords, rclcpp::Time timestamp, int id = NAN)
    {
        PoseData *pose_data = new PoseData;
        pose_data->setNodeType(ROBOT_TYPE);
        if (!(id == NAN))
        {
        pose_data->setUniqueID(id);
        }
        pose_data->setIdealCoords(ideal_coords);
        pose_data->setTimestamp(timestamp);
        node->setUserData(pose_data);
    }


    /**
     * Create a custom node message to be pubilshed for graph visualization.
     * @param type The type of node (robot pose or landmark).
     * @param id The id of the graph node.
     * @param stamp The timestamp of the current data.
     * @param x The x coordinate of the node, in global reference frame (map).
     * @param y The y coordinate of the node, in global reference frame (map).
     * @param color (Optional) The color of the represented cone. Only for nodes representing cone position.
     * @param theta (Optional) The orientation of the node. Only for nodes representing robot poses. //TODO: check if this is necessary
     */
    graph_based_slam::msg::GraphNode nodeMessage(std::string type, int id, rclcpp::Time stamp, float x, float y, std::string color = "", float theta = NAN)
    {
        graph_based_slam::msg::GraphNode new_node_msg;
        new_node_msg.header.frame_id = "map";
        new_node_msg.header.stamp = stamp;
        new_node_msg.x = x;
        new_node_msg.y = y;
        new_node_msg.type = type;
        if (!(color == ""))
        {
        new_node_msg.color = color;
        }
        if (!(theta == NAN))
        {
        new_node_msg.theta = theta;
        }
        new_node_msg.id = id;
        return new_node_msg;
    }

};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  rclcpp::executors::MultiThreadedExecutor executor;
  slam_node_ptr = std::make_shared<GraphSLAMReal>(options);
  executor.add_node(slam_node_ptr);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}