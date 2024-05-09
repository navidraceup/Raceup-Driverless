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

/**
 * This class implements a graph-based SLAM module, with incremental optimization.
 * Once a certain number of new nodes have been added to the graph, the corresponding subgraph is locally optimized.
 * Then, once a every a certain time, the graph is optimized globally.
 */
class incrementalGraphSLAM : public rclcpp::Node
{

public:
  // variables

  int cone_unique_ID = 0;
  int total_cones = 0;
  int added_pose_vertices = 0;

  bool init_cones = false;
  bool moving_car = false; // flag useful to avoid collecting data when car is not moving

  // thresholds for data association
  double DIST_THRESHOLD = 2.5; // 0.3 for 0.1 stddev
  // time for data quantization
  double TIME_THRESHOLD = 1.0;
  // threshold for optimization
  double OPTIM_THRESHOLD = 10.0;

  int optim_c = 0;
  bool doing_optim = false;

  // for incoming data quantization
  int msg_count = 0;

  int cnt = 0;

  // variables to handle data quantization over time
  std::chrono::steady_clock::time_point cones_timer, odom_timer, wheel_timer, optim_timer;

  g2o::VertexSE2 *first_fixed_pose;
  geometry_msgs::msg::TransformStamped last_odom_transf; // latest published odom transform

  Eigen::Vector2d position_noise; // odometry noise, along x and y
  Eigen::Vector2d cone_noise;     // cone position noise, along x and y

  // useful for graph visualization
  std::vector<graph_based_slam::msg::GraphNode> all_nodes_msg;
  std::vector<graph_based_slam::msg::GraphEdge> all_edges_msg;

  // Variables to handle the modification of the graph while doing optimization
  // Nodes and edges created while optimizing the graph, will be added later, when finished
  std::vector<ConeToAdd> pending_cones;        // hold info about pending cone nodes
  std::vector<ObsEdgeToAdd> pending_obs_edges; // hold info about pending observation edges
  std::vector<PoseToAdd> pending_poses;        // hold info about pending pose nodes

  // Gaussian noise params
  double noise_mean = 0.0;
  double noise_stddev = 0.1; // with 0.3, already 50-60 cm of noise on data in worst cases
  double orientation_mean = 0.0;
  double orientation_stddev = 0.00;

  // Ackermann motion model integration
  racecar_simulator::CarParams car_params;

  // constructor
  explicit incrementalGraphSLAM(const rclcpp::NodeOptions &options) : Node("incrementalGraphSLAM_node", options)
  {
    // param to use sim_time
    this->set_parameter(rclcpp::Parameter("use_sim_time", true));

    graph_handler_ptr.reset(new GraphHandling());

    // initialize timers for msg quantization
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    odom_timer = now;
    wheel_timer = now;
    cones_timer = now;
    optim_timer = now;

    // declare and acquire `target_frame` parameter
    target_frame = this->declare_parameter<std::string>("target_frame", "base_footprint");

    // noise params
    // TODO: tune these values
    position_noise[0] = 0.2;
    position_noise[1] = 0.1;
    position_noise[2] = 0.00000001; // heading noise: this should be zero, but it creates problems with the inverse
    cone_noise[0] = 0.1;
    cone_noise[1] = 0.1;

    //  initialize tf2 components
    tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    tf_buffer = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    data_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    timer_cb_group = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    timer_optim = this->create_wall_timer(180s, std::bind(&incrementalGraphSLAM::timer_callback, this), timer_cb_group); // TODO: handle callback group

    // get data of ground truth wheel speed
    gt_wheel_sub = this->create_subscription<eufs_msgs::msg::WheelSpeedsStamped>(
        "/ground_truth/wheel_speeds", 10, std::bind(&incrementalGraphSLAM::wheel_topic_callback, this, _1));

    // initialize publisher for the pose graph
    graph_pub = this->create_publisher<graph_based_slam::msg::PoseGraph>("pose_graph", 1);

    kin_cmd_sub = this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>("/cmd", 10, std::bind(&incrementalGraphSLAM::kin_callback, this, _1));

    sync_init();

    // real track topic subscription (for metrics)
    gt_track_sub = this->create_subscription<visualization_msgs::msg::MarkerArray>(
        "/ground_truth/track/viz", 10, std::bind(&incrementalGraphSLAM::gt_track_callback, this, _1));

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

    // add pose data
    addPoseData(initial_pose, zero_transform, zero_stamp, FIRST_NODE_ID);

    initial_pose->setFixed(true);
    first_fixed_pose = initial_pose;

    // initialize kinematic parameters
    car_params.wheelbase = 1.580; // taken from URDF
  }

private:
  std::unique_ptr<GraphHandling> graph_handler_ptr;

  rclcpp::Subscription<eufs_msgs::msg::WheelSpeedsStamped>::SharedPtr gt_wheel_sub;
  message_filters::Subscriber<eufs_msgs::msg::ConeArrayWithCovariance> cones_sub;
  // message_filters::Subscriber<eufs_msgs::msg::CarState> odom_sub;
  message_filters::Subscriber<nav_msgs::msg::Odometry> odom_sub;
  rclcpp::Subscription<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr kin_cmd_sub;

  // subscribe to ground truth track
  rclcpp::Subscription<visualization_msgs::msg::MarkerArray>::SharedPtr gt_track_sub;

  rclcpp::Publisher<graph_based_slam::msg::PoseGraph>::SharedPtr graph_pub;

  rclcpp::TimerBase::SharedPtr timer_optim;
  rclcpp::CallbackGroup::SharedPtr timer_cb_group, data_cb_group;

  typedef message_filters::sync_policies::ApproximateTime<eufs_msgs::msg::ConeArrayWithCovariance, nav_msgs::msg::Odometry> approx_policy;
  std::shared_ptr<message_filters::Synchronizer<approx_policy>> syncApprox;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener{nullptr};
  std::unique_ptr<tf2_ros::Buffer> tf_buffer;
  std::string target_frame;

  // hold kin model commands
  // newer cmds are in the front part, while older ones are in the back
  std::deque<ackermann_msgs::msg::AckermannDriveStamped> last_kin_cmds;
  std::mutex kin_cmd_mtx;
  ackermann_msgs::msg::AckermannDriveStamped first_command_ever;
  bool first_cmd = false;

  // cone metrics
  int num_cones_mapped, num_cones_gt;              // compare the number of detected cones with the real
  double true_pos, true_neg, false_pos, false_neg; // metrics about cone association
  double true_pos_mh, true_neg_mh, false_pos_mh, false_neg_mh;
  int total_assoc_test; // num of times data association has been checked
  bool gt_cones_done = false;

  /**
   * @brief Initialize synchronization-related stuff (subscriptions, policy, etc..)
   */
  void sync_init()
  {
    rmw_qos_profile_t custom_qos_profile = rmw_qos_profile_default;
    custom_qos_profile.depth = 1;
    custom_qos_profile.reliability = rmw_qos_reliability_policy_t::RMW_QOS_POLICY_RELIABILITY_RELIABLE;
    custom_qos_profile.history = rmw_qos_history_policy_t::RMW_QOS_POLICY_HISTORY_KEEP_LAST;
    custom_qos_profile.durability = rmw_qos_durability_policy_t::RMW_QOS_POLICY_DURABILITY_VOLATILE;

    // cones_sub.subscribe(this, "/fusion/cones", custom_qos_profile);
    cones_sub.subscribe(this, "/ground_truth/cones", custom_qos_profile);
    // odom_sub.subscribe(this, "/odometry_integration/car_state", custom_qos_profile);
    odom_sub.subscribe(this, "/ground_truth/odom", custom_qos_profile);

    syncApprox = std::make_shared<message_filters::Synchronizer<approx_policy>>(approx_policy(1), cones_sub, odom_sub);
    syncApprox->registerCallback(&incrementalGraphSLAM::sync_callback, this);
  }

  /**
   * @brief Callback containing the ground truth track visualization markers, used to set the ground-truth number of cones.
   */
  void gt_track_callback(const visualization_msgs::msg::MarkerArray::SharedPtr track_msg)
  {
    if (!gt_cones_done)
    {
      num_cones_gt = track_msg->markers.size();
      gt_cones_done = true;
    }
  }

  /**
   * @brief Called when a WheelSpeedsStamped msg is received.
   * @param msg ROS message that provides info about steering (in rad) and speeds of each wheel (in RPM).
   */
  void kin_callback(const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg)
  {
    // RCLCPP_INFO(this->get_logger(), "CALLBACK WITH KINEMATIC CMD");
    kin_cmd_mtx.lock();
    if (first_cmd == false)
    {
      std::cout << "first cmd ever set\n";
      first_command_ever = *msg;
      first_cmd = true;
    }
    // keep at maximum 10 msgs queued
    if (last_kin_cmds.size() == 10)
    {
      // remove oldest element
      last_kin_cmds.pop_back();
    }
    last_kin_cmds.push_front(*msg);
    kin_cmd_mtx.unlock();
  }

  /**
   * Callback to synchronize the reception of cones and odometry messages.
   *
   * @param cones_msg ROS message representing an array 2D cone locations (z = 0), with covariances.
   * @param odom_msg ROS message representing an estimate of position and velocity in free space.
   */
  void sync_callback(eufs_msgs::msg::ConeArrayWithCovariance::SharedPtr cones_msg, nav_msgs::msg::Odometry::SharedPtr odom_msg)
  {
    msg_count++;

    // save ground truth and ideal odom
    nav_msgs::msg::Odometry ideal_odom;
    ideal_odom.header = odom_msg->header;
    ideal_odom.child_frame_id = odom_msg->child_frame_id;
    ideal_odom.pose = odom_msg->pose;
    ideal_odom.twist = odom_msg->twist;

    // save ground truth and ideal cones
    eufs_msgs::msg::ConeArrayWithCovariance ideal_cones;
    ideal_cones.header = cones_msg->header;
    ideal_cones.big_orange_cones = cones_msg->big_orange_cones;
    ideal_cones.orange_cones = cones_msg->orange_cones;
    ideal_cones.blue_cones = cones_msg->blue_cones;
    ideal_cones.yellow_cones = cones_msg->yellow_cones;
    ideal_cones.unknown_color_cones = cones_msg->unknown_color_cones;

    // add noise to cones
    cones_msg->big_orange_cones = addNoiseToCones(cones_msg->big_orange_cones);
    cones_msg->orange_cones = addNoiseToCones(cones_msg->orange_cones);
    cones_msg->blue_cones = addNoiseToCones(cones_msg->blue_cones);
    cones_msg->yellow_cones = addNoiseToCones(cones_msg->yellow_cones);

    // for (int i = 0; i < static_cast<int>(ideal_cones.blue_cones.size()); i++)
    // {
    //   std::cout << "ideal_blue[" << ideal_cones.blue_cones[i].point.x << "," << ideal_cones.blue_cones[i].point.y << "]\n";
    // }
    // for (int i = 0; i < static_cast<int>(cones_msg->blue_cones.size()); i++)
    // {
    //   std::cout << "noisy_blue[" << cones_msg->blue_cones[i].point.x << "," << cones_msg->blue_cones[i].point.y << "]\n";
    // }

    // RCLCPP_INFO(this->get_logger(), "++++++++++++++++++++++++++++++++++++++++++++++++++++");
    // RCLCPP_INFO(this->get_logger(), "ENTERED THE SYNC CALLBACK. GETTING DATA FROM GT CONES AND NOISY ODOMETRY");
    // RCLCPP_INFO(this->get_logger(), "Odometry msg time %ds and %dns", odom_msg->header.stamp.sec, odom_msg->header.stamp.nanosec);
    // RCLCPP_INFO(this->get_logger(), "Cones    msg time %ds and %dns", cones_msg->header.stamp.sec, cones_msg->header.stamp.nanosec);
    // RCLCPP_INFO(this->get_logger(), "++++++++++++++++++++++++++++++++++++++++++++++++++++");

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

    // prepare the real odometry transformation
    geometry_msgs::msg::TransformStamped t_odom;
    t_odom.header.stamp = now_time;
    t_odom.header.frame_id = "map";
    t_odom.child_frame_id = "base_footprint";

    t_odom.transform.translation.x = odom_msg->pose.pose.position.x;
    t_odom.transform.translation.y = odom_msg->pose.pose.position.y;
    t_odom.transform.translation.z = odom_msg->pose.pose.position.z;

    t_odom.transform.rotation.x = odom_msg->pose.pose.orientation.x;
    t_odom.transform.rotation.y = odom_msg->pose.pose.orientation.y;
    t_odom.transform.rotation.z = odom_msg->pose.pose.orientation.z;
    t_odom.transform.rotation.w = odom_msg->pose.pose.orientation.w;

    last_odom_transf = t_odom;

    // add noise to odom
    t_odom.transform = addNoiseToOdom(t_odom.transform);
    // std::cout << "ideal odom:\n";
    // std::cout << "x=" << ideal_odom.pose.pose.position.x << "\n";
    // std::cout << "y=" << ideal_odom.pose.pose.position.y << "\n";
    // std::cout << "noisy odom:\n";
    // std::cout << "x=" << t_odom.transform.translation.x << "\n";
    // std::cout << "y=" << t_odom.transform.translation.y << "\n";

    tf_broadcaster->sendTransform(t_odom);
    g2o::VertexSE2 *prev_pose = graph_handler_ptr->getLastPoseNode();

    if (msg_count == 8)
    {
      msg_count = 0;
      // ------------------------------------ ODOMETRY DATA PROCESSING ------------------------------------

      // RCLCPP_INFO(this->get_logger(), "----------------------------------------------------------------------------");
      // RCLCPP_INFO(this->get_logger(), "RECEIVED DATA FROM ODOMETRY, in frame %s", odom_msg->header.frame_id.c_str());
      // RCLCPP_INFO(this->get_logger(), "----------------------------------------------------------------------------");

      // odom_timer = odom_timer + std::chrono::duration_cast<std::chrono::microseconds>(now - odom_timer);
      odom_timer = now;

      // if the car is moving, add nodes about its position in the graph (odom data is already in map frame, no need to transform)
      geometry_msgs::msg::TransformStamped t_r_m_odom;
      if (moving_car)
      {
        // g2o::VertexSE2 *prev_pose = graph_handler_ptr->getLastPoseNode(); // retrieve previous node to create the edge after

        // const auto &car_pose = t_odom.transform;
        //  const auto &car_pose = odom_msg->pose.pose;

        // RETRIEVE THE KINEMATIC STATE OF THE VEHICLE

        // make a copy of the save commands for two reasons:
        //    1) don't lock the mutex for too long
        //    2) avoid errors in the original deque (don't modify it, read only)
        std::deque<ackermann_msgs::msg::AckermannDriveStamped> current_cmds;
        kin_cmd_mtx.lock();
        current_cmds = last_kin_cmds;
        kin_cmd_mtx.unlock();

        // retrieve only the commands suitable for integration
        std::stack<std::pair<ackermann_msgs::msg::AckermannDriveStamped, double>> cmds_to_integrate;

        // TODO: check if it is better to use an iterator on elements instead of emptying the deque
        // retrieve timestamp of last added pose (to compute the dt)
        // here sec and nanosec are the same value in two different units --> DON'T SUM!
        auto prev_pose_time = (dynamic_cast<NodeData *>(prev_pose->userData())->getTimestamp()).seconds(); // <------------------- time when integration starts (prev_pose inserted)

        // sync the timestamp of the first node
        // otherwise the first pose has time = 0, while the first cmd has time n (it starts after the simulation, when increasing velocity or steering)
        if (prev_pose_time == 0)
        {
          // fill the gap wrt the oldest cmd
          // std::cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
          // here it is correct to sum the sec and nanosec
          // prev_pose_time += first_command_ever.header.stamp.sec + (first_command_ever.header.stamp.nanosec * (1e-9)); // TODO: check this
          // std::cout << "first_cmd time is sec " << first_command_ever.header.stamp.sec << ", nanosec " << first_command_ever.header.stamp.nanosec * (1e-9) << "\n";
        }

        bool first_cmd_before_odom = false;
        // std::cout << "\n\n\n\n**************************************\n";
        // std::cout << "prev pose node time is " << prev_pose_time << "\n";
        //  std::cout << "seconds = " << (dynamic_cast<NodeData *>(prev_pose->userData())->getTimestamp()).seconds() << ", nanoseconds = " << (dynamic_cast<NodeData *>(prev_pose->userData())->getTimestamp()).nanoseconds() * (1e-9) << "\n";

        double odom_msg_time = odom_msg->header.stamp.sec + (odom_msg->header.stamp.nanosec * (1e-9));

        // TODO: maybe initialize start_time = prev_pose_time, end_time = odom_msg->header.stamp.sec (to handle the case in which I only go to the "else if")
        double start_time, end_time;
        start_time = prev_pose_time;
        end_time = odom_msg_time;

        while (!current_cmds.empty())
        {
          // std::cout << "____________________________________first part _______________________________\n";
          ackermann_msgs::msg::AckermannDriveStamped tmp_cmd = current_cmds.front();
          current_cmds.pop_front();
          // save cmds with timestamp between previous node and current time
          double tmp_cmd_time = tmp_cmd.header.stamp.sec + tmp_cmd.header.stamp.nanosec * (1e-9);
          // std::cout << "current sub-cmd time is " << tmp_cmd_time << "\n";
          if (tmp_cmd_time < odom_msg_time && tmp_cmd_time > prev_pose_time)
          {
            // end_time = odom_msg_time;
            start_time = tmp_cmd_time;
            // std::cout << "current sub-cmd start time is " << start_time << "\n";
            // std::cout << "current sub-cmd end time is " << end_time << "\n";
            double tmp_dt = abs(start_time - end_time); // TODO: check the sign here
            // std::cout << "current sub-cmd dt is " << tmp_dt << "\n";
            cmds_to_integrate.push(std::pair<ackermann_msgs::msg::AckermannDriveStamped, double>(tmp_cmd, tmp_dt));
            end_time = start_time; // go the the previous time span
          }
          // save the first command before timestamp of the previous node
          else if ((tmp_cmd_time < prev_pose_time) && (!first_cmd_before_odom))
          {
            first_cmd_before_odom = true;
            start_time = prev_pose_time;
            // std::cout << "first cmd before prev_node time\n";
            // std::cout << "current sub-cmd start time is " << start_time << "\n";
            // std::cout << "current sub-cmd end time is " << end_time << "\n";
            //  end_time should be already correctly set from the previous iterations TODO: check
            double tmp_dt = abs(start_time - end_time); // TODO: check the sign here
            // std::cout << "current sub-cmd dt is " << tmp_dt << "\n";
            cmds_to_integrate.push(std::pair<ackermann_msgs::msg::AckermannDriveStamped, double>(tmp_cmd, tmp_dt));
            // stop processing residual cmds, if any
            // empty the saved commands
            // current_cmds.clear();
          }
          if (first_cmd_before_odom)
            break;
        }
        // std::cout << "now time is " << odom_msg_time << "\n";

        // empty the saved commands
        current_cmds.clear();

        // now, integrate the extracted cmds
        // double sub_prev_time = prev_pose_time;

        // TODO: remove, just for debugging
        double total_dt = 0;

        // new state is updated iteratively by integrating all the cmds
        // in the end, it should contain the result of the total integration
        racecar_simulator::CarState new_state;
        bool first_sub_int = true;
        while (!cmds_to_integrate.empty())
        {
          // std::cout << "_______________________________ second part _______________________________\n";
          //  sub-integration of each single piece of motion
          auto curr_cmd = cmds_to_integrate.top();
          cmds_to_integrate.pop(); // remove each element once it is processed
          double steer_ang_vel = curr_cmd.first.drive.steering_angle_velocity;
          double accel = curr_cmd.first.drive.acceleration;
          double dt = curr_cmd.second;
          total_dt += dt;
          // std::cout << "current cmd timestamp is " << curr_cmd.first.header.stamp.sec + curr_cmd.first.header.stamp.nanosec * (1e-9) << "\n";
          // std::cout << "sub-dt in seconds is " << dt << "\n";
          racecar_simulator::CarState curr_state;
          curr_state.steer_angle = curr_cmd.first.drive.steering_angle;
          curr_state.velocity = curr_cmd.first.drive.speed;

          if (first_sub_int)
          {
            // if integrating the first cmd, then the previous pose is actually the previous pose node of the graph
            curr_state.x = prev_pose->estimate().translation()[0];
            curr_state.y = prev_pose->estimate().translation()[1];
            curr_state.theta = prev_pose->estimate().rotation().angle();
            first_sub_int = false;
          }
          else
          {
            // otherwise, the previous pose is the result of the previous sub-integration
            curr_state.x = new_state.x;
            curr_state.y = new_state.y;
            curr_state.theta = new_state.theta;
          }

          // std::cout << "old x = " << curr_state.x << "\n";
          // std::cout << "old y = " << curr_state.y << "\n";
          new_state = racecar_simulator::KSKinematics::update(curr_state, accel, steer_ang_vel, car_params, dt);
          // std::cout << "new x = " << new_state.x << "\n";
          // std::cout << "new y = " << new_state.y << "\n";
        }

        // check difference between the two computed poses
        std::cout << "Pose computed by ackermann is [" << new_state.x << "," << new_state.y << "]\n";
        std::cout << "Pose computed by odom is [" << ideal_t_robot_map.transform.translation.x << "," << ideal_t_robot_map.transform.translation.y << "]\n";

        // compute the transformation between base footprint and map, according to integrated odometry

        t_r_m_odom.header.stamp = now_time;
        t_r_m_odom.header.frame_id = "map";
        t_r_m_odom.child_frame_id = "base_footprint";

        t_r_m_odom.transform.translation.x = new_state.x;
        t_r_m_odom.transform.translation.y = new_state.y;
        t_r_m_odom.transform.translation.z = 0;

        double new_orientation = new_state.theta; // NB: in rad
        // std::cout << "orientation angle is " << new_orientation << "\n\n";

        // construct the quaternion from the 2D angle
        Eigen::Quaterniond new_rot(Eigen::AngleAxisd(new_orientation, Eigen::Vector3d::UnitZ().normalized()));
        t_r_m_odom.transform.rotation.x = new_rot.x();
        t_r_m_odom.transform.rotation.y = new_rot.y();
        t_r_m_odom.transform.rotation.z = new_rot.z();
        t_r_m_odom.transform.rotation.w = new_rot.w();

        last_odom_transf = t_r_m_odom;

        tf_broadcaster->sendTransform(t_r_m_odom);

        // TODO:TODO:TODO:TODO:TODO:TODO:TODO:

        // const auto &car_pose = t_odom.transform;
        const auto &car_pose = t_r_m_odom.transform;

        // real (noisy) rotation angle
        Eigen::Quaterniond quat;
        quat.w() = car_pose.rotation.w;
        quat.x() = car_pose.rotation.x;
        quat.y() = car_pose.rotation.y;
        quat.z() = car_pose.rotation.z;
        double rotation_angle = quat.toRotationMatrix().eulerAngles(0, 1, 2)[2]; // yaw, around z

        // ideal rotation angle
        Eigen::Quaterniond ideal_quat;
        ideal_quat.w() = ideal_t_robot_map.transform.rotation.w;
        ideal_quat.x() = ideal_t_robot_map.transform.rotation.x;
        ideal_quat.y() = ideal_t_robot_map.transform.rotation.y;
        ideal_quat.z() = ideal_t_robot_map.transform.rotation.z;
        double ideal_rotation_angle = ideal_quat.toRotationMatrix().eulerAngles(0, 1, 2)[2]; // yaw, around z

        // real (noisy) pose
        // 2D isometry: 2*2 matrix for rotation (linear), 2D vector for translation (translation)
        Eigen::Isometry2d pose = Eigen::Isometry2d::Identity();
        // pose.translation() = Eigen::Vector2d(car_pose.translation.x, car_pose.translation.y);
        // pose.linear() = Eigen::Rotation2Dd(rotation_angle).toRotationMatrix();
        //  TODO:TODO:TODO:TODO:TODO:TODO:TODO:
        pose.translation() = Eigen::Vector2d(new_state.x, new_state.y);
        pose.linear() = Eigen::Rotation2Dd(new_state.theta).toRotationMatrix();
        // TODO:TODO:TODO:TODO:TODO:TODO:TODO:

        // ideal pose
        Eigen::Isometry2d ideal_pose = Eigen::Isometry2d::Identity();
        ideal_pose.translation() = Eigen::Vector2d(ideal_t_robot_map.transform.translation.x, ideal_t_robot_map.transform.translation.y);
        ideal_pose.linear() = Eigen::Rotation2Dd(ideal_rotation_angle).toRotationMatrix();

        g2o::VertexSE2 *new_node = graph_handler_ptr->createPoseVertex(pose);

        if (!doing_optim)
        {
          g2o::VertexSE2 *curr_node = graph_handler_ptr->addPoseVertex(new_node);
          added_pose_vertices++;

          // message for new vertex visualization
          all_nodes_msg.push_back(nodeMessage(ROBOT_TYPE, static_cast<int>(curr_node->id()), cones_msg->header.stamp, pose.translation()[0], pose.translation()[1], "", rotation_angle));

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

          // construct the message for the new edge
          all_edges_msg.push_back(edgeMessage(ROBOT_TYPE, static_cast<int>(new_edge->vertices()[0]->id()), static_cast<int>(new_edge->vertices()[1]->id())));
        }
        else
        {
          PoseToAdd pending_pose;
          pending_pose.pose_node = new_node;
          pending_pose.data_header = odom_msg->header;
          pending_pose.real_pose = pose;
          pending_pose.rot_angle = rotation_angle;
          pending_pose.ideal_pose = ideal_t_robot_map.transform;
          pending_pose.prev_pose = prev_pose;
          pending_poses.push_back(pending_pose);
        }

        //************************************************** LOCAL OPTIMIZATION ***********************************************
        // perform local optimization every 10 new nodes
        if (added_pose_vertices >= 10 && !doing_optim)
        {
          doing_optim = true;

          // restore the counter
          added_pose_vertices = 0;

          // save the portion of the graph not fixed (the one to optimize)
          g2o::HyperGraph::EdgeSet edges_to_optim;
          auto pre_opt_edges = graph_handler_ptr->getEdges();
          for (g2o::HyperGraph::EdgeSet::iterator it = pre_opt_edges.begin(); it != pre_opt_edges.end(); ++it)
          {
            g2o::OptimizableGraph::Edge *edge = (g2o::OptimizableGraph::Edge *)(*it);
            NodeData *vert_i_data = dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(edge->vertices()[0])->userData());
            NodeData *vert_j_data = dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(edge->vertices()[1])->userData());
            bool vert_i_fixed, vert_j_fixed;
            // check if vertex_i is fixed
            if (vert_i_data->getNodeType() == ROBOT_TYPE)
            {
              g2o::VertexSE2 *vert_i = dynamic_cast<g2o::VertexSE2 *>(edge->vertices()[0]);
              vert_i_fixed = vert_i->fixed();
            }
            else
            {
              g2o::VertexPointXY *vert_i = dynamic_cast<g2o::VertexPointXY *>(edge->vertices()[0]);
              vert_i_fixed = vert_i->fixed();
            }
            // check if vertex_j is fixed
            if (vert_j_data->getNodeType() == ROBOT_TYPE)
            {
              g2o::VertexSE2 *vert_j = dynamic_cast<g2o::VertexSE2 *>(edge->vertices()[1]);
              vert_j_fixed = vert_j->fixed();
            }
            else
            {
              g2o::VertexPointXY *vert_j = dynamic_cast<g2o::VertexPointXY *>(edge->vertices()[1]);
              vert_j_fixed = vert_j->fixed();
            }

            // if at least one of the two vertices is not fixed, optimize this edge
            if ((!vert_i_fixed) || (!vert_j_fixed))
            {
              edges_to_optim.insert(edge);
            }
          }

          // perform local optimization
          std::cout << "before local optim\n";
          graph_handler_ptr->localOptimization(1000, edges_to_optim);

          // fix all the optimized vertices
          auto opt_vertices = graph_handler_ptr->getNodes();
          for (int i = 0; i < opt_vertices.size(); i++)
          {
            auto vert = opt_vertices[i];
            NodeData *vert_data = dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(vert)->userData());
            if (vert_data->getNodeType() == ROBOT_TYPE)
            {
              g2o::VertexSE2 *pose_vert = dynamic_cast<g2o::VertexSE2 *>(vert);
              pose_vert->setFixed(true);
            }
            else
            {
              g2o::VertexPointXY *cone_vert = dynamic_cast<g2o::VertexPointXY *>(vert);
              cone_vert->setFixed(true);
            }
          }

          // release the optimization flag
          doing_optim = false;
        }
        //********************************************************************************************************************
      }

      //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
      // ------------------------------------ CONES DATA PROCESSING -----------------------------------
      // RCLCPP_INFO(this->get_logger(), "----------------------------------------------------------------------------");
      // RCLCPP_INFO(this->get_logger(), "RECEIVED CONES DATA");
      // RCLCPP_INFO(this->get_logger(), "----------------------------------------------------------------------------");

      cones_timer = now;

      // NB: TF2 frames are: source is the frame of the data, while target is the frame to transform to
      std::string to_frame = "map";                  //(called target frame in tf2 params)
      std::string from_frame = target_frame.c_str(); //(called source frame in tf2 params)  //base_footprint

      geometry_msgs::msg::TransformStamped t_robot_map;
      // t_robot_map = t_odom;
      t_robot_map = t_r_m_odom;
      // transform cones data into map frame
      // try
      // {
      //   // tf has been inverted because markers were not correctly visualized otherwise
      //   t_robot_map = tf_buffer->lookupTransform(to_frame, from_frame, tf2::TimePointZero);
      //   std::cout<< "tr[" << t_robot_map.transform.translation.x
      //            << ","   << t_robot_map.transform.translation.y
      //            << ","   << t_robot_map.transform.translation.z << "]\n";

      //   std::cout<< "to[" << t_odom.transform.translation.x
      //            << ","   << t_odom.transform.translation.y
      //            << ","   << t_odom.transform.translation.z << "]\n";

      //   // RCLCPP_INFO(this->get_logger(), "SUCCESSFULL TRANSFORM from frame %s to frame %s",
      //   //             t_robot_map.header.frame_id.c_str(), t_robot_map.child_frame_id.c_str());
      // }
      // catch (const tf2::TransformException &exception)
      // {
      //   // RCLCPP_INFO(
      //   //   this->get_logger(), "Could not transform %s to %s: %s",
      //   //   to_frame.c_str(), from_frame.c_str(), exception.what());
      //   std::cout << "============================ TF_ROBOT_MAP NOT FOUND " << cnt << " =========================\n";
      //   cnt++;
      //   return;
      // }

      /** To process data of perceived cones:
       * 1. if first scan, add all of them to the graph by doing initialization
       * 2. otherwise, check for data association
       * 3. if no association, add a new node, otherwise add only an edge
       */

      //************************************************************** INITIAL SCAN **************************************************************

      // add initial data about first scan (initialize the graph)
      if (!init_cones)
      {
        cones_timer = now; // update the time of the last processed cone callback
                           // retrieve the node corresponding to the current car pose
        g2o::VertexSE2 *curr_car_pose = graph_handler_ptr->getLastPoseNode();
        // process big orange cones
        for (int i = 0; i < static_cast<int>(cones_msg->big_orange_cones.size()); i++)
        {
          // RCLCPP_INFO(this->get_logger(), "Adding big orange cone with unique ID = %d", cone_unique_ID);
          cone_unique_ID++;

          // transform to map frame
          geometry_msgs::msg::PointStamped cone_map_frame = cone_to_map(cones_msg->header, cones_msg->big_orange_cones[i].point, t_robot_map);

          geometry_msgs::msg::PointStamped ideal_cone_map_frame = cone_to_map(ideal_cones.header, ideal_cones.big_orange_cones[i].point, ideal_t_robot_map);

          g2o::VertexPointXY *cone_node = graph_handler_ptr->createLandmarkVertex(Eigen::Vector2d(cone_map_frame.point.x, cone_map_frame.point.y));

          if (!doing_optim)
          {
            // if not doing optimization, modify the graph and add the new elements

            g2o::VertexPointXY *added_vertex = graph_handler_ptr->addLandmarkVertex(cone_node);
            addConeData(added_vertex, cone_unique_ID, ORANGE_CONE, cones_msg->header.stamp, ideal_cone_map_frame);

            // message for new vertex visualization
            all_nodes_msg.push_back(nodeMessage(LANDMARK_TYPE, static_cast<int>(added_vertex->id()), cones_msg->header.stamp, cone_map_frame.point.x, cone_map_frame.point.y, ORANGE_CONE));

            total_cones++;

            // add an edge between the current pose and the observed cones (pose-landmark edge)
            g2o::EdgeSE2PointXY *added_edge = add_cone_edge(cones_msg->big_orange_cones[i].point, added_vertex, curr_car_pose);
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
          geometry_msgs::msg::PointStamped cone_map_frame = cone_to_map(cones_msg->header, cones_msg->blue_cones[i].point, t_robot_map);

          geometry_msgs::msg::PointStamped ideal_cone_map_frame = cone_to_map(ideal_cones.header, ideal_cones.blue_cones[i].point, ideal_t_robot_map);

          g2o::VertexPointXY *cone_node = graph_handler_ptr->createLandmarkVertex(Eigen::Vector2d(cone_map_frame.point.x, cone_map_frame.point.y));

          if (!doing_optim)
          {
            g2o::VertexPointXY *added_vertex = graph_handler_ptr->addLandmarkVertex(cone_node);
            total_cones++;

            // message for new vertex visualization
            all_nodes_msg.push_back(nodeMessage(LANDMARK_TYPE, static_cast<int>(added_vertex->id()), cones_msg->header.stamp, cone_map_frame.point.x, cone_map_frame.point.y, BLUE_CONE));

            addConeData(added_vertex, cone_unique_ID, BLUE_CONE, cones_msg->header.stamp, ideal_cone_map_frame);

            // add an edge between the current pose and the observed cones (pose-landmark edge)
            g2o::EdgeSE2PointXY *added_edge = add_cone_edge(cones_msg->blue_cones[i].point, added_vertex, curr_car_pose);
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
          geometry_msgs::msg::PointStamped cone_map_frame = cone_to_map(cones_msg->header, cones_msg->yellow_cones[i].point, t_robot_map);

          g2o::VertexPointXY *cone_node = graph_handler_ptr->createLandmarkVertex(Eigen::Vector2d(cone_map_frame.point.x, cone_map_frame.point.y));

          geometry_msgs::msg::PointStamped ideal_cone_map_frame = cone_to_map(ideal_cones.header, ideal_cones.yellow_cones[i].point, ideal_t_robot_map);

          if (!doing_optim)
          {
            g2o::VertexPointXY *added_vertex = graph_handler_ptr->addLandmarkVertex(cone_node);
            total_cones++;

            // message for new vertex visualization
            all_nodes_msg.push_back(nodeMessage(LANDMARK_TYPE, static_cast<int>(added_vertex->id()), cones_msg->header.stamp, cone_map_frame.point.x, cone_map_frame.point.y, YELLOW_CONE));

            addConeData(added_vertex, cone_unique_ID, YELLOW_CONE, cones_msg->header.stamp, ideal_cone_map_frame);

            // add an edge between the current pose and the observed cones (pose-landmark edge)
            g2o::EdgeSE2PointXY *added_edge = add_cone_edge(cones_msg->yellow_cones[i].point, added_vertex, curr_car_pose);
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
        // std::cout << "INITIALIZATION DONE - CAR NOT MOVING \n Number of nodes in the graph = " << num_cones << "\n";
        // std::cout << "INITIALIZATION DONE - CAR NOT MOVING \n Number of edges in the graph = " << num_edges << "\n";
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
      if (cones_msg->big_orange_cones.size() != 0)
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
        g2o::EdgeSE2 *new_edge = graph_handler_ptr->createPoseEdge(first_fixed_pose, last_pose, rel_pose, info_mat);
        graph_handler_ptr->addPoseEdge(new_edge);

        // construct the message for the new edge
        all_edges_msg.push_back(edgeMessage(ROBOT_TYPE, static_cast<int>(new_edge->vertices()[0]->id()), static_cast<int>(new_edge->vertices()[1]->id())));
      }
      //---------------------------------------------------- LOOP CLOSURE CHECK END------------------------------------------------------------------

      //  orange cones
      for (int i = 0; i < cones_msg->big_orange_cones.size(); i++)
      {
        // transform to map frame the real observation (noisy)
        geometry_msgs::msg::PointStamped cone_map_frame = cone_to_map(cones_msg->header, cones_msg->big_orange_cones[i].point, t_robot_map);
        // std::cout << "real observation: " << cones_msg->big_orange_cones[i].point.x << "," << cones_msg->big_orange_cones[i].point.y << "\n";
        //  transform to map frame the ideal observation (ground truth)
        geometry_msgs::msg::PointStamped ideal_cone_map_frame = cone_to_map(ideal_cones.header, ideal_cones.big_orange_cones[i].point, ideal_t_robot_map);
        // std::cout << "ideal observation: " << ideal_cones.big_orange_cones[i].point.x << "," << ideal_cones.big_orange_cones[i].point.y << "\n";

        // test association
        // g2o::VertexPointXY *assoc_found = perform_data_assoc(ideal_cone_map_frame.point);
        g2o::VertexPointXY *assoc_found = perform_data_assoc(ideal_cone_map_frame.point, cone_map_frame.point, curr_pose_node_id);
        if (assoc_found != nullptr)
        {
          // std::cout << "orange_association[" << dynamic_cast<g2o::VertexPointXY *>(assoc_found)->estimate().x()
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
            all_nodes_msg.push_back(nodeMessage(LANDMARK_TYPE, static_cast<int>(added_vertex->id()), cones_msg->header.stamp, cone_map_frame.point.x, cone_map_frame.point.y, ORANGE_CONE));

            addConeData(added_vertex, cone_unique_ID, ORANGE_CONE, cones_msg->header.stamp, ideal_cone_map_frame);

            // add an edge between the current pose and the observed cones (pose-landmark edge)
            g2o::EdgeSE2PointXY *added_edge = add_cone_edge(cones_msg->big_orange_cones[i].point, added_vertex, last_pose);
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
            g2o::EdgeSE2PointXY *added_edge = add_cone_edge(cones_msg->big_orange_cones[i].point, assoc_found, last_pose);
          }
          else
          {
            ObsEdgeToAdd pending_edge;
            pending_edge.rel_obs = cones_msg->big_orange_cones[i].point;
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
        geometry_msgs::msg::PointStamped cone_map_frame = cone_to_map(cones_msg->header, cones_msg->blue_cones[i].point, t_robot_map);
        // transform to map frame the ideal observation (ground truth)
        geometry_msgs::msg::PointStamped ideal_cone_map_frame = cone_to_map(ideal_cones.header, ideal_cones.blue_cones[i].point, ideal_t_robot_map);

        // test association
        // g2o::VertexPointXY *assoc_found = perform_data_assoc(ideal_cone_map_frame.point);
        g2o::VertexPointXY *assoc_found = perform_data_assoc(ideal_cone_map_frame.point, cone_map_frame.point, curr_pose_node_id);
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

            addConeData(added_vertex, cone_unique_ID, BLUE_CONE, cones_msg->header.stamp, ideal_cone_map_frame);

            // add an edge between the current pose and the observed cones (pose-landmark edge)
            g2o::EdgeSE2PointXY *added_edge = add_cone_edge(cones_msg->blue_cones[i].point, added_vertex, last_pose);
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
            g2o::EdgeSE2PointXY *added_edge = add_cone_edge(cones_msg->blue_cones[i].point, assoc_found, last_pose);
          }
          else
          {

            ObsEdgeToAdd pending_edge;
            pending_edge.rel_obs = cones_msg->blue_cones[i].point;
            pending_edge.cone_vertex = assoc_found;
            pending_edge.pose_vertex = last_pose;

            pending_obs_edges.push_back(pending_edge);
          }
        }
      }

      //  yellow cones
      for (int i = 0; i < cones_msg->yellow_cones.size(); i++)
      {
        // transform to map frame
        geometry_msgs::msg::PointStamped cone_map_frame = cone_to_map(cones_msg->header, cones_msg->yellow_cones[i].point, t_robot_map);

        // transform to map frame the ideal observation (ground truth)
        geometry_msgs::msg::PointStamped ideal_cone_map_frame = cone_to_map(ideal_cones.header, ideal_cones.yellow_cones[i].point, ideal_t_robot_map);

        // test association
        // g2o::VertexPointXY *assoc_found = perform_data_assoc(ideal_cone_map_frame.point);
        g2o::VertexPointXY *assoc_found = perform_data_assoc(ideal_cone_map_frame.point, cone_map_frame.point, curr_pose_node_id);
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

            addConeData(added_vertex, cone_unique_ID, YELLOW_CONE, cones_msg->header.stamp, ideal_cone_map_frame);

            // add an edge between the current pose and the observed cones (pose-landmark edge)
            g2o::EdgeSE2PointXY *added_edge = add_cone_edge(cones_msg->yellow_cones[i].point, added_vertex, last_pose);
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
            g2o::EdgeSE2PointXY *added_edge = add_cone_edge(cones_msg->yellow_cones[i].point, assoc_found, last_pose);
          }
          else
          {
            ObsEdgeToAdd pending_edge;
            pending_edge.rel_obs = cones_msg->yellow_cones[i].point;
            pending_edge.cone_vertex = assoc_found;
            pending_edge.pose_vertex = last_pose;
            pending_obs_edges.push_back(pending_edge);
          }
        }
      }
    }
    //@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    // publish pose graph msg for visualization
    graph_based_slam::msg::PoseGraph pose_graph_msg;
    pose_graph_msg.header.frame_id = "map";
    pose_graph_msg.header.stamp = rclcpp::Node::now();
    pose_graph_msg.graph_edges = all_edges_msg;
    pose_graph_msg.graph_nodes = all_nodes_msg;
    graph_pub->publish(pose_graph_msg);
  }

  /**
   * Once every a fixed amount of time, run a global optimization of the entire graph
   */
  void timer_callback()
  {
    RCLCPP_INFO(this->get_logger(), "**************** Entering optimization callback! ******************** ");

    float error_cones_before_optim = computeEuclideanConeError();
    // std::vector<float> error_pose_before_optim = computeEuclideanPoseError();

    RCLCPP_INFO(this->get_logger(), "Cones error before optimization: %f", error_cones_before_optim);
    // RCLCPP_INFO(this->get_logger(), "Position error before optimization: %f", error_pose_before_optim[0]);
    // RCLCPP_INFO(this->get_logger(), "Orientation error before optimization: %f", error_pose_before_optim[1]);

    doing_optim = true;

    optim_timer = std::chrono::steady_clock::now();

    // before global optimization, unfix al the nodes
    auto vertices = graph_handler_ptr->getNodes();
    for (int i = 0; i < vertices.size(); i++)
    {
      auto vert = vertices[i];
      NodeData *vert_data = dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(vert)->userData());
      if (vert_data->getNodeType() == ROBOT_TYPE)
      {
        g2o::VertexSE2 *pose_vert = dynamic_cast<g2o::VertexSE2 *>(vert);
        PoseData *pose_data = dynamic_cast<PoseData *>(vert_data);
        // keep fixed the initial node
        if (pose_data->getUniqueID() == FIRST_NODE_ID)
        {
          pose_vert->setFixed(true);
        }
        else
          pose_vert->setFixed(false);
      }
      else
      {
        g2o::VertexPointXY *cone_vert = dynamic_cast<g2o::VertexPointXY *>(vert);
        cone_vert->setFixed(false);
      }
    }
    // very high n. iterations, it stops when the error does not decrease anymore
    graph_handler_ptr->globalOptimization(1000);

    optim_c++;
    std::cout << "finished graph optimization \n";

    float error_cones_after_optim = computeEuclideanConeError();
    // std::vector<float> error_pose_after_optim = computeEuclideanPoseError();

    RCLCPP_INFO(this->get_logger(), "Cones error after optimization: %f", error_cones_after_optim);
    // RCLCPP_INFO(this->get_logger(), "Position error after optimization: %f", error_pose_after_optim[0]);
    // RCLCPP_INFO(this->get_logger(), "Orientation error befoafterre optimization: %f", error_pose_after_optim[1]);

    std::cout << "True Positive: " << true_pos << "/" << total_assoc_test << " = " << true_pos / total_assoc_test * 100 << "%\n";
    std::cout << "True Negative: " << true_neg << "/" << total_assoc_test << " = " << true_neg / total_assoc_test * 100 << "%\n";
    std::cout << "False Positive: " << false_pos << "/" << total_assoc_test << " = " << false_pos / total_assoc_test * 100 << "%\n";
    std::cout << "False Negative: " << false_neg << "/" << total_assoc_test << " = " << false_neg / total_assoc_test * 100 << "%\n";

    std::cout << "---------------------------------------------------------------------\n";

    std::cout << "True Positive: " << true_pos_mh << "/" << total_assoc_test << " = " << true_pos_mh / total_assoc_test * 100 << "%\n";
    std::cout << "True Negative: " << true_neg_mh << "/" << total_assoc_test << " = " << true_neg_mh / total_assoc_test * 100 << "%\n";
    std::cout << "False Positive: " << false_pos_mh << "/" << total_assoc_test << " = " << false_pos_mh / total_assoc_test * 100 << "%\n";
    std::cout << "False Negative: " << false_neg_mh << "/" << total_assoc_test << " = " << false_neg_mh / total_assoc_test * 100 << "%\n";

    // TODO: change folder/filename
    saveTrajectoryToTxt("traj_eval/optimized_estimated_traj.txt", "est");
    saveTrajectoryToTxt("traj_eval/ground_truth_traj.txt", "gt");

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
    // for (int i = 0; i < pending_odom_edges.size(); i++)
    // {
    //   // g2o::EdgeSE2* added_edge = add_pose_edge(TODO:TODO: TODO: TODO: TODO: TODO: fill here);
    // }

    pending_cones.clear();
    pending_obs_edges.clear();
    pending_poses.clear();
    // pending_odom_edges.clear();

    all_nodes_msg.clear();
    all_edges_msg.clear();

    // publish pose graph msg for visualization
    auto optimized_vertices = graph_handler_ptr->getNodes();
    for (int i = 0; i < optimized_vertices.size(); i++)
    {
      auto vert = optimized_vertices[i];

      NodeData *vert_data = dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(vert)->userData());

      double x_vert, y_vert;

      if (vert_data->getNodeType() == LANDMARK_TYPE)
      {
        x_vert = dynamic_cast<g2o::VertexPointXY *>(vert)->estimate().x();
        y_vert = dynamic_cast<g2o::VertexPointXY *>(vert)->estimate().y();

        all_nodes_msg.push_back(nodeMessage(LANDMARK_TYPE, static_cast<int>(vert->id()), vert_data->getTimestamp(), x_vert, y_vert, dynamic_cast<ConeData *>(vert_data)->getColor()));

        num_cones_mapped++;
      }
      else
      {
        x_vert = dynamic_cast<g2o::VertexSE2 *>(vert)->estimate().translation()[0];
        y_vert = dynamic_cast<g2o::VertexSE2 *>(vert)->estimate().translation()[1];

        all_nodes_msg.push_back(nodeMessage(ROBOT_TYPE, static_cast<int>(vert->id()), vert_data->getTimestamp(), x_vert, y_vert));
      }
    }

    std::cout << " MAPPED cones / GROUND TRUTH cones:" << num_cones_mapped << "/" << num_cones_gt << "\n";

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
   * @brief Called when a WheelSpeedsStamped msg is received.
   * @param msg ROS message that provides info about steering (in rad) and speeds of each wheel (in RPM).
   */
  void wheel_topic_callback(const eufs_msgs::msg::WheelSpeedsStamped::SharedPtr msg)
  {
    // RCLCPP_INFO(this->get_logger(), "RECEIVED DATA FROM WHEEL SPEEDS");

    // check if the car is moving: all wheels must have speed different from zero
    float rf = msg->speeds.rf_speed;
    float lf = msg->speeds.lf_speed;
    float rb = msg->speeds.rb_speed;
    float lb = msg->speeds.lb_speed;
    moving_car = rf && lf && rb && lb;
  }

  // ##################################################### USEFUL METHODS #####################################################
  /**
   * @brief Generate a random noise value, following Gaussian distribution.
   * @param mean The mean of the desired Gaussian distribution.
   * @param stddev The standard deviation of the desired Gaussian distribution.
   *
   * @return The value of noise to add to data.
   */
  float generateGaussianNoise(double mean, double stddev)
  {
    float noisy_value;

    // Define random generator with Gaussian distribution
    std::random_device r;
    std::default_random_engine generator(r());
    std::normal_distribution<double> dist(mean, stddev);

    return dist(generator);
  }

  /**
   * @brief Add Gaussian noise to a car pose.
   * @param odom_transform The input odometry data.
   * @return The noisy odometry data.
   */
  geometry_msgs::msg::Transform addNoiseToOdom(geometry_msgs::msg::Transform odom_transform)
  {
    float position_noise = generateGaussianNoise(noise_mean, noise_stddev);
    float orientation_noise = generateGaussianNoise(orientation_mean, orientation_stddev);
    geometry_msgs::msg::Transform noisy_odom;
    noisy_odom.translation.x = odom_transform.translation.x + position_noise;
    noisy_odom.translation.y = odom_transform.translation.y + position_noise;

    noisy_odom.rotation.x = odom_transform.rotation.x + orientation_noise;
    noisy_odom.rotation.y = odom_transform.rotation.y + orientation_noise;
    noisy_odom.rotation.z = odom_transform.rotation.z + orientation_noise;
    noisy_odom.rotation.w = odom_transform.rotation.w + orientation_noise;

    return noisy_odom;
  }

  /**
   * @brief Add Gaussian noise to a list of cones.
   * @param cones_list The input cones positions.
   * @return The noisy cones positions.
   */
  std::vector<eufs_msgs::msg::ConeWithCovariance> addNoiseToCones(std::vector<eufs_msgs::msg::ConeWithCovariance> cones_list)
  {
    std::vector<eufs_msgs::msg::ConeWithCovariance> noisy_cones_list;
    float noise = generateGaussianNoise(noise_mean, noise_stddev);
    for (int i = 0; i < cones_list.size(); i++)
    {
      eufs_msgs::msg::ConeWithCovariance noisy_cone;
      noisy_cone.point.x = cones_list[i].point.x + noise;
      noisy_cone.point.y = cones_list[i].point.y + noise;

      // TODO: check this
      noisy_cone.covariance[0] = std::pow(noise_stddev, 2);
      noisy_cone.covariance[3] = std::pow(noise_stddev, 2);
      noisy_cone.covariance[1] = 0.0;
      noisy_cone.covariance[2] = 0.0;

      noisy_cones_list.push_back(noisy_cone);
    }
    return noisy_cones_list;
  }

  /**
   * @brief Construct transformation matrix from TF2 transform.
   * @param t A transformation as translation and rotatiom components.
   * @return A 2D transformation matrix representing the given transformation.
   */
  Eigen::Matrix3d transf_to_2Dmatrix(geometry_msgs::msg::TransformStamped t)
  {
    Eigen::Quaterniond quat;
    quat.x() = t.transform.rotation.x;
    quat.y() = t.transform.rotation.y;
    quat.z() = t.transform.rotation.z;
    quat.w() = t.transform.rotation.w;

    double rot_angle = quat.toRotationMatrix().eulerAngles(0, 1, 2)[2]; // rotation around z
    Eigen::Vector2d translation;
    translation.x() = t.transform.translation.x;
    translation.y() = t.transform.translation.y;
    Eigen::Matrix3d t_matrix;
    t_matrix.setIdentity();
    t_matrix.block<2, 2>(0, 0) = Eigen::Rotation2Dd(rot_angle).toRotationMatrix();
    t_matrix.block<2, 1>(0, 2) = translation;

    return t_matrix;
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
    double DIST_THRESHOLD_MH = 25.0;
    g2o::VertexPointXY *nn_cone_ideal;
    g2o::VertexPointXY *nn_cone_est;
    //g2o::VertexPointXY *nn_cone_mh = nullptr;

    //  coords of the observation to test
    // use the ideal coordinates for data association, then use the real one for the edge
    double x_obs_ideal = ideal_obs_to_test.x;
    double y_obs_ideal = ideal_obs_to_test.y;

    double x_obs_est = est_obs_to_test.x;
    double y_obs_est = est_obs_to_test.y;

    auto graph_vertices = graph_handler_ptr->getNodes();
    auto graph = graph_handler_ptr->getGraph();

    //****************
    PoseData *curr_data = dynamic_cast<PoseData *>(dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(graph->vertex(curr_pose_node_id))->userData()));
    
    Eigen::Quaterniond q(curr_data->getIdealCoords().rotation.w,
                         curr_data->getIdealCoords().rotation.x,
                         curr_data->getIdealCoords().rotation.y,
                         curr_data->getIdealCoords().rotation.z);

    g2o::SE2 curr_se2(curr_data->getIdealCoords().translation.x,
                      curr_data->getIdealCoords().translation.y,
                      q.toRotationMatrix().eulerAngles(0, 1, 2)[2]);

    Eigen::Vector2d obs_estim = curr_se2.inverse() * Eigen::Vector2d(x_obs_est, y_obs_est);

    //***************
    //graph->setVerbose(false);
    //graph->initializeOptimization();
    //graph->optimize(1);
    //graph->setVerbose(true);
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

        // compute Mahalanobis distance
        // std::cout << "curr_land_node_id " << curr_land_node_id << std::endl;
        // std::cout << "curr_pose_node_id " << curr_pose_node_id << std::endl;
        // auto graph = graph_handler_ptr->getGraph();

        // MH TENTATIVE
        //double mahal_dist_est = computeGatingChi2(obs_estim, *graph, curr_pose_node_id, curr_land_node_id);
        //if (mahal_dist_est <= min_MAHAL_NN)
        //{
        //  min_MAHAL_NN = mahal_dist_est;
        //  nn_cone_mh = dynamic_cast<g2o::VertexPointXY *>(tmp_vertex);
        //}

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

    //std::cout << "MinMahalDist " << min_MAHAL_NN << "\n";
    //if (nn_cone_mh)
    //  std::cout << "ID Mh = " << nn_cone_mh->id() << "\n";
//
    //if (nn_cone_ideal)
    //  std::cout << "ID Ideal = " << nn_cone_ideal->id() << "\n";
    //else
    //  std::cout << "NO ASSOCITION \n";
//
    //std::cout << "--------------------------------------\n";

    //  COMPUTE TP, TN, FP, FN
  
    // both ideal and est association found
    if ((min_dist_NN_ideal < DIST_THRESHOLD) && (min_dist_NN_est < DIST_THRESHOLD))
    {
      // retrieve the ID of the cone associated both with ideal data and with estimated data
      ConeData *assoc_cone_data_ideal = dynamic_cast<ConeData *>(dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(nn_cone_ideal)->userData()));
      ConeData *assoc_cone_data_est = dynamic_cast<ConeData *>(dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(nn_cone_est)->userData()));

      // TRUE POSITIVE (same IDs)
      if (assoc_cone_data_est->getUniqueID() == assoc_cone_data_ideal->getUniqueID())
      {
        true_pos++;
      }
      else // TRUE NEGATIVE (different IDs)
      {
        false_pos++;
      }
      //return nn_cone_ideal;
      return nn_cone_est;
    }
    // ideal associates, estimated NOT
    else if ((min_dist_NN_ideal < DIST_THRESHOLD) && !(min_dist_NN_est < DIST_THRESHOLD))
    {
      false_neg++;

      //return nn_cone_ideal;
      return nullptr;
    }
    // ideal does NOT associate, estimated yes
    else if (!(min_dist_NN_ideal < DIST_THRESHOLD) && (min_dist_NN_est < DIST_THRESHOLD))
    {
      false_pos++;

      // with ideal data assoc, change otherwise
      //return nullptr; //
      return nn_cone_est;
    }
    // neither ideal and est associates
    else
    {
      true_neg++;
      return nullptr;
    }


    /** TENTATIVE 
    if ((min_dist_NN_ideal < DIST_THRESHOLD) && (min_MAHAL_NN < DIST_THRESHOLD_MH))
    {
      // retrieve the ID of the cone associated both with ideal data and with estimated data
      ConeData *assoc_cone_data_ideal = dynamic_cast<ConeData *>(dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(nn_cone_ideal)->userData()));
      ConeData *assoc_cone_data_est = dynamic_cast<ConeData *>(dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(nn_cone_mh)->userData()));

      // TRUE POSITIVE (same IDs)
      if (assoc_cone_data_est->getUniqueID() == assoc_cone_data_ideal->getUniqueID())
      {
        true_pos_mh++;
      }
      else
      {
        false_pos_mh++;
      }
      return nn_cone_mh;
    }
    else if ((min_dist_NN_ideal < DIST_THRESHOLD) && !(min_MAHAL_NN < DIST_THRESHOLD_MH))
    {
      false_neg_mh++;
      return nullptr;
    }
    else if (!(min_dist_NN_ideal < DIST_THRESHOLD) && (min_MAHAL_NN < DIST_THRESHOLD_MH))
    {
      false_pos_mh++;
      return nn_cone_mh;
    }
    else
    {
      true_neg_mh++;
      return nullptr;
    }
    /**/

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

  /**
   * Transform coordinates of a cone into map frame (global reference frame).
   * @param msg_header Header of the received message, containing reference frame and timestamp of data.
   * @param cone_coords Coordinates of a cone in base footprint frame (robot frame).
   * @param robot_map_tf Rigid body transformation between the data reference frame and the target one.
   * @return Coordinates of the considered cone in map reference frame.
   */
  geometry_msgs::msg::PointStamped cone_to_map(std_msgs::msg::Header msg_header, geometry_msgs::msg::Point cone_coords, geometry_msgs::msg::TransformStamped robot_map_tf)
  {
    geometry_msgs::msg::PointStamped cone_bf, cone_map;
    cone_bf.header.frame_id = msg_header.frame_id;
    cone_bf.header.stamp = msg_header.stamp;
    cone_bf.point = cone_coords;
    tf2::doTransform(cone_bf, cone_map, robot_map_tf);

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
   * Compute a simple metric: euclidean distance between estimated cones positions and ground truth ones.
   * @return The mean position error of the cones in the graph.
   */
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
        // std::cout << "current euclidean error = " << euclid_dist << "\n";

        total_error += euclid_dist;
      }
    }
    std::cout << "num_cones_nodes = " << n_landmark_nodes << "\n";
    std::cout << "total error = " << total_error << "\n";
    // compute the mean error for a cone
    return total_error / n_landmark_nodes;
  }

  // /**
  //  * Compute a simple metric: euclidean distance between estimated car positions and ground truth ones.
  //  * @return The mean position and orientation error of the car poses in the graph.
  //  */
  // std::vector<float> computeEuclideanPoseError()
  // {
  //   float total_error_pos, total_error_orient;
  //   int n_pose_nodes = 0;

  //   // retrieve nodes of the graph
  //   auto graph_vertices = graph_handler_ptr->getNodes();
  //   for (int i = 0; i < graph_vertices.size(); i++)
  //   {
  //     auto tmp_vertex = graph_vertices[i];

  //     // consider only car pose nodes
  //     NodeData *node_data = dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(tmp_vertex)->userData());
  //     if (node_data->getNodeType() == ROBOT_TYPE)
  //     {
  //       n_pose_nodes++;
  //       // compute euclidean distance between ideal and estimated positions
  //       g2o::VertexSE2 *tmp = dynamic_cast<g2o::VertexSE2 *>(tmp_vertex);
  //       PoseData *pose_data = dynamic_cast<PoseData *>(node_data);

  //       double x_ideal = pose_data->getIdealCoords().translation.x;
  //       double y_ideal = pose_data->getIdealCoords().translation.y;

  //       double x_estimated = tmp->estimate().translation()[0];
  //       double y_estimated = tmp->estimate().translation()[1];

  //       double euclid_dist_position = sqrt(pow(x_estimated - x_ideal, 2) + pow(y_estimated - y_ideal, 2));

  //       total_error_pos += euclid_dist_position;

  //       Eigen::Quaterniond quat_ideal;
  //       quat_ideal.x() = pose_data->getIdealCoords().rotation.x;
  //       quat_ideal.y() = pose_data->getIdealCoords().rotation.y;
  //       quat_ideal.z() = pose_data->getIdealCoords().rotation.z;
  //       quat_ideal.w() = pose_data->getIdealCoords().rotation.w;
  //       double ideal_rot_angle = quat_ideal.toRotationMatrix().eulerAngles(0, 1, 2)[2]; // yaw, around z

  //       double estimated_rot_angle = tmp->estimate().rotation().angle(); // yaw, around z

  //       // TODO: check orientation error
  //       total_error_orient += abs(estimated_rot_angle - ideal_rot_angle);
  //     }
  //   }
  //   std::cout << "num_cones_nodes = " << n_pose_nodes << "\n";
  //   std::cout << "total error position= " << total_error_pos << "\n";
  //   std::cout << "total error orientation= " << total_error_orient << "\n";

  //   std::vector<float> car_errors;
  //   // compute the mean positioning error
  //   car_errors.push_back(total_error_pos / n_pose_nodes);
  //   // compute the mean orientation error
  //   car_errors.push_back(total_error_orient / n_pose_nodes);

  //   return car_errors;
  // }

  /**
   * Save trajectory to txt.
   * @param filename The path of the file to be saved.
   * @param type Flag to know what trajectory has to be saved.
   *
   */
  void saveTrajectoryToTxt(std::string filename, std::string type)
  {

    // open file for writing and write data
    std::ofstream save_traj(filename, std::ofstream::out);

    if (save_traj.is_open())
    {
      double tx, ty, tz, qx, qy, qz, qw;
      rclcpp::Time timestamp;

      // retrieve nodes of the graph
      auto graph_vertices = graph_handler_ptr->getNodes();
      for (int i = 0; i < graph_vertices.size(); i++)
      {
        auto tmp_vertex = graph_vertices[i];

        // consider only car pose nodes
        NodeData *node_data = dynamic_cast<NodeData *>(dynamic_cast<g2o::OptimizableGraph::Vertex *>(tmp_vertex)->userData());
        if (node_data->getNodeType() == ROBOT_TYPE)
        {
          g2o::VertexSE2 *tmp = dynamic_cast<g2o::VertexSE2 *>(tmp_vertex);

          timestamp = node_data->getTimestamp();

          if (type == "gt")
          {
            // ground-truth trajectory
            PoseData *pose_data = dynamic_cast<PoseData *>(node_data);
            double x_ideal = pose_data->getIdealCoords().translation.x;
            double y_ideal = pose_data->getIdealCoords().translation.y;
            // auto rot_ideal = pose_data->getIdealCoords().linear();

            tx = x_ideal;
            ty = y_ideal;
            tz = 0;
            qx = pose_data->getIdealCoords().rotation.x;
            qy = pose_data->getIdealCoords().rotation.y;
            qz = pose_data->getIdealCoords().rotation.z;
            qw = pose_data->getIdealCoords().rotation.w;
          }
          else if (type == "est")
          {
            // estimated trajectory
            double x_est = tmp->estimate().translation()[0];
            double y_est = tmp->estimate().translation()[1];
            double z_est = tmp->estimate().translation()[2];
            auto rot = tmp->estimate().rotation().angle();
            Eigen::Quaterniond quat_est(Eigen::AngleAxisd(rot, Eigen::Vector3d::UnitZ().normalized())); // TODO: check this

            tx = x_est;
            ty = y_est;
            tz = z_est;
            qx = quat_est.x();
            qy = quat_est.y();
            qz = quat_est.z();
            qw = quat_est.w();
          }

          save_traj << timestamp.nanoseconds() << " " << tx << " " << ty << " " << tz << " " << qx << " " << qy << " " << qz << " " << qw << "\n";
        }
      }
      save_traj.close();
    }
    else
    {
      std::cout << "Problem writing the file!\n";
    }
    save_traj.close();
  }
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  rclcpp::executors::MultiThreadedExecutor executor;
  slam_node_ptr = std::make_shared<incrementalGraphSLAM>(options);
  executor.add_node(slam_node_ptr);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}