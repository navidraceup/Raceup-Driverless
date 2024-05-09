// ROS2 headers
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point.hpp>

// Eigen headers
#include <Eigen/Dense>

// GPS headers
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <geodesy/utm.h>
#include <geodesy/wgs84.h>
#include <geographic_msgs/msg/geo_point_stamped.hpp>

// Project headers
#include <graph_based_slam/msg/utm_gps.hpp>

/**
 * This node converts GPS coordinates (latitude, longitude) into a more suitable UTM format.
 */

rclcpp::Publisher<graph_based_slam::msg::UtmGPS>::SharedPtr utm_pub;
rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr pose_pub;

/**
 * @brief Called when a message is received from the GPS.
 * @param gps_msg A reference to the received GPS message.
 */
void gps_callback(sensor_msgs::msg::NavSatFix::SharedPtr gps_msg)
{
  geographic_msgs::msg::GeoPointStamped geo_pt;
  geo_pt.header = gps_msg->header;
  geo_pt.position.latitude = gps_msg->latitude;
  geo_pt.position.longitude = gps_msg->longitude;
  geo_pt.position.altitude = NAN; // since we work in 2D, altitude is not specified

  geodesy::UTMPoint utm_pt; // easting=x, northing=y, altitude=z
  geodesy::fromMsg(geo_pt.position, utm_pt);
  utm_pt.easting = 0;

  // message creation and publishing
  // TODO: check what is better:
  
  // option 1: use geodesy::toGeometry() and then publish a standard point or pose
  geometry_msgs::msg::Point pt_from_utm = geodesy::toGeometry(utm_pt);
  pose_pub->publish(pt_from_utm);

  // option 2: custom message publishing UTM point/data
  graph_based_slam::msg::UtmGPS utm_coords_msg;
  utm_coords_msg.utm_x = utm_pt.easting;
  utm_coords_msg.utm_y = utm_pt.northing;
  utm_coords_msg.utm_z = utm_pt.altitude;
  utm_coords_msg.utm_band = utm_pt.band;
  utm_coords_msg.utm_zone = utm_pt.zone;
  utm_pub->publish(utm_coords_msg);
}

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::Node::SharedPtr node_ptr = rclcpp::Node::make_shared("convert_GPS");

  rclcpp::Subscription<sensor_msgs::msg::NavSatFix>::SharedPtr gps_sub = node_ptr->create_subscription<sensor_msgs::msg::NavSatFix>("/gps", 10, gps_callback);
  utm_pub = node_ptr->create_publisher<graph_based_slam::msg::UtmGPS>("utm_gps", 1);
  pose_pub = node_ptr->create_publisher<geometry_msgs::msg::Point>("gps_pose", 1);
  rclcpp::spin(node_ptr);
  rclcpp::shutdown();
  return 0;
}