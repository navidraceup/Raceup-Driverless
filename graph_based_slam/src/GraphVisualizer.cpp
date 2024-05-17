// C++ standard headers
#include <string>
#include <vector>
#include <fstream>

// ROS headers
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

// project headers
#include <graph_based_slam/msg/graph_edge.hpp>
#include <graph_based_slam/msg/graph_node.hpp>
#include <graph_based_slam/msg/pose_graph.hpp>

rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr graph_viz_pub;
int id = 0;


/**
 * Callback when a custom message containing the pose graph is received.
 * It handles the visualization of the structure, by creating and publishing markers on RViz.
 * @param graph_msg The message containing the pose graph to visualize.
 */
void graph_callback(const graph_based_slam::msg::PoseGraph::SharedPtr graph_msg)
{
  // process the graph here and visualize
  // std::cout << "Pose graph message received!\n";
  // std::cout << "pose graph edges size is: " << graph_msg->graph_edges.size() << "\n";
  // std::cout << "pose graph nodes size is: " << graph_msg->graph_nodes.size() << "\n";

  visualization_msgs::msg::MarkerArray::Ptr markers_array(new visualization_msgs::msg::MarkerArray());
  visualization_msgs::msg::Marker marker_vertex;
  visualization_msgs::msg::Marker marker_edge;

  // convert vertices into Marker Array
  int num_vertices = graph_msg->graph_nodes.size();
  // std::cout << num_vertices << " vertices\n";
  if (num_vertices)
  {
    marker_vertex.action = visualization_msgs::msg::Marker::MODIFY;
    marker_vertex.header = graph_msg->header;
    marker_vertex.type = visualization_msgs::msg::Marker::SPHERE;
    marker_vertex.scale.x = 1;
    marker_vertex.scale.y = 1;
    marker_vertex.scale.z = 1;
    marker_vertex.lifetime = rclcpp::Duration(0);
    marker_vertex.ns = "vertex";

    for (int i = 0; i < num_vertices; i++)
    {
      // TODO: handle orientation for robot poses? maybe triangles?

      if (graph_msg->graph_nodes[i].type == "robot")
      {
        // robot pose node
        marker_vertex.color.r = 0.0;
        marker_vertex.color.g = 1.0;
        marker_vertex.color.b = 0.0;
        marker_vertex.color.a = 1.0;
        marker_vertex.pose.position.x = graph_msg->graph_nodes[i].x;
        marker_vertex.pose.position.y = graph_msg->graph_nodes[i].y;
        marker_vertex.pose.position.z = 0;
        marker_vertex.id = graph_msg->graph_nodes[i].id;
      }
      else
      {
        // landmark position node
        marker_vertex.color.r = 1.0;
        marker_vertex.color.g = 0.0;
        marker_vertex.color.b = 0.0;
        marker_vertex.color.a = 1.0;
        marker_vertex.pose.position.x = graph_msg->graph_nodes[i].x;
        // std::cout << "x is " << marker_vertex.pose.position.x << "\n";
        marker_vertex.pose.position.y = graph_msg->graph_nodes[i].y;
        // std::cout << "y is " << marker_vertex.pose.position.y << "\n";
        marker_vertex.pose.position.z = 0;
        marker_vertex.id = graph_msg->graph_nodes[i].id;
      }
      markers_array->markers.push_back(marker_vertex);
    }
  }

  // convert edges into Marker Array
  int num_edges = graph_msg->graph_edges.size();
  std::cout << num_edges << " edges\n";
  if (num_edges)
  {
    marker_edge.action = visualization_msgs::msg::Marker::MODIFY;
    marker_edge.header = graph_msg->header;
    marker_edge.ns = "edge";
    marker_edge.type = visualization_msgs::msg::Marker::LINE_LIST;
    marker_edge.scale.x = 0.1; // line width
    marker_vertex.lifetime = rclcpp::Duration(0);
    marker_edge.id = 0;

    for (int i = 0; i < num_edges; i++)
    {
      if (graph_msg->graph_edges[i].type == "robot")
      {
        // pose-pose edge
        marker_edge.color.r = 0.0;
        marker_edge.color.b = 0.5;
        marker_edge.color.g = 0.8;
        marker_edge.color.a = 1.0;
      }
      else
      {
        // cone-pose edge
        marker_edge.color.r = 1.0;
        marker_edge.color.b = 0.8;
        marker_edge.color.g = 0.1;
        marker_edge.color.a = 1.0;
      }
      geometry_msgs::msg::Point v_i, v_j;
      v_i.x = graph_msg->graph_nodes[graph_msg->graph_edges[i].vertex_i].x;
      v_i.y = graph_msg->graph_nodes[graph_msg->graph_edges[i].vertex_i].y;
      v_i.z = 0;

      v_j.x = graph_msg->graph_nodes[graph_msg->graph_edges[i].vertex_j].x;
      v_j.y = graph_msg->graph_nodes[graph_msg->graph_edges[i].vertex_j].y;
      v_j.z = 0;

      marker_edge.points.push_back(v_i);
      marker_edge.points.push_back(v_j);

      markers_array->markers.push_back(marker_edge);
    }
  }

  // publish the graph visualization message
  graph_viz_pub->publish(*markers_array);
}


int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::Node::SharedPtr node_ptr = rclcpp::Node::make_shared("graph_visualizer");

  rclcpp::Subscription<graph_based_slam::msg::PoseGraph>::SharedPtr graph_sub = node_ptr->create_subscription<graph_based_slam::msg::PoseGraph>("/pose_graph", 10, graph_callback);
  graph_viz_pub = node_ptr->create_publisher<visualization_msgs::msg::MarkerArray>("graph_viz", 1);
  

  rclcpp::spin(node_ptr);
  rclcpp::shutdown();
  return 0;
}