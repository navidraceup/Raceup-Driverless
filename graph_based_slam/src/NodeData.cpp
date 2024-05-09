
#include "graph_based_slam/NodeData.h"

NodeData::NodeData() : g2o::HyperGraph::Data()
{
}

NodeData::~NodeData() {}

void NodeData::setTimestamp(rclcpp::Time ts)
{
    this->timestamp = ts;
}
void NodeData::setNodeType(const std::string &n_type)
{
    this->node_type = n_type;
}