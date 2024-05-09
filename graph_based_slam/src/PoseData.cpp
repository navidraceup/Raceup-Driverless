#include "graph_based_slam/PoseData.h"

PoseData::PoseData() : NodeData()
{
}
PoseData::~PoseData() {}

bool PoseData::write(std::ostream &os) const
{
    // not yet implemeted
    return false;
}
bool PoseData::read(std::istream &is)
{ // not yet implemeted
    return false;
}

void PoseData::setUniqueID(const int &id)
{
    this->unique_ID = id;
}

void PoseData::setIdealCoords(const geometry_msgs::msg::Transform &ideal_data)
{
    this->ideal_coords.translation = ideal_data.translation;
    this->ideal_coords.rotation = ideal_data.rotation;
}