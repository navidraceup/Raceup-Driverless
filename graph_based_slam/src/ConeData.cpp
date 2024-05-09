#include "graph_based_slam/ConeData.h"

ConeData::ConeData() : NodeData()
{
}
ConeData::~ConeData() {}

bool ConeData::write(std::ostream &os) const
{
    // TODO: not yet implemeted
    return false;
}
bool ConeData::read(std::istream &is)
{
    // TODO: not yet implemeted
    return false;
}

void ConeData::setColor(const std::string &color)
{
    this->cone_color = color;
}

void ConeData::setUniqueID(const int &id)
{
    this->unique_ID = id;
}

void ConeData::setIdealCoords(const geometry_msgs::msg::PointStamped &ideal_data)
{
    this->ideal_coords.point.x = ideal_data.point.x;
    this->ideal_coords.point.y = ideal_data.point.y;
}