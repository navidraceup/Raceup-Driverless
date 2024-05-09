// Standard headers
#include <stdint.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <sched.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/mman.h>

// ROS2 headers
#include <rclcpp/rclcpp.hpp>

// Project headers
#include <graph_based_slam/msg/can_data.hpp>

// CANLIB headers
#include <canlib.h>
#include <can_stuff/can_functions.h>
#include <can_stuff/IDS.h>
#include <can_msgs/msg/frame.h>

#define TWO_PI 6.283185307179586476925286766559
#define RPM_TO_KMH (TWO_PI * 0.00095)

#define INT 1
#define FLOAT 2
#define BOOL 3

const int AMK_VAL_1_IDS[4] = {0x283, 0x284, 0x287, 0x288};
const int AMK_VAL_2_IDS[4] = {0x285, 0x286, 0x289, 0x28A};

uint64_t time_k = 0;

// Battery_pack_tension
float battery_pack_tension;
// BMS_LV
float bms_lv[8]; // first 6 are tension last two are temps
// IMU
float accelerations[3]; // g
float omegas[3];        // rad/s
// SMU
float temperatures[5];
uint32_t suspensions[4];
// BMS
float max_bms_voltage;
float min_bms_voltage;
float mean_bms_voltage;
float max_bms_temp;
float min_bms_temp;
float mean_bms_temp;
float max_temp_nslave;
// LEM
float lem_current;
// AMK
motorValues1 motorVal1[4]; //  0 --> FL, 1 --> FR, 2 --> RL, 3 --> RR
motorValues2 motorVal2[4];
// GENERAL
int throttle;
int brake;
int steering;
int actualVelRPM;

using std::placeholders::_1;

rclcpp::Node::SharedPtr node_ptr;

class KvaserParser : public rclcpp::Node
{

public:
    // constructor
    KvaserParser(const rclcpp::NodeOptions &options) : Node("kvaser_parser", options)
    {

        can_sub = this->create_subscription<graph_based_slam::msg::CanData>("/can_data_out", 10, std::bind(&KvaserParser::parseCanMsg, this, _1));
    }

    virtual ~KvaserParser(){};

private:
    rclcpp::Subscription<graph_based_slam::msg::CanData>::SharedPtr can_sub;

    /**
     * Read all the incoming messages and republish them as custom ROS2 messages.
     */
    void parseCanMsg(const graph_based_slam::msg::CanData::SharedPtr can_msg)
    {

        if (can_msg->id == 0x60 || can_msg->id == 0x61 || can_msg->id == 0x62)
        {

            uint8_t msg_body_int[8];
            for (int i = 0; i < can_msg->msg_body.size(); i++)
            {
                msg_body_int[i] = (uint8_t)can_msg->msg_body[i];
            }
            imu_parser(msg_body_int, (int)can_msg->id);
            printf("acc_x %d %d %.3f\r\n", ACC_X_ID, FLOAT, accelerations[0]);
            printf("acc_y %d %d %.3f\r\n", ACC_Y_ID, FLOAT, accelerations[1]);
            printf("acc_z %d %d %.3f\r\n", ACC_Z_ID, FLOAT, accelerations[2]);
            printf("omega_x %d %d %.3f\r\n", OMG_X_ID, FLOAT, omegas[0]);
            printf("omega_y %d %d %.3f\r\n", OMG_Y_ID, FLOAT, omegas[1]);
            printf("omega_z %d %d %.3f\r\n", OMG_Z_ID, FLOAT, omegas[2]);
        }
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    node_ptr = std::make_shared<KvaserParser>(options);
    rclcpp::spin(node_ptr);
    rclcpp::shutdown();
    return 0;
}