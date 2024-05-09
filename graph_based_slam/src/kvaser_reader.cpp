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

// ROS2 stuff
using std::placeholders::_1;

rclcpp::Node::SharedPtr node_ptr;

class KvaserReader : public rclcpp::Node
{

public:
    // constructor
    KvaserReader(const rclcpp::NodeOptions &options) : Node("kvaser_reader", options)
    {
        declareParameters();
        setupRealTime(this->get_parameter("sched_priority").as_int());

        can_pub = this->create_publisher<graph_based_slam::msg::CanData>("can_data_out", 1); // topic with CAN data, not parsed

        canHandle hnd;
        canStatus stat;
        int channel_number = 0;
        canInitializeLibrary();
        std::cout << "Opening channel %d\n", channel_number;
        hnd = canOpenChannel(channel_number, canOPEN_ACCEPT_VIRTUAL);
        if (hnd < 0)
        {
            KvaserReader::Check("canOpenChannel", (canStatus)hnd);
            exit(1);
        }
        std::cout << "Setting bitrate and going on bus on \n";
        stat = canSetBusParams(hnd, canBITRATE_500K, 0, 0, 0, 0, 0);
        KvaserReader::Check("canSetBusParams", stat);
        stat = canBusOn(hnd);
        KvaserReader::Check("canBusOn", stat);

        dumpMessageLoop(hnd, channel_number);
    }

    virtual ~KvaserReader(){};

private:
    rclcpp::Publisher<graph_based_slam::msg::CanData>::SharedPtr can_pub;

    /**
     * Declares the ROS2 parameters of this node.
     */
    void declareParameters()
    {
        this->declare_parameter<int>("sched_priority", 99);
    }

    /**
     * Set real time priority for this node. This is fundamental to have data as soon as they are available.
     * @param priority The priority to be assigned to this process. 99 means realtime, it is the max accepted value.
     */
    void setupRealTime(int32_t priority)
    {
        if (priority > 99)
        {
            priority = 99;
        }
        printf("Setting realtime scheduling with priority : %d\n", priority);
        struct sched_param schp;
        memset(&schp, 0, sizeof(schp));
        schp.sched_priority = priority;
        struct rlimit rt_limit = {priority, priority};

        if (setrlimit(RLIMIT_RTPRIO, &rt_limit) || sched_setscheduler(0, SCHED_FIFO, &schp))
        {
            fprintf(stderr, "\n**********************ALERT**********************\n");
            fprintf(stderr, "Unable to get realtime scheduling priority.\n");
            fprintf(stderr, "This is BAD if you are trying to capture data.\n");
            fprintf(stderr, "To enable realtime scheduling, add the following line:\n");
            fprintf(stderr, "<user_name> hard rtprio 99\n");
            fprintf(stderr, "to the /etc/security/limits.conf file\n");
        }
        else
        {
            std::cout << "priority has been correctly set!\n";
        }
    }

    void print_timestamp(canStatus ans)
    {
        if (ans == canOK)
        {
            printf("time(ms) --> %lud\r\n", time_k);
        }
    }

    // The check method takes a canStatus (which is an enumerable) and the method
    // name as a string argument. If the status is an error code, it will print it.
    // Most Canlib method return a status, and checking it with a method like this
    // is a useful practice to avoid code duplication.
    void Check(const char *id, canStatus stat)
    {
        if (stat != canOK)
        {
            char buf[50];
            buf[0] = '\0';
            canGetErrorText(stat, buf, sizeof(buf));
            printf("%s: failed, stat=%d (%s)\n", id, (int)stat, buf);
            exit(1);
        }
    }


    void dumpMessageLoop(canHandle hnd, int channel_number)
    {
        // First declare some variables for holding the incoming messages. The
        // incoming messages consist of the same parameters as an outgoing message,
        // i.e. identifier (id), body (msg), length (dlc), and flags), as well as a
        // timestamp.
        canStatus stat = canOK;
        canStatus ans = canOK;
        long id;
        unsigned int dlc, flags;
        unsigned char msg[8];
        unsigned long int timestamp;
        printf("Listening for messages on channel %d, press any key to close\n", channel_number);
        // Start a loop that loops until a key is pressed.
        while (rclcpp::ok())
        {
            // Call the canReadWait method to wait for a message on the channel. This
            // method has a timeout parameter which in this case is set to 100 ms. If a
            // message is received during this time, it will return the status code
            // canOK and the message will be written to the output parameters. If no
            // message is received, it will return canERR_NOMSG.
            stat = canReadWait(hnd, &id, msg, &dlc, &flags, &timestamp, 100);

            //************************************* msg creation ***************************************
            // this publishes the message without distinguishing its content
            // the parsing part will be done after (at message receipt)
            graph_based_slam::msg::CanData can_msg;
            can_msg.header.stamp = this->get_clock()->now(); // use a timestamp consistent with thw other sensors
            // TODO: handle the frame_id
            can_msg.id = id;
            can_msg.length = dlc;
            can_msg.flags = flags;
            can_msg.msg_body[0] = msg[0];
            can_msg.msg_body[1] = msg[1];
            can_msg.msg_body[2] = msg[2];
            can_msg.msg_body[3] = msg[3];
            can_msg.msg_body[4] = msg[4];
            can_msg.msg_body[5] = msg[5];
            can_msg.msg_body[6] = msg[6];
            can_msg.msg_body[7] = msg[7];
            can_msg.msg_body[8] = msg[8];
            can_msg.timestamp = timestamp;

            can_pub->publish(can_msg);
            //*****************************************************************************************

            // Check that the returned status is OK (which means that a message has been received)
            /*
            if (stat == canOK)
            {
                // If the message contains an error flag (which implies a different kind
                // of error than if an error signal had been returned), an error message
                // will be shown.
                if (flags & canMSG_ERROR_FRAME)
                {
                    printf("%d %d %d\r\n", 999, 999, 999);
                }
                // If no error flag was found, the program prints the message.
                else
                {
                    ans = kvReadTimer64(hnd, &time_k);

                    //printf("Id: %, Msg: %#04x %u %u %u %u %u %u %u length: %u Flags: %lu\n",
                    //       id, dlc, msg[0], msg[1], msg[2], msg[3], msg[4],
                     //      msg[5], msg[6], msg[7], timestamp);


                    switch (id)
                    {

                        // BMS_LV
                    case 0x55:
                    case 0x56:
                        print_timestamp(ans);
                        bms_lv_parser((uint8_t *)msg, (int)id);

                        printf("%d %d %.3f\r\n", BMS_LV_0_ID, FLOAT, bms_lv[0]);
                        printf("%d %d %.3f\r\n", BMS_LV_1_ID, FLOAT, bms_lv[1]);
                        printf("%d %d %.3f\r\n", BMS_LV_2_ID, FLOAT, bms_lv[2]);
                        printf("%d %d %.3f\r\n", BMS_LV_3_ID, FLOAT, bms_lv[3]);
                        printf("%d %d %.3f\r\n", BMS_LV_4_ID, FLOAT, bms_lv[4]);
                        printf("%d %d %.3f\r\n", BMS_LV_5_ID, FLOAT, bms_lv[5]);
                        printf("%d %d %.3f\r\n", BMS_LV_6_ID, FLOAT, bms_lv[6]);
                        printf("%d %d %.3f\r\n", BMS_LV_7_ID, FLOAT, bms_lv[7]);

                        break;

                        // IMU
                    case 0x60:
                    case 0x61:
                    case 0x62:
                        print_timestamp(ans);
                        imu_parser((uint8_t *)msg, (int)id);

                        printf("%d %d %.3f\r\n", ACC_X_ID, FLOAT, accelerations[0]);
                        printf("%d %d %.3f\r\n", ACC_Y_ID, FLOAT, accelerations[1]);
                        printf("%d %d %.3f\r\n", ACC_Z_ID, FLOAT, accelerations[2]);

                        printf("%d %d %.3f\r\n", OMG_X_ID, FLOAT, omegas[0]);
                        printf("%d %d %.3f\r\n", OMG_Y_ID, FLOAT, omegas[1]);
                        printf("%d %d %.3f\r\n", OMG_Z_ID, FLOAT, omegas[2]);

                        break;

                        // SMU temp and susp
                    case 0x100:
                    case 0x101:
                        print_timestamp(ans);
                        smu_parser((uint8_t *)msg, (int)id);

                        printf("%d %d %.3f\r\n", TEMP_PRE_RADIATOR_ID, FLOAT, temperatures[0]);
                        printf("%d %d %.3f\r\n", TEMP_PRE_COLDPLATE_ID, FLOAT, temperatures[1]);
                        printf("%d %d %.3f\r\n", TEMP_POST_COLDPLATE_ID, FLOAT, temperatures[2]);
                        printf("%d %d %.3f\r\n", TEMP_PRE_MOTOR_ID, FLOAT, temperatures[3]);
                        printf("%d %d %.3f\r\n", TEMP_POST_MOTOR_ID, FLOAT, temperatures[4]);

                        printf("%d %d %.3f\r\n", SUSP_FL_ID, FLOAT, (float)suspensions[0]);
                        printf("%d %d %.3f\r\n", SUSP_FR_ID, FLOAT, (float)suspensions[1]);
                        printf("%d %d %.3f\r\n", SUSP_RL_ID, FLOAT, (float)suspensions[2]);
                        printf("%d %d %.3f\r\n", SUSP_RR_ID, FLOAT, (float)suspensions[3]);

                        break;

                        // BMS Voltage
                    case 0x110:
                        print_timestamp(ans);
                        bms_voltage_parser((uint8_t *)msg);

                        printf("%d %d %.3f\r\n", MAX_BMS_VOLTAGE_ID, FLOAT, max_bms_voltage);
                        printf("%d %d %.3f\r\n", MIN_BMS_VOLTAGE_ID, FLOAT, min_bms_voltage);
                        printf("%d %d %.3f\r\n", MEAN_BMS_VOLTAGE_ID, FLOAT, mean_bms_voltage);

                        break;
                        // BMS Temp
                    case 0x111:
                        print_timestamp(ans);
                        bms_temp_parser((uint8_t *)msg);

                        printf("%d %d %.3f\r\n", MAX_BMS_TEMP_ID, FLOAT, max_bms_temp);
                        printf("%d %d %.3f\r\n", MIN_BMS_TEMP_ID, FLOAT, min_bms_temp);
                        printf("%d %d %.3f\r\n", MEAN_BMS_TEMP_ID, FLOAT, mean_bms_temp);
                        printf("%d %d %.3f\r\n", MAX_NSLAVE_TEMP_ID, FLOAT, max_temp_nslave);

                        break;

                        // Battery_pack_tension + total power
                    case 0x120:
                        print_timestamp(ans);
                        total_tension_parser((uint8_t *)msg);

                        printf("%d %d %.3f\r\n", CAR_VOLTAGE_ID, FLOAT, battery_pack_tension);

                        printf("%d %d %.3f\r\n", TOTAL_POWER_ID, FLOAT, battery_pack_tension * lem_current);

                        // aggiungerr total power da tabella
                        // togliere "\r\n da tutto
                        break;

                        // Car general info
                    case 0x130:
                        print_timestamp(ans);
                        general_info_parser((uint8_t *)msg);

                        // ID, tipo, msg
                        printf("%d %d %d\r\n", THROTTLE_ID, INT, throttle);
                        printf("%d %d %d\r\n", BRAKE_ID, INT, brake);
                        printf("%d %d %d\r\n", STEERING_ID, INT, steering);
                        printf("%d %d %f\r\n", ACTUALVELKMH_ID, FLOAT, actualVelRPM * RPM_TO_KMH); // TODO: check if RPM or kmh (kmh = RPM * 2 pi * 0.00095)
                        break;

                        // LEM
                    case 0x3C2:
                        print_timestamp(ans);
                        lem_parser(msg);

                        printf("%d %d %.3f\r\n", LEM_ID, FLOAT, lem_current);

                        break;

                        // AMK
                    case 0x283:
                        print_timestamp(ans);
                        amk_parser((uint8_t *)msg, (int)id);
                        printf("%d %d %d\r\n", AMK_STATUS_0_ID, INT, motorVal1[0].AMK_bSystemReady);
                        printf("%d %d %.3f\r\n", AMK_ACTUAL_VELOCITY_0_ID, FLOAT, motorVal1[0].AMK_ActualVelocity);
                        printf("%d %d %.3f\r\n", AMK_TORQUE_CURRENT_0_ID, FLOAT, motorVal1[0].AMK_TorqueCurrent);
                        printf("%d %d %u\r\n", AMK_VOLTAGE_0_ID, INT, motorVal1[0].AMK_Voltage);
                        printf("%d %d %.3f\r\n", AMK_CURRENT_0_ID, FLOAT, motorVal1[0].AMK_Current);
                        break;
                    case 0x284:
                        print_timestamp(ans);
                        amk_parser((uint8_t *)msg, (int)id);
                        printf("%d %d %d\r\n", AMK_STATUS_1_ID, INT, motorVal1[1].AMK_bSystemReady);
                        printf("%d %d %.3f\r\n", AMK_ACTUAL_VELOCITY_1_ID, FLOAT, motorVal1[1].AMK_ActualVelocity);
                        printf("%d %d %.3f\r\n", AMK_TORQUE_CURRENT_1_ID, FLOAT, motorVal1[1].AMK_TorqueCurrent);
                        printf("%d %d %u\r\n", AMK_VOLTAGE_1_ID, INT, motorVal1[1].AMK_Voltage);
                        printf("%d %d %.3f\r\n", AMK_CURRENT_1_ID, FLOAT, motorVal1[1].AMK_Current);
                        break;
                    case 0x285:
                        print_timestamp(ans);
                        amk_parser((uint8_t *)msg, (int)id);
                        printf("%d %d %.3f\r\n", AMK_MOTOR_TEMP_0_ID, FLOAT, motorVal2[0].AMK_TempMotor);
                        printf("%d %d %.3f\r\n", AMK_INVERTER_TEMP_0_ID, FLOAT, motorVal2[0].AMK_TempInverter);
                        printf("%d %d %.3f\r\n", AMK_IGBT_TEMP_0_ID, FLOAT, motorVal2[0].AMK_TempIGBT);
                        break;
                    case 0x286:
                        print_timestamp(ans);
                        amk_parser((uint8_t *)msg, (int)id);
                        printf("%d %d %.3f\r\n", AMK_MOTOR_TEMP_1_ID, FLOAT, motorVal2[1].AMK_TempMotor);
                        printf("%d %d %.3f\r\n", AMK_INVERTER_TEMP_1_ID, FLOAT, motorVal2[1].AMK_TempInverter);
                        printf("%d %d %.3f\r\n", AMK_IGBT_TEMP_1_ID, FLOAT, motorVal2[1].AMK_TempIGBT);
                        break;
                    case 0x287:
                        print_timestamp(ans);
                        amk_parser((uint8_t *)msg, (int)id);
                        printf("%d %d %d\r\n", AMK_STATUS_2_ID, INT, motorVal1[2].AMK_bSystemReady);
                        printf("%d %d %.3f\r\n", AMK_ACTUAL_VELOCITY_2_ID, FLOAT, motorVal1[2].AMK_ActualVelocity);
                        printf("%d %d %.3f\r\n", AMK_TORQUE_CURRENT_2_ID, FLOAT, motorVal1[2].AMK_TorqueCurrent);
                        printf("%d %d %u\r\n", AMK_VOLTAGE_2_ID, INT, motorVal1[2].AMK_Voltage);
                        printf("%d %d %.3f\r\n", AMK_CURRENT_2_ID, FLOAT, motorVal1[2].AMK_Current);
                        break;
                    case 0x288:
                        print_timestamp(ans);
                        amk_parser((uint8_t *)msg, (int)id);
                        printf("%d %d %d\r\n", AMK_STATUS_3_ID, INT, motorVal1[3].AMK_bSystemReady);
                        printf("%d %d %.3f\r\n", AMK_ACTUAL_VELOCITY_3_ID, FLOAT, motorVal1[3].AMK_ActualVelocity);
                        printf("%d %d %.3f\r\n", AMK_TORQUE_CURRENT_3_ID, FLOAT, motorVal1[3].AMK_TorqueCurrent);
                        printf("%d %d %u\r\n", AMK_VOLTAGE_3_ID, INT, motorVal1[3].AMK_Voltage);
                        printf("%d %d %.3f\r\n", AMK_CURRENT_3_ID, FLOAT, motorVal1[3].AMK_Current);
                        break;
                    case 0x289:
                        print_timestamp(ans);
                        amk_parser((uint8_t *)msg, (int)id);
                        printf("%d %d %.3f\r\n", AMK_MOTOR_TEMP_2_ID, FLOAT, motorVal2[2].AMK_TempMotor);
                        printf("%d %d %.3f\r\n", AMK_INVERTER_TEMP_2_ID, FLOAT, motorVal2[2].AMK_TempInverter);
                        printf("%d %d %.3f\r\n", AMK_IGBT_TEMP_2_ID, FLOAT, motorVal2[2].AMK_TempIGBT);
                        break;
                    case 0x28A:
                        print_timestamp(ans);
                        amk_parser((uint8_t *)msg, (int)id);
                        printf("%d %d %.3f\r\n", AMK_MOTOR_TEMP_3_ID, FLOAT, motorVal2[3].AMK_TempMotor);
                        printf("%d %d %.3f\r\n", AMK_INVERTER_TEMP_3_ID, FLOAT, motorVal2[3].AMK_TempInverter);
                        printf("%d %d %.3f\r\n", AMK_IGBT_TEMP_3_ID, FLOAT, motorVal2[3].AMK_TempIGBT);
                        break;
                    }
                }
            }
            // Break the loop if something goes wrong, i.e. if we get a status code
            // that is not canOK (taken care of above) and not canERR_NOMSG
            else if (stat != canERR_NOMSG)
            {
                Check("canRead", stat);
                break;
            }*/
        }
        std::cout << "Going of bus and closing channel\n";
        stat = canBusOff(hnd);
        Check("canBusOff", stat);
        stat = canClose(hnd);
        Check("canClose", stat);
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    node_ptr = std::make_shared<KvaserReader>(options);
    rclcpp::spin(node_ptr);
    rclcpp::shutdown();
    return 0;
}