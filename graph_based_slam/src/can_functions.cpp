#include <can_stuff/can_functions.h>

// bmslv start *************************************************************
void bms_lv_parser(uint8_t *bmslv_values, int id)
{
    int i = 0;
    if (id == 0x55) // Set to 55 instead of MSG_ID_BMS_LV_1 for testing
    {
        for (i = 0; i < 8; i = i + 2)
        {
            bms_lv[i / 2] = (bmslv_values[i] & 0xff) | ((bmslv_values[i + 1] & 0xff) << 8);
            bms_lv[i / 2] *= 0.0001;
        }
    }
    else if (id == 0x56)
    {
        for (i = 0; i < 8; i = i + 2)
        {
            bms_lv[(i + 8) / 2] = (bmslv_values[i] & 0xff) | ((bmslv_values[i + 1] & 0xff) << 8);
            bms_lv[(i + 8) / 2] *= 0.0001;
        }
        bms_lv[6] = convert_temp_lv(bms_lv[6]);
        bms_lv[7] = convert_temp_lv(bms_lv[7]);
    }
}

float convert_temp_lv(float cell_volt)
{
    float x = ((log((5 - cell_volt) / cell_volt)) / 3435);
    x = 1 / ((1 / 298.15) - x);
    return x - 273.15;
}
// bmslv end ******************************************************************************************************************************

// IMU start *********************************************************************************************************************************

void imu_parser(uint8_t imu_values[], int id) // passiamo accelerations[] e omegas[] perche usiamo sti array per printare in seriale
{
    // Uint16 imu_msg_temp[8];
    uint16_t imu_values_16[8];
    convert8_to_16(imu_values_16, imu_values, 8);

    uint32_t aux_1 = 0;
    uint32_t aux_2 = 0;

    aux_1 |= ((int32_t)(imu_values_16[3]) << 24);
    aux_1 |= ((int32_t)(imu_values_16[2]) << 16);
    aux_1 |= ((int32_t)(imu_values_16[1]) << 8);
    aux_1 |= ((int32_t)(imu_values_16[0]) << 0);

    aux_2 |= ((int32_t)(imu_values_16[7]) << 24);
    aux_2 |= ((int32_t)(imu_values_16[6]) << 16);
    aux_2 |= ((int32_t)(imu_values_16[5]) << 8);
    aux_2 |= ((int32_t)(imu_values_16[4]) << 0);

    float float_aux_1;
    float float_aux_2;

    union
    {
        uint32_t u;
        float f;
    } temp;
    temp.u = aux_1;
    float_aux_1 = temp.f;
    temp.u = aux_2;
    float_aux_2 = temp.f;
    // Serial.println(float_aux_1);
    // Serial.println(float_aux_2);

    if (id == 0x60) // MSG_ID_IMU_1
    {
        accelerations[X] = accelerations[X] - 0.5 * (accelerations[X] - float_aux_1);
        accelerations[Y] = accelerations[Y] - 0.5 * (accelerations[Y] - (-uint32_to_float(aux_2)));
    }
    else if (id == 0x61) // MSG_ID_IMU_2
    {
        accelerations[Z] = accelerations[Z] - 0.5 * (accelerations[Z] - (uint32_to_float(aux_1)));
        omegas[X] = uint32_to_float(aux_2);
    }
    else if (id == 0x62) // MSG_ID_IMU_3
    {
        omegas[Y] = uint32_to_float(aux_1);
        omegas[Z] = uint32_to_float(aux_2);
    }
}

float uint32_to_float(uint32_t u)
{
    union
    {
        uint32_t u;
        float f;
    } temp;
    temp.u = u;

    return temp.f;
}
// IMU end *******************************************************************************************************************************

// SMU start *******************************************************************************************************************************

void smu_parser(uint8_t smu_values[], int id)
{

    uint64_t aux = 0;
    int i;
    int NUM_SMU_TEMP = 5;
    int NUM_SMU_SUSP = 4;

    for (i = 7; i >= 0; i--)
    {
        aux = aux << 8;
        aux |= (0xFF & smu_values[i]);
    }

    if (id == 0x100)
    {                                      // MSG_ID_SMU_TEMPERATURES
        for (i = 0; i < NUM_SMU_TEMP; i++) // NUM_SMU_TEMP = 5;
        {
            temperatures[i] = ConvertTempToKelvin(0x3FF & aux);
            aux >>= 10;
        }
    }
    else if (id == 0x101)
    {                                      // MSG_ID_SMU_SUSPENSIONS
        for (i = 0; i < NUM_SMU_SUSP; i++) // NUM_SMU_SUSP = 4;
        {
            suspensions[i] = (uint32_t)(0x3FF & aux);
            aux >>= 10;
        }
    }
}
float ConvertTempToKelvin(int adc_read)
{
    float div = (float)((float)adc_read / 1023);
    return (3977 * 298.15) / (3977 - (298.25 * log((1 - div) / div)));
}

// SMU end *******************************************************************************************************************************

// BMS Voltage start ****************************************************************************************************************************

void bms_voltage_parser(uint8_t bms_values[])
{
    uint16_t tmp = 0; // cambiare unint16 con uint8 (come ha spiegato edo ex: casting )
    uint16_t bms_values_16[6];
    convert8_to_16(bms_values_16, bms_values, 6);

    tmp = (bms_values_16[0] | (bms_values_16[1] << 8));
    max_bms_voltage = convertBMSvoltage(tmp);
    tmp = (bms_values_16[2] | (bms_values_16[3] << 8));
    min_bms_voltage = convertBMSvoltage(tmp);
    tmp = (bms_values_16[4] | (bms_values_16[5] << 8));
    mean_bms_voltage = convertBMSvoltage(tmp);
}

void bms_temp_parser(uint8_t bms_values[])
{
    uint16_t tmp = 0;
    uint16_t bms_values_16[7];
    convert8_to_16(bms_values_16, bms_values, 7);

    tmp = (bms_values_16[0] | (bms_values_16[1] << 8));
    max_bms_temp = convertBMStemp(tmp);
    tmp = (bms_values_16[2] | (bms_values_16[3] << 8));
    min_bms_temp = convertBMStemp(tmp);
    tmp = (bms_values_16[4] | (bms_values_16[5] << 8));
    mean_bms_temp = convertBMStemp(tmp);
    max_temp_nslave = (int)bms_values_16[6];
}

void convert8_to_16(uint16_t destination[], uint8_t input[], int input_length)
{
    for (int i = 0; i < input_length; i++)
    {
        destination[i] = input[i];
    }
}

float convertBMSvoltage(uint16_t voltage)
{
    return (3000 + (1.2 * voltage));
}

float convertBMStemp(uint16_t temp)
{
    return 27 * (-1) + 0.2 * temp;
}

// BMS Voltage end ****************************************************************************************************************************

// LEM start ****************************************************************************************************************************

void lem_parser(unsigned char lem_values[])
{
    unsigned long int reassembled_data = 0;

    uint16_t lem_values_16[8];
    convert8_to_16(lem_values_16, lem_values, 8);

    uint16_t tmp = lem_values[0];
    tmp ^= 1 << 7;
    reassembled_data |= ((uint32_t)(tmp) << 24);
    reassembled_data |= ((uint32_t)(lem_values[1]) << 16);
    reassembled_data |= ((uint32_t)(lem_values[2]) << 8);
    reassembled_data |= ((uint32_t)(lem_values[3]) << 0);
    lem_current = (int32_t)(reassembled_data) / 1000.0;
}

// LEM end ****************************************************************************************************************************

// AMK start ****************************************************************************************************************************
void amk_parser(uint8_t *RXB_AmkVal_Data, int id)
{

    if (id == 0x283 || id == 0x284 || id == 0x287 || id == 0x288)
    {
        read_AMK_Values1((uint8_t *)RXB_AmkVal_Data, getMotorIndex(id));
    }
    else if (id == 0x285 || id == 0x286 || id == 0x289 || id == 0x28A)
    {
        read_AMK_Values2((uint8_t *)RXB_AmkVal_Data, getMotorIndex(id));
    }
}

void read_AMK_Values1(uint8_t canMsg[], int indexMotor)
{

    float AMK_CURR_SCALE = (107.2f / 16384.0f);

    if (indexMotor != 0 && indexMotor != 1 && indexMotor != 2 && indexMotor != 3)
        return;

    motorVal1[indexMotor].AMK_bSystemReady = getBit(canMsg[1], 0);
    motorVal1[indexMotor].AMK_bError = getBit(canMsg[1], 1);
    motorVal1[indexMotor].AMK_bWarn = getBit(canMsg[1], 2);
    motorVal1[indexMotor].AMK_bQuitDcOn = getBit(canMsg[1], 3);
    motorVal1[indexMotor].AMK_bDcOn = getBit(canMsg[1], 4);
    motorVal1[indexMotor].AMK_bQuitInverterOn = getBit(canMsg[1], 5);
    motorVal1[indexMotor].AMK_bInverterOn = getBit(canMsg[1], 6);
    motorVal1[indexMotor].AMK_bDerating = getBit(canMsg[1], 7);

    motorVal1[indexMotor].AMK_ActualVelocity = unsigned_to_signed((canMsg[2] | canMsg[3] << 8));
    motorVal1[indexMotor].AMK_TorqueCurrent = (canMsg[4] | canMsg[5] << 8) * AMK_CURR_SCALE;
    motorVal1[indexMotor].AMK_Voltage = (canMsg[6] | canMsg[7] << 8);

    // see PDK
    float torqueCurr = motorVal1[indexMotor].AMK_TorqueCurrent;
    // float magnCurr = motorVal1[indexMotor].AMK_MagnetizingCurrent;
    // float curr = 0.243 * torqueCurr + 0.0009 * torqueCurr * magnCurr;
    motorVal1[indexMotor].AMK_Current = torqueCurr;
}

void read_AMK_Values2(uint8_t canMsg[], int indexMotor)
{

    if (indexMotor != 0 && indexMotor != 1 && indexMotor != 2 && indexMotor != 3)
        return;

    motorVal2[indexMotor].AMK_TempMotor = (canMsg[0] | canMsg[1] << 8) / 10;
    motorVal2[indexMotor].AMK_TempInverter = (canMsg[2] | canMsg[3] << 8) / 10;
    motorVal2[indexMotor].AMK_TempIGBT = (canMsg[6] | canMsg[7] << 8) / 10;
    motorVal2[indexMotor].AMK_ErrorInfo = (canMsg[4] | canMsg[5] << 8);
}

int getMotorIndex(int id)
{
    int index = -1;
    int i;
    for (i = 0; i < 4; i++)
    {
        if (id == AMK_VAL_1_IDS[i] || id == AMK_VAL_2_IDS[i])
            index = i;
    }

    return index;
}

uint8_t getBit(char c, int bitNumber)
{
    return (c & (1 << bitNumber)) != 0;
}

int16_t unsigned_to_signed(uint16_t value)
{
    union
    {
        uint16_t u;
        int16_t s;
    } temp;
    temp.u = value;
    return temp.s;
}

// AMK end ****************************************************************************************************************************

void total_tension_parser(uint8_t *payload)
{
    uint16_t tmp[2];
    convert8_to_16(tmp, payload, 2);
    battery_pack_tension = (tmp[0] | (tmp[1] << 8));
}

//******************************************************************************************************

void general_info_parser(uint8_t *payload)
{

    uint16_t tmp = 0;
    uint16_t payload_16[8];
    convert8_to_16(payload_16, payload, 8);

    throttle = (payload_16[0] | (payload_16[1] << 8));
    brake = (payload_16[2] | (payload_16[3] << 8));
    steering = (payload_16[4] | (payload_16[5] << 8));
    actualVelRPM = (payload_16[6] | (payload_16[7] << 8));
}