 #pragma once
#include <ap_fixed.h>
typedef ap_fixed<32,24, AP_RND, AP_SAT_SYM> fixed;

// #include "constants.h"
#include "fixedptc.h"
#include <vector>
// https://sourceforge.net/projects/fixedptc/
#define MAX_VEHICLES_PER_LANE 16384
#define LANE_MAX 4
//#define dt FIXEDPT_QUARTER

//#define LANES_(i) &lanes[(i) * MAX_VEHICLES_PER_LANE]
//#define LANES(i, j) lanes[(i) * MAX_VEHICLES_PER_LANE + (j)]
//#define LANES_GLOBAL(i, j) lanes_global[(i) * MAX_VEHICLES_PER_LANE + (j)]
#define MAX_AGENT_SIZE MAX_VEHICLES_PER_LANE * LANE_MAX
//#define REMOVER(i, j) remover[(i) * MAX_VEHICLES_PER_LANE + (j)]
//#define LANE_AGENT_COUNT(i, j) laneAgentCount[LANE_MAX * (j) + i] // i: lane, j: state

// typedef struct{
//     short id;
//     short lane;
//     short front, back, left_front, left_back, right_front, right_back;
//     fixedpt velocity;
//     fixedpt position;
//     short pad[6];
// }CLVehicle;

//typedef struct{
//  int id, lane, velocity, position;
//    int front_current, rear_current, front_left, rear_left, front_right, rear_right;
//    int pad0, pad1, pad2, pad3, pad4, pad5;
//}CLVehInt;

class CLVehInt{
public:
    int id, lane;
    fixed velocity, position;
//  int front_current, rear_current, front_left, rear_left, front_right, rear_right;
//  int pad0, pad1, pad2, pad3, pad4, pad5;
    CLVehInt(){
        id = -1;
        lane = 0;
        velocity = 0;
        position = INT_MAX;
    }
};

// typedef struct{
//     int front,back,left_front,left_back,right_front,right_back;
//     // int pad[2];
// }ptr;

#if(C!=1)
typedef struct{
    short id;
    short front, back, left_front, left_back, right_front, right_back;
    float velocity;
    float position;
}CLVehicleFloat;
#endif

fixed idm(fixed v, fixed dVFront, int vdesired, fixed ds){
    fixed free_road_term, interaction_term, vRat, ss, temp;
    if(ds <= 0) return fixed(-5); // return something less than minimum allowed acceleration
    // if (v<vdesired){
    vRat = v * fixed(0.0625);// /vdesired; // 0.02 originally
    free_road_term = fixed(1.8);
    // } else {
        // vRat = vdesired/v;
        // free_road_term = -2.0f;
    // }
    fixedpt vRat2 = vRat * vRat;
    vRat = vRat2 * vRat2;
    free_road_term = free_road_term * ( 1 - vRat );

    ss = v * (fixed)1.5 + ( ( v * dVFront ) >> 2 ) + (fixed)2; // multiplication by 0.26 instead of division by 3.79 // UP: division by 4
    temp = ss / ds;
    fixed temp2 = temp * temp;
    if(temp2 == INT_MAX) return -5;
    interaction_term = (fixed)-1.8 * temp2;
    return free_road_term + interaction_term;
}



bool updateAgent(CLVehInt& veh, CLVehInt& veh_new, fixed acceleration, fixed dt, fixed ds){
    veh_new.id = veh.id;
    veh_new.lane = veh.lane;
    acceleration = acceleration > (fixed)5 ? (fixed)5 : acceleration;
    acceleration = acceleration < (fixed)-5 ? (fixed)-5 : acceleration;
    fixed dv = acceleration * dt;
    fixed v = veh.velocity + dv;
    v = v < (fixed)0 ? (fixed)0 : v;
    fixed distance = v * dt;

    ds = ds < (fixed)0 ? (fixed)0 : ds;
//    distance = fixed(distance > ( ds * fixed(0.9) ) ? ( ds * fixed(0.9) ) : distance);
    if(distance > ds * fixed(0.9)){
        distance = ds * fixed(0.9);
        v = distance / dt;
    }
//    v = distance > ( ds * (fixed)0.9 ) ? ( distance / dt ) : v;

    veh_new.position = veh.position + distance;
    veh_new.velocity = v;

    return true;
}


fixed calculateIncentive(fixed accThisNew, fixed accThisOld, fixed politeness, fixed accObsNew, fixed accObsOld, fixed accFreeNew, fixed accFreeOld ){
//    return ( accThisNew - accThisOld + fixedpt_xmul( politeness, ( accObsNew - accObsOld + accFreeNew - accFreeOld ) ) );
    fixed a = accObsNew - accObsOld + accFreeNew - accFreeOld;
    fixed b = politeness * a + accThisNew - accThisOld;
//  b = b + accThisNew;
//  b = b - accThisOld;
    return b;
}