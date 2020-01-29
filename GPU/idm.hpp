 #pragma once

// https://sourceforge.net/projects/fixedptc/
#include "fixedptc.h"

#include <vector>
#define MAX_VEHICLES_PER_LANE 16384
#define LANE_MAX 4
#define MAX_AGENT_SIZE MAX_VEHICLES_PER_LANE * LANE_MAX

class CLVehInt{
public:
    int id, lane;
    fixedpt velocity, position;
    CLVehInt(){
        id = -1;
        lane = 0;
        velocity = 0;
        position = FIXEDPT_MAX;
    }
};


fixedpt idm(fixedpt v, fixedpt dVFront, int vdesired, fixedpt ds){
    fixedpt free_road_term, interaction_term, vRat, ss, temp;
    if(ds <= 0) return fixedpt_rconst(-5); // return something less than minimum allowed acceleration
    // if (v<vdesired){
        vRat = fixedpt_xmul( v, fixedpt_rconst(0.0625) );// /vdesired; // 0.02 originally
        free_road_term = FIXEDPT_1_8;
    // } else {
        // vRat = vdesired/v;
        // free_road_term = -2.0f;
    // }
    fixedpt vRat2 = fixedpt_mul( vRat, vRat );
    vRat = fixedpt_mul( vRat2 , vRat2 );
    free_road_term = fixedpt_xmul( free_road_term, FIXEDPT_ONE - vRat );

    ss = fixedpt_mul( v, FIXEDPT_1_5 ) + ( fixedpt_mul( v, dVFront ) >> 2 ) + FIXEDPT_TWO; // multiplication by 0.26 instead of division by 3.79 // UP: division by 4
    temp = fixedpt_div( ss , ds );
    fixedpt temp2 = fixedpt_mul(temp, temp);
    if(temp2 == FIXEDPT_MAX) return -5;
    interaction_term = -fixedpt_mul( FIXEDPT_1_8 , temp2 );
    return free_road_term + interaction_term;
}


bool updateAgent(CLVehInt& veh, CLVehInt& veh_new, fixedpt acceleration, fixedpt dt, fixedpt ds){
    veh_new.id = veh.id;
    veh_new.lane = veh.lane;
    acceleration = acceleration > FIXEDPT_FIVE  ? FIXEDPT_FIVE : acceleration;
    acceleration = acceleration < -FIXEDPT_FIVE ? -FIXEDPT_FIVE : acceleration;
    fixedpt dv = fixedpt_mul( acceleration, dt );
    fixedpt v = veh.velocity + dv;
    v = v < 0 ? 0 : v;
    fixedpt distance = fixedpt_xmul( v, dt );
   
    ds = ds < 0 ? 0 : ds;
    distance = distance > fixedpt_xmul( ds, fixedpt_rconst(0.9) ) ? fixedpt_xmul( ds, fixedpt_rconst(0.9) ) : distance;
    v = distance > fixedpt_xmul( ds, fixedpt_rconst(0.9) ) ? fixedpt_div( distance, dt ) : v;

    veh_new.position = veh.position + distance;
    veh_new.velocity = v;

    return true;
}


fixedpt calculateIncentive(fixedpt accThisNew, fixedpt accThisOld, fixedpt politeness, fixedpt accObsNew, fixedpt accObsOld, fixedpt accFreeNew, fixedpt accFreeOld ){
    return ( accThisNew - accThisOld + fixedpt_xmul( politeness, ( accObsNew - accObsOld + accFreeNew - accFreeOld ) ) );
}