#include "../fixedptc.h"

typedef struct {
    int id;
    int lane;
    fixedpt velocity;
    fixedpt position;
} CLVehInt;

__constant fixedpt vehicleSize = FIXEDPT_FIVE;
__constant unsigned char arg_ind_r[9] = {0,0,2,2,4,4,6,6,6};
__constant unsigned char arg_ind_l[9] = {1,6,6,3,5,6,3,1,5};
__constant fixedpt politeness = FIXEDPT_ONE;
__constant fixedpt safe_decel = fixedpt_rconst(-3.0);
__constant fixedpt incThreshold = FIXEDPT_ONE;

fixedpt fixedpt_inline_mul(fixedpt A, fixedpt B)
{
    return (((fixedptd)A * (fixedptd)B) >> FIXEDPT_FBITS);
}

fixedpt fixedpt_inline_div(fixedpt A, fixedpt B)
{
    return (((fixedptd)A << FIXEDPT_FBITS) / (fixedptd)B);
}

CLVehInt updateAgent(CLVehInt veh, fixedpt acceleration, fixedpt dt, fixedpt ds){
    CLVehInt veh_new;
    veh_new.id = veh.id;
    veh_new.lane = veh.lane;
    acceleration = acceleration > FIXEDPT_FIVE  ? FIXEDPT_FIVE : acceleration;
    acceleration = acceleration < -FIXEDPT_FIVE ? -FIXEDPT_FIVE : acceleration;
    fixedpt dv = fixedpt_inline_mul( acceleration, dt );
    fixedpt v = veh.velocity + dv;
    v = v < 0 ? 0 : v;
    fixedpt distance = fixedpt_inline_mul( v, dt );
   
    ds = ds < 0 ? 0 : ds;
    distance = distance > fixedpt_inline_mul( ds, FIXEDPT_0_9 ) ? fixedpt_inline_mul( ds, FIXEDPT_0_9 ) : distance;
    v = distance > fixedpt_inline_mul( ds, FIXEDPT_0_9 ) ? fixedpt_inline_div( distance, dt ) : v;

    veh_new.position = veh.position + distance;
    veh_new.velocity = v;

    return veh_new;
}

fixedpt calculateIncentive(fixedpt accThisNew, fixedpt accThisOld, fixedpt politeness, fixedpt accObsNew, fixedpt accObsOld, fixedpt accFreeNew, fixedpt accFreeOld ){
    return ( accThisNew - accThisOld + fixedpt_inline_mul( politeness, ( accObsNew - accObsOld + accFreeNew - accFreeOld ) ) );
}

__kernel void sim(
        __global CLVehInt * vehs,
        __global CLVehInt * vehs_new,
        __global int * lane_length,
        __global int * lane_length_new,
        int agentCount,
        int laneCount       
        )
{
    unsigned gid = get_global_id(0);
    if (gid > agentCount-1) return;

    execute_iteration:
    {
        CLVehInt vehi = vehs[gid];
        int current_lane = vehi.lane;
        int left_lane = vehi.lane-1;
        int right_lane = vehi.lane+1;

        CLVehInt rear_left, front_left, rear_current, front_current, rear_right, front_right;
        
        rear_left.id = -1; 
        front_left.id = -1; 
        rear_current.id = -1; 
        front_current.id = -1; 
        rear_right.id = -1; 
        front_right.id = -1; 

        //bool found_current= false, found_left = (left_lane>-1)?false:true, found_right = (right_lane<laneCount)?false:true;
        
        front_current = (gid+1<agentCount)?vehs[gid+1]:front_current;
        if (front_current.lane != vehi.lane){
            front_current.id = -1;
        }
        rear_current = (gid>0)?vehs[gid-1]:rear_current;
        if (rear_current.lane != vehi.lane){
            rear_current.id = -1;
        }
        if (left_lane>-1)
        {
            int base=0;
            for (int lane_i = 0;lane_i<left_lane;lane_i++){
                base += lane_length[lane_i];
            }

            for (int index = 0;index < lane_length[left_lane];index++){
                CLVehInt neighbour = vehs[base+index];
                if (neighbour.position >= vehi.position)
                {
                    front_left = neighbour;
                    rear_left = (index>0)?vehs[base+index-1]:rear_left;
                    break;
                }
            }
            if (front_left.id == -1 && lane_length[left_lane]>0){
                rear_left = vehs[base+lane_length[left_lane]-1];
            }
        }
        
        if (right_lane<laneCount)
        {
            int base=0;
            for (int lane_i = 0;lane_i<right_lane;lane_i++){
                base += lane_length[lane_i];
            }

            for (int index = 0;index < lane_length[right_lane];index++){
                CLVehInt neighbour = vehs[base+index];
                if (neighbour.position >= vehi.position)
                {
                    front_right = neighbour;
                    rear_right = (index>0)?vehs[base+index-1]:rear_right;
                    break;
                }
            }
            if (front_right.id == -1 && lane_length[right_lane]>0){
                rear_right = vehs[base+lane_length[right_lane]-1];
            }
        }
        
        CLVehInt veh_new = vehi;

        fixedpt incentives[2];

        bool want_left = true;
        bool want_right = true;

        fixedpt args[9][3];
        fixedpt results[9]; 

        fixedpt vels[7];
        fixedpt pos[7];

        vels[0] = (rear_left.id==-1)?(fixedpt)0:rear_left.velocity;
        vels[1] = (front_left.id==-1)?(fixedpt)0:front_left.velocity;
        vels[2] = (rear_current.id==-1)?(fixedpt)0:rear_current.velocity;
        vels[3] = (front_current.id==-1)?(fixedpt)0:front_current.velocity;
        vels[4] = (rear_right.id==-1)?(fixedpt)0:rear_right.velocity;
        vels[5] = (front_right.id==-1)?(fixedpt)0:front_right.velocity;
        vels[6] = vehi.velocity;

        pos[0] = (rear_left.id==-1)?(fixedpt)0:rear_left.position;
        pos[1] = (front_left.id==-1)?(fixedpt)0:front_left.position;
        pos[2] = (rear_current.id==-1)?(fixedpt)0:rear_current.position;
        pos[3] = (front_current.id==-1)?(fixedpt)0:front_current.position;
        pos[4] = (rear_right.id==-1)?(fixedpt)0:rear_right.position;
        pos[5] = (front_right.id==-1)?(fixedpt)0:front_right.position;
        pos[6] = vehi.position;

        //printf("veh %d position %d,%f rear_left %f front_left %f rear_current %f front_current %f rear_right %f front_right %f mine %f\n",vehi.id,vehi.lane,vehi.position,pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],pos[6]);
        //printf("veh %d position %d,%f rear_left %d front_left %d rear_current %d front_current %d rear_right %d front_right %d mine %d\n",vehi.id,vehi.lane,vehi.position,rear_left.id, front_left.id, rear_current.id, front_current.id, rear_right.id, front_right.id, vehi.id);

        //printf("veh %d vels %f %f %f %f %f %f %f\n",vehi.id,vels[0],vels[1],vels[2],vels[3],vels[4],vels[5],vels[6]);

        set_args:
        for(short i = 0; i < 9; i++)
        {
            args[i][0] = vels[arg_ind_r[i]];
            args[i][1] = vels[arg_ind_l[i]] - vels[arg_ind_r[i]];
            args[i][2] = pos[arg_ind_l[i]] - pos[arg_ind_r[i]] - FIXEDPT_FIVE;
            results[i] = 0;
        }

        find_results:
        for(short i = 0; i < 9; i++)
        {
            fixedpt ds = args[i][2];
            if( (vehi.lane==0 && (i==0||i==1||i==7) ) ){
                want_left = false;
                continue;
            }
            if( (vehi.lane==laneCount-1 && (i==4||i==5||i==8) ) ){
                want_right = false;
                continue;
            }
            if( (rear_left.id==-1&&(i==0||i==1)) || (/*isTail(vehi)*/ (rear_current.id == -1) &&(i==2||i==3)) || (rear_right.id==-1&&(i==4||i==5)) ){
                continue;
            }
            if( (front_left.id==-1&&(i==0||i==7)) || /*(isHead(vehi)*/ ( (front_current.id == -1) &&(i==3||i==6) ) || ( front_right.id==-1&&(i==4||i==8) ) ){
                ds = FIXEDPT_1024;
            }

            fixedpt free_road_term, interaction_term, vRat, ss, temp;
            if(ds <= 0){
                results[i] = FIXEDPT_MINUS_5; // return something less than minimum allowed acceleration
                continue;
            }

            vRat =  fixedpt_inline_mul( args[i][0], FIXEDPT_0_0_6_2_5 );// /vdesired; // 0.02 originally
            free_road_term = FIXEDPT_1_8;

            fixedpt vRat2 =  fixedpt_inline_mul( vRat, vRat );
            vRat =  fixedpt_inline_mul( vRat2 , vRat2 );
            free_road_term = fixedpt_inline_mul( FIXEDPT_1_8, FIXEDPT_ONE - vRat );

            ss =  fixedpt_inline_mul(args[i][0], FIXEDPT_1_5) + ( fixedpt_inline_mul( args[i][0], args[i][1] ) >> 2 ) + FIXEDPT_TWO; // multiplication by 0.26 instead of division by 3.79 // UP: division by 4
            temp = fixedpt_inline_div( ss, ds );
            fixedpt temp2 = fixedpt_inline_mul(temp, temp);
            if( temp2 == FIXEDPT_MAX ){
                results[i] = FIXEDPT_MINUS_5;
                continue;
            }
            interaction_term = fixedpt_inline_mul( FIXEDPT_MINUS_1_8 , temp2 );
            results[i]= free_road_term+interaction_term;
    //             printf("%d %d\n",i,results[i]);
        }
        want_left = want_left && (results[1] >= safe_decel);
        want_right = want_right && (results[5] >= safe_decel);

        calculate_incentives:
        for(int i = 0; i < 2; i++)
        {
            incentives[i] = calculateIncentive( results[7+i], results[6], politeness, results[1 + 4*i], results[4*i], results[3], results[2] );
        }
        want_left = want_left && (incentives[0] > incThreshold);
        want_right = want_right && (incentives[1] > incThreshold);

        bool prefer_left = true;

        ////////////////
        //// ACT v3 ////
        ////////////////
        int direction = 0;
        bool left = !want_right || incentives[0] > incentives[1];
        left = left || (incentives[0] == incentives[1] && prefer_left);
        left = left && want_left;
        int lane = veh_new.lane;

        if(left){
            direction = -1;
        }
        else if(want_right){
            direction = 1;
        }
        if(direction != 0){
            CLVehInt front = (direction == 1) ? front_right : front_left;
            fixedpt result = left ? results[7] : results[8];
            fixedpt pos_ = left ? pos[1] : pos[5];
            veh_new = updateAgent(vehi, result, FIXEDPT_QUARTER, (front.id != -1) ? pos_-vehi.position-FIXEDPT_FIVE : FIXEDPT_1024);
        } else {
            veh_new = updateAgent(vehi, results[6], FIXEDPT_QUARTER, (front_current.id != -1) ?  pos[3]-vehi.position-FIXEDPT_FIVE : FIXEDPT_1024);
        }
        atomic_dec(&lane_length_new[veh_new.lane]);
        veh_new.lane += direction;
        atomic_inc(&lane_length_new[veh_new.lane]);
        vehs_new[gid] = veh_new;
    }

}
bool compare(int lane1, int lane2, fixedpt p1, fixedpt p2) {
    if (lane1 > lane2) return true;
    else if (lane1 == lane2){
        if (p1>p2) return true;
    }
    return false;
}
__kernel void sort(__global CLVehInt * vehs_new, __global CLVehInt * vehs, int inc, int dir) {
    //printf("here");
    int i = get_global_id(0);
    int j = i ^ inc;
    if (i>j) return;
    CLVehInt ag1 = vehs_new[i];
    CLVehInt ag2 = vehs_new[j];
    bool smaller = compare(ag1.lane,ag2.lane,ag1.position, ag2.position);
    bool swap = smaller ^ (j < i) ^ ((dir & i) != 0);
    if (swap) {
        CLVehInt tmp = vehs_new[i];
        vehs_new[i] = vehs_new[j];
        vehs_new[j] = tmp;
        tmp = vehs[i];
        vehs[i] = vehs[j];
        vehs[j] = tmp;
    }
}
__kernel void conflict_resolver(__global CLVehInt * vehs_new, __global CLVehInt * vehs, __global int * lane_length_new, __global int * lane_length, int agentCount, __global bool *isConflict) {
    int gid = get_global_id(0);
    if (gid > agentCount-2) return;

    CLVehInt vehi = vehs_new[gid];
    CLVehInt vehi_next = vehs_new[gid+1];
    if (vehi_next.lane!=vehi.lane) return;
    if (vehi_next.position > vehi.position && vehi_next.position < vehi.position+vehicleSize) {
        //printf("Thread %d Agent %d rolled back my %d other %d\n", gid, vehs[gid].id,vehi.position,vehi_next.position);
        if (vehi.lane != vehs[gid].lane){
            atomic_dec(&lane_length_new[vehs_new[gid].lane]);
            vehs_new[gid].lane = vehs[gid].lane;
            atomic_inc(&lane_length_new[vehs_new[gid].lane]);
        } else {
            vehs_new[gid].position = vehi_next.position - FIXEDPT_5_1;
        }
        *isConflict = true;
    }
    //for (int index = gid+1;index < agentCount;index++){
    //    CLVehInt neighbour = vehs_new[index];
     //   if (neighbour.lane == vehi.lane) {
    //        if (neighbour.position > vehi.position && neighbour.position < vehi.position+vehicleSize) {
                //printf("Thread %d Agent %d rolled back my %f other %f\n", gid, vehs[gid].id,vehi.position,neighbour.position);
                //if (vehi.lane != vehs[gid].lane)
                //{
                //    atomic_dec(&lane_length_new[vehi.lane]);
                //    vehs_new[gid].lane = vehs[gid].lane;
                //    atomic_inc(&lane_length_new[vehi.lane]);
                //} else {
                //    vehs_new[gid].position = neighbour.position - 5.1;
                //}
    //            *isConflict = true;
    //            break;
    //        }
    //    }
   // }

}

__kernel void update(        
        __global CLVehInt * vehs,
        __global CLVehInt * vehs_new,
        __global int * lane_length,
        __global int * lane_length_new,
        int agentCount,
        int laneCount)     
{
    int gid = get_global_id(0);
    if (gid > agentCount) return;
    //printf("lane %d %d\n",agentCount,laneCount);

    if (gid == 0){
        for (int pp=0;pp<laneCount;pp++){
            //printf("lane %d old %d\n",pp,lane_length[pp] );
            lane_length[pp] = lane_length_new[pp];
            //printf("lane %d new %d\n",pp,lane_length[pp] );
        }
    } 

    vehs[gid] = vehs_new[gid];
    vehs_new[gid].id = -1;

    //printf("Thread %d Agent %d %d %f\n", gid, vehs[gid].id, vehs[gid].lane, vehs[gid].position);
}