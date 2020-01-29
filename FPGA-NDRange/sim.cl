#include "fixedptc.h"

typedef struct {
    int id;
    int lane;
    fixedpt velocity;
    fixedpt position;
} CLVehInt;

#define MAX_VEHICLES_PER_LANE 16384
#define LANE_MAX 4
#define MAX_AGENT_SIZE MAX_VEHICLES_PER_LANE * LANE_MAX

__constant fixedpt vehicleSize = FIXEDPT_FIVE;
__constant unsigned char arg_ind_r[9] = {0,0,2,2,4,4,6,6,6};
__constant unsigned char arg_ind_l[9] = {1,6,6,3,5,6,3,1,5};
__constant fixedpt politeness = FIXEDPT_ONE;
__constant fixedpt safe_decel = fixedpt_rconst(-3.0);
__constant fixedpt incThreshold = FIXEDPT_ONE;

__attribute__((always_inline))
fixedpt_inline_mul(fixedpt A, fixedpt B)
{
    return (((fixedptd)A * (fixedptd)B) >> FIXEDPT_FBITS);
}

/* Divides two fixedpt numbers, returns the result. */
__attribute__((always_inline))
fixedpt_inline_div(fixedpt A, fixedpt B)
{
    return (((fixedptd)A << FIXEDPT_FBITS) / (fixedptd)B);
}

__attribute__((always_inline))
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

__attribute__((always_inline))
fixedpt calculateIncentive(fixedpt accThisNew, fixedpt accThisOld, fixedpt politeness, fixedpt accObsNew, fixedpt accObsOld, fixedpt accFreeNew, fixedpt accFreeOld ){
    return ( accThisNew - accThisOld + fixedpt_inline_mul( politeness, ( accObsNew - accObsOld + accFreeNew - accFreeOld ) ) );
}


__attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void init(
        __global CLVehInt * vehs,
        __global CLVehInt * vehs_min_1,
        __global CLVehInt * vehs_min_2,
        __global int * current_index,
        __global int * current_index_new
        )
{
    __attribute__((opencl_unroll_hint(5)))
    for(int i = 0; i < LANE_MAX+1; i++){
        current_index[i]=0;
        current_index_new[i]=0;
    }
    __attribute__((opencl_unroll_hint(4)))
    for(int p = 0; p < LANE_MAX; p++){
        vehs_min_1[p + 1] = vehs[p*MAX_VEHICLES_PER_LANE];
        vehs_min_2[p + 1].id = -1;
    }
    vehs_min_1[0].id = -1;
    vehs_min_2[0].id = -1;
}

__attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void sim(
        __global CLVehInt * vehs,
        __global CLVehInt * vehs_new,
        __global int * lane_length,
        __global int * lane_length_new,
        __global CLVehInt * vehs_min_1,
        __global CLVehInt * vehs_min_2,
        __global int * current_index,
        __global int * current_index_new,
        short agentCount,
        short laneCount
        )
{
    __attribute__((xcl_pipeline_workitems)) {
        int min_displ_lane = -1;
        CLVehInt vehi;
        vehi.position = FIXEDPT_MAX;

        CLVehInt rear_left, front_left, rear_current, front_current, rear_right, front_right;
        rear_left.id = -1;
        front_left.id = -1;
        rear_current.id = -1;
        front_current.id = -1;
        rear_right.id = -1;
        front_right.id = -1;

        //__attribute__((opencl_unroll_hint(4)))
        find_min_displ_veh:for(int j = 1; j < LANE_MAX + 1; j++){
            CLVehInt cur_veh = vehs_min_1[j];
            if(cur_veh.position < vehi.position){
                vehi = cur_veh;
                min_displ_lane = j;
            }
        }

        rear_current = vehs_min_2[min_displ_lane];
        vehs_min_2[min_displ_lane] = vehi;
        current_index[min_displ_lane] += 1;
        vehs_min_1[min_displ_lane] = vehs[MAX_VEHICLES_PER_LANE * (min_displ_lane - 1) + current_index[min_displ_lane]];
        front_current = vehs_min_1[min_displ_lane];
        front_right = vehs_min_1[min_displ_lane + 1];
        front_left = vehs_min_1[min_displ_lane - 1];
        rear_left = vehs_min_2[min_displ_lane - 1];
        rear_right = vehs_min_2[min_displ_lane + 1];

        CLVehInt veh_new = vehi;

        fixedpt incentives[2];

        bool want_left = true;
        bool want_right = true;

        fixedpt args[9][3] __attribute__((xcl_array_partition(complete, 1)));
        fixedpt results[9] ;

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

//       printf("pos %d %d %d %d %d %d %d\n",pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],pos[6]);
//       printf("vels %d %d %d %d %d %d %d\n",vels[0],vels[1],vels[2],vels[3],vels[4],vels[5],vels[6]);
//       printf("veh %d position %d,%f rear_left %d front_left %d rear_current %d front_current %d rear_right %d front_right %d mine %d\n",vehi.id,vehi.lane,vehi.position,rear_left.id, front_left.id, rear_current.id, front_current.id, rear_right.id, front_right.id, vehi.id);
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
        for(short i = 0; i < 2; i++)
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
        int c_index = current_index_new[lane + 1 + direction];
        if(direction != 0){
            if(c_index <= 0 || veh_new.position > vehs_new[ (lane + direction) * MAX_VEHICLES_PER_LANE + c_index - 1].position + vehicleSize){
                current_index_new[lane + 1 + direction]++;
                CLVehInt front = (direction == 1) ? front_right : front_left;
                fixedpt result = left ? results[7] : results[8];
                fixedpt pos_ = left ? pos[1] : pos[5];
                veh_new = updateAgent(vehi, result, FIXEDPT_QUARTER, (front.id != -1) ? pos_-vehi.position-FIXEDPT_FIVE : FIXEDPT_1024);
            }else{
                c_index = current_index_new[lane + 1];
                direction = 0;
            }
        }
        if(direction == 0){
            current_index_new[lane+1] += 1;
            veh_new = updateAgent(vehi, results[6], FIXEDPT_QUARTER, (front_current.id != -1) ?  pos[3]-vehi.position-FIXEDPT_FIVE : FIXEDPT_1024);
        }
        lane_length_new[veh_new.lane]--;
        veh_new.lane += direction;
        lane_length_new[veh_new.lane]++;
        vehs_new[(lane + direction)*MAX_VEHICLES_PER_LANE+c_index] = veh_new;
    }
}

__attribute__ ((reqd_work_group_size(1, 1, 1)))
__kernel void update(
        __global CLVehInt * vehs,
        __global CLVehInt * vehs_new,
        __global int * lane_length,
        __global int * lane_length_new
        )
{
    __attribute__((opencl_unroll_hint(4)))
    for(int i = 0; i < LANE_MAX; i++){
        lane_length[i] = lane_length_new[i];
    }

    __attribute__((xcl_pipeline_loop))
    update_vehs1:
    for(int i = 0; i < MAX_AGENT_SIZE; i++){
        vehs[i] = vehs_new[i];
    }

    __attribute__((xcl_pipeline_loop))
    update_vehs2:
    for(int i = 0; i < MAX_AGENT_SIZE; i++){
        vehs_new[i].id = -1;
        vehs_new[i].position = FIXEDPT_MAX;
        vehs_new[i].velocity = 0;
        vehs_new[i].lane = 0;
    }
}
