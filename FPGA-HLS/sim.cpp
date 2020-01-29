#include <ap_fixed.h>
#include <string>
#include "idm.hpp"

fixed vehicleSize = 5;

const unsigned char arg_ind_r[9] = {0,0,2,2,4,4,6,6,6};
const unsigned char arg_ind_l[9] = {1,6,6,3,5,6,3,1,5};

//__attribute__((max_global_work_dim(0)))
extern "C" {


void find_min_displ_veh(const CLVehInt vehs_min[], CLVehInt & veh_min, int & lane_min, int current_index[]){
    #pragma HLS inline
    find_min:
    for(int j = 1; j < LANE_MAX + 1; j++){
        CLVehInt cur_veh = vehs_min[j];
        if(cur_veh.position < veh_min.position){
            veh_min = cur_veh;
            lane_min = j;
        }
    }
}

void sim(
        CLVehInt * vehs_global,
        int * lane_length_global,
        short agentCount,
        short laneCount,
        short iteration
        )
{
#pragma HLS INTERFACE m_axi port=vehs_global offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=vehs_global  bundle=control
#pragma HLS INTERFACE m_axi port=lane_length_global offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=lane_length_global bundle=control
#pragma HLS INTERFACE s_axilite port=agentCount bundle=control
#pragma HLS INTERFACE s_axilite port=laneCount bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=iteration bundle=control

#pragma HLS data_pack variable=vehs_global

    CLVehInt vehs_new[MAX_AGENT_SIZE];
    CLVehInt vehs[MAX_AGENT_SIZE];
    CLVehInt veh_null;

#pragma HLS array_partition variable=vehs block factor=8
#pragma HLS array_partition variable=vehs_new block factor=8

#pragma HLS data_pack variable=vehs
#pragma HLS data_pack variable=vehs_new

    int current_index[LANE_MAX + 2];
    int current_index_new[LANE_MAX + 2];
    int lane_length[LANE_MAX];

    int lane_length_new[LANE_MAX];
#pragma HLS array_partition variable=lane_length_new
#pragma HLS array_partition variable=lane_length
#pragma HLS array_partition variable=current_index
#pragma HLS array_partition variable=current_index_new

    read_lane_length:
    for(int i = 0; i < LANE_MAX; i++){
    #pragma HLS unroll
        lane_length[i] = lane_length_global[i];
        current_index[i]=0;
        current_index_new[i]=0;
    }
    current_index_init:
    for(int i = LANE_MAX; i < LANE_MAX + 2; i++){
    #pragma HLS unroll
        current_index[i]=0;
        current_index_new[i]=0;
    }
    read_lane_length_2:
    for(int i = 0; i < LANE_MAX; i++){
    #pragma HLS unroll
        lane_length_new[i] = lane_length_global[i];
    }

    read_vehicles:
    for(int i = 0; i < MAX_AGENT_SIZE; i++){
    #pragma HLS pipeline II=1
        vehs[i] = vehs_global[i];
    }

    fixed politeness = 1; //0.1
    fixed safe_decel = -3;//fixedpt_rconst(-3.0);
    fixed incThreshold = 1;//fixedpt_rconst(1);

    execute_simulation:
    for(int it = 0; it < iteration; it++){
    #pragma HLS inline region
        CLVehInt vehs_min_1[LANE_MAX + 2];
        CLVehInt vehs_min_2[LANE_MAX + 2];
        #pragma HLS array_partition variable=vehs_min_1
        #pragma HLS array_partition variable=vehs_min_2

        for(int p = 0; p < LANE_MAX; p++){
        #pragma HLS unroll
            vehs_min_1[p + 1] = vehs[p*MAX_VEHICLES_PER_LANE];
        }

        execute_iteration:
        for(int index = 0; index < agentCount; index++){
        #pragma HLS pipeline II=1

            int min_displ_lane = -1;

            CLVehInt vehi;
            CLVehInt rear_left, front_left, rear_current, front_current, rear_right, front_right;

            find_min_displ_veh(vehs_min_1, vehi, min_displ_lane, current_index);

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

            fixed incentives[2];

            bool want_left = true;
            bool want_right = true;

            fixed args[9][3];
            fixed results[9];
        #pragma HLS array partition variable=results
            fixed vels[7];
            fixed pos[7];

            vels[0] = (rear_left.id==-1)?(fixed)0:rear_left.velocity;
            vels[1] = (front_left.id==-1)?(fixed)0:front_left.velocity;
            vels[2] = (rear_current.id==-1)?(fixed)0:rear_current.velocity;
            vels[3] = (front_current.id==-1)?(fixed)0:front_current.velocity;
            vels[4] = (rear_right.id==-1)?(fixed)0:rear_right.velocity;
            vels[5] = (front_right.id==-1)?(fixed)0:front_right.velocity;
            vels[6] = vehi.velocity;

            pos[0] = (rear_left.id==-1)?(fixed)0:rear_left.position;
            pos[1] = (front_left.id==-1)?(fixed)0:front_left.position;
            pos[2] = (rear_current.id==-1)?(fixed)0:rear_current.position;
            pos[3] = (front_current.id==-1)?(fixed)0:front_current.position;
            pos[4] = (rear_right.id==-1)?(fixed)0:rear_right.position;
            pos[5] = (front_right.id==-1)?(fixed)0:front_right.position;
            pos[6] = vehi.position;

    //         printf("pos %d %d %d %d %d %d %d\n",pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],pos[6]);
    //         printf("vels %d %d %d %d %d %d %d\n",vels[0],vels[1],vels[2],vels[3],vels[4],vels[5],vels[6]);

            set_args:
            for(short i = 0; i < 9; i++)
            {
            #pragma HLS unroll
                args[i][0] = vels[arg_ind_r[i]];
                args[i][1] = vels[arg_ind_l[i]] - vels[arg_ind_r[i]];
                args[i][2] = pos[arg_ind_l[i]] - pos[arg_ind_r[i]] - fixed(5);
                results[i] = 0;
            }

            find_results:
            for(short i = 0; i < 9; i++)
            {
            #pragma HLS unroll
                fixed ds = args[i][2];
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
                    ds = (fixed)1024;
                }

                fixed free_road_term, interaction_term, vRat, ss, temp;
                if(ds <= 0){
                    results[i] = fixed(-5); // return something less than minimum allowed acceleration
                    continue;
                }

                vRat = args[i][0] * fixed(0.0625);// /vdesired; // 0.02 originally
                free_road_term = fixed(1.8);

                fixed vRat2 = vRat * vRat;
                vRat = vRat2 * vRat2;
                free_road_term = free_road_term * ( 1 - vRat );

                ss =  args[i][0] * (fixed)1.5 + ( ( args[i][0] * args[i][1] ) >> 2 ) + (fixed)2; // multiplication by 0.26 instead of division by 3.79 // UP: division by 4
                temp = ss / ds;
                fixed temp2 = temp * temp;
                if(temp2 == INT_MAX){
                    results[i]=fixed(-5);
                    continue;
                }
                interaction_term = (fixed)-1.8 * temp2;
                results[i]=free_road_term + interaction_term;
    //             printf("%d %d\n",i,results[i]);
            }

            want_left = want_left && (results[1] >= safe_decel);
            want_right = want_right && (results[5] >= safe_decel);

            calculate_incentives:
            for(short i = 0; i < 2; i++)
            {
            #pragma HLS unroll
            #pragma HLS inline region
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
            int lane2 = lane;
            int lane3 = lane2;
            if(left){
                direction = -1;
            }
            else if(want_right){
                direction = 1;
            }
            int c_index = current_index_new[lane + 1 + direction];
            if(direction != 0){
                if(c_index <= 0 || veh_new.position > vehs_new[ (lane3 + direction) * MAX_VEHICLES_PER_LANE + c_index - 1].position + vehicleSize){
                    current_index_new[lane3 + 1 + direction]++;
                    CLVehInt & front = (direction == 1) ? front_right : front_left;
                    fixed & result = left ? results[7] : results[8];
                    fixed & pos_ = left ? pos[1] : pos[5];
                    updateAgent(vehi, veh_new, result, (fixed)0.25, (front.id != -1) ? /*vehs[vehi.front_right].position*/fixed( pos_-vehi.position-(fixed)5 ) : (fixed)1024);
                }else{
                    c_index = current_index_new[lane + 1];
                    direction = 0;
                }
            }
            if(direction == 0){
                current_index_new[lane2+1] += 1;
                updateAgent(vehi, veh_new, results[6], (fixed)0.25, (front_current.id != -1) ? /*vehs[vehi.front_current].position*/fixed( pos[3]-vehi.position-(fixed)5 ) : (fixed)1024);
            }
            lane_length_new[veh_new.lane]--;
            veh_new.lane += direction;
            lane_length_new[veh_new.lane]++;
            vehs_new[(lane3 + direction)*MAX_VEHICLES_PER_LANE+c_index] = veh_new;

        }
        /// UPDATE ARRAYS
        update_arrays:
        for(int i=0; i < LANE_MAX; i++){
        #pragma HLS unroll
            current_index[i+1] = 0;
            current_index_new[i+1] = 0;
            lane_length[i] = lane_length_new[i];
        }

        update_vehs:
        for(int i = 0; i < MAX_AGENT_SIZE; i++){
            vehs[i] = vehs_new[i];
            vehs_new[i] = veh_null;
        }

    }

    write_vehs:
    for(int i = 0; i < MAX_AGENT_SIZE; i++){
    #pragma HLS pipeline II=1
        vehs_global[i] = vehs[i];
    }

    write_lane_length:
    for(int i = 0; i < LANE_MAX; i++){
    #pragma HLS unroll
        lane_length_global[i] = lane_length_new[i];
    }
}
}
