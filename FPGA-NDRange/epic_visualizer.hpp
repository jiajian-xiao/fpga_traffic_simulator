#pragma once

#include "idm.hpp"
#include "fixedptc.h"
#include "stdio.h"
#include <vector>


void epic_visualizer_ptr(CLVehInt vehs[], int lane_length[], int length, int laneCount, short agentCount){
    // bool fixedpoint = true;
    // int laneCount = 3;
    printf("\n%d %d %d %d", lane_length[0], lane_length[1], lane_length[2], lane_length[3]);
    for (short i = 0; i < LANE_MAX; i++){
        printf("\nindex-id-lane-pos-vel\n");
        for(int j = 0; j < /*lane_length[i]*/7; j++){
            int index = i*MAX_VEHICLES_PER_LANE+j;
            printf("%d %d %d %d %d\n",index,vehs[index].id,vehs[index].lane,vehs[index].position, vehs[index].velocity );
        }
    }
}

void epic_visualizer(CLVehInt vehs[], int lane_length[], int length, int laneCount){
    fixedpt position = 5;
    int index;
    while(position < length){
        printf("=");
        position += 5.0;
    }
    position = 5;
    printf("\n");
    for(int i = 0; i < laneCount; i++){
        for(int j = 0; j < lane_length[i]; j++){
            index = i*MAX_VEHICLES_PER_LANE+j;
            while(position < vehs[index].position){
                position += 5;
                printf("-");
            }
            printf("%c", (vehs[index].id + 65) );
            position += 5;
        }
        while(position < length){
            printf("-");
            position += 5;
        }
        printf("\n");
        position = 5;
        // printf("%d\n",lane_length[j]);
    }
}
