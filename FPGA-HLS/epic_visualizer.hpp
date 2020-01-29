#pragma once

#include "idm.hpp"
#include "fixedptc.h"
#include "stdio.h"
#include <vector>

// #if (C != 1)
// #include <vector>
// #include <algorithm>
// #include "stdio.h"

// #define ANSI_COLOR_RED \x1b[31m
// #define ANSI_COLOR_GREEN \x1b[32m
// #define ANSI_COLOR_YELLOW \x1b[33m
// #define ANSI_COLOR_BLUE \x1b[34m
// #define ANSI_COLOR_MAGENTA \x1b[35m
// #define ANSI_COLOR_CYAN \x1b[36m
// #define ANSI_COLOR_RESET \x1b[0m
// #define ANSI_COLOR_WHITE \033[01;37m

// void epic_visualizer(const CLVehicle * vehs, int length, int vehcount, int laneCount = 3, bool fixedpoint = true){
//     bool state=true;
//     int proportion;
//     int size;
//     std::vector< std::vector<int> > road;
//     road.resize(laneCount);

//     do{
//         size = length / proportion;
//         proportion++;
//     }while(size > length);
//     proportion--;

//     printf("\n");
//     for(int k=0;k<size;k++){
//         printf("\033[01;37m=");
//     }
//     printf("\n\x1b[0m");

//     for(size_t i = 0; i < vehcount; i++)
//     {
//         road[vehs[i].lane[state]].push_back(fixedpoint ? ( fixedpt_tofloat(vehs[i].position[state]) ) : vehs[i].position[state]);
//     }
//     for(size_t i = 0; i < laneCount; i++){
//         std::transform(road[i].begin(), road[i].end(), road[i].begin(), [proportion](int& c){return c / proportion;});
//         std::sort(road[i].begin(), road[i].end());
//     }

//     int j;
//     for(size_t k = 0; k < laneCount; k++){
//         j = 0;
//         for(size_t i = 0; i < size; i++)
//         {
//             if(road[k].size() == 0){
//                 for(int k=0;k<size;k++){
//                     printf("-");
//                 }
//                 printf("\n");
//                 break;
//             }
//             if(i == road[k][j]){
//                 printf("0");
//                 j++;
//                 if(j == road[k].size()){
//                     j--;
//                 }
//             }else{
//             printf("-");
//             }
//         }
//         printf("\n");
//     }

//     for(int k=0;k<size;k++){
//         printf("=");
//     }
//     printf("\n");
// }
// #endif 

//void epic_visualizer_C(const CLVehicle * lanes, int length, int vehcount, int laneCount, bool state){  // assuming sorted vehicles
//    bool fixedpoint = true;
//    // int laneCount = 3;
//    int proportion = 1;
//    int size;
//
//    do{
//        size = length / proportion;
//        proportion++;
//    }while(size > 128);
//    proportion--;
//
//    int elements[LANE_MAX];
//    for(int i = 0; i < laneCount; i++)
//    {
//        elements[i] = 0;
//    }
//    for(int i = 0; i < laneCount; i++)
//    {
//        for(int j = 1; j < MAX_VEHICLES_PER_LANE; j++)
//        {
//            if( LANES(i,j).id == 0 ){
//                break;
//            }
//            elements[i]++;
//        }
//    }
//
//    printf("\033[01;37m");
//    for(int k=0;k<size;k++){
//        printf("=");
//    }
//    printf("\n\x1b[0m");
//
//    for(int i = 0; i < laneCount; i++)
//    {
//        for(int j = 0; j < size; j++)
//        {
//            if(elements[i] == 0){
//                    printf("-");
//                    continue;
//            }
//            if(j == fixedpt_toint(LANES(i,elements[i]).position[!state])/proportion ){
//                printf("\x1b[33m%d\x1b[0m", LANES(i, elements[i]).id);
//                elements[i]--;
//            }else{
//                printf("-");
//            }
//        }
//        printf("\n");
//    }
//
//    printf("\033[01;37m");
//    for(int k=0;k<size;k++){
//        printf("=");
//    }
//    printf("\n\x1b[0m");
//}

// void epic_visualizer_ptr(CLVehInt vehs[], int length, int laneCount, short agentCount){
//     // bool fixedpoint = true;
//     // int laneCount = 3;
//     int proportion = 1;
//     int size;
//     short tails[LANE_MAX];
//     for (short i = 0; i < LANE_MAX; i++)
//     {
//         tails[i] = findTail(i, vehs, agentCount);
//     }

//     do{
//         size = length / proportion;
//         proportion++;
//     }while(size > 128);
//     proportion--;

//     printf("\033[01;37m");
//     for(int k=0;k<size;k++){
//         printf("=");
//     }
//     printf("\n\x1b[0m");

//     for(int i = 0; i < laneCount; i++)
//     {
//      if(tails[i] == 0){
//          for(int j = 0; j < size; j++){
//              printf("-");
//          }
//      }else{
//             for(int j = 0; j < size; j++) // excludes tails[i].position == j
//             {
//                 if(j == fixedpt_toint(vehs[tails[i]].position) / proportion ){
//                     printf("\x1b[33m%d\x1b[0m", tails[i]);
//                     tails[i] = vehs[tails[i]].front_current;
//                 }else{
//                     printf("-");
//                 }
//             }
//      }

//         printf("\n");
//     }
//     printf("\033[01;37m");
//     for(int k=0;k<size;k++){
//         printf("=");
//     }
//     printf("\n\x1b[0m");
// }

void epic_visualizer_ptr(CLVehInt vehs[], int lane_length[], int length, int laneCount, short agentCount){
    // bool fixedpoint = true;
    // int laneCount = 3;
    printf("\n%d %d %d %d", lane_length[0], lane_length[1], lane_length[2], lane_length[3]);
    for (short i = 0; i < LANE_MAX; i++){
        printf("\nindex-id-lane-pos-vel\n");
        for(int j = 0; j < /*lane_length[i]*/7; j++){
            int index = i*MAX_VEHICLES_PER_LANE+j;
            printf("%d %d %d %.2f %.2f\n",index,vehs[index].id,vehs[index].lane,float(vehs[index].position), float(vehs[index].velocity) );
        }
    }
}

void epic_visualizer(CLVehInt vehs[], int lane_length[], int length, int laneCount){
    fixed position = 5;
    int index;
    while(position < length){
        printf("=");
        position += (fixed)5;
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
    }
}
