#include <iostream>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <stdlib.h>
#include <cstdlib>
#include <signal.h>
#include <random>
#include <array>
#include <algorithm>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <malloc.h>
#include <iomanip>
#include <math.h>
#include <CL/cl_ext.h>
#include <ap_fixed.h>

#include "constants.h"
#include "idm.hpp"
#include "epic_visualizer.hpp"
#include "xcl2.hpp"

using std::cout;
using std::cin;
using std::ios;
using std::stoi;
namespace pt = boost::posix_time;

const unsigned char arg_ind_r[9] = {0,0,2,2,4,4,6,6,6};
const unsigned char arg_ind_l[9] = {1,6,6,3,5,6,3,1,5};


///////// PROBLEM VARIABLES ///////////

const fixed vehicleSize=5;
short laneCount = 4;
short agentCount = 20;
// CLVehicle * vehs;
std::vector< CLVehInt, aligned_allocator<CLVehInt> >vehsInt_it(MAX_AGENT_SIZE);
std::vector< int, aligned_allocator<int> >lane_length_it(LANE_MAX);


	///cpu variables
	CLVehInt vehs_new[MAX_AGENT_SIZE];
	CLVehInt vehs[MAX_AGENT_SIZE];
	int current_index[LANE_MAX+2] = {0};
	int current_index_new[LANE_MAX+2] = {0};
	int lane_length[LANE_MAX];
	int lane_length_new[LANE_MAX];


//	typedef ap_fixed<32,24, AP_RND, AP_SAT_SYM> fixed;
//	typedef class ap_fixed<32,24, AP_RND, AP_SAT_SYM> fixed;
	fixed politeness = 1; //0.1
	fixed safe_decel = -3;//fixedpt_rconst(-3.0);
	fixed incThreshold = 1;//fixedpt_rconst(1);
	fixed args[9][3];

/////////// TIME VARIABLES /////////

double total_execution = 0;
double sendToFPGA, sendToHost, sendToFPGA_start, sendToFPGA_end, sendToHost_start, sendToHost_end;
double neighbors, neighbors_start, neighbors_end;
double FPGA, FPGA_start, FPGA_end;
double mem_write_start, mem_write_end, mem_write, fpga_start_time, fpga_end_time, execution_time, mem_read_start, mem_read_end, mem_read, ex_kernel_start, ex_kernel_end, ex_kernel, iteration_start, iteration_end;
double cpu_start_time, cpu_end_time, cpu_total;
cl_ulong time_exec; /// measures kernel execution time through profiler




//////////// OUTPUT STREAMS /////////

//std::ofstream clock_cycles("Results/clock_cycles.txt");
std::ofstream cpu_time("Results/cpu_time.txt", ios::app);
//std::ofstream fpga_time("Results/fpga_time.txt");
std::ofstream agentCount_ofstream("Results/agentcount.txt", ios::app);
std::ofstream sendTime_ofstream("Results/sendTime.txt", ios::app) ;
std::ofstream receiveTime_ofstream("Results/receiveTime.txt", ios::app);
std::ofstream totalMemcpy_ofstream("Results/totalMemcpy.txt", ios::app);
std::ofstream totalExecution_ofstream("Results/totalExecution.txt", ios::app);
std::ofstream simulation_ofstream("Results/simulation.txt", ios::app);
//std::ofstream neighbors_ofstream("Results/neighbors.txt", ios::app);
// std::ofstream positions("Results/positions.txt");


/////////// HELPER FUNCTIONS ///////////

template <class T>
void printArray(T * & array, int size){
        for(int i = 0; i < size; i++)
    {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

template <class U>
void writeArray(std::ofstream & out, U * array, int size){ 
    for(int i = 0; i < size; i++)
    {
        out << std::setprecision(4) << std::right << std::setw(10) << array[i];
    }  
    out << std::endl; 
}

float rand_float() {
	return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}

int get_index(int lane_idx, int current_index_offset, int laneCount, int *lane_length, int current_index[]){

  if (lane_idx < 0 || lane_idx >= laneCount)
    return -1;

  int vehIdx = current_index[lane_idx] + current_index_offset;

  if (vehIdx < 0 || vehIdx >= lane_length[lane_idx])
    return -1;

  return lane_idx*MAX_VEHICLES_PER_LANE + vehIdx;
}

// High-resolution timer.
double getCurrentTimestamp() {
#ifdef _WIN32 // Windows
  // Use the high-resolution performance counter.

  static LARGE_INTEGER ticks_per_second = {};
  if(ticks_per_second.QuadPart == 0) {
    // First call - get the frequency.
    QueryPerformanceFrequency(&ticks_per_second);
  }

  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);

  double seconds = double(counter.QuadPart) / double(ticks_per_second.QuadPart);
  return seconds;
#else         // Linux
  timespec a;
  clock_gettime(CLOCK_MONOTONIC, &a);
  return (double(a.tv_nsec) * 1.0e-9) + double(a.tv_sec);
#endif
}

/////////// FUNCTION DECLARATIONS ///////////////

void initialiseProblem();
void initialiseKernel();
int executeKernel();
bool run_model();
void releaseKernel();
void startSim();

//////////////////////////////////////////

cl_int err = 0;
int globalWorkSize = 1;
int localWorkSize = 1;
bool useCpu = false;
std::string aocx;
bool scan = false;
short iteration = 1;
bool emulate = false;        // select Kernel if to be worked on computer
bool singleWorkItem = true; // select Kernel(swi) and adapt iteration count  /// TRUE BY DEFAULT -- change if not-so-pure version is to be used --
bool &swi = singleWorkItem;

//cl_mem_ext_ptr_t vehsExt, pointersExt;

int main(int argc, char *argv[])
{

	/////////// CL VARIABLES //////////

	std::vector<cl::Memory> inoutBufVec;
    std::vector<cl::Memory> outBufVec;
    cl::Context m_context;
    cl::CommandQueue m_command_queue;
    cl::Buffer vehs_buffer; // vehs;
    cl::Program m_program;
    cl::Kernel m_kernel;



    printf("Simulation starts");
    for(short i=1; i < argc; i++){
        if( strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "-cpu") == 0 ){
            useCpu = true;
        }
        else if(strcmp(argv[i], "-w") == 0){
            localWorkSize = stoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-l") == 0){
            laneCount = stoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-scan") == 0){
            scan = true;
        }
        else if(strcmp(argv[i], "-it") == 0){
            iteration = stoi(argv[++i]);
        }
        else if(strcmp(argv[i], "-emulate") == 0){
            emulate=true;
        }
        else if(strcmp(argv[i], "-swi") == 0 ){
            singleWorkItem = true;
        }
        else if(strcmp(argv[i], "-aocx") == 0 ){
            aocx = std::string(argv[++i]);
        }
        else if(strcmp(argv[i], "-agents") == 0 ){
            agentCount = stoi(argv[++i]);
            globalWorkSize = agentCount;
            if(agentCount > MAX_AGENT_SIZE){std::cout << std::endl << "too many agents" << std::endl; return 0;}
        }
    }

    /// INITIALISE PROBLEM

	for(int i = 0; i < laneCount; i++){
		lane_length_it[i] = 0;
	}

	for(int i = 0; i < agentCount/2; i++){
		vehsInt_it[i].id = i;
		vehsInt_it[i].position =  ( 6 * (i) );
		vehsInt_it[i].lane = 0;
		lane_length_it[0]++;
		vehsInt_it[MAX_VEHICLES_PER_LANE + i].id = i+agentCount/2;
		vehsInt_it[MAX_VEHICLES_PER_LANE + i].position = ( 6 * (i) + 1 );
		vehsInt_it[MAX_VEHICLES_PER_LANE + i].lane = 1;
		lane_length_it[1]++;
	}
	
	/// INITIALISE KERNEL

    if(useCpu != true){
		std::vector<cl::Device> devices = xcl::get_xil_devices();
		cl::Device device = devices[0];
		std::cout << "Initialising Kernel on the FPGA" << std::endl;

		OCL_CHECK(err, m_context = cl::Context(device, NULL, NULL, NULL, &err));
		OCL_CHECK(err, m_command_queue = cl::CommandQueue(m_context, device, CL_QUEUE_PROFILING_ENABLE, &err));
		OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));
		std::cout << "Found Device=" << device_name.c_str() << std::endl;

		/* Create Kernel Program from the binary */
		std::string binaryFile = xcl::find_binary_file(device_name, "sim");
		cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
		devices.resize(1);
		OCL_CHECK(err, m_program = cl::Program(m_context, devices, bins, NULL, &err));
		OCL_CHECK(err, m_kernel = cl::Kernel(m_program, "sim", &err));

		/* Create Memory Buffer */

		OCL_CHECK(err, cl::Buffer vehsInt_buffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
				sizeof(CLVehInt) * MAX_AGENT_SIZE, vehsInt_it.data()/*&vehsExt*/ , &err));
		OCL_CHECK(err, cl::Buffer laneLength_buffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
				sizeof(int) * (LANE_MAX), lane_length_it.data()/*&vehsExt*/ , &err));

		inoutBufVec.push_back(vehsInt_buffer);
		inoutBufVec.push_back(laneLength_buffer);

		int narg = 0;
		OCL_CHECK(err, err = m_kernel.setArg(narg++, vehsInt_buffer));
		OCL_CHECK(err, err = m_kernel.setArg(narg++, laneLength_buffer));
		// OCL_CHECK(err, err = m_kernel.setArg(narg++, pointers_buffer));
		OCL_CHECK(err, err = m_kernel.setArg(narg++, agentCount));
		OCL_CHECK(err, err = m_kernel.setArg(narg++, laneCount));
		OCL_CHECK(err, err = m_kernel.setArg(narg++, iteration));
    }

    /// EXECUTE FPGA KERNEL

	if( useCpu == false ){

		if(singleWorkItem) { globalWorkSize = 1; localWorkSize = 1; }

		sendToHost = sendToFPGA = FPGA = neighbors = 0;

		/// Simulation Starts

		fpga_start_time = getCurrentTimestamp();

		sendToFPGA_start = getCurrentTimestamp();

		OCL_CHECK(err, err = m_command_queue.enqueueMigrateMemObjects(inoutBufVec, 0));
		OCL_CHECK(err, err = m_command_queue.finish());

		sendToFPGA_end = getCurrentTimestamp();

		sendToFPGA = (sendToFPGA_end - sendToFPGA_start);

		OCL_CHECK(err, err = m_command_queue.enqueueTask(m_kernel));
		OCL_CHECK(err, err = m_command_queue.finish());


		sendToHost_start = getCurrentTimestamp();

		OCL_CHECK(err, err = m_command_queue.enqueueMigrateMemObjects(inoutBufVec, CL_MIGRATE_MEM_OBJECT_HOST));
		OCL_CHECK(err, err = m_command_queue.finish());

		sendToHost_end = getCurrentTimestamp();

		fpga_end_time = getCurrentTimestamp();

		sendToHost = (sendToHost_end - sendToHost_start);

		/// Simulation Ends

	}

	/// EXECUTE ON CPU

	else if (useCpu == true)
	{
		CLVehInt vehNULL;

		// INITIALIZE VARIABLES
		for(int i=0; i<laneCount; i++){
			lane_length[i] = lane_length_it[i];
			lane_length_new[i] = lane_length_it[i];
			for(int j = 0; j<lane_length_new[i];j++){
				vehs[i*MAX_VEHICLES_PER_LANE+j] = vehsInt_it[i*MAX_VEHICLES_PER_LANE+j];
			}
		}

		for(int i=0; i<MAX_AGENT_SIZE; i++){
			vehs_new[i] = vehNULL;
		}
//		epic_visualizer(vehs, lane_length, 8000, laneCount);
		cpu_start_time = getCurrentTimestamp();



		for(int i=0; i<iteration;i++){
			// printf("iter %d \n", i);
			run_model();
//			epic_visualizer(vehs, lane_length, 8000, laneCount);
			for(int i = 0; i < MAX_AGENT_SIZE; i++){
				vehs[i] = vehs_new[i];
				vehs_new[i] = vehNULL;
			}
			for(int i=0; i < LANE_MAX; i++){
				current_index[i+1] = 0;
				current_index_new[i+1] = 0;
				lane_length[i] = lane_length_new[i];
				// printf("Lane %d length %d ", i, lane_length_new[i]);
				// for(int j=0; j<5;j++){
				// 	printf("AGENT %d ",vehs[i*MAX_VEHICLES_PER_LANE+j].id);
				// }
				// printf("\n");
			}
		}

		cpu_end_time = getCurrentTimestamp();

		for(int i=0; i<laneCount;i++){
			lane_length_it[i] = lane_length_new[i];
			for(int j=0; j<lane_length_new[i];j++){
				vehsInt_it[i*MAX_VEHICLES_PER_LANE+j] = vehs[i*MAX_VEHICLES_PER_LANE+j];
			}
		}

		cpu_total = cpu_end_time - cpu_start_time;
		std::cout << "CPU Execution Time: " << cpu_total*1e6 << std::endl;

		total_execution += cpu_total;
		cpu_time << agentCount << " " << cpu_total*1e6 << std::endl;

	}

    /// SHOW RESULT

	epic_visualizer_ptr(vehsInt_it.data(), lane_length_it.data(), 128, laneCount, agentCount);
	epic_visualizer(vehsInt_it.data(), lane_length_it.data(), 8000, laneCount);

	/// WRITE RESULTS

	execution_time = fpga_end_time - fpga_start_time;

//	neighbors_ofstream << " " << (neighbors / iteration) * 1e6;
	simulation_ofstream << agentCount << " " << ( (sendToHost_start - sendToFPGA_end)/iteration ) * 1e6 << std::endl;
	agentCount_ofstream << " " << agentCount;
	totalMemcpy_ofstream << agentCount << " " << (sendToFPGA + sendToHost) * 1e6 << std::endl;
	sendTime_ofstream << agentCount << " " << sendToFPGA * 1e6 << std::endl;
	receiveTime_ofstream << agentCount << " " << sendToHost * 1e6 << std::endl;
	totalExecution_ofstream << agentCount << " " << (execution_time)*1e6 << std::endl;
	// time_exec = getStartEndTime(kernel_execution);

	if(useCpu == true){
		total_execution = cpu_total;
	}else{
		total_execution = execution_time;
	}
	std::cout << (useCpu ? "CPU":"FPGA") << " Total Execution Time: " << total_execution*1e6 << std::endl;

	if(useCpu == false){
		std::cout << "Total Simulation Run Time: " << (sendToHost_start - sendToFPGA_end)*1e6 << std::endl;
		std::cout << "Total Send-to-FPGA Time: " << sendToFPGA*1e6 << std::endl;
		std::cout << "Total Receive-from-FPGA Time: " << sendToHost*1e6 << std::endl;
	}

	return 0;

}

bool run_model(){
	CLVehInt vehNULL = CLVehInt();
	CLVehInt vehs_min_1[LANE_MAX + 2]={vehNULL};
    CLVehInt vehs_min_2[LANE_MAX + 2]={vehNULL};
//    while (true) {
	for(int p = 0; p < laneCount; p++){
		vehs_min_1[p + 1] = vehs[p*MAX_VEHICLES_PER_LANE];
	}

	for(int index = 0; index < agentCount; index++){

		int min_displ_lane = 1;

		CLVehInt vehi;
		CLVehInt rear_left, front_left, rear_current, front_current, rear_right, front_right;
//
		find_min_displ_veh:
		vehi = vehs_min_1[1];
		for(int j = 2; j < LANE_MAX + 1; j++){
			if(vehi.position > vehs_min_1[j].position){
				vehi = vehs_min_1[j];
				min_displ_lane = j;
			}
		}
//		printf("%d\n",vehi.id);
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
        fixed results[9]; //= {[0 ... 8] = 32};
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

////         printf("pos %d %d %d %d %d %d %d\n",pos[0],pos[1],pos[2],pos[3],pos[4],pos[5],pos[6]);
////         printf("vels %d %d %d %d %d %d %d\n",vels[0],vels[1],vels[2],vels[3],vels[4],vels[5],vels[6]);

        for(short i = 0; i < 9; i++)
        {
            args[i][0] = vels[arg_ind_r[i]];
            args[i][1] = vels[arg_ind_l[i]] - vels[arg_ind_r[i]];
            args[i][2] = pos[arg_ind_l[i]] - pos[arg_ind_r[i]] - fixed(5);
            results[i] = 0;
        }

        for(short i = 0; i < 9; i++)
        {
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
//            results[i] = idm(args[i][0], args[i][1], 50, ds);

            fixed free_road_term, interaction_term, vRat, ss, temp;
            if(ds <= 0){
                results[i] = fixed(-5); // return something less than minimum allowed acceleration
                continue;
            }
            // if (v<vdesired){
                vRat = args[i][0] * fixed(0.0625);// /vdesired; // 0.02 originally
                free_road_term = fixed(1.8);
            // } else {
                // vRat = vdesired/v;
                // free_road_term = -2.0f;
            // }
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
   return true;
}
