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
#include <fixedptc.h>

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

const unsigned char arg_ind_r[9] = {0,0,2,2,4,4,6,6,6};
const unsigned char arg_ind_l[9] = {1,6,6,3,5,6,3,1,5};


///////// PROBLEM VARIABLES ///////////

const float vehicleSize=5;
short laneCount = 4;
short agentCount = 20;
// CLVehicle * vehs;
std::vector< CLVehInt, aligned_allocator<CLVehInt> >vehsInt_it(MAX_AGENT_SIZE);
std::vector< CLVehInt, aligned_allocator<CLVehInt> >vehsInt_it_new(MAX_AGENT_SIZE);

std::vector< int, aligned_allocator<int> >lane_length_it(LANE_MAX);
std::vector< int, aligned_allocator<int> >lane_length_it_new(LANE_MAX);


///cpu variables
CLVehInt vehs_new[MAX_AGENT_SIZE];
CLVehInt vehs[MAX_AGENT_SIZE];
int current_index[LANE_MAX+2] = {0};
int current_index_new[LANE_MAX+2] = {0};
int lane_length[LANE_MAX];
int lane_length_new[LANE_MAX];
float politeness = 1; //0.1
float safe_decel = -3;//fixedpt_rconst(-3.0);
float incThreshold = 1;//fixedpt_rconst(1);
float args[9][3];

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

	std::vector<cl::Memory> inBufVec;
    std::vector<cl::Memory> outBufVec;
    cl::Context m_context;
    cl::CommandQueue m_command_queue;
    cl::Buffer vehs_buffer; // vehs;
    cl::Program m_program;
    cl::Kernel m_kernel;
    cl::Kernel init_kernel;
    cl::Kernel update_kernel;



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
		lane_length_it_new[i] = 0;
	}

	for(int i = 0; i < agentCount/2; i++){
		vehsInt_it[i].id = i;
		vehsInt_it[i].position =  ( fixedpt_mul(fixedpt_rconst(6),fixedpt_rconst(i)) );
		vehsInt_it[i].lane = 0;
		lane_length_it[0]++;
		lane_length_it_new[0]++;
		vehsInt_it[MAX_VEHICLES_PER_LANE + i].id = i+agentCount/2;
		vehsInt_it[MAX_VEHICLES_PER_LANE + i].position = ( fixedpt_mul(fixedpt_rconst(6),fixedpt_rconst(i)) + FIXEDPT_ONE );
		vehsInt_it[MAX_VEHICLES_PER_LANE + i].lane = 1;
		lane_length_it[1]++;
		lane_length_it_new[1]++;
	}

	/// INITIALISE KERNEL

    if(!useCpu){
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
		OCL_CHECK(err, init_kernel = cl::Kernel(m_program, "init", &err));
		OCL_CHECK(err, update_kernel = cl::Kernel(m_program, "update", &err));

		/* Create Memory Buffer */

		OCL_CHECK(err, cl::Buffer vehsInt_buffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
				sizeof(CLVehInt) * MAX_AGENT_SIZE, vehsInt_it.data()/*&vehsExt*/ , &err));
		OCL_CHECK(err, cl::Buffer vehsInt_new_buffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
				sizeof(CLVehInt) * MAX_AGENT_SIZE, vehsInt_it_new.data()/*&vehsExt*/ , &err));
		OCL_CHECK(err, cl::Buffer laneLength_buffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
				sizeof(int) * (LANE_MAX), lane_length_it.data()/*&vehsExt*/ , &err));
		OCL_CHECK(err, cl::Buffer laneLength_new_buffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
				sizeof(int) * (LANE_MAX), lane_length_it_new.data()/*&vehsExt*/ , &err));
		OCL_CHECK(err, cl::Buffer veh_min_buffer1(m_context, CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
				sizeof(CLVehInt) * (LANE_MAX+1), NULL/*&vehsExt*/ , &err));
		OCL_CHECK(err, cl::Buffer veh_min_buffer2(m_context, CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
				sizeof(CLVehInt) * (LANE_MAX+1), NULL/*&vehsExt*/ , &err));
		OCL_CHECK(err, cl::Buffer veh_current_buffer(m_context, CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
				sizeof(int) * (LANE_MAX+1), NULL/*&vehsExt*/ , &err));
		OCL_CHECK(err, cl::Buffer veh_current_buffer_new(m_context, CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
				sizeof(int) * (LANE_MAX+1), NULL/*&vehsExt*/ , &err));

		inBufVec.push_back(vehsInt_buffer);
		inBufVec.push_back(vehsInt_new_buffer);
		inBufVec.push_back(laneLength_buffer);
		inBufVec.push_back(laneLength_new_buffer);

		outBufVec.push_back(vehsInt_buffer);
		outBufVec.push_back(laneLength_buffer);

		int narg = 0;
		OCL_CHECK(err, err = m_kernel.setArg(narg++, vehsInt_buffer));
		OCL_CHECK(err, err = m_kernel.setArg(narg++, vehsInt_new_buffer));
		OCL_CHECK(err, err = m_kernel.setArg(narg++, laneLength_buffer));
		OCL_CHECK(err, err = m_kernel.setArg(narg++, laneLength_new_buffer));
		OCL_CHECK(err, err = m_kernel.setArg(narg++, veh_min_buffer1));
		OCL_CHECK(err, err = m_kernel.setArg(narg++, veh_min_buffer2));
		OCL_CHECK(err, err = m_kernel.setArg(narg++, veh_current_buffer));
		OCL_CHECK(err, err = m_kernel.setArg(narg++, veh_current_buffer_new));
		OCL_CHECK(err, err = m_kernel.setArg(narg++, agentCount));
		OCL_CHECK(err, err = m_kernel.setArg(narg++, laneCount));

		narg = 0;
		OCL_CHECK(err, err = init_kernel.setArg(narg++, vehsInt_buffer));
		OCL_CHECK(err, err = init_kernel.setArg(narg++, veh_min_buffer1));
		OCL_CHECK(err, err = init_kernel.setArg(narg++, veh_min_buffer2));
		OCL_CHECK(err, err = init_kernel.setArg(narg++, veh_current_buffer));
		OCL_CHECK(err, err = init_kernel.setArg(narg++, veh_current_buffer_new));

		narg = 0;
		OCL_CHECK(err, err = update_kernel.setArg(narg++, vehsInt_buffer));
		OCL_CHECK(err, err = update_kernel.setArg(narg++, vehsInt_new_buffer));
		OCL_CHECK(err, err = update_kernel.setArg(narg++, laneLength_buffer));
		OCL_CHECK(err, err = update_kernel.setArg(narg++, laneLength_new_buffer));

		globalWorkSize = agentCount;
		localWorkSize = 1;

		sendToHost = sendToFPGA = FPGA = neighbors = 0;

		/// Simulation Starts

		fpga_start_time = getCurrentTimestamp();

		sendToFPGA_start = getCurrentTimestamp();

		OCL_CHECK(err, err = m_command_queue.enqueueMigrateMemObjects(inBufVec, 0));
		OCL_CHECK(err, err = m_command_queue.finish());

		sendToFPGA_end = getCurrentTimestamp();

		sendToFPGA = (sendToFPGA_end - sendToFPGA_start);

		for (int i=0;i<iteration;i++) {
			// OCL_CHECK(err, err = m_command_queue.enqueueTask(m_kernel));
			OCL_CHECK(err, err = m_command_queue.enqueueTask(init_kernel));
			OCL_CHECK(err, err = m_command_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, cl::NDRange(globalWorkSize), cl::NDRange(localWorkSize), NULL, NULL));
			OCL_CHECK(err, err = m_command_queue.enqueueTask(update_kernel));
		}
		OCL_CHECK(err, err = m_command_queue.finish());


		sendToHost_start = getCurrentTimestamp();

		OCL_CHECK(err, err = m_command_queue.enqueueMigrateMemObjects(outBufVec, CL_MIGRATE_MEM_OBJECT_HOST));
		OCL_CHECK(err, err = m_command_queue.finish());

		sendToHost_end = getCurrentTimestamp();

		fpga_end_time = getCurrentTimestamp();

		sendToHost = (sendToHost_end - sendToHost_start);

		/// Simulation Ends

	}
	else
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
			// printf("iter %` \n", i);
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
   return true;
}
