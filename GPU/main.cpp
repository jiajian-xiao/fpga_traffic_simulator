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
#include <stdio.h>

#include "CL/cl.h"
#include <boost/program_options.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <malloc.h>
#include <iomanip>
#include <math.h>
#include <CL/cl_ext.h>

#include "constants.h"
#include "idm.hpp"
#include "epic_visualizer.hpp"


const unsigned char arg_ind_r[9] = {0,0,2,2,4,4,6,6,6};
const unsigned char arg_ind_l[9] = {1,6,6,3,5,6,3,1,5};

using std::cout;
using std::cin;
using std::ios;
using std::stoi;
namespace pt = boost::posix_time;


///////// PROBLEM VARIABLES ///////////

const float vehicleSize=5;
int laneCount = 4;
int agentCount = 1024;
// CLVehicle * vehs;
std::vector< CLVehInt> vehsInt_it(MAX_AGENT_SIZE);
std::vector< int> lane_length_it(LANE_MAX);
std::vector< int> current_index_it(LANE_MAX+2);
std::vector< CLVehInt>vehMin_it(LANE_MAX+2);

//    std::vector< CLVehInt, aligned_allocator<CLVehInt> >vehsIntNew(MAX_AGENT_SIZE);
// std::vector< ptr, aligned_allocator<ptr> > pointers(MAX_AGENT_SIZE);
//    CLVehInt * vehsInt;
//    CLVehicleFloat * vehiclesFloat;
// double dt = 250.0/1000.0;

///cpu variables
CLVehInt vehs_new[MAX_AGENT_SIZE];
CLVehInt vehs[MAX_AGENT_SIZE];
int current_index[LANE_MAX+2] = {0};
int current_index_new[LANE_MAX+2] = {0};
int lane_length[LANE_MAX];
int lane_length_new[LANE_MAX];


//	typedef ap_float<32,24, AP_RND, AP_SAT_SYM> float;
//	typedef class ap_float<32,24, AP_RND, AP_SAT_SYM> float;
float politeness = 1; //0.1
float safe_decel = -3;//floatpt_rconst(-3.0);
float incThreshold = 1;//floatpt_rconst(1);
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

// void writeVehicles(std::ofstream & positions, std::ofstream & velocities, bool floatpoint=false){
//     for(size_t i = 0; i < agentsize; i++)
//     {
//         if(useCpu != true){
//             positions << std::setprecision(8) << std::right << std::setw(15) << floatpt_tofloat( vehicles[i].position[1] );
//             velocities << std::setprecision(8) << std::right << std::setw(15) << floatpt_tofloat( vehicles[i].velocity[1] );
//         }else{
//             positions << std::setprecision(8) << std::right << std::setw(15) << vehiclesFloat[i].position[1] ;
//             velocities << std::setprecision(8) << std::right << std::setw(15) << vehiclesFloat[i].velocity[1] ;
//         }
//     }
//     positions << std::endl;
//     velocities << std::endl;
// }

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
size_t globalWorkSize = 1;
size_t localWorkSize = 1;
bool useCpu = false;
std::string aocx;
bool scan = false;
short iteration = 100;
bool emulate = false;        // select Kernel if to be worked on computer
bool singleWorkItem = true; // select Kernel(swi) and adapt iteration count  /// TRUE BY DEFAULT -- change if not-so-pure version is to be used --
bool &swi = singleWorkItem;

//cl_mem_ext_ptr_t vehsExt, pointersExt;

int main(int argc, char *argv[])
{

//	vehsExt.flags = XCL_MEM_DDR_BANK2;
//	pointersExt.flags = XCL_MEM_DDR_BANK3;
//	vehsExt.param = pointersExt.param = 0;
//	vehsExt.obj = vehsInt.data();
//	pointersExt.obj = pointers.data();

    // waitMilliseconds(5000);
    // std::getchar();


	/////////// CL VARIABLES //////////

//	std::vector<cl::Memory> inoutBufVec;
//    std::vector<cl::Memory> outBufVec;
//    cl::Context m_context;
//    cl::CommandQueue m_command_queue;
//    cl::Buffer vehs_buffer; // vehs;
//    cl::Program m_program;
//    cl::Kernel m_kernel;



    printf("Simulation starts\n");
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
 
 	for(int i = 0; i < LANE_MAX + 2; i++){
        current_index_it[i] = 0;
    }

	for(int i = 0; i < agentCount/2; i++){
		vehsInt_it[i].id = i;
		vehsInt_it[i].position =  ( fixedpt_mul(fixedpt_rconst(6),fixedpt_rconst(i))  );
		vehsInt_it[i].lane = 0;
		lane_length_it[0]++;
		// vehsInt_it[i+1].id = i+1;
		// vehsInt_it[i+1].position = ( 6 * (i) + 1 );
		// vehsInt_it[i+1].lane = 1;
		// lane_length_it[1]++;
	}
	for(int i = agentCount/2; i < agentCount; i++){
		vehsInt_it[i].id = i;
		vehsInt_it[i].position =  ( fixedpt_mul(fixedpt_rconst(6), fixedpt_rconst(i-agentCount/2)) + FIXEDPT_ONE );
		vehsInt_it[i].lane = 1;
		lane_length_it[1]++;
		// vehsInt_it[i+1].id = i+1;
		// vehsInt_it[i+1].position = ( 6 * (i) + 1 );
		// vehsInt_it[i+1].lane = 1;
		// lane_length_it[1]++;
	}
	// epic_visualizer_ptr(vehsInt_it.data(), lane_length_it.data(), 128, laneCount, agentCount);
	
	/// INITIALISE KERNEL

    if (useCpu != true) {
        cl_int ret;
		cl_device_id device_id = NULL;
		cl_uint num_of_platforms;
		cl_uint num_of_devices=0;
		clGetPlatformIDs(0, NULL, &num_of_platforms);
		cl_platform_id platform_ids[num_of_platforms];
		ret = clGetPlatformIDs( num_of_platforms, platform_ids, NULL );
		cl_command_queue command_queue;

		char *source_str;
		size_t source_size;
		FILE *fp;
		fp = fopen("../sim.cl", "r");
		source_str = (char*)malloc(0x100000);
		source_size = fread(source_str, 1, 0x100000, fp);
		fclose(fp);
    	int devId = 0;
        clGetDeviceIDs( platform_ids[devId], CL_DEVICE_TYPE_ALL, 1, &device_id, NULL );
        cl_context_properties contextProperties[] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties) platform_ids[devId],
            0
        };
        cl_context context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_ALL, NULL, NULL, &ret);
        command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
        cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &ret);
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        if (ret!=0) {
            char buildLog[16384];
            clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                                  sizeof(buildLog), buildLog, NULL);
            std::cerr << "Error in kernel: " << buildLog << std::endl;
            exit(1);
        }
        cl_kernel sim_kernel = clCreateKernel(program, "sim", &ret);
		cl_kernel sort_kernel = clCreateKernel(program, "sort", &ret);
		cl_kernel conflict_kernel = clCreateKernel(program, "conflict_resolver", &ret);
		cl_kernel update_kernel = clCreateKernel(program, "update", &ret);

		cl_mem agents_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(CLVehInt)*MAX_AGENT_SIZE, NULL , &ret);
		cl_mem agents_buff_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(CLVehInt)*MAX_AGENT_SIZE, NULL , &ret);
		cl_mem lane_length_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*LANE_MAX, NULL , &ret);
		cl_mem lane_length_buff_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*LANE_MAX, NULL , &ret);
        cl_mem isConflict = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool), NULL , &ret);

        ret  = clSetKernelArg(sim_kernel, 0, sizeof(cl_mem), &agents_mem);
        ret |= clSetKernelArg(sim_kernel, 1, sizeof(cl_mem), &agents_buff_mem);
        ret |= clSetKernelArg(sim_kernel, 2, sizeof(cl_mem), &lane_length_mem);
        ret |= clSetKernelArg(sim_kernel, 3, sizeof(cl_mem), &lane_length_buff_mem);
        ret |= clSetKernelArg(sim_kernel, 4, sizeof(int), &agentCount);
        ret |= clSetKernelArg(sim_kernel, 5, sizeof(int), &laneCount);

        ret  = clSetKernelArg(sort_kernel, 0, sizeof(cl_mem), &agents_buff_mem);
        ret |= clSetKernelArg(sort_kernel, 1, sizeof(cl_mem), &agents_mem);

        ret  = clSetKernelArg(conflict_kernel, 0, sizeof(cl_mem), &agents_buff_mem);
        ret |= clSetKernelArg(conflict_kernel, 1, sizeof(cl_mem), &agents_mem);
        ret |= clSetKernelArg(conflict_kernel, 2, sizeof(cl_mem), &lane_length_buff_mem);
        ret |= clSetKernelArg(conflict_kernel, 3, sizeof(cl_mem), &lane_length_mem);
        ret |= clSetKernelArg(conflict_kernel, 4, sizeof(int), &agentCount);
        ret |= clSetKernelArg(conflict_kernel, 5, sizeof(cl_mem), &isConflict);

		ret  = clSetKernelArg(update_kernel, 0, sizeof(cl_mem), &agents_mem);
		ret |= clSetKernelArg(update_kernel, 1, sizeof(cl_mem), &agents_buff_mem);
		ret |= clSetKernelArg(update_kernel, 2, sizeof(cl_mem), &lane_length_mem);
		ret |= clSetKernelArg(update_kernel, 3, sizeof(cl_mem), &lane_length_buff_mem);
		ret |= clSetKernelArg(update_kernel, 4, sizeof(int), &agentCount);
		ret |= clSetKernelArg(update_kernel, 5, sizeof(int), &laneCount);

		double gpu_start_time = getCurrentTimestamp();

		ret = clEnqueueWriteBuffer(command_queue, agents_mem, CL_TRUE, 0, sizeof(CLVehInt)*MAX_AGENT_SIZE, vehsInt_it.data(), 0, NULL, NULL);
		ret = clEnqueueWriteBuffer(command_queue, agents_buff_mem, CL_TRUE, 0, sizeof(CLVehInt)*MAX_AGENT_SIZE, vehsInt_it.data(), 0, NULL, NULL);
		ret = clEnqueueWriteBuffer(command_queue, lane_length_mem, CL_TRUE, 0, sizeof(int)*LANE_MAX, lane_length_it.data(), 0, NULL, NULL);
		ret = clEnqueueWriteBuffer(command_queue, lane_length_buff_mem, CL_TRUE, 0, sizeof(int)*LANE_MAX, lane_length_it.data(), 0, NULL, NULL);
    
		globalWorkSize = agentCount;
		localWorkSize = globalWorkSize>128?128:globalWorkSize/8;
		
		for(int i = 0; i < iteration; i++) {
			double think_start_time = getCurrentTimestamp();
			ret = clEnqueueNDRangeKernel(command_queue, sim_kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
			clFinish(command_queue);
			double think_end_time = getCurrentTimestamp();

			printf("%d\n",i );
	        bool conflictFlag = false;

	        double conflict_start_time = getCurrentTimestamp();
	        do {
		        for (int length = 1; length < globalWorkSize; length <<= 1)
		            for (int inc = length; inc > 0; inc >>= 1) {
		                int dir = length << 1;
		                clSetKernelArg(sort_kernel, 2, sizeof(int), &inc);
		                clSetKernelArg(sort_kernel, 3, sizeof(int), &dir);
		                ret = clEnqueueNDRangeKernel(command_queue, sort_kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
		                clFinish(command_queue);
		            }
		        conflictFlag = false;
		        clEnqueueWriteBuffer(command_queue, isConflict, CL_TRUE, 0, sizeof(bool), &conflictFlag, 0, NULL, NULL);
	            clEnqueueNDRangeKernel(command_queue, conflict_kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
	            clEnqueueReadBuffer(command_queue, isConflict, CL_TRUE, 0, sizeof(bool), &conflictFlag, 0, NULL, NULL);
	            clFinish(command_queue);
	        } while (conflictFlag);
			double conflict_end_time = getCurrentTimestamp();

			ret = clEnqueueNDRangeKernel(command_queue, update_kernel, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
			clFinish(command_queue);

			double update_end_time = getCurrentTimestamp();

			printf("Think %f Conflict %f Update %f\n", think_end_time - think_start_time, conflict_end_time - conflict_start_time, update_end_time - conflict_end_time);
		}
		ret = clEnqueueWriteBuffer(command_queue, agents_mem, CL_TRUE, 0, sizeof(CLVehInt)*MAX_AGENT_SIZE, vehsInt_it.data(), 0, NULL, NULL);
		ret = clEnqueueWriteBuffer(command_queue, lane_length_mem, CL_TRUE, 0, sizeof(int)*LANE_MAX, lane_length_it.data(), 0, NULL, NULL);

		double gpu_end_time = getCurrentTimestamp();

		double duration = (gpu_end_time - gpu_start_time);
		printf("duration %f\n", duration*1e6);
		exit(0);
		// std::vector<cl::Device> devices = xcl::get_xil_devices();
		// cl::Device device = devices[0];
		// std::cout << "Initialising Kernel on the FPGA" << std::endl;

		// OCL_CHECK(err, m_context = cl::Context(device, NULL, NULL, NULL, &err));
		// OCL_CHECK(err, m_command_queue = cl::CommandQueue(m_context, device, CL_QUEUE_PROFILING_ENABLE, &err));
		// OCL_CHECK(err, std::string device_name = device.getInfo<CL_DEVICE_NAME>(&err));
		// std::cout << "Found Device=" << device_name.c_str() << std::endl;

		// /* Create Kernel Program from the binary */
		// std::string binaryFile = xcl::find_binary_file(device_name, "sim");
		// cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
		// devices.resize(1);
		// OCL_CHECK(err, m_program = cl::Program(m_context, devices, bins, NULL, &err));
		// OCL_CHECK(err, m_kernel = cl::Kernel(m_program, "sim", &err));

		// /* Create Memory Buffer */

		// OCL_CHECK(err, cl::Buffer vehsInt_buffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
		// 		sizeof(CLVehInt) * MAX_AGENT_SIZE, vehsInt_it.data(), &err));
		// OCL_CHECK(err, cl::Buffer vehsInt_new_buffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
		// 		sizeof(CLVehInt) * MAX_AGENT_SIZE, vehsInt_it.data(), &err));
		// OCL_CHECK(err, cl::Buffer laneLength_buffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
		// 		sizeof(int) * (LANE_MAX), lane_length_it.data(), &err));
		// OCL_CHECK(err, cl::Buffer laneLength_new_buffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
		// 		sizeof(int) * (LANE_MAX), lane_length_it.data(), &err));
		// OCL_CHECK(err, cl::Buffer current_buffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
		// 		sizeof(int) * (LANE_MAX+2), current_index_it.data(), &err));
		// OCL_CHECK(err, cl::Buffer current_new_buffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
		// 		sizeof(int) * (LANE_MAX+2), current_index_it.data(), &err));
		// OCL_CHECK(err, cl::Buffer min1_buffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  /*|CL_MEM_EXT_PTR_XILINX*/,
		// 		sizeof(CLVehInt) * (LANE_MAX+2), vehMin_it.data(), &err));
		// OCL_CHECK(err, cl::Buffer min2_dbuffer(m_context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE  |CL_MEM_EXT_PTR_XILINX,
		// 		sizeof(CLVehInt) * (LANE_MAX+2), vehMin_it.data(), &err));

		// inoutBufVec.push_back(vehsInt_buffer);
		// inoutBufVec.push_back(vehsInt_new_buffer);
		// inoutBufVec.push_back(laneLength_buffer);
		// inoutBufVec.push_back(laneLength_new_buffer);
		// inoutBufVec.push_back(current_buffer);
		// inoutBufVec.push_back(current_new_buffer);
		// inoutBufVec.push_back(min1_buffer);
		// inoutBufVec.push_back(min2_dbuffer);

		// int narg = 0;
		// OCL_CHECK(err, err = m_kernel.setArg(narg++, vehsInt_buffer));
		// OCL_CHECK(err, err = m_kernel.setArg(narg++, vehsInt_new_buffer));
		// OCL_CHECK(err, err = m_kernel.setArg(narg++, laneLength_buffer));
		// OCL_CHECK(err, err = m_kernel.setArg(narg++, laneLength_new_buffer));
		// OCL_CHECK(err, err = m_kernel.setArg(narg++, current_buffer));
		// OCL_CHECK(err, err = m_kernel.setArg(narg++, current_new_buffer));
		// OCL_CHECK(err, err = m_kernel.setArg(narg++, min1_buffer));
		// OCL_CHECK(err, err = m_kernel.setArg(narg++, min2_buffer));
		// OCL_CHECK(err, err = m_kernel.setArg(narg++, agentCount));
		// OCL_CHECK(err, err = m_kernel.setArg(narg++, laneCount));


		// if(singleWorkItem) { globalWorkSize = 1; localWorkSize = 1; }

		// sendToHost = sendToFPGA = FPGA = neighbors = 0;

		// /// Simulation Starts

		// fpga_start_time = getCurrentTimestamp();

		// sendToFPGA_start = getCurrentTimestamp();

		// OCL_CHECK(err, err = m_command_queue.enqueueMigrateMemObjects(inoutBufVec, 0));
		// OCL_CHECK(err, err = m_command_queue.finish());

		// sendToFPGA_end = getCurrentTimestamp();

		// sendToFPGA = (sendToFPGA_end - sendToFPGA_start);

		// for(int i = 0; i < iteration; i++){
		// 	// OCL_CHECK(err, err = m_command_queue.enqueueTask(m_kernel));
		// 	m_command_queue.enqueueNDRangeKernel(m_kernel,
  //                         cl::NDRange(0),           // offset
  //                         cl::NDRange(agentCount),             // global
		// 				  cl::NDRange(32),
  //                         NULL, NULL);
		// 	OCL_CHECK(err, err = m_command_queue.finish());
		// }

		// sendToHost_start = getCurrentTimestamp();

		// OCL_CHECK(err, err = m_command_queue.enqueueMigrateMemObjects(inoutBufVec, CL_MIGRATE_MEM_OBJECT_HOST));
		// OCL_CHECK(err, err = m_command_queue.finish());

		// sendToHost_end = getCurrentTimestamp();

		// fpga_end_time = getCurrentTimestamp();

		// sendToHost = (sendToHost_end - sendToHost_start);

		/// Simulation Ends

	}
	/// EXECUTE ON CPU
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

bool run_model() {
    return true;
}
