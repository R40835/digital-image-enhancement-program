#pragma once
#define CL_HPP_ENABLE_EXCEPTIONS

#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <CL/cl2.hpp>

using namespace std;


string get_platform_name(int platform_id) {
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platform_id < 0 || platform_id >= platforms.size()) {
		throw std::out_of_range("Invalid platform ID");
	}
	return platforms.at(platform_id).getInfo<CL_PLATFORM_NAME>();
}


string get_device_name(int platform_id, int device_id) {
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	if (platform_id < 0 || platform_id >= platforms.size()) {
		throw std::out_of_range("Invalid platform ID");
	}
	vector<cl::Device> devices;
	platforms[platform_id].getDevices((cl_device_type)CL_DEVICE_TYPE_ALL, &devices);
	if (device_id < 0 || device_id >= devices.size()) {
		throw std::out_of_range("Invalid device ID");
	}
	return devices.at(device_id).getInfo<CL_DEVICE_NAME>();
}


cl::Context get_context(int platform_id, int device_id) {
	vector<cl::Platform> platforms;

	cl::Platform::get(&platforms);

	for (unsigned int i = 0; i < platforms.size(); i++)
	{
		vector<cl::Device> devices;
		platforms[i].getDevices((cl_device_type)CL_DEVICE_TYPE_ALL, &devices);

		for (unsigned int j = 0; j < devices.size(); j++)
		{
			if ((i == platform_id) && (j == device_id))
				return cl::Context({ devices[j] });
		}
	}

	return cl::Context();
}

string ListPlatformsDevices() {
	stringstream sstream;
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	sstream << "Found " << platforms.size() << " platform(s):\n\n";

	for (const auto& platform : platforms) {
		sstream << "Platform " << platform() << ", " << platform.getInfo<CL_PLATFORM_NAME>() << ", version: " << platform.getInfo<CL_PLATFORM_VERSION>();
		sstream << ", vendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << endl;

		vector<cl::Device> devices;
		platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
		sstream << "\tFound " << devices.size() << " device(s):\n";

		for (const auto& device : devices) {
			sstream << "\t\tDevice " << device() << ", " << device.getInfo<CL_DEVICE_NAME>() << ", version: " << device.getInfo<CL_DEVICE_VERSION>();
			sstream << ", vendor: " << device.getInfo<CL_DEVICE_VENDOR>();

			cl_device_type device_type = device.getInfo<CL_DEVICE_TYPE>();
			sstream << ", type: ";
			if (device_type & CL_DEVICE_TYPE_DEFAULT) sstream << "DEFAULT ";
			if (device_type & CL_DEVICE_TYPE_CPU) sstream << "CPU ";
			if (device_type & CL_DEVICE_TYPE_GPU) sstream << "GPU ";
			if (device_type & CL_DEVICE_TYPE_ACCELERATOR) sstream << "ACCELERATOR ";

			sstream << ", compute units: " << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
			sstream << ", clock freq [MHz]: " << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>();
			sstream << ", max memory size [B]: " << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
			sstream << ", max allocatable memory [B]: " << device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();
			sstream << endl;
		}
		sstream << endl;
	}
	sstream << "----------------------------------------------------------------\n";
	return sstream.str();
}

//measuring time with different metrics
enum ProfilingResolution {
	PROF_NS = 1,
	PROF_US = 1000,
	PROF_MS = 1000000,
	PROF_S = 1000000000
};

string GetProfilingInfo(const cl::Event& evn, ProfilingResolution res) {
	stringstream sstream;

	auto queued_time = evn.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>() - evn.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
	auto submitted_time = evn.getProfilingInfo<CL_PROFILING_COMMAND_START>() - evn.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
	auto executed_time = evn.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evn.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	auto total_time = evn.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evn.getProfilingInfo <CL_PROFILING_COMMAND_QUEUED>();

	sstream << "Queued " << queued_time / res;
	sstream << ", Submitted " << submitted_time / res;
	sstream << ", Executed " << executed_time / res;
	sstream << ", Total " << total_time / res;

	switch (res) {
	case PROF_NS: sstream << " [ns]"; break;
	case PROF_US: sstream << " [us]"; break;
	case PROF_MS: sstream << " [ms]"; break;
	case PROF_S: sstream << " [s]"; break;
	default: break;
	}

	return sstream.str();
}


//error handling
const char* get_error_string(cl_int error) {
	switch (error) {
		// run-time and JIT compiler errors
	case 0: return "CL_SUCCESS";
	case -1: return "CL_DEVICE_NOT_FOUND";
	case -2: return "CL_DEVICE_NOT_AVAILABLE";
	case -3: return "CL_COMPILER_NOT_AVAILABLE";
	case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case -5: return "CL_OUT_OF_RESOURCES";
	case -6: return "CL_OUT_OF_HOST_MEMORY";
	case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case -8: return "CL_MEM_COPY_OVERLAP";
	case -9: return "CL_IMAGE_FORMAT_MISMATCH";
	case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case -11: return "CL_BUILD_PROGRAM_FAILURE";
	case -12: return "CL_MAP_FAILURE";
	case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case -15: return "CL_COMPILE_PROGRAM_FAILURE";
	case -16: return "CL_LINKER_NOT_AVAILABLE";
	case -17: return "CL_LINK_PROGRAM_FAILURE";
	case -18: return "CL_DEVICE_PARTITION_FAILED";
	case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

		// compile-time errors
	case -30: return "CL_INVALID_VALUE";
	case -31: return "CL_INVALID_DEVICE_TYPE";
	case -32: return "CL_INVALID_PLATFORM";
	case -33: return "CL_INVALID_DEVICE";
	case -34: return "CL_INVALID_CONTEXT";
	case -35: return "CL_INVALID_QUEUE_PROPERTIES";
	case -36: return "CL_INVALID_COMMAND_QUEUE";
	case -37: return "CL_INVALID_HOST_PTR";
	case -38: return "CL_INVALID_MEM_OBJECT";
	case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case -40: return "CL_INVALID_IMAGE_SIZE";
	case -41: return "CL_INVALID_SAMPLER";
	case -42: return "CL_INVALID_BINARY";
	case -43: return "CL_INVALID_BUILD_OPTIONS";
	case -44: return "CL_INVALID_PROGRAM";
	case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case -46: return "CL_INVALID_KERNEL_NAME";
	case -47: return "CL_INVALID_KERNEL_DEFINITION";
	case -48: return "CL_INVALID_KERNEL";
	case -49: return "CL_INVALID_ARG_INDEX";
	case -50: return "CL_INVALID_ARG_VALUE";
	case -51: return "CL_INVALID_ARG_SIZE";
	case -52: return "CL_INVALID_KERNEL_ARGS";
	case -53: return "CL_INVALID_WORK_DIMENSION";
	case -54: return "CL_INVALID_WORK_GROUP_SIZE";
	case -55: return "CL_INVALID_WORK_ITEM_SIZE";
	case -56: return "CL_INVALID_GLOBAL_OFFSET";
	case -57: return "CL_INVALID_EVENT_WAIT_LIST";
	case -58: return "CL_INVALID_EVENT";
	case -59: return "CL_INVALID_OPERATION";
	case -60: return "CL_INVALID_GL_OBJECT";
	case -61: return "CL_INVALID_BUFFER_SIZE";
	case -62: return "CL_INVALID_MIP_LEVEL";
	case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case -64: return "CL_INVALID_PROPERTY";
	case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
	case -66: return "CL_INVALID_COMPILER_OPTIONS";
	case -67: return "CL_INVALID_LINKER_OPTIONS";
	case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

		// extension errors
	case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
	case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
	case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
	case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
	case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
	case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
	default: return "Unknown OpenCL error";
	}
}


void check_error(cl_int error) {
	if (error != CL_SUCCESS) {
		cerr << "\n" << get_error_string(error) << endl;
		exit(1);
	}
}