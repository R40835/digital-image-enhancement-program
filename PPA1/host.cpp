#include "Utils.h"
#include "CImg.h"
#include <iostream>
#include <vector>


using namespace cimg_library;

//for printing
template <typename T>
ostream& operator<< (ostream& out, const vector<T>& v) {
	out << '[';
	if (!v.empty()) {
		out << v[0];
		for (size_t i = 1; i < v.size(); ++i) {
			out << ", " << v[i];
		}
	}
	out << ']';
	return out;
}

int main(int argc, char *argv[]) {
	int platform_id = 0;
	int device_id = 0;
	string image_path = "test.pgm"; //Change this to for a different Image. options: test.pgm, test_large.pgm

	//detect any potential exceptions
	try {
		//displaying the image provided
		CImg<unsigned char> input(image_path.c_str());
		CImgDisplay display_input(input, "input");

		//making host operations
		//Selecting computing devices
		cl::Context context = get_context(platform_id, device_id);

		//printing the selected device
		std::cout << "app runing on " << get_platform_name(platform_id) << ", " << get_device_name(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//Loading device code
		cl::Program::Sources sources;

		const string file_name = "kernels/kernels.cl";
		ifstream file(file_name);
		string* source_code = new string(istreambuf_iterator<char>(file), (istreambuf_iterator<char>()));
		sources.push_back((*source_code).c_str());

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		//showing error
		catch (const cl::Error& err) {
			cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>()[0];
			cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device);
			std::string options = program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(device);
			std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
			std::cerr << "Build Status: " << (status == CL_BUILD_SUCCESS ? "Success" : "Failure") << std::endl;
			std::cerr << "Build Options:\n" << options << std::endl;
			std::cerr << "Build Log:\n" << log << std::endl;
			throw err;
		}

		//allocating memory
		//setting host input
		std::vector<int> H(256);
		size_t histsize = H.size() * sizeof(int);

		//setting device buffers
		cl::Buffer device_image_input = cl::Buffer(context, CL_MEM_READ_ONLY, input.size());
		cl::Buffer device_histogram_output = cl::Buffer(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer device_cumulative_histogram_output = cl::Buffer(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer device_LUT_output = cl::Buffer(context, CL_MEM_READ_WRITE, histsize);
		cl::Buffer device_image_output = cl::Buffer(context, CL_MEM_READ_WRITE, input.size());

		//copying images to device memory
		cl::Event image_input_event, histogram_output_event;
		queue.enqueueWriteBuffer(device_image_input, CL_TRUE, 0, input.size(), input.data(), nullptr, &image_input_event);
		queue.enqueueFillBuffer(device_histogram_output, 0, 0, histsize, nullptr, &histogram_output_event);

		//Setting up and executing the kernel
		//plotting a histogram of the frequency of pixel values in the picture provided
		cl::Kernel kernel_histogram = cl::Kernel(program, "histogram");
		kernel_histogram.setArg(0, device_image_input);
		kernel_histogram.setArg(1, device_histogram_output);

		cl::Event prof_event1;

		queue.enqueueNDRangeKernel(kernel_histogram, cl::NullRange, cl::NDRange(input.size()), cl::NullRange, NULL, &prof_event1);
		queue.enqueueReadBuffer(device_histogram_output, CL_TRUE, 0, histsize, &H[0]);

		std::vector<int> CH(256);

		queue.enqueueFillBuffer(device_cumulative_histogram_output, 0, 0, histsize);

		//plotting a cumulative histogram of the total pixels in the picture across pixel values
		cl::Kernel kernel_cum_hist = cl::Kernel(program, "cum_hist");
		kernel_cum_hist.setArg(0, device_histogram_output);
		kernel_cum_hist.setArg(1, device_cumulative_histogram_output);

		cl::Event prof_event2;

		queue.enqueueNDRangeKernel(kernel_cum_hist, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &prof_event2);
		queue.enqueueReadBuffer(device_cumulative_histogram_output, CL_TRUE, 0, histsize, &CH[0]);

		std::vector<int> LUT(256);

		queue.enqueueFillBuffer(device_LUT_output, 0, 0, histsize);

		//creating a new histogram as a look up table (LUT) of the new pixel values. (normalising the cumulative histogram)
		cl::Kernel kernel_LUT = cl::Kernel(program, "LUT");
		kernel_LUT.setArg(0, device_cumulative_histogram_output);
		kernel_LUT.setArg(1, device_LUT_output);

		cl::Event prof_event3;

		queue.enqueueNDRangeKernel(kernel_LUT, cl::NullRange, cl::NDRange(histsize), cl::NullRange, NULL, &prof_event3);
		queue.enqueueReadBuffer(device_LUT_output, CL_TRUE, 0, histsize, &LUT[0]);

		//allocating the new pixel values from the LUT to the output image to make the output of higher contrast than the input
		cl::Kernel kernel_re_project = cl::Kernel(program, "re_project");
		kernel_re_project.setArg(0, device_image_input);
		kernel_re_project.setArg(1, device_LUT_output);
		kernel_re_project.setArg(2, device_image_output);

		cl::Event prof_event4;

		//printing values from each histogram with the kernel execution times and memory transfers
		vector<unsigned char> output_buffer(input.size());
		queue.enqueueNDRangeKernel(kernel_re_project, cl::NullRange, cl::NDRange(input.size()), cl::NullRange, NULL, &prof_event4);
		queue.enqueueReadBuffer(device_image_output, CL_TRUE, 0, output_buffer.size(), &output_buffer.data()[0]);

		//Histogram
		std::cout << "-------------------------------------------------------------------------------------------------------------------\n";
		std::cout << "Histogram = " << H << std::endl;
		std::cout << "Histogram kernel execution time (ns): " << prof_event1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event1.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetProfilingInfo(prof_event1, ProfilingResolution::PROF_US) << endl;
		std::cout << "-------------------------------------------------------------------------------------------------------------------\n";

		//Cumulative Histogram
		std::cout << "Cumulative Histogram = " << CH << std::endl;
		std::cout << "Cumulative Histogram kernel execution time (ns): " << prof_event1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event2.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetProfilingInfo(prof_event2, ProfilingResolution::PROF_US) << endl;
		std::cout << "-------------------------------------------------------------------------------------------------------------------\n";

		//LUT
		std::cout << "LUT = " << LUT << std::endl;
		std::cout << "LUT kernel execution time (ns): " << prof_event1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event3.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetProfilingInfo(prof_event3, ProfilingResolution::PROF_US) << endl;
		std::cout << "-------------------------------------------------------------------------------------------------------------------\n";

		//Vector Kernel
		std::cout << "Vector kernel execution time (ns): " << prof_event1.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event4.getProfilingInfo<CL_PROFILING_COMMAND_START>() << std::endl;
		std::cout << GetProfilingInfo(prof_event4, ProfilingResolution::PROF_US) << endl;

		//diplaying the output image
		CImg<unsigned char> output(output_buffer.data(), input.width(), input.height(), input.depth(), input.spectrum());
		CImgDisplay display_output(output, "output");

		//key 'Q' to close the images displayed
		while (!display_input.is_closed() && !display_output.is_closed() && !display_input.is_keyQ() && !display_output.is_keyQ()) {
			//waiting for events each 1 ms
			display_input.wait(1);
			display_output.wait(1);
		}
	}
	//showing errors
	catch (const cl::Error& e) {
		std::cerr << "ERROR: " << e.what() << ": " << get_error_string(e.err()) << std::endl;
	}
	catch (CImgException& e) {
		std::cerr << "ERROR: " << e.what() << std::endl;
	}

	return 0;
}
