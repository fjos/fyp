//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// Sample code for FYP project based on distributed OpenCL
// distributedCL, the API this code showcases, is available at
// https://github.com/fjos/fyp
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// Sample demonstrating non-blocking distributed OpenCL commands
// Program continues execution even with enqueued write buffer
// Write buffer will not complete until Machine 0 finishes sleeping
// A second queue is created on Machine 1 and 2 that finishes without
// being blocked by the write
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// Code based on Apple's OpenCL hello world tutorial available at
// https://developer.apple.com/library/mac/samplecode/OpenCL_Hello_World_Example
// using error checking from Code Project's OpenCL tutorial available at
// http://www.codeproject.com/Articles/110685/Part-OpenCL-Portable-Parallelism


#include <unistd.h>
#include <iostream>

#include "distributedCL.h"
using namespace std;

#ifdef __APPLE__
#include "OpenCL/opencl.h"
#else
#include "CL/cl.h"
#endif

#define DATA_SIZE (1024 * 1240)

const char *KernelSource = "\n"
                           "__kernel void square(                    \n"
                           "   __global float* input,                \n"
                           "   __global float* output,               \n"
                           "   const unsigned int count)             \n"
                           "{                                        \n"
                           "   int i = get_global_id(0);             \n"
                           "   if(i < count)                         \n"
                           "       output[i] = input[i] * input[i];  \n"
                           "}                                        \n"
                           "\n";

int main(int argc, char *argv[])
{

    distCL::distributedCL distributedCL(&argc, &argv);

    int devType = CL_DEVICE_TYPE_GPU;

    if (argc > 1)
    {
        devType = CL_DEVICE_TYPE_CPU;
        cout << "Using: CL_DEVICE_TYPE_CPU" << endl;
    }
    else
    {
        cout << "Using: CL_DEVICE_TYPE_GPU" << endl;
    }

    cl_int err; // error code returned from api calls

    size_t global; // global domain size for our calculation
    size_t local;  // local domain size for our calculation

    cl_platform_id cpPlatform; // OpenCL platform
    cl_device_id device_id;    // compute device id
    cl_context context;        // compute context
    cl_command_queue commands; // compute command queue
    cl_program program;        // compute program
    cl_kernel kernel;          // compute kernel

    // Connect to a compute device
    err = distributedCL.GetPlatformIDs(1, &cpPlatform, NULL, {});
    if (err != CL_SUCCESS)
    {
        cerr << "Error: Failed to find a platform!" << endl;
        return EXIT_FAILURE;
    }

    // Get a device of the appropriate type
    err = distributedCL.GetDeviceIDs(cpPlatform, devType, 1, &device_id, NULL, {});
    if (err != CL_SUCCESS)
    {
        cerr << "Error: Failed to create a device group!" << endl;
        return EXIT_FAILURE;
    }

    // Create a compute context
    err = distributedCL.CreateContext(&context, 0, 1, &device_id, NULL, NULL, {});
    if (err != CL_SUCCESS)
    {
        cerr << "Error: Failed to create a compute context!" << endl;
        return EXIT_FAILURE;
    }

    // Create a command queue
    err = distributedCL.CreateCommandQueue(&commands, context, device_id, 0, {});
    if (err != CL_SUCCESS)
    {
        cerr << "Error: Failed to create a command commands!" << endl;
        return EXIT_FAILURE;
    }

    // Create the compute program from the source buffer
    err = distributedCL.CreateProgramWithSource(&program, context, 1,
                                                (const char **)&KernelSource,
                                                NULL, {});
    if (err != CL_SUCCESS)
    {
        cerr << "Error: Failed to create compute program!" << endl;
        return EXIT_FAILURE;
    }

    // Build the program executable
    err = distributedCL.BuildProgram(program, 0, NULL, NULL, NULL, NULL, {});
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[2048];

        cerr << "Error: Failed to build program executable!" << endl;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer), buffer, &len);
        cerr << buffer << endl;
        exit(1);
    }

    // Create the compute kernel in the program
    err = distributedCL.CreateKernel(&kernel, program, "square", {});
    if (!kernel || err != CL_SUCCESS)
    {
        cerr << "Error: Failed to create compute kernel!" << endl;
        exit(1);
    }

    if (distributedCL.world_rank == 0)
        printf("Creating barrier on machines 0,1,2\n");
    data_barrier<float> barrier = distributedCL.CreateBarrier<float>(DATA_SIZE, DATA_SIZE / 2, context, { 0, 1, 2 });

    // create data for the run
    // float *data = new float[DATA_SIZE];    // original data set given to device
    // float *results = new float[DATA_SIZE]; // results returned from device
    unsigned int correct; // number of correct results returned
    cl_mem input;         // device memory used for the input array
    cl_mem output;        // device memory used for the output array

    // Fill the vector with random float values
    unsigned int count = DATA_SIZE;

    for (int i = 0; i < DATA_SIZE; i++)
        barrier.data[i] = rand() / (float)RAND_MAX;

    // Create the device memory vectors
    //
    err = distributedCL.CreateBuffer(&input, context, CL_MEM_READ_ONLY,
                                     sizeof(float) * count, NULL, {});
    err = distributedCL.CreateBuffer(&output, context, CL_MEM_WRITE_ONLY,
                                     sizeof(float) * count, NULL, {});
    if (!input || !output)
    {
        cerr << "Error: Failed to allocate device memory!" << endl;
        exit(1);
    }

    if (distributedCL.world_rank == 0)
    {
        printf("Machine 0 sleeping for 10 seconds\n");
        err = sleep(10);
    }
    // printf("Creating barrier on machines 0,1,2\n");

    // Transfer the input vector into device memory
    distCL_event receive_event, kernel_event, read_event;

    printf("Machine %d EnqueueWriteBuffer\n", distributedCL.world_rank);
    err = distributedCL.EnqueueWriteBuffer(commands,
                                           input,
                                           CL_FALSE,
                                           0,
                                           count,
                                           &barrier, 0,
                                           0, NULL,
                                           NULL, NULL,
                                           0, { 0, 1, 2 });

    // Set the arguments to the compute kernel
    err = 0;
    err = distributedCL.SetKernelArg(kernel, 0, sizeof(cl_mem), &input, {});
    err |= distributedCL.SetKernelArg(kernel, 1, sizeof(cl_mem), &output, {});
    err |= distributedCL.SetKernelArg(kernel, 2, sizeof(unsigned int), &count, {});
    if (err != CL_SUCCESS)
    {
        cerr << "Error: Failed to set kernel arguments! " << err << endl;
        exit(1);
    }

    // Get the maximum work group size for executing the kernel on the device
    err = clGetKernelWorkGroupInfo(kernel, device_id,
                                   CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(local), &local, NULL);
    if (err != CL_SUCCESS)
    {
        cerr << "Error: Failed to retrieve kernel work group info! "
             << err << endl;
        exit(1);
    }

    // Execute the kernel over the vector using the
    // maximum number of work group items for this device
    global = count;
    printf("Machine %d EnqueueNDRangeKernel\n", distributedCL.world_rank);
    err = distributedCL.EnqueueNDRangeKernel(commands, kernel,
                                             1, NULL, &global, &local,
                                             0, NULL, NULL, {});
    if (err != CL_SUCCESS)
    {
        cerr << "Error: Failed to execute kernel!" << endl;
        return EXIT_FAILURE;
    }

    // Wait for all commands to complete

    // Read back the results from the device to verify the output
    //

    data_barrier<float> outputbarrier = distributedCL.CreateBarrier<float>(DATA_SIZE, DATA_SIZE / 2, context, { 0, 1, 2 });

    printf("Machine %d first EnqueueReadBuffer\n", distributedCL.world_rank);
    // distCL_event *event = new distCL_event[1];
    err = distributedCL.EnqueueReadBuffer(commands, output,
                                          CL_FALSE, 0, count, &outputbarrier, 0,
                                          1, &kernel_event,
                                          NULL, &read_event,
                                          2, { 2 });
    err = distributedCL.EnqueueReadBuffer(commands, output,
                                          CL_FALSE, 0, count, &outputbarrier, 0,
                                          1, &kernel_event,
                                          NULL, &read_event,
                                          1, { 1 });
    err = distributedCL.EnqueueReadBuffer(commands, output,
                                          CL_FALSE, 0, count, &outputbarrier, 0,
                                          1, &kernel_event,
                                          NULL, &read_event,
                                          0, { 0 });

    //-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //    Asynchronous activity on machines 1 and 2
    //-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    if (distributedCL.world_rank == 1 || distributedCL.world_rank == 2)
    {
        printf("Second command queue on machines %d\n", distributedCL.world_rank);
        data_barrier<float> asyncbarrier = distributedCL.CreateBarrier<float>(DATA_SIZE, DATA_SIZE / 2, context, { 1, 2 });

        cl_command_queue command2;
        err = distributedCL.CreateCommandQueue(&command2, context, device_id, 0, { 1, 2 });
        if (err != CL_SUCCESS)
        {
            cerr << "Error: Failed to create a command commands!" << endl;
            return EXIT_FAILURE;
        }

        distCL_event async_event;

        printf("Machine %d called second EnqueueReadBuffer\n", distributedCL.world_rank);
        err = distributedCL.EnqueueReadBuffer(command2, input,
                                              CL_FALSE, 0, count, &asyncbarrier, 0,
                                              0, NULL,
                                              &async_event, NULL,
                                              2, { 1, 2 });

        distributedCL.WaitForEvents(1, &async_event, { 1, 2 });
        printf("Machine %d completed second EnqueueReadBuffer\n", distributedCL.world_rank);
    }

    printf("Machine %d waiting for first EnqueueReadBuffer\n", distributedCL.world_rank);
    distributedCL.WaitForEvents(1, &read_event, {});
    // delete [] event;
    if (err != CL_SUCCESS)
    {
        std::cerr << "Error: Failed to read output array! " << err << std::endl;
        exit(1);
    }

    // Validate our results
    correct = 0;

    for (int i = 0; i < DATA_SIZE; i++)
    {
        if (outputbarrier.data[i] == barrier.data[i] * barrier.data[i])
            correct++;
    }

    // Print a brief summary detailing the results
    cout << "Computed " << correct << "/" << DATA_SIZE << " correct values on " << distributedCL.world_rank << endl;
    cout << "Computed " << 100.f * (float)correct / (float)DATA_SIZE
         << "% correct values" << endl;

    // Shutdown and cleanup
    // delete [] data; delete [] results;

    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    distributedCL.Finalize();

    return 0;
}
