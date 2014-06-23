//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// Sample code for FYP project based on distributed OpenCL
// distributedCL, the API this code showcases, is available at
// https://github.com/fjos/fyp
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// Sample demonstrating simulation work split between two machines
// This sample divides the work between two machines, exchanging
// rows every cycle to maintain correctness
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
// Base code provided by Dr. D.B. Thomas for his course High Performance
// Computing for Engineers.
//-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-



#include "heat.hpp"

#include <stdexcept>
#include <cmath>
#include <cstdint>
#include <memory>
#include <cstdio>
#include <fstream>
#include <streambuf>

#define __CL_ENABLE_EXCEPTIONS

#include "distributedCL.h"

namespace hpce
{

//! Reference world stepping program
/*! \param dt Amount to step the world by.  Note that large steps will be unstable.
    \param n Number of times to step the world
    \note Overall time increment will be n*dt
*/

std::string LoadSource(const char *fileName)
{
    std::string baseDir = "src";
    if (getenv("HPCE_CL_SRC_DIR"))
    {
        baseDir = getenv("HPCE_CL_SRC_DIR");
    }

    std::string fullName = baseDir + "/" + fileName;

    std::ifstream src(fullName, std::ios::in | std::ios::binary);
    if (!src.is_open())
        throw std::runtime_error("LoadSource : Couldn't load cl file from '" + fullName + "'.");

    return std::string(
        (std::istreambuf_iterator<char>(src)), // Node the extra brackets.
        std::istreambuf_iterator<char>());
}

void StepWorldDistributed(world_t &world, float dt, unsigned n)
{
    distCL::distributedCL distributedCL(NULL, NULL);
    int devType = CL_DEVICE_TYPE_CPU;

    cl_int err;
    // Get platforms
    cl_platform_id cpPlatform; // OpenCL platform
    cl_device_id device_id;    // compute device id
    cl_context context;        // compute context
    cl_command_queue commands; // compute command queue
    cl_program program;        // compute program
    cl_kernel kernel;          // compute kernel

    err = distributedCL.GetPlatformIDs(1, &cpPlatform, NULL, {});
    err = distributedCL.GetDeviceIDs(cpPlatform, devType, 1, &device_id, NULL, {});
    err = distributedCL.CreateContext(&context, 0, 1, &device_id, NULL, NULL, {});
    err = distributedCL.CreateCommandQueue(&commands, context, device_id, 0, {});

    // Load kernel to string
    std::string kernelSource = LoadSource("step_world_kernel.cl");
    const char *src = kernelSource.c_str();

    err = distributedCL.CreateProgramWithSource(&program, context, 1,
                                                &src,
                                                NULL, {});

    err = distributedCL.BuildProgram(program, 0, NULL, NULL, NULL, NULL, {});

    cl_mem buffProperties, buffState, buffBuffer;

    distributedCL.BroadcastValue(1, &world.w, 0);
    distributedCL.BroadcastValue(1, &world.h, 0);
    distributedCL.BroadcastValue(1, &world.alpha, 0);

    size_t split_world = world.w * (world.h / 2) + world.w;
    size_t world_size = world.w * world.h;

    data_barrier<uint32_t> properties = distributedCL.CreateBarrier<uint32_t>(world_size, world.w, context, { 0, 1 });
    if (distributedCL.world_rank == 0)
    {
        std::copy(&world.properties[0], &world.properties[0] + world_size, properties.data);
    }

    data_barrier<float> state = distributedCL.CreateBarrier<float>(world_size, world.w, context, { 0, 1 });
    if (distributedCL.world_rank == 0)
    {
        std::copy(&world.state[0], &world.state[0] + world_size, state.data);
    }

    err = distributedCL.CreateBuffer(&buffProperties, context, CL_MEM_READ_ONLY,
                                     split_world * sizeof(uint32_t), NULL, {});
    err = distributedCL.CreateBuffer(&buffState, context, CL_MEM_READ_WRITE,
                                     split_world * sizeof(float), NULL, {});
    err = distributedCL.CreateBuffer(&buffBuffer, context, CL_MEM_READ_WRITE,
                                     split_world * sizeof(float), NULL, {});

    if (!buffProperties || !buffState || !buffBuffer)
    {
        std::cerr << "Error: Failed to allocate device memory!" << std::endl;
        exit(1);
    }

    err = distributedCL.CreateKernel(&kernel, program, "kernel_xy", {});

    float outer = world.alpha * dt; // We spread alpha to other cells per time
    float inner = 1 - outer / 4;    // Anything that doesn't spread stays

    err = distributedCL.SetKernelArg(kernel, 0, sizeof(float),
                                     &inner, {});

    err |= distributedCL.SetKernelArg(kernel, 1, sizeof(float),
                                      &outer, {});
    err |= distributedCL.SetKernelArg(kernel, 3, sizeof(cl_mem),
                                      &buffProperties, {});

    if (err != CL_SUCCESS)
    {
        std::cerr << "Error: Failed to set kernel arguments! " << err << std::endl;
        exit(1);
    }

    size_t row = world.w;
    size_t mid_world = world.h/2;

    err = distributedCL.EnqueueWriteBuffer(commands, buffProperties, CL_TRUE,
                                           0, split_world,
                                           &properties, 0,
                                           0, NULL,
                                           NULL, NULL,
                                           0, { 0 });

    err = distributedCL.EnqueueWriteBuffer(commands, buffState, CL_TRUE,
                                           0, split_world,
                                           &state, 0,
                                           0, NULL,
                                           NULL, NULL,
                                           0, { 0 });

    err = distributedCL.EnqueueWriteBuffer(commands, buffProperties, CL_TRUE,
                                           0, split_world,
                                           &properties, row * (mid_world-1),
                                           0, NULL,
                                           NULL, NULL,
                                           0, { 1 });

    err = distributedCL.EnqueueWriteBuffer(commands, buffState, CL_TRUE,
                                           0, split_world,
                                           &state, row * (mid_world-1),
                                           0, NULL,
                                           NULL, NULL,
                                           0, { 1 });

    size_t *offset_0 = new size_t[2];
    offset_0[0] = 0;
    offset_0[1] = 0;

    size_t *offset_1 = new size_t[2];
    offset_1[0] = 0;
    offset_1[1] = 1;

    // size_t *global = new size_t[2];
    // global[0] = world.w;
    // global[1] = world.h;
    // global[1] = world.h/2+1;

    size_t *global = new size_t[2];
    global[0] = world.w;
    global[1] =mid_world;

    for (unsigned t = 0; t < n; ++t)
    {
        err = distributedCL.SetKernelArg(kernel, 2, sizeof(cl_mem),
                                         &buffState, {});
        err = distributedCL.SetKernelArg(kernel, 4, sizeof(cl_mem),
                                         &buffBuffer, {});


        err = distributedCL.EnqueueReadBuffer(commands, buffState, CL_TRUE,
                                              row * (mid_world-1), row,
                                              &state, row * (mid_world-1),
                                              0, NULL,
                                              NULL, NULL,
                                              0, { 1 });

        err = distributedCL.EnqueueWriteBuffer(commands, buffState, CL_TRUE,
                                               0, row,
                                               &state, row * (mid_world-1),
                                               0, NULL,
                                               NULL, NULL,
                                               1, { 1 });

        err = distributedCL.EnqueueReadBuffer(commands, buffState, CL_TRUE,
                                              row, row,
                                              &state, row * (mid_world),
                                              0, NULL,
                                              NULL, NULL,
                                              1, { 0 });

        err = distributedCL.EnqueueWriteBuffer(commands, buffState, CL_TRUE,
                                               row * (mid_world), row,
                                               &state, row * (mid_world),
                                               0, NULL,
                                               NULL, NULL,
                                               0, { 0 });

        err = distributedCL.EnqueueNDRangeKernel(commands, kernel,
                                                 2,
                                                 offset_0,
                                                 global,
                                                 NULL,
                                                 0,
                                                 NULL,
                                                 NULL, { 0 });

        err = distributedCL.EnqueueNDRangeKernel(commands, kernel,
                                                 2,
                                                 offset_1,
                                                 global,
                                                 NULL,
                                                 0,
                                                 NULL,
                                                 NULL, { 1 });

        err = distributedCL.EnqueueBarrier(commands, {});


        // rows 0-5, 4-9
        // machine 0: read buffer row 4 into state 4
        // machine 1: read buffer row 1 into state 5
        // machine 0: read state 5 into row 5
        // machine 1: read state 4 into row 0

        std::swap(buffState, buffBuffer);
        world.t += dt;
    }

    delete[] global;
    delete[] offset_0;
    delete[] offset_1;

    // err = clEnqueueReadBuffer(commands, buffState, CL_TRUE, 0, split_world, &world.state[0], 0, NULL, NULL);
    err = distributedCL.EnqueueReadBuffer(commands, buffState, CL_TRUE, 0,
                                          split_world, &state, row * (mid_world-1), 0, NULL,
                                          NULL, NULL,
                                          1, { 0 });

    err = distributedCL.EnqueueReadBuffer(commands, buffState, CL_TRUE, 0,
                                          split_world-row, &state, 0, 0, NULL,
                                          NULL, NULL,
                                          0, { 0 });

 

    if (distributedCL.world_rank == 0)
    {
        std::copy(state.data, state.data + world_size, &world.state[0]);
    }
    clReleaseMemObject(buffProperties);
    clReleaseMemObject(buffState);
    clReleaseMemObject(buffBuffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    distributedCL.Finalize();
}

}; // namepspace hpce

int main(int argc, char *argv[])
{
    distCL::distributedCL distributedCL(&argc, &argv);
    float dt = 0.1;
    unsigned n = 1;
    bool binary = false;

    if (argc > 1)
    {
        dt = strtof(argv[1], NULL);
    }
    if (argc > 2)
    {
        n = atoi(argv[2]);
    }
    if (argc > 3)
    {
        if (atoi(argv[3]))
            binary = true;
    }

    try
    {
        hpce::world_t world;
        if (distributedCL.world_rank == 0)
        {
            world = hpce::LoadWorld(std::cin);
            std::cerr << "Loaded world with w=" << world.w << ", h=" << world.h << std::endl;

            std::cerr << "Stepping by dt=" << dt << " for n=" << n << std::endl;
        }

        hpce::StepWorldDistributed(world, dt, n);

        if (distributedCL.world_rank == 0)
        {
            hpce::SaveWorld(std::cout, world, binary);
        }
        distributedCL.Finalize();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception : " << e.what() << std::endl;
        distributedCL.Finalize();
        return 1;
    }

    return 0;
}
