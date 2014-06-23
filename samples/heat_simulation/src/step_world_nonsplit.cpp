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

template <typename data_type>
void sync_data(data_type *data, int source_machine)
{
    int num_elements = sizeof(data) / sizeof(data_type);
    MPI_Bcast(&num_elements, 1, MPI_INT, source_machine, MPI_COMM_WORLD);
    MPI_Bcast(data, num_elements, convert_type(get_abstraction_data_type<data_type>()), source_machine,
              MPI_COMM_WORLD);
};

namespace hpce
{
namespace fs1910
{
//! Reference world stepping program
/*! \param dt Amount to step the world by.  Note that large steps will be unstable.
    \param n Number of times to step the world
    \note Overall time increment will be n*dt
*/

void kernel_xy(uint32_t x, uint32_t y, uint32_t w, float inner, float outer, 
    const float *world_state, const uint32_t *world_properties, float *buffer)
{
    unsigned index = y * w + x;

    if ((world_properties[index] & Cell_Fixed) || (world_properties[index] & Cell_Insulator))
    {
        // Do nothing, this cell never changes (e.g. a boundary, or an interior fixed-value heat-source)
        buffer[index] = world_state[index];
    }
    else
    {
        float contrib = inner;
        float acc = inner * world_state[index];

        // Cell above
        if (!(world_properties[index - w] & Cell_Insulator))
        {
            contrib += outer;
            acc += outer * world_state[index - w];
        }

        // Cell below
        if (!(world_properties[index + w] & Cell_Insulator))
        {
            contrib += outer;
            acc += outer * world_state[index + w];
        }

        // Cell left
        if (!(world_properties[index - 1] & Cell_Insulator))
        {
            contrib += outer;
            acc += outer * world_state[index - 1];
        }

        // Cell right
        if (!(world_properties[index + 1] & Cell_Insulator))
        {
            contrib += outer;
            acc += outer * world_state[index + 1];
        }

        // Scale the accumulate value by the number of places contributing to it
        float res = acc / contrib;
        // Then clamp to the range [0,1]
        res = std::min(1.0f, std::max(0.0f, res));
        buffer[index] = res;
    }
}

std::string LoadSource(const char *fileName)
{
    std::string baseDir = "src/fs1910";
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

void StepWorldV4DoubleBuffered(world_t &world, float dt, unsigned n)
{
    distCL::distributedCL distributedCL(NULL, NULL);
    int devType = CL_DEVICE_TYPE_GPU;

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
    std::string kernelSource = LoadSource("step_world_v3_kernel.cl");
    const char *src = kernelSource.c_str();

    err = distributedCL.CreateProgramWithSource(&program, context, 1,
                                                &src,
                                                NULL, {});

    err = distributedCL.BuildProgram(program, 0, NULL, NULL, NULL, NULL, {});

    cl_mem buffProperties, buffState, buffBuffer;

    sync_data(&world.w, 0);
    sync_data(&world.h, 0);
    sync_data(&world.alpha, 0);

    fprintf(stderr, "%d %d %f\n", world.w, world.h, world.alpha);

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
                                     world_size * sizeof(uint32_t), NULL, {});
    err = distributedCL.CreateBuffer(&buffState, context, CL_MEM_READ_WRITE,
                                     world_size * sizeof(float), NULL, {});
    err = distributedCL.CreateBuffer(&buffBuffer, context, CL_MEM_READ_WRITE,
                                     world_size * sizeof(float), NULL, {});

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

    err = distributedCL.EnqueueWriteBuffer(commands, buffProperties, CL_TRUE,
                                           0, world_size,
                                           &properties, 0,
                                           0, NULL,
                                           NULL, NULL,
                                           0, { 0, 1 });

    err = distributedCL.EnqueueWriteBuffer(commands, buffState, CL_TRUE,
                                           0, world_size,
                                           &state, 0,
                                           0, NULL,
                                           NULL, NULL,
                                           0, { 0, 1 });


    size_t *offset = new size_t[2];
    offset[0] = 0;
    offset[1] = 0;

    size_t *global = new size_t[2];
    global[0] = world.w;
    global[1] = world.h;

    size_t row = world.w;

    for (unsigned t = 0; t < n; ++t)
    {
        err = distributedCL.SetKernelArg(kernel, 2, sizeof(cl_mem),
                                         &buffState, {});
        err = distributedCL.SetKernelArg(kernel, 4, sizeof(cl_mem),
                                         &buffBuffer, {});

        err = distributedCL.EnqueueNDRangeKernel(commands, kernel,
                                                 2,
                                                 offset,
                                                 global,
                                                 NULL,
                                                 0,
                                                 NULL,
                                                 NULL, {});


        //  err = distributedCL.EnqueueReadBuffer(commands, buffState, CL_TRUE,
        //                                       row * 4, row,
        //                                       &state, row * 4,
        //                                       0, NULL,
        //                                       NULL, NULL,
        //                                       0, { 1 });

        // // if(distributedCL.world_rank == 1)
        // {
        //   fprintf(stderr, "%u   ", t);
        //   for (int i = 0; i <row; ++i)
        //   {
        //     fprintf(stderr, "%f ", state.data[row*4 + i]);
        //   }
        //   fprintf(stderr, "\n");
        // }
        clEnqueueBarrier(commands);
        
        std::swap(buffState, buffBuffer);
        world.t += dt;
    }

    delete[] global;
    delete[] offset;

    // err = clEnqueueReadBuffer(commands, buffState, CL_TRUE, 0, world_size, &world.state[0], 0, NULL, NULL);
    err = distributedCL.EnqueueReadBuffer(commands, buffState, CL_TRUE, 0,
                                          world_size, &state, 0, 0, NULL,
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
}; // namespace fs1910

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

        hpce::fs1910::StepWorldV4DoubleBuffered(world, dt, n);

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
