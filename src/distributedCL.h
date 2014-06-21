#include "data_barrier.h"
namespace distCL
{
//----------------------------------------------------------//
//----------------------------------------------------------//
//                  distCL class definition                 //
//----------------------------------------------------------//
//----------------------------------------------------------//

class distributedCL
{
  public:
    distributedCL() {};
    distributedCL(int *argc, char ***argv);
    ~distributedCL() {};

    void Init(int *argc, char ***argv);
    void Finalize();

    template <typename data_type>
    data_barrier<data_type> CreateBarrier(int size_x, int granularity,
                                          cl_context context,
                                          const init_list &target_machines);
    template <typename data_type>
    cl_int EnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer,
                              cl_bool blocking_write, size_t offset, size_t cb,
                              data_barrier<data_type> *barrier,
                              size_t barrier_offset,
                              cl_uint num_events_in_wait_list,
                              const distCL_event *event_wait_list,
                              distCL_event *send_event, distCL_event *recv_event,
                              int source_machine, init_list target_machines);

    template <typename data_type>
    cl_int EnqueueWriteBuffer(cl_command_queue command_queue, cl_mem buffer,
                              cl_bool blocking_write, size_t offset, size_t cb,
                              data_barrier<data_type> *barrier,
                              cl_uint num_events_in_wait_list,
                              const distCL_event *event_wait_list,
                              distCL_event *send_event, distCL_event *recv_event,
                              int source_machine, init_list target_machines);

    template <typename data_type>
    cl_int EnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer,
                             cl_bool blocking_read, size_t offset, size_t cb,
                             data_barrier<data_type> *barrier,
                             size_t barrier_offset,
                             cl_uint num_events_in_wait_list,
                             const distCL_event *event_wait_list,
                             distCL_event *send_event, distCL_event *recv_event,
                             int source_machine, init_list target_machines);

    template <typename data_type>
    cl_int EnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer,
                             cl_bool blocking_read, size_t offset, size_t cb,
                             data_barrier<data_type> *barrier,
                             cl_uint num_events_in_wait_list,
                             const distCL_event *event_wait_list,
                             distCL_event *send_event, distCL_event *recv_event,
                             int source_machine, init_list target_machines);

    cl_int EnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel,
                                cl_uint work_dim,
                                const size_t *global_work_offset,
                                const size_t *global_work_size,
                                const size_t *local_work_size,
                                cl_uint num_events_in_wait_list,
                                const distCL_event *event_wait_list,
                                distCL_event *event, init_list target_machines);

    cl_int GetPlatformIDs(cl_uint num_entries, cl_platform_id *platforms,
                          cl_uint *num_platforms, init_list target_machines);

    cl_int GetDeviceIDs(cl_platform_id platform, cl_device_type device_type,
                        cl_uint num_entries, cl_device_id *devices,
                        cl_uint *num_devices, init_list target_machines);

    //----------------------------------------------------------//
    //              Custom pass by reference CL command         //
    //----------------------------------------------------------//

    cl_int CreateContext(cl_context *context, cl_context_properties *properties,
                         cl_uint num_devices, const cl_device_id *devices,
                         void (*pfn_notify)(const char *errinfo,
                                            const void *private_info, size_t cb,
                                            void *user_data),
                         void *user_data, init_list target_machines);

    //----------------------------------------------------------//
    //              Custom pass by reference CL command         //
    //----------------------------------------------------------//

    cl_int CreateCommandQueue(cl_command_queue *command_queue, cl_context context,
                              cl_device_id device,
                              cl_command_queue_properties properties,
                              init_list target_machines);

    //----------------------------------------------------------//
    //              Custom pass by reference CL command         //
    //----------------------------------------------------------//

    cl_int CreateProgramWithSource(cl_program *program, cl_context context,
                                   cl_uint count, const char **strings,
                                   const size_t *lengths,
                                   init_list target_machines);

    cl_int BuildProgram(cl_program program, cl_uint num_devices,
                        const cl_device_id *device_list, const char *options,
                        void (*pfn_notify)(cl_program, void *user_data),
                        void *user_data, init_list target_machines);

    //----------------------------------------------------------//
    //              Custom pass by reference CL command         //
    //----------------------------------------------------------//

    cl_int CreateKernel(cl_kernel *kernel, cl_program program,
                        const char *kernel_name, init_list target_machines);

    //----------------------------------------------------------//
    //              Custom pass by reference CL command         //
    //----------------------------------------------------------//

    cl_int CreateBuffer(cl_mem *buffer, cl_context context, cl_mem_flags flags,
                        size_t size, void *host_ptr, init_list target_machines);

    cl_int WaitForEvents(cl_uint num_events, const distCL_event *event_list,
                         init_list target_machines);

    cl_int SetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size,
                        const void *arg_value, init_list target_machines);

    cl_int Finish(cl_command_queue command_queue, init_list target_machines);

    int world_rank, root_process, world_size;
    int tag_value;
};

//----------------------------------------------------------//
//----------------------------------------------------------//
//                  INTERNAL FUNCTIONS                      //
//----------------------------------------------------------//
//----------------------------------------------------------//
void check_source_and_target_valid(init_list shared_machine_list,
                                   init_list target_machines,
                                   int source_machine);
bool world_rank_in_list(const init_list &shared_machine_list, int world_rank);
cl_int wait_for_distCL_event(const distCL_event *event);

distributedCL::distributedCL(int *argc, char ***argv)
{
    Init(argc, argv);
}

void distributedCL::Init(int *argc, char ***argv)
{
    tag_value = 0;
    int initialized, provided, ierr;

    MPI_Initialized(&initialized);
    if (!initialized)
    {
        ierr = MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
        // IERR CATCH
    }

    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // IERR CATCH
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // IERR CATCH
}

void distributedCL::Finalize()
{
    int finalized, ierr;
    MPI_Finalized(&finalized);
    if (!finalized)
    {
        ierr = MPI_Finalize();
    }
}

template <typename data_type>
data_barrier<data_type> distributedCL::CreateBarrier(
    int size_x, int granularity, cl_context context,
    const init_list &target_machines)
{
    int temp_tag_value = tag_value;
    tag_value += size_x / granularity + 2;
    bool id_in_list = false;
    for (auto i = begin(target_machines); i < end(target_machines); ++i)
    {
        if (*i >= world_size)
            std::runtime_error("Targeting non-existent machine.");
        if (*i == world_rank)
        {
            id_in_list = true;
        }
    }
    if (id_in_list)
    {
        return data_barrier<data_type>(target_machines, size_x, granularity,
                                       temp_tag_value, world_rank, context);
    }
    return data_barrier<data_type>();
}

template <typename data_type>
cl_int distributedCL::EnqueueWriteBuffer(
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
    size_t offset, size_t cb, data_barrier<data_type> *barrier,
    cl_uint num_events_in_wait_list, const distCL_event *event_wait_list,
    distCL_event *send_event, distCL_event *recv_event, int source_machine,
    init_list target_machines)
{
    return EnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb,
                              barrier, 0, num_events_in_wait_list,
                              event_wait_list, send_event, recv_event,
                              source_machine, target_machines);
}

template <typename data_type>
cl_int internal_send(data_barrier<data_type> *barrier, size_t chunk_offset,
                     size_t chunks_sent, distCL_event *send_event,
                     int world_rank, init_list target_machines);

template <typename data_type>
cl_int distributedCL::EnqueueWriteBuffer(
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_write,
    size_t offset, size_t cb, data_barrier<data_type> *barrier,
    size_t barrier_offset, cl_uint num_events_in_wait_list,
    const distCL_event *event_wait_list, distCL_event *send_event,
    distCL_event *recv_event, int source_machine, init_list target_machines)
{
    cl_int err = 0;
    size_t chunk_offset;
    size_t chunks_sent;

    check_source_and_target_valid(barrier->shared_machine_list, target_machines,
                                  source_machine);

    if (target_machines.size() == 0 && barrier->shared_machine_list.size() != 0)
    {
        target_machines = barrier->shared_machine_list;
    }

    if (world_rank_in_list(target_machines, world_rank) ||
        world_rank == source_machine)
    {
        err = WaitForEvents(num_events_in_wait_list, event_wait_list, {});

        if (err != CL_SUCCESS)
        {
            return err;
        }

        if (barrier_offset % barrier->chunk_size != 0 ||
            (barrier_offset + cb) % barrier->chunk_size != 0)
            std::runtime_error("Invalid offset, must align with chunk granularity");

        // Check offset falls within appropriate boundaries.
        chunk_offset = barrier_offset / barrier->chunk_size;
        chunks_sent = cb / barrier->chunk_size;

        cb *= sizeof(data_type);
    }

    if (world_rank == source_machine)
    {
        if (send_event != NULL)
        {
            err = internal_send(barrier, chunk_offset, chunks_sent, send_event,
                                world_rank, target_machines);
        }
        else
        {
            distCL_event temp_event;
            err = internal_send(barrier, chunk_offset, chunks_sent, &temp_event,
                                world_rank, target_machines);
            WaitForEvents(1, &temp_event, {});
            delete[] temp_event.events;
        }
        if (world_rank_in_list(target_machines, world_rank))
        {
            if (recv_event != NULL)
            {
                cl_event r_event;
                err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write,
                                           offset, cb, &barrier->data[barrier_offset],
                                           0, NULL, &r_event);
                recv_event->size = 1;
                recv_event->events = new std::shared_ptr<cl_event>[1];
                recv_event->events[0] = std::make_shared<cl_event>(r_event);
            }
            else
            {
                err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write,
                                           offset, cb, &barrier->data[barrier_offset],
                                           0, NULL, NULL);
            }
        }
    }
    else
    {
        for (auto target_machine = begin(target_machines);
             target_machine < end(target_machines); ++target_machine)
        {

            if (*target_machine == world_rank && *target_machine != source_machine)
            {

                int errcode_ret;
                cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);
                std::shared_ptr<cl_event> temp_event =
                    std::make_shared<cl_event>(new_event);
                barrier->receive_data(temp_event, source_machine);

                if (recv_event != NULL)
                {
                    cl_event r_event;
                    recv_event->size = 1;
                    recv_event->events = new std::shared_ptr<cl_event>[1];
                    recv_event->events[0] = std::make_shared<cl_event>(r_event);
                    err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write,
                                               offset, cb, &barrier->data[barrier_offset],
                                               1, temp_event.get(),
                                               recv_event->events[0].get());
                }
                else
                {
                    err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write,
                                               offset, cb, &barrier->data[barrier_offset],
                                               1, temp_event.get(), NULL);
                }
            }
        }
    }
    return err;
}

template <typename data_type>
cl_int internal_send(data_barrier<data_type> *barrier, size_t chunk_offset,
                     size_t chunks_sent, distCL_event *send_event,
                     int world_rank, init_list target_machines)
{
    int event_id = 0;
    int errcode_ret = CL_SUCCESS;

    send_event->size = target_machines.size();
    send_event->events = new std::shared_ptr<cl_event>[send_event->size];

    for (auto target_machine = begin(target_machines);
         target_machine < end(target_machines); ++target_machine)
    {

        if (*target_machine != world_rank)
        {
            cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);
            send_event->events[event_id] = std::make_shared<cl_event>(new_event);

            barrier->send_data(send_event->events[event_id], *target_machine,
                               chunk_offset, chunks_sent);
            event_id++;
        }
        else
        {
            send_event->size -= 1;
        }
    }
    return errcode_ret;
}

template <typename data_type>
cl_int distributedCL::EnqueueReadBuffer(
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
    size_t offset, size_t cb, data_barrier<data_type> *barrier,
    cl_uint num_events_in_wait_list, const distCL_event *event_wait_list,
    distCL_event *send_event, distCL_event *recv_event, int source_machine,
    init_list target_machines)
{
    return EnqueueReadBuffer(command_queue, buffer, blocking_read, offset, cb,
                             barrier, 0, num_events_in_wait_list, event_wait_list,
                             send_event, recv_event, source_machine,
                             target_machines);
}

template <typename data_type>
cl_int internal_send(data_barrier<data_type> *barrier, size_t chunk_offset,
                     size_t chunks_sent, std::shared_ptr<cl_event> lock,
                     distCL_event *send_event, int world_rank,
                     init_list target_machines)
{
    int event_id = 0;
    int errcode_ret = CL_SUCCESS;

    send_event->size = target_machines.size();
    send_event->events = new std::shared_ptr<cl_event>[send_event->size];

    for (auto target_machine = begin(target_machines);
         target_machine < end(target_machines); ++target_machine)
    {

        if (*target_machine != world_rank)
        {

            cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);
            send_event->events[event_id] = std::make_shared<cl_event>(new_event);

            barrier->send_data(lock, send_event->events[event_id], *target_machine,
                               chunk_offset, chunks_sent);
            event_id++;
        }
        else
        {
            send_event->size -= 1;
        }
    }
    return errcode_ret;
}

template <typename data_type>
cl_int internal_recv(data_barrier<data_type> *barrier, distCL_event *recv_event,
                     int source_machine)
{
    recv_event->size = 1;
    recv_event->events = new std::shared_ptr<cl_event>[1];

    int errcode_ret;
    cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);
    recv_event->events[0] = std::make_shared<cl_event>(new_event);

    barrier->receive_data(recv_event->events[0], source_machine);
    return errcode_ret;
}

template <typename data_type>
cl_int distributedCL::EnqueueReadBuffer(
    cl_command_queue command_queue, cl_mem buffer, cl_bool blocking_read,
    size_t offset, size_t cb, data_barrier<data_type> *barrier,
    size_t barrier_offset, cl_uint num_events_in_wait_list,
    const distCL_event *event_wait_list, distCL_event *send_event,
    distCL_event *recv_event, int source_machine, init_list target_machines)
{
    cl_int err = 0;
    int chunk_offset;
    int chunks_sent;

    check_source_and_target_valid(barrier->shared_machine_list, target_machines,
                                  source_machine);

    if (target_machines.size() == 0)
    {
        target_machines = barrier->shared_machine_list;
    }

    if (world_rank_in_list(target_machines, world_rank) ||
        world_rank == source_machine)
    {
        err = WaitForEvents(num_events_in_wait_list, event_wait_list, {});

        if (err != CL_SUCCESS)
        {
            return err;
        }

        if (barrier_offset % barrier->chunk_size != 0 ||
            (barrier_offset + cb) % barrier->chunk_size != 0)
            std::runtime_error("Invalid offset, must align with chunk granularity");

        // Check offset falls within appropriate boundaries.
        chunk_offset = barrier_offset / barrier->chunk_size;
        chunks_sent = cb / barrier->chunk_size;

        cb *= sizeof(data_type);
    }

    if (world_rank == source_machine)
    {
        cl_event lock_event;
        std::shared_ptr<cl_event> lock = std::make_shared<cl_event>(lock_event);

        err = clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, cb,
                                  &barrier->data[barrier_offset], 0, NULL,
                                  lock.get());

        if (send_event != NULL)
        {
            err = internal_send(barrier, chunk_offset, chunks_sent, lock, send_event,
                                world_rank, target_machines);
        }
        else
        {
            distCL_event temp_event;
            err = internal_send(barrier, chunk_offset, chunks_sent, lock, &temp_event,
                                world_rank, target_machines);
            WaitForEvents(1, &temp_event, {});
            delete[] temp_event.events;
        }
    }
    else
    {
        if (world_rank_in_list(target_machines, world_rank))
        {
            if (world_rank != source_machine)
            {
                if (recv_event != NULL)
                {
                    err = internal_recv(barrier, recv_event, source_machine);
                }
                else
                {
                    distCL_event temp_event;
                    err = internal_recv(barrier, &temp_event, source_machine);

                    WaitForEvents(1, &temp_event, {});
                    delete[] temp_event.events;
                }
            }
        }
        err = CL_SUCCESS;
    }
    return err;
}

cl_int distributedCL::EnqueueNDRangeKernel(
    cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, cl_uint num_events_in_wait_list,
    const distCL_event *event_wait_list, distCL_event *event,
    init_list target_machines)
{
    cl_int err = 0;
    if (world_rank_in_list(target_machines, world_rank) ||
        target_machines.size() == 0)
    {
        err = WaitForEvents(num_events_in_wait_list, event_wait_list, {});
        if (err != CL_SUCCESS)
        {
            return err;
        }

        if (event != NULL)
        {
            event->size = 1;
            event->events = new std::shared_ptr<cl_event>[1];
            err = clEnqueueNDRangeKernel(
                command_queue, kernel, work_dim, global_work_offset, global_work_size,
                local_work_size, 0, NULL, event->events[0].get());
        }
        else
        {
            err = clEnqueueNDRangeKernel(command_queue, kernel, work_dim,
                                         global_work_offset, global_work_size,
                                         local_work_size, 0, NULL, NULL);
        }
    }
    else
    {
        err = CL_SUCCESS;
    }

    return err;
}

cl_int distributedCL::GetPlatformIDs(cl_uint num_entries,
                                     cl_platform_id *platforms,
                                     cl_uint *num_platforms,
                                     init_list target_machines)
{
    cl_int err;
    if (world_rank_in_list(target_machines, world_rank) ||
        target_machines.size() == 0)
    {
        err = clGetPlatformIDs(num_entries, platforms, num_platforms);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}

cl_int distributedCL::GetDeviceIDs(cl_platform_id platform,
                                   cl_device_type device_type,
                                   cl_uint num_entries, cl_device_id *devices,
                                   cl_uint *num_devices,
                                   init_list target_machines)
{
    cl_int err;
    if (world_rank_in_list(target_machines, world_rank) ||
        target_machines.size() == 0)
    {
        err = clGetDeviceIDs(platform, device_type, num_entries, devices,
                             num_devices);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}

cl_int distributedCL::CreateContext(
    cl_context *context, cl_context_properties *properties, cl_uint num_devices,
    const cl_device_id *devices,
    void (*pfn_notify)(const char *errinfo, const void *private_info, size_t cb,
                       void *user_data),
    void *user_data, init_list target_machines)
{
    cl_int err;
    if (world_rank_in_list(target_machines, world_rank) ||
        target_machines.size() == 0)
    {
        *context = clCreateContext(properties, num_devices, devices, pfn_notify,
                                   user_data, &err);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}

cl_int distributedCL::CreateCommandQueue(cl_command_queue *command_queue,
                                         cl_context context,
                                         cl_device_id device,
                                         cl_command_queue_properties properties,
                                         init_list target_machines)
{
    cl_int err;
    if (world_rank_in_list(target_machines, world_rank) ||
        target_machines.size() == 0)
    {
        *command_queue = clCreateCommandQueue(context, device, properties, &err);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}

cl_int distributedCL::CreateProgramWithSource(cl_program *program,
                                              cl_context context, cl_uint count,
                                              const char **strings,
                                              const size_t *lengths,
                                              init_list target_machines)
{
    cl_int err;
    if (world_rank_in_list(target_machines, world_rank) ||
        target_machines.size() == 0)
    {
        *program =
            clCreateProgramWithSource(context, count, strings, lengths, &err);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}

cl_int distributedCL::BuildProgram(
    cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options, void (*pfn_notify)(cl_program, void *user_data),
    void *user_data, init_list target_machines)
{
    cl_int err;
    if (world_rank_in_list(target_machines, world_rank) ||
        target_machines.size() == 0)
    {
        err = clBuildProgram(program, num_devices, device_list, options, pfn_notify,
                             user_data);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}

cl_int distributedCL::CreateKernel(cl_kernel *kernel, cl_program program,
                                   const char *kernel_name,
                                   init_list target_machines)
{
    cl_int err;
    if (world_rank_in_list(target_machines, world_rank) ||
        target_machines.size() == 0)
    {
        *kernel = clCreateKernel(program, kernel_name, &err);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}

cl_int distributedCL::CreateBuffer(cl_mem *buffer, cl_context context,
                                   cl_mem_flags flags, size_t size,
                                   void *host_ptr, init_list target_machines)
{
    cl_int err;
    if (world_rank_in_list(target_machines, world_rank) ||
        target_machines.size() == 0)
    {
        *buffer = clCreateBuffer(context, flags, size, host_ptr, &err);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}

cl_int distributedCL::WaitForEvents(cl_uint num_events,
                                    const distCL_event *event_list,
                                    init_list target_machines)
{
    cl_int err;

    if ((world_rank_in_list(target_machines, world_rank) ||
         target_machines.size() == 0) &&
        (event_list != NULL))
    {
        for (int i = 0; i < num_events; ++i)
        {
            err = wait_for_distCL_event(&event_list[i]);
            if (err != CL_SUCCESS)
                return err;
        }
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}

cl_int distributedCL::SetKernelArg(cl_kernel kernel, cl_uint arg_index,
                                   size_t arg_size, const void *arg_value,
                                   init_list target_machines)
{
    cl_int err;
    if (world_rank_in_list(target_machines, world_rank) ||
        target_machines.size() == 0)
    {
        err = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}

cl_int distributedCL::Finish(cl_command_queue command_queue,
                             init_list target_machines)
{
    cl_int err;
    if (world_rank_in_list(target_machines, world_rank) ||
        target_machines.size() == 0)
    {
        err = clFinish(command_queue);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}

//----------------------------------------------------------
//----------------------------------------------------------
//                  INTERNAL FUNCTIONS
//----------------------------------------------------------
//----------------------------------------------------------

cl_int wait_for_distCL_event(const distCL_event *event)
{
    cl_int err;
    for (int i = 0; i < event->size; ++i)
    {
        err = clWaitForEvents(1, event->events[i].get());
        if (err != CL_SUCCESS)
        {
            return err;
        }
    }
    return err;
}

bool world_rank_in_list(const init_list &shared_machine_list, int world_rank)
{
    for (auto i = begin(shared_machine_list); i < end(shared_machine_list); ++i)
    {
        if (*i == world_rank)
        {
            return true;
        }
    }

    return false;
}

void check_source_and_target_valid(init_list shared_machine_list,
                                   init_list target_machines,
                                   int source_machine)
{
    bool valid_target = true;
    for (auto i = begin(target_machines); i < end(target_machines); ++i)
    {
        if (!world_rank_in_list(shared_machine_list, *i))
        {
            valid_target = false;
        }
    }
    if (!world_rank_in_list(shared_machine_list, source_machine))
    {
        throw std::runtime_error("Source machine does not share memory barrier.");
    }

    if (!valid_target)
        throw std::runtime_error("Target machine does not share memory barrier.");
}
};
