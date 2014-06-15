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

    template<typename data_type>
    data_barrier<data_type> create_barrier(const std::initializer_list<int> &target_machines, int size_x, int granularity, cl_context context);

    template <typename data_type>
    cl_int EnqueueWriteBuffer(cl_command_queue command_queue,
                              cl_mem buffer,
                              cl_bool blocking_write,
                              size_t offset,
                              size_t cb,
                              data_barrier<data_type> *barrier,
                              size_t barrier_offset,
                              cl_uint num_events_in_wait_list,
                              const distCL_event *event_wait_list,
                              distCL_event *send_event,
                              distCL_event *recv_event,
                              int source_machine,
                              init_list target_machines);

    template <typename data_type>
    cl_int EnqueueWriteBuffer(cl_command_queue command_queue,
                              cl_mem buffer,
                              cl_bool blocking_write,
                              size_t offset,
                              size_t cb,
                              data_barrier<data_type> *barrier,
                              cl_uint num_events_in_wait_list,
                              const distCL_event *event_wait_list,
                              distCL_event *send_event,
                              distCL_event *recv_event,
                              int source_machine,
                              init_list target_machines);

    template <typename data_type>
    cl_int EnqueueReadBuffer(cl_command_queue command_queue,
                             cl_mem buffer,
                             cl_bool blocking_read,
                             size_t offset,
                             size_t cb,
                             data_barrier<data_type> *barrier,
                             cl_uint num_events_in_wait_list,
                             const distCL_event *event_wait_list,
                             distCL_event *send_event,
                             distCL_event *recv_event,
                             int source_machine,
                             init_list target_machines);

    cl_int EnqueueNDRangeKernel ( cl_command_queue command_queue,
                                  cl_kernel kernel,
                                  cl_uint work_dim,
                                  const size_t *global_work_offset,
                                  const size_t *global_work_size,
                                  const size_t *local_work_size,
                                  cl_uint num_events_in_wait_list,
                                  const distCL_event *event_wait_list,
                                  distCL_event *event,
                                  init_list target_machines);

    cl_int GetPlatformIDs(cl_uint num_entries,
                          cl_platform_id *platforms,
                          cl_uint *num_platforms,
                          init_list target_machines);


    cl_int GetDeviceIDs(  cl_platform_id platform,
                          cl_device_type device_type,
                          cl_uint num_entries,
                          cl_device_id *devices,
                          cl_uint *num_devices,
                          init_list target_machines);


    //----------------------------------------------------------//
    //              Custom pass by reference CL command         //
    //----------------------------------------------------------//

    cl_int CreateContext(cl_context *context,
                         cl_context_properties *properties,
                         cl_uint num_devices,
                         const cl_device_id *devices,
                         void (*pfn_notify) (
                             const char *errinfo,
                             const void *private_info,
                             size_t cb,
                             void *user_data
                         ),
                         void *user_data,
                         init_list target_machines);

    //----------------------------------------------------------//
    //              Custom pass by reference CL command         //
    //----------------------------------------------------------//

    cl_int CreateCommandQueue(  cl_command_queue *command_queue,
                                cl_context context,
                                cl_device_id device,
                                cl_command_queue_properties properties,
                                init_list target_machines);

    //----------------------------------------------------------//
    //              Custom pass by reference CL command         //
    //----------------------------------------------------------//

    cl_int CreateProgramWithSource ( cl_program *program,
                                     cl_context context,
                                     cl_uint count,
                                     const char **strings,
                                     const size_t *lengths,
                                     init_list target_machines);


    cl_int BuildProgram ( cl_program program,
                          cl_uint num_devices,
                          const cl_device_id *device_list,
                          const char *options,
                          void (*pfn_notify)(cl_program, void *user_data),
                          void *user_data,
                          init_list target_machines);

    //----------------------------------------------------------//
    //              Custom pass by reference CL command         //
    //----------------------------------------------------------//

    cl_int  CreateKernel ( cl_kernel *kernel,
                           cl_program  program,
                           const char *kernel_name,
                           init_list target_machines);

    //----------------------------------------------------------//
    //              Custom pass by reference CL command         //
    //----------------------------------------------------------//

    cl_int CreateBuffer ( cl_mem *buffer,
                          cl_context context,
                          cl_mem_flags flags,
                          size_t size,
                          void *host_ptr,
                          init_list target_machines);

    cl_int WaitForEvents(cl_uint num_events,
                         const distCL_event *event_list,
                         init_list target_machines);

    cl_int SetKernelArg ( cl_kernel kernel,
                          cl_uint arg_index,
                          size_t arg_size,
                          const void *arg_value,
                          init_list target_machines);


    cl_int Finish (   cl_command_queue command_queue,
                      init_list target_machines);

    // clCreateBuffer
    // clSetKernelArg(
    // clGetKernelWorkGroupInfo


    int my_id, root_process, num_processes;
    MPI_Status status;
    int tag_value;


};


//----------------------------------------------------------//
//----------------------------------------------------------//
//                  INTERNAL FUNCTIONS                      //
//----------------------------------------------------------//
//----------------------------------------------------------//
void check_source_and_target_valid(init_list shared_machine_list, init_list target_machines, int source_machine);
bool my_id_in_list(const init_list &shared_machine_list, int my_id);
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
        ierr = MPI_Init_thread(argc, argv, MPI_THREAD_SERIALIZED, &provided);
        //IERR CATCH
    }

    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    //IERR CATCH
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    //IERR CATCH
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


template<typename data_type>
data_barrier<data_type> distributedCL::create_barrier(const std::initializer_list<int> &target_machines, int size_x, int granularity, cl_context context)
{
    int temp_tag_value = tag_value;
    tag_value += size_x / granularity + 2;
    bool id_in_list = false;
    for (auto i = begin(target_machines); i < end(target_machines); ++i)
    {
        if (*i == my_id)
        {
            id_in_list = true;
        }
    }
    if (id_in_list)
    {
        return data_barrier<data_type>(target_machines, size_x, granularity, temp_tag_value, my_id, context);
    }
    return data_barrier<data_type>();

}

template <typename data_type>
cl_int distributedCL::EnqueueWriteBuffer(cl_command_queue command_queue,
        cl_mem buffer,
        cl_bool blocking_write,
        size_t offset,
        size_t cb,
        data_barrier<data_type> *barrier,
        cl_uint num_events_in_wait_list,
        const distCL_event *event_wait_list,
        distCL_event *send_event,
        distCL_event *recv_event,
        int source_machine,
        init_list target_machines)
{
    cl_int err = 0;
    int chunk_offset;
    int chunks_sent;
    bool target_self = false;

    check_source_and_target_valid(barrier->shared_machine_list, target_machines, source_machine);

    if (target_machines.size() == 0)
    {
        target_machines = barrier->shared_machine_list;
    }

    if (my_id_in_list(target_machines, my_id) || my_id == source_machine)
    {
        err = WaitForEvents(num_events_in_wait_list, event_wait_list, {});

        if (err != CL_SUCCESS)
        {
            return err;
        }



        if (offset % barrier->chunk_size != 0 || (offset + cb) % barrier->chunk_size != 0 )
            std::runtime_error("Invalid offset, must align with chunk granularity");

        // Check offset falls within appropriate boundaries.
        chunk_offset = offset / barrier->chunk_size;
        chunks_sent = cb / barrier->chunk_size;

        cb *= sizeof(data_type);
    }

    if (my_id == source_machine)
    {
        if (send_event != NULL)
        {
            int event_id = 0;

            send_event->size = target_machines.size();
            send_event->events = new std::shared_ptr<cl_event>[send_event->size];

            for (auto target_machine = begin(target_machines);
                    target_machine < end(target_machines);
                    ++target_machine)
            {

                if (*target_machine != my_id)
                {
                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);
                    send_event->events[event_id] = std::make_shared<cl_event>(new_event);

                    barrier->send_data(send_event->events[event_id], *target_machine, chunk_offset, chunks_sent);
                    event_id++;
                }
                else
                {
                    send_event->size -= 1;
                    target_self = true;

                }
            }
        }
        else
        {
            distCL_event temp_event;
            int event_id = 0;

            temp_event.size = target_machines.size();
            temp_event.events = new std::shared_ptr<cl_event>[temp_event.size];

            for (auto target_machine = begin(target_machines);
                    target_machine < end(target_machines);
                    ++target_machine)
            {
                if (*target_machine != my_id)
                {
                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);

                    temp_event.events[event_id] = std::make_shared<cl_event>(new_event);

                    barrier->send_data(temp_event.events[event_id], *target_machine, chunk_offset, chunks_sent);
                    event_id++;
                }
                else
                {
                    temp_event.size -= 1;
                    target_self = true;
                }
            }
            WaitForEvents(event_id, &temp_event, {});
            delete [] temp_event.events;
        }
        if (target_self == true)
        {
            if (recv_event != NULL)
            {
                cl_event r_event;
                err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 0, NULL, &r_event);
                recv_event->size = 1;
                recv_event->events = new std::shared_ptr<cl_event>[1];
                recv_event->events[0] = std::make_shared<cl_event>(r_event);
            }
            else
            {
                err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 0, NULL, NULL);
            }
        }
    }
    else
    {
        for (auto target_machine = begin(target_machines);
                target_machine < end(target_machines);
                ++target_machine)
        {

            if (*target_machine == my_id && *target_machine != source_machine)
            {

                distCL_event temp_event;
                temp_event.size = 1;
                temp_event.events = new std::shared_ptr<cl_event>[1];
                int errcode_ret;
                cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);
                temp_event.events[0] = std::make_shared<cl_event>(new_event);

                barrier->receive_data(temp_event.events[0], source_machine);

                if (recv_event != NULL)
                {
                    cl_event r_event;
                    err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 1, temp_event.events[0].get(), &r_event);
                    recv_event->size = 1;
                    recv_event->events = new std::shared_ptr<cl_event>[1];
                    recv_event->events[0] = std::make_shared<cl_event>(r_event);

                }
                else
                {
                    err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 1, temp_event.events[0].get(), NULL);
                }
                delete [] temp_event.events;
            }
        }
    }
    return err;
}


template <typename data_type>
cl_int distributedCL::EnqueueWriteBuffer(cl_command_queue command_queue,
        cl_mem buffer,
        cl_bool blocking_write,
        size_t offset,
        size_t cb,
        data_barrier<data_type> *barrier,
        size_t barrier_offset,
        cl_uint num_events_in_wait_list,
        const distCL_event *event_wait_list,
        distCL_event *send_event,
        distCL_event *recv_event,
        int source_machine,
        init_list target_machines)
{
    cl_int err = 0;
    int chunk_offset;
    int chunks_sent;
    bool target_self = false;

    check_source_and_target_valid(barrier->shared_machine_list, target_machines, source_machine);

    if (target_machines.size() == 0)
    {
        target_machines = barrier->shared_machine_list;
    }

    if (my_id_in_list(target_machines, my_id) || my_id == source_machine)
    {
        err = WaitForEvents(num_events_in_wait_list, event_wait_list, {});

        if (err != CL_SUCCESS)
        {
            return err;
        }



        if (barrier_offset % barrier->chunk_size != 0 || (barrier_offset + cb) % barrier->chunk_size != 0 )
            std::runtime_error("Invalid offset, must align with chunk granularity");

        // Check offset falls within appropriate boundaries.
        chunk_offset = barrier_offset / barrier->chunk_size;
        chunks_sent = cb / barrier->chunk_size;

        cb *= sizeof(data_type);
    }

    if (my_id == source_machine)
    {
        if (send_event != NULL)
        {
            int event_id = 0;

            send_event->size = target_machines.size();
            send_event->events = new std::shared_ptr<cl_event>[send_event->size];

            for (auto target_machine = begin(target_machines);
                    target_machine < end(target_machines);
                    ++target_machine)
            {

                if (*target_machine != my_id)
                {
                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);
                    send_event->events[event_id] = std::make_shared<cl_event>(new_event);

                    barrier->send_data(send_event->events[event_id], *target_machine, chunk_offset, chunks_sent);
                    event_id++;
                }
                else
                {
                    send_event->size -= 1;
                    target_self = true;

                }
            }
        }
        else
        {
            distCL_event temp_event;
            int event_id = 0;

            temp_event.size = target_machines.size();
            temp_event.events = new std::shared_ptr<cl_event>[temp_event.size];

            for (auto target_machine = begin(target_machines);
                    target_machine < end(target_machines);
                    ++target_machine)
            {
                if (*target_machine != my_id)
                {
                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);

                    temp_event.events[event_id] = std::make_shared<cl_event>(new_event);

                    barrier->send_data(temp_event.events[event_id], *target_machine, chunk_offset, chunks_sent);
                    event_id++;
                }
                else
                {
                    temp_event.size -= 1;
                    target_self = true;
                }
            }
            WaitForEvents(event_id, &temp_event, {});
            delete [] temp_event.events;
        }
        if (target_self == true)
        {
            if (recv_event != NULL)
            {
                cl_event r_event;
                err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, &barrier->data[barrier_offset], 0, NULL, &r_event);
                recv_event->size = 1;
                recv_event->events = new std::shared_ptr<cl_event>[1];
                recv_event->events[0] = std::make_shared<cl_event>(r_event);
            }
            else
            {
                err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, &barrier->data[barrier_offset], 0, NULL, NULL);
            }
        }
    }
    else
    {
        for (auto target_machine = begin(target_machines);
                target_machine < end(target_machines);
                ++target_machine)
        {

            if (*target_machine == my_id && *target_machine != source_machine)
            {

                distCL_event temp_event;
                temp_event.size = 1;
                temp_event.events = new std::shared_ptr<cl_event>[1];
                int errcode_ret;
                cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);
                temp_event.events[0] = std::make_shared<cl_event>(new_event);

                barrier->receive_data(temp_event.events[0], source_machine);

                if (recv_event != NULL)
                {
                    cl_event r_event;
                    err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, &barrier->data[barrier_offset], 1, temp_event.events[0].get(), &r_event);
                    recv_event->size = 1;
                    recv_event->events = new std::shared_ptr<cl_event>[1];
                    recv_event->events[0] = std::make_shared<cl_event>(r_event);

                }
                else
                {
                    err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, &barrier->data[barrier_offset], 1, temp_event.events[0].get(), NULL);
                }
                delete [] temp_event.events;
            }
        }
    }
    return err;
}

template <typename data_type>
cl_int distributedCL::EnqueueReadBuffer(cl_command_queue command_queue,
                                        cl_mem buffer,
                                        cl_bool blocking_read,
                                        size_t offset,
                                        size_t cb,
                                        data_barrier<data_type> *barrier,
                                        cl_uint num_events_in_wait_list,
                                        const distCL_event *event_wait_list,
                                        distCL_event *send_event,
                                        distCL_event *recv_event,
                                        int source_machine,
                                        init_list target_machines)
{
    cl_int err = 0;
    int chunk_offset;
    int chunks_sent;

    check_source_and_target_valid(barrier->shared_machine_list, target_machines, source_machine);

    if (target_machines.size() == 0)
    {
        target_machines = barrier->shared_machine_list;
    }

    if (my_id_in_list(target_machines, my_id) || my_id == source_machine)
    {
        err = WaitForEvents(num_events_in_wait_list, event_wait_list, {});

        if (err != CL_SUCCESS)
        {
            return err;
        }



        if (offset % barrier->chunk_size != 0 || (offset + cb) % barrier->chunk_size != 0 )
            std::runtime_error("Invalid offset, must align with chunk granularity");

        // Check offset falls within appropriate boundaries.
        chunk_offset = offset / barrier->chunk_size;
        chunks_sent = cb / barrier->chunk_size;

        cb *= sizeof(data_type);
    }

    if (my_id == source_machine)
    {
        std::shared_ptr<cl_event> lock(new cl_event);

        err = clEnqueueReadBuffer (command_queue,
                                   buffer,
                                   blocking_read,
                                   offset,
                                   cb,
                                   barrier->data,
                                   0,
                                   NULL,
                                   lock.get());

        if (send_event != NULL)
        {
            int event_id = 0;

            send_event->size = target_machines.size();
            send_event->events = new std::shared_ptr<cl_event>[send_event->size];

            for (auto target_machine = begin(target_machines);
                    target_machine < end(target_machines);
                    ++target_machine)
            {

                if (*target_machine != my_id)
                {
                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);
                    send_event->events[event_id] = std::make_shared<cl_event>(new_event);

                    barrier->send_data(lock, send_event->events[event_id], *target_machine, chunk_offset, chunks_sent);
                    event_id++;
                }
                else
                {
                    send_event->size -= 1;
                }
            }
        }
        else
        {
            distCL_event temp_event;
            int event_id = 0;

            temp_event.size = target_machines.size();
            temp_event.events = new std::shared_ptr<cl_event>[temp_event.size];

            for (auto target_machine = begin(target_machines);
                    target_machine < end(target_machines);
                    ++target_machine)
            {

                if (*target_machine != my_id)
                {
                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);
                    temp_event.events[event_id] = std::make_shared<cl_event>(new_event);

                    barrier->send_data(lock, temp_event.events[event_id], *target_machine, chunk_offset, chunks_sent);
                    event_id++;
                }
                else
                {
                    temp_event.size -= 1;
                }
            }
            WaitForEvents(event_id, &temp_event, {});
            delete [] temp_event.events;
        }
    }
    else
    {
        if (my_id_in_list(target_machines, my_id))
        {
            if (my_id != source_machine)
            {
                if (recv_event != NULL)
                {
                    recv_event->size = 1;
                    recv_event->events =  new std::shared_ptr<cl_event>[recv_event->size];

                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);
                    recv_event->events[0] = std::make_shared<cl_event>(new_event);

                    barrier->receive_data(recv_event->events[0], source_machine);

                }
                else
                {
                    distCL_event temp_event;
                    temp_event.size = 1;
                    temp_event.events =  new std::shared_ptr<cl_event>[temp_event.size];

                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(barrier->context, &errcode_ret);
                    temp_event.events[0] = std::make_shared<cl_event>(new_event);

                    barrier->receive_data(temp_event.events[0], source_machine);

                    WaitForEvents(1, &temp_event, {});
                    delete [] temp_event.events;
                }

            }
        }
        err = CL_SUCCESS;
    }
    return err;
}

cl_int distributedCL::EnqueueNDRangeKernel(cl_command_queue command_queue,
        cl_kernel kernel,
        cl_uint work_dim,
        const size_t *global_work_offset,
        const size_t *global_work_size,
        const size_t *local_work_size,
        cl_uint num_events_in_wait_list,
        const distCL_event *event_wait_list,
        distCL_event *event,
        init_list target_machines)
{
    cl_int err = 0;
    if (my_id_in_list(target_machines, my_id) || target_machines.size() == 0)
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
            err = clEnqueueNDRangeKernel (command_queue,
                                          kernel,
                                          work_dim,
                                          global_work_offset,
                                          global_work_size,
                                          local_work_size,
                                          0,
                                          NULL,
                                          event->events[0].get());
        }
        else
        {
            err = clEnqueueNDRangeKernel (command_queue,
                                          kernel,
                                          work_dim,
                                          global_work_offset,
                                          global_work_size,
                                          local_work_size,
                                          0,
                                          NULL,
                                          NULL);

        }

    }
    else
    {
        err = CL_SUCCESS;
    }

    return err;
}

cl_int distributedCL::GetPlatformIDs(    cl_uint num_entries,
        cl_platform_id *platforms,
        cl_uint *num_platforms,
        init_list target_machines)
{
    cl_int err;
    if (my_id_in_list(target_machines, my_id) || target_machines.size() == 0)
    {
        err = clGetPlatformIDs(     num_entries,
                                    platforms,
                                    num_platforms);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}




cl_int distributedCL::GetDeviceIDs(  cl_platform_id platform,
                                     cl_device_type device_type,
                                     cl_uint num_entries,
                                     cl_device_id *devices,
                                     cl_uint *num_devices,
                                     init_list target_machines)
{
    cl_int err;
    if (my_id_in_list(target_machines, my_id) || target_machines.size() == 0)
    {
        err = clGetDeviceIDs(platform, device_type, num_entries, devices, num_devices);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}

cl_int distributedCL::CreateContext(cl_context *context,
                                    cl_context_properties *properties,
                                    cl_uint num_devices,
                                    const cl_device_id *devices,
                                    void (*pfn_notify) (
                                        const char *errinfo,
                                        const void *private_info,
                                        size_t cb,
                                        void *user_data
                                    ),
                                    void *user_data,
                                    init_list target_machines)
{
    cl_int err;
    if (my_id_in_list(target_machines, my_id) || target_machines.size() == 0)
    {
        *context = clCreateContext(properties, num_devices, devices, pfn_notify, user_data, &err);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;
}


cl_int distributedCL::CreateCommandQueue(  cl_command_queue *command_queue,
        cl_context context,
        cl_device_id device,
        cl_command_queue_properties properties,
        init_list target_machines)
{
    cl_int err;
    if (my_id_in_list(target_machines, my_id) || target_machines.size() == 0)
    {
        *command_queue = clCreateCommandQueue(context, device, properties, &err);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;

}

cl_int distributedCL::CreateProgramWithSource ( cl_program *program,
        cl_context context,
        cl_uint count,
        const char **strings,
        const size_t *lengths,
        init_list target_machines)
{
    cl_int err;
    if (my_id_in_list(target_machines, my_id) || target_machines.size() == 0)
    {
        *program = clCreateProgramWithSource(context, count, strings, lengths, &err);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;

}

cl_int distributedCL::BuildProgram ( cl_program program,
                                     cl_uint num_devices,
                                     const cl_device_id *device_list,
                                     const char *options,
                                     void (*pfn_notify)(cl_program, void *user_data),
                                     void *user_data,
                                     init_list target_machines)
{
    cl_int err;
    if (my_id_in_list(target_machines, my_id) || target_machines.size() == 0)
    {
        err = clBuildProgram(program, num_devices, device_list, options, pfn_notify, user_data);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;

}

cl_int  distributedCL::CreateKernel ( cl_kernel *kernel,
                                      cl_program  program,
                                      const char *kernel_name,
                                      init_list target_machines)
{
    cl_int err;
    if (my_id_in_list(target_machines, my_id) || target_machines.size() == 0)
    {
        *kernel = clCreateKernel(program, kernel_name, &err);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;

}

cl_int distributedCL::CreateBuffer ( cl_mem *buffer,
                                     cl_context context,
                                     cl_mem_flags flags,
                                     size_t size,
                                     void *host_ptr,
                                     init_list target_machines)
{
    cl_int err;
    if (my_id_in_list(target_machines, my_id) || target_machines.size() == 0)
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


    if ((my_id_in_list(target_machines, my_id) || target_machines.size() == 0) && (event_list != NULL))
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

cl_int distributedCL::SetKernelArg ( cl_kernel kernel,
                                     cl_uint arg_index,
                                     size_t arg_size,
                                     const void *arg_value,
                                     init_list target_machines)

{
    cl_int err;
    if (my_id_in_list(target_machines, my_id) || target_machines.size() == 0)
    {
        err = clSetKernelArg(kernel, arg_index, arg_size, arg_value);
    }
    else
    {
        err = CL_SUCCESS;
    }
    return err;

}


cl_int distributedCL::Finish (   cl_command_queue command_queue,
                                 init_list target_machines)
{
    cl_int err;
    if (my_id_in_list(target_machines, my_id) || target_machines.size() == 0)
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

bool my_id_in_list(const init_list &shared_machine_list, int my_id)
{
    for (auto i = begin(shared_machine_list); i < end(shared_machine_list); ++i)
    {
        if (*i == my_id)
        {
            return true;
        }
    }

    return false;
}

void check_source_and_target_valid(init_list shared_machine_list, init_list target_machines, int source_machine)
{
    bool valid_target = true;
    for (auto i = begin(target_machines);
            i < end(target_machines); ++i)
    {
        if (!my_id_in_list(shared_machine_list, *i))
        {
            valid_target = false;
        }
    }
    if (!my_id_in_list(shared_machine_list, source_machine))
    {
        throw std::runtime_error("Source machine does not share memory barrier.");
    }

    if (!valid_target)
        throw std::runtime_error("Target machine does not share memory barrier.");

}

};
