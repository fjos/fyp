#include "distributedCL.h"
namespace distCL
{

//HELPER FUNCTIONS
void check_source_and_target_valid(init_list shared_machine_list, init_list target_machines, int source_machine);

//functions
void WaitForEvents(int num_events, distCL_event *events);
void wait_for_distCL_event(distCL_event * event);
// void add_event(distCL_event_list *event_list, distCL_event *event);



template <typename data_type> cl_int EnqueueWriteBuffer(cl_command_queue command_queue,
        cl_mem buffer,
        cl_bool blocking_write,
        size_t offset,
        size_t cb,
        int source_machine,
        init_list target_machines,
        data_barrier<data_type> *barrier,
        distCL_event *send_event,
        distCL_event *recv_event,
        cl_context context)
{
    check_source_and_target_valid(barrier->shared_machine_list, target_machines, source_machine);


    if (offset % barrier->chunk_size != 0 || (offset + cb) % barrier->chunk_size != 0 )
        std::runtime_error("Invalid offset, must align with chunk granularity");

    // Check offset falls within appropriate boundaries.
    int chunk_offset = offset / barrier->chunk_size;
    int chunks_sent = cb / barrier->chunk_size;

    cb *= sizeof(data_type);
    cl_int err = 0;
    bool target_self = false;

    if (barrier->machine_id == source_machine)
    {
        if (send_event != NULL)
        {
            int event_id = 0;

            send_event->size = target_machines.size();
            send_event->event = new std::shared_ptr<cl_event>[send_event->size];

            for (auto target_machine = begin(target_machines);
                    target_machine < end(target_machines);
                    ++target_machine)
            {

                if (*target_machine != barrier->machine_id)
                {
                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(context, &errcode_ret);
                    send_event->event[event_id] = std::make_shared<cl_event>(new_event);

                    barrier->send_data(send_event->event[event_id], *target_machine, chunk_offset, chunks_sent);
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
            temp_event.event = new std::shared_ptr<cl_event>[temp_event.size];

            for (auto target_machine = begin(target_machines);
                    target_machine < end(target_machines);
                    ++target_machine)
            {
                if (*target_machine != barrier->machine_id)
                {
                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(context, &errcode_ret);

                    temp_event.event[event_id] = std::make_shared<cl_event>(new_event);

                    barrier->send_data(temp_event.event[event_id], *target_machine, chunk_offset, chunks_sent);
                    event_id++;
                }
                else
                {
                    temp_event.size -= 1;
                    target_self = true;
                }
            }
            // for (int i = 0; i < temp_event.size; ++i)
            // {
            //     clWaitForEvents(1,  temp_event.event[i].get());
            // }
        }
        if (target_self == true)
        {
            if (recv_event != NULL)
            {
                recv_event->size = 1;
                recv_event->event =  new std::shared_ptr<cl_event>[recv_event->size];
                err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 0, NULL, recv_event->event[0].get());
            }
            else
            {
                err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 0, NULL, NULL);
            }
        }
        // if (target_local_machine == true)
        // {
        //     std::shared_ptr<distCL_event> recv_event(new distCL_event);

        //     if (recv_list != NULL)
        //     {
        //         recv_list->events.push_back(recv_event);
        //         recv_list->events.back()->event_temp = std::make_shared<cl_event>();
        //         err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 0, NULL, recv_list->events.back()->event_temp.get());

        //     }
        //     else
        //     {
        //         recv_event->event_temp = std::make_shared<cl_event>();
        //         err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 0, NULL, recv_event->event_temp.get());
        //         distCL::wait_for_distCL_event(recv_event);
        //     }
        // }
    }
    else
    {
        for (auto target_machine = begin(target_machines);
                target_machine < end(target_machines);
                ++target_machine)
        {

            if (*target_machine == barrier->machine_id && *target_machine != source_machine)
            {

                if (recv_event != NULL)
                {
                    recv_event->size = 2;
                    recv_event->event =  new std::shared_ptr<cl_event>[recv_event->size];

                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(context, &errcode_ret);
                    recv_event->event[0] = std::make_shared<cl_event>(new_event);

                    barrier->receive_data(recv_event->event[0], source_machine);

                    err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 1, recv_event->event[0].get(), recv_event->event[1].get());


                }
                else
                {
                    distCL_event temp_event;
                    temp_event.size = 1;
                    temp_event.event =  new std::shared_ptr<cl_event>[temp_event.size];

                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(context, &errcode_ret);
                    temp_event.event[0] = std::make_shared<cl_event>(new_event);

                    barrier->receive_data(temp_event.event[0], source_machine);
                    err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 1, temp_event.event[0].get(), NULL);

                }
            }
        }
    }
    return err;
}

template <typename data_type> cl_int EnqueueReadBuffer(cl_command_queue command_queue,
        cl_mem buffer,
        cl_bool blocking_read,
        size_t offset,
        size_t cb,
        int source_machine,
        init_list target_machines,
        data_barrier<data_type> *barrier,
        distCL_event *send_event,
        distCL_event *recv_event,
        cl_context context )
{
    check_source_and_target_valid(barrier->shared_machine_list, target_machines, source_machine);

    if (offset % barrier->chunk_size != 0 || (offset + cb) % barrier->chunk_size != 0 )
        std::runtime_error("Invalid offset, must align with chunk granularity");

    // Check offset falls within appropriate boundaries.
    int chunk_offset = offset / barrier->chunk_size;
    int chunks_sent = cb / barrier->chunk_size;

    cb *= sizeof(data_type);
    cl_int err = 0;

    if (barrier->machine_id == source_machine)
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
            send_event->event = new std::shared_ptr<cl_event>[send_event->size];

            for (auto target_machine = begin(target_machines);
                    target_machine < end(target_machines);
                    ++target_machine)
            {

                if (*target_machine != barrier->machine_id)
                {
                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(context, &errcode_ret);
                    send_event->event[event_id] = std::make_shared<cl_event>(new_event);

                    barrier->send_data(lock, send_event->event[event_id], *target_machine, chunk_offset, chunks_sent);
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
            temp_event.event = new std::shared_ptr<cl_event>[temp_event.size];

            for (auto target_machine = begin(target_machines);
                    target_machine < end(target_machines);
                    ++target_machine)
            {

                if (*target_machine != barrier->machine_id)
                {
                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(context, &errcode_ret);
                    temp_event.event[event_id] = std::make_shared<cl_event>(new_event);

                    barrier->send_data(lock, temp_event.event[event_id], *target_machine, chunk_offset, chunks_sent);
                    event_id++;
                }
                else
                {
                    temp_event.size -= 1;
                }
            }
            // for (int i = 0; i < temp_event.size; ++i)
            // {
            //     clWaitForEvents(1,  temp_event.event[i].get());
            // }

        }
    }
    else
    {

        for (auto target_machine = begin(target_machines);
                target_machine < end(target_machines);
                ++target_machine)
        {

            if (*target_machine == barrier->machine_id && *target_machine != source_machine)
            {

                if (recv_event != NULL)
                {
                    recv_event->size = 1;
                    recv_event->event =  new std::shared_ptr<cl_event>[recv_event->size];

                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(context, &errcode_ret);
                    recv_event->event[0] = std::make_shared<cl_event>(new_event);

                    barrier->receive_data(recv_event->event[0], source_machine);

                }
                else
                {
                    distCL_event temp_event;
                    temp_event.size = 1;
                    temp_event.event =  new std::shared_ptr<cl_event>[temp_event.size];

                    int errcode_ret;
                    cl_event new_event = clCreateUserEvent(context, &errcode_ret);
                    temp_event.event[0] = std::make_shared<cl_event>(new_event);

                    barrier->receive_data(temp_event.event[0], source_machine);
                    clWaitForEvents(1, temp_event.event[0].get());
                }
            }
        }
        err = CL_SUCCESS;
    }
    return err;
}

void WaitForEvents(int num_events, distCL_event *events)
{
    for (int i = 0; i < num_events; ++i)
    {
        wait_for_distCL_event(&events[i]);
    }
}
void wait_for_distCL_event(distCL_event * event)
{
    for (int i = 0; i < event->size; ++i)
    {
        clWaitForEvents(1, event->event[i].get());
    }
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