#include "distributedCL.h"
#include <unistd.h>
namespace distCL
{

//HELPER FUNCTIONS
void check_source_and_target_valid(init_list shared_machine_list, init_list target_machines, int source_machine);

//functions
void WaitForEvents(distCL_event_list *event_list);
void wait_for_distCL_event(std::shared_ptr<distCL_event> event);
// void add_event(distCL_event_list *event_list, distCL_event *event);


template <typename data_type> cl_int EnqueueWriteBuffer(cl_command_queue command_queue,
        cl_mem buffer,
        cl_bool blocking_write,
        size_t offset,
        size_t cb,
        int source_machine,
        init_list target_machines,
        data_barrier<data_type> *barrier,
        distCL_event_list *event_wait_list,
        distCL_event_list *send_list,
        distCL_event_list *recv_list,
        cl_context context)
{
    check_source_and_target_valid(barrier->shared_machine_list, target_machines, source_machine);

    distCL::WaitForEvents(event_wait_list);


    if (offset % barrier->chunk_size != 0 || (offset + cb) % barrier->chunk_size != 0 )
        std::runtime_error("Invalid offset, must align with chunk granularity");

    // Check offset falls within appropriate boundaries.
    int chunk_offset = offset / barrier->chunk_size;
    int chunks_sent = cb / barrier->chunk_size;

    cb *= sizeof(data_type);
    cl_int err = 0;
    if (barrier->machine_id == source_machine)
    {
        bool target_local_machine = false;
        for (auto target_machine = begin(target_machines);
                target_machine < end(target_machines);
                ++target_machine)
        {
            if (*target_machine == barrier->machine_id)
            {
                target_local_machine = true;
            }
            else
            {
                distCL_event send_event;
                barrier->send_data(&send_event, *target_machine, chunk_offset, chunks_sent);


                //ADD SEND_LIST STUFF
                // if (send_list != NULL)
                // {
                //     send_list->events.push_back(std::move(&send_event));
                // }
            }
        }

        if (target_local_machine == true)
        {
            std::shared_ptr<distCL_event> recv_event(new distCL_event);

            if (recv_list != NULL)
            {
                recv_list->events.push_back(recv_event);
                recv_list->events.back()->event_temp = std::make_shared<cl_event>();
                err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 0, NULL, recv_list->events.back()->event_temp.get());

            }
            else
            {
                recv_event->event_temp = std::make_shared<cl_event>();
                err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 0, NULL, recv_event->event_temp.get());
                distCL::wait_for_distCL_event(recv_event);
            }
        }
    }
    else
    {
        for (auto target_machine = begin(target_machines);
                target_machine < end(target_machines); ++target_machine)
        {
            if (*target_machine == barrier->machine_id && *target_machine != source_machine)
            {
                if (recv_list != NULL)
                {
                    std::shared_ptr<distCL_event> recv_event(new distCL_event);
                    recv_list->events.push_back(recv_event);

                    cl_int errcode_ret;

                    recv_list->events.back()->lock = std::make_shared<cl_event>();
                    recv_list->events.back()->event_temp = std::make_shared<cl_event>();

                    *recv_list->events.back()->lock = clCreateUserEvent(context, &errcode_ret);
                    //ERRCODE CHECK


                    barrier->receive_data(recv_list->events.back(), source_machine, recv_list->events.back()->lock);

                    err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 1, recv_list->events.back()->lock.get(), recv_list->events.back()->event_temp.get());
                }
                else
                {
                    std::shared_ptr<distCL_event> recv_event(new distCL_event);
                    cl_int errcode_ret;
                    recv_event->lock = std::make_shared<cl_event>();
                    recv_event->event_temp = std::make_shared<cl_event>();

                    *recv_event->lock = clCreateUserEvent(context, &errcode_ret);
                    //ERRCODE CHECK

                    barrier->receive_data(recv_event, source_machine, recv_event->lock);

                    err = clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, barrier->data, 1, recv_event->lock.get(), recv_event->event_temp.get());

                    distCL::wait_for_distCL_event(recv_event);
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
        distCL_event_list *event_wait_list,
        distCL_event_list *send_list,
        distCL_event_list *recv_list,
        cl_context context )
{
    check_source_and_target_valid(barrier->shared_machine_list, target_machines, source_machine);

    distCL::WaitForEvents(event_wait_list);


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

        if (send_list != NULL)
        {
            for (auto target_machine = begin(target_machines);
                    target_machine < end(target_machines);
                    ++target_machine)
            {

                if (*target_machine != barrier->machine_id)
                {
                    // distCL_event send_event;
                    // barrier->send_data(&send_event, *target_machine, chunk_offset, chunks_sent);

                    std::shared_ptr<distCL_event> send_event(new distCL_event);
                    send_list->events.push_back(send_event);

                    cl_int errcode_ret;

                    send_list->events.back()->lock = lock;
                    send_list->events.back()->event_temp = std::make_shared<cl_event>();

                    barrier->send_data(send_list->events.back(), *target_machine, chunk_offset, chunks_sent);
                }
                // // else
                // // {

                // std::shared_ptr<distCL_event> send_event(new distCL_event);

                // send_event->lock = lock;
                // send_event->event_temp = std::make_shared<cl_event>();


                // barrier->send_data(send_event, *target_machine, chunk_offset, chunks_sent);

                // // printf("sent to %d\n", *target_machine);
                // distCL::wait_for_distCL_event(send_event);
                // send_event.reset();
                // // }
            }
        }
        else
        {
            distCL_event_list temp_list;
            for (auto target_machine = begin(target_machines);
                    target_machine < end(target_machines);
                    ++target_machine)
            {

                if (*target_machine != barrier->machine_id)
                {
                    std::shared_ptr<distCL_event> send_event(new distCL_event);
                    temp_list.events.push_back(send_event);

                    temp_list.events.back()->lock = lock;
                    temp_list.events.back()->event_temp.reset(new cl_event);

                    barrier->send_data(temp_list.events.back(), *target_machine, chunk_offset, chunks_sent);
                }
            }
            distCL::WaitForEvents(&temp_list);
        }
    }
    else
    {
        // usleep(100000000);

        for (auto target_machine = begin(target_machines);
                target_machine < end(target_machines);
                ++target_machine)
        {

            if (*target_machine == barrier->machine_id && *target_machine != source_machine)
            {

                if (recv_list != NULL)
                {

                    std::shared_ptr<distCL_event> recv_event(new distCL_event);
                    recv_list->events.push_back(recv_event);
                    barrier->receive_data(recv_list->events.back(), source_machine);

                }
                else
                {


                    std::shared_ptr<distCL_event> recv_event(new distCL_event);
                    barrier->receive_data(recv_event, source_machine);

                    distCL::wait_for_distCL_event(recv_event);
                }
            }
        }
        err = CL_SUCCESS;
    }
    // err = CL_FALSE;
    return err;
}

void WaitForEvents(distCL_event_list *event_list)
{
    if (event_list != NULL)
    {
        while (!event_list->events.empty())
        {
            wait_for_distCL_event(event_list->events.front());
            event_list->events.front().reset();
            event_list->events.pop_front();
        }
    }
}


void wait_for_distCL_event(std::shared_ptr<distCL_event> event)
{

    while (!event->event_mpi.empty())
    {
        MPI_Wait(&event->event_mpi.back(), MPI_STATUS_IGNORE);
        event->event_mpi.pop_back();
    }
    while (!event->event_thread.empty())
    {
        event->event_thread.back().join();
        event->event_thread.pop_back();
    }

    if (event->event_temp.get() != NULL)
    {
        clWaitForEvents(1, event->event_temp.get());
        event->event_temp.reset();
    }
    if (event->lock != NULL)
    {
        // while(!event->lock.unique())
        // {}
        event->lock.reset();

    }
}

// void enqueueWriteBuffer(data_barrier *barrier, int source_machine, init_list target_machines)
// {

//     check_source_and_target_valid(barrier->shared_machine_list, target_machines, source_machine);

//     if (barrier.machine_id == source_machine)
//     {
//         for (auto target_machine = begin(target_machines);
//                 target_machine < end(target_machines);
//                 ++target_machine)
//         {
//             if (target_machine == barrier.machine_id)
//             {
//                 //CL FUNCTION CALL

//             }
//             else
//             {
//                 barrier.send_data(target_machine);
//             }
//         }
//     }
//     else
//     {
//         for (auto target_machine = begin(target_machines);
//                 target_machine < end(target_machines); ++target_machine)
//         {
//             if (target_machine != barrier.machine_id)
//             {
//                 distCL recv_request;
//                 barrier.receive_data(&recv_request, source_machine);
//                 wait_for_distCL_event(&recv_request);

//                 //CL FUNCTION CALL
//             }
//         }
//     }
// }



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