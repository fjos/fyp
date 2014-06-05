#include "distributedCL.h"
namespace distCL
{

//HELPER FUNCTIONS
void check_source_and_target_valid(init_list shared_machine_list, init_list target_machines, int source_machine);

//functions
void enqueue_barrier(distCL_request request);

void enqueueWriteBuffer()
{

    check_source_and_target_valid(barrier.shared_machine_list, target_machines, source_machine);

    if (barrier.machine_id == source_machine)
    {
        for (auto target_machine = begin(target_machines);
        target_machine < end(target_machines); ++target_machine)
        {
            if (target_machine == barrier.machine_id)
            {
                //CL FUNCTION CALL

            }
            else
            {
                barrier.send_data(target_machine);
            }
        }
    }
    else
    {
        for (auto target_machine = begin(target_machines);
        target_machine < end(target_machines); ++target_machine)
        {
            if (target_machine != barrier.machine_id)
            {
                distCL recv_request;
                barrier.receive_data(&recv_request, source_machine);
                wait_for_distCL_request(&recv_request);

                //CL FUNCTION CALL
            }
        }
    }
}

void enqueue_barrier(distCL_request *request)
{
    wait_for_distCL_request(request);
}

void check_source_and_target_valid(init_list shared_machine_list, init_list target_machines, int source_machine)
{
    for (auto i = begin(shared_machine_list);
            i < end(shared_machine_list); ++i)
    {
        if (!my_id_in_list(target_machines, *i))
        {
            throw std::runtime_error("Target machine does not share memory barrier.");
        }
    }
    if (!my_id_in_list(shared_machine_list, source_machine))
    {
        throw std::runtime_error("Source machine does not share memory barrier.");
    }
}

};

