#include <mpi.h>
#include <vector>
#include <list>
#include <sstream>
#include <iostream>
#include <thread>
#include <functional>
#include "mpi_data_conversion.h"

#define SEND_MACHINE 1000000
#define RECV_MACHINE 100000

#define init_list std::initializer_list<int>


typedef struct distCL_event
{
    int size;
    std::shared_ptr<cl_event> *event;
    // distCL_event() : size(0)
    // {

    // }
    // ~distCL_event()
    // {
    //             delete[] event;
    // }
    // distCL_event(cl_context context)
    // {
    //     *event = new_event;
    // }
} distCL_event;


template <typename data_type> class data_barrier
{
public:
    int data_size;
    int chunk_size;
    int number_chunks;
    init_list shared_machine_list;
    int message_tag;
    int machine_id;
    data_type *data;
    data_type *previous_data;

    data_barrier() {};
    data_barrier(const init_list &shared_machine_list, int size_x, int granularity, int tag_value, int id);
    virtual ~data_barrier();

    void dumpData(bool show_data, bool show_stats)
    {
        std::stringstream output;
        if (show_data == true)
        {
            output << "Data " << machine_id << " :"  << std::endl;
            for (int i = 0; i < data_size; ++i)
            {
                output << data[i] << " ";
            }
            output << std::endl;

            output << "Data Previous " << machine_id << " :" << std::endl;
            for (int i = 0; i < data_size; ++i)
            {
                output << previous_data[i] << " ";
            }
            output << std::endl;
        }
        if (show_stats == true)
        {
            output << "Data Size     : " << data_size << std::endl;
            output << "Chunk Size    : " << chunk_size << std::endl;
            output << "Number Chunks : " << number_chunks << std::endl;
        }

        std::cout << output.str();
    }

    void send_data(std::shared_ptr<cl_event> send_event, int target_machine, int offset, int chunks_sent);
    void send_data(std::shared_ptr<cl_event> lock, std::shared_ptr<cl_event> send_event, int target_machine, int offset, int chunks_sent);
    void receive_data(std::shared_ptr<cl_event> recv_event, int source_machine);
    // void receive_data(int source_machine);
};


template <typename data_type> data_barrier<data_type>::data_barrier(const std::initializer_list<int> &machines, int size_x, int granularity, int tag_value, int id)
{
    data_size = size_x;
    number_chunks = size_x / granularity;
    chunk_size = granularity;
    message_tag = tag_value;
    machine_id = id;
    shared_machine_list = machines;
    data = new data_type[size_x];
    previous_data = new data_type[size_x];
}

template <typename data_type> data_barrier<data_type>::~data_barrier()
{
    delete []data;
    delete []previous_data;
}

template <typename data_type> void send_thread_lock(std::shared_ptr<cl_event> lock,
        std::shared_ptr<cl_event> send_event,
        data_type *data,
        int message_tag,
        int number_chunks,
        int chunk_size,
        int machine_id,
        std::vector<int> chunks_to_send,
        int target_machine)
{
    clWaitForEvents(1, lock.get());


    int ierr;
    unsigned chunk_list_size = chunks_to_send.size();
    MPI_Request new_request;

    std::vector<MPI_Request> event_mpi;
    event_mpi.push_back(new_request);
    ierr = MPI_Isend(&chunk_list_size, 1, MPI_UNSIGNED, target_machine, machine_id * SEND_MACHINE + target_machine * RECV_MACHINE + message_tag + number_chunks, MPI_COMM_WORLD, &event_mpi.back());
    //IERR CATCH

    event_mpi.push_back(new_request);
    ierr = MPI_Isend(&chunks_to_send.front(), chunk_list_size, MPI_INT, target_machine, machine_id * SEND_MACHINE + target_machine * RECV_MACHINE +  message_tag + number_chunks + 1, MPI_COMM_WORLD, &event_mpi.back());
    //IERR CATCH


    while (!chunks_to_send.empty())
    {
        int chunk_id = chunks_to_send.back();
        chunks_to_send.pop_back();

        // std::copy(data + chunk_id * chunk_size,
        //           data + (chunk_id + 1)*chunk_size,
        //           previous_data + chunk_id * chunk_size);
        event_mpi.push_back(new_request);

        ierr = MPI_Isend(data + chunk_id * chunk_size,
                         chunk_size,
                         convert_type(get_abstraction_data_type<data_type>()),
                         target_machine,
                         machine_id * SEND_MACHINE + target_machine * RECV_MACHINE + message_tag + chunk_id,
                         MPI_COMM_WORLD,
                         &event_mpi.back());
    }
    clSetUserEventStatus(*send_event, CL_SUCCESS);
}


template <typename data_type> void data_barrier<data_type>::send_data(std::shared_ptr<cl_event> lock, std::shared_ptr<cl_event> send_event, int target_machine, int offset, int chunks_sent)
{
    if (offset + chunks_sent > number_chunks)
        std::runtime_error("offset + chunks_sent value out of bounds");
    std::vector<int> chunks_to_send;
    for (int chunk_id = offset; chunk_id < (offset + chunks_sent); ++chunk_id)
    {
        chunks_to_send.push_back(chunk_id);
    }

    std::thread t(send_thread_lock<data_type>,
                  lock,
                  send_event,
                  data,
                  message_tag,
                  number_chunks,
                  chunk_size,
                  machine_id,
                  chunks_to_send,
                  target_machine);
    t.detach();
}



template <typename data_type> void send_thread(std::shared_ptr<cl_event> send_event,
        data_type *data,
        int message_tag,
        int number_chunks,
        int chunk_size,
        int machine_id,
        std::vector<int> chunks_to_send,
        int target_machine)
{
    int ierr;
    unsigned chunk_list_size = chunks_to_send.size();
    MPI_Request new_request;

    std::vector<MPI_Request> event_mpi;
    event_mpi.push_back(new_request);
    ierr = MPI_Isend(&chunk_list_size, 1, MPI_UNSIGNED, target_machine, machine_id * SEND_MACHINE + target_machine * RECV_MACHINE + message_tag + number_chunks, MPI_COMM_WORLD, &event_mpi.back());
    //IERR CATCH

    event_mpi.push_back(new_request);
    ierr = MPI_Isend(&chunks_to_send.front(), chunk_list_size, MPI_INT, target_machine, machine_id * SEND_MACHINE + target_machine * RECV_MACHINE +  message_tag + number_chunks + 1, MPI_COMM_WORLD, &event_mpi.back());
    //IERR CATCH


    while (!chunks_to_send.empty())
    {
        int chunk_id = chunks_to_send.back();
        chunks_to_send.pop_back();

        // std::copy(data + chunk_id * chunk_size,
        //           data + (chunk_id + 1)*chunk_size,
        //           previous_data + chunk_id * chunk_size);
        event_mpi.push_back(new_request);

        ierr = MPI_Isend(data + chunk_id * chunk_size,
                         chunk_size,
                         convert_type(get_abstraction_data_type<data_type>()),
                         target_machine,
                         machine_id * SEND_MACHINE + target_machine * RECV_MACHINE + message_tag + chunk_id,
                         MPI_COMM_WORLD,
                         &event_mpi.back());
    }
    clSetUserEventStatus(*send_event, CL_SUCCESS);
}


template <typename data_type> void data_barrier<data_type>::send_data(std::shared_ptr<cl_event> send_event, int target_machine, int offset, int chunks_sent)
{
    if (offset + chunks_sent > number_chunks)
        std::runtime_error("offset + chunks_sent value out of bounds");
    std::vector<int> chunks_to_send;
    for (int chunk_id = offset; chunk_id < (offset + chunks_sent); ++chunk_id)
    {
        chunks_to_send.push_back(chunk_id);
    }
    std::thread t(send_thread<data_type>,
                  send_event,
                  data,
                  message_tag,
                  number_chunks,
                  chunk_size,
                  machine_id,
                  chunks_to_send,
                  target_machine);
    t.detach();
}


template <typename data_type> void receive_thread( std::shared_ptr<cl_event> recv_event, data_type *data, int message_tag, int number_chunks, int chunk_size, int source_machine, int machine_id)
{
    int ierr;
    unsigned chunk_list_size = 0;

    MPI_Status status;
    ierr = MPI_Recv(&chunk_list_size, 1, MPI_UNSIGNED, source_machine, source_machine * SEND_MACHINE + machine_id * RECV_MACHINE + message_tag + number_chunks, MPI_COMM_WORLD, &status);
    //IERR CATCH

    std::vector<int> chunks_to_receive(chunk_list_size);
    ierr = MPI_Recv(&chunks_to_receive[0], chunk_list_size, MPI_INT, source_machine, source_machine * SEND_MACHINE + machine_id * RECV_MACHINE + message_tag + number_chunks + 1, MPI_COMM_WORLD, &status);
    //IERR CATCH


    while (!chunks_to_receive.empty())
    {
        int chunk_id = chunks_to_receive.back();
        chunks_to_receive.pop_back();
        ierr = MPI_Recv(data + chunk_id * chunk_size, chunk_size, convert_type(get_abstraction_data_type<data_type>()), source_machine, source_machine * SEND_MACHINE + machine_id * RECV_MACHINE + message_tag + chunk_id, MPI_COMM_WORLD, &status);
    }

    clSetUserEventStatus(*recv_event, CL_SUCCESS);
}

template <typename data_type> void data_barrier<data_type>::receive_data(std::shared_ptr<cl_event> recv_event, int source_machine)
{
    std::thread t(receive_thread<data_type>, recv_event, data, message_tag, number_chunks, chunk_size, source_machine, machine_id);
    t.detach();
}



// template <typename data_type> void data_barrier<data_type>::sync_barrier(distCL_event_list *send_list, distCL_event_list *recv_list)
// {
//     if (my_id_in_list(shared_machine_list, machine_id))
//     {

//         std::vector<int> chunks_changed;
//         for (int chunk_id = 0; chunk_id < number_chunks; ++chunk_id)
//         {
//             bool chunk_changed = false;
//             for (int offset = 0; offset < chunk_size; ++offset)
//             {
//                 if (data[chunk_id * chunk_size + offset] != previous_data[chunk_id * chunk_size + offset])
//                 {
//                     chunk_changed = true;
//                 }
//             }
//             if (chunk_changed)
//             {
//                 chunks_changed.push_back(chunk_id);
//             }
//         }
//         for (auto i = begin(shared_machine_list); i < end(shared_machine_list); ++i)
//         {
//             if (*i != machine_id)
//             {
//                 distCL_event send_event;
//                 send_data(&send_event, chunks_changed, *i);
//                 // If send_list provided, push event
//                 if (send_list != NULL)
//                     send_list->events.push_back(std::move(&send_event));
//             }
//         }
//         for (auto i = begin(shared_machine_list); i < end(shared_machine_list); ++i)
//         {
//             if (*i != machine_id)
//             {
//                 distCL_event recv_event;;
//                 receive_data(&recv_event, *i);
//                 // If recv_list provided, push event
//                 if (recv_list != NULL)
//                     recv_list->events.push_back(std::move(&recv_event));
//             }
//         }
//     }
// }

