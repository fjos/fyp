#include <mpi.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <thread>
#include <functional>
#include "mpi_data_conversion.h"

#define SEND_MACHINE 1000000
#define RECV_MACHINE 100000

#define init_list std::initializer_list<int>

typedef struct distCL_request
{
    std::vector<std::thread> request_thread;
    std::vector<MPI_Request> request_mpi;
} distCL_request;


void wait_for_distCL_request(distCL_request *request)
{
    while (!request->request_mpi.empty())
    {
        MPI_Wait(&request->request_mpi.back(), MPI_STATUS_IGNORE);
        request->request_mpi.pop_back();
    }
    while (!request->request_thread.empty())
    {
        request->request_thread.back().join();
        request->request_thread.pop_back();
    }
}

bool my_id_in_list(const std::initializer_list<int> &shared_machine_list, int my_id)
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

    void sync_barrier(distCL_request *request);
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

    void send_data(distCL_request *request, std::vector<int> chunks_to_send, int target_machine);
    void send_data(distCL_request *request, int target_machine);
    void receive_data(distCL_request *request, int source_machine);
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

template <typename data_type> void data_barrier<data_type>::send_data(distCL_request *request, std::vector<int> chunks_to_send, int target_machine)
{
    int ierr;
    unsigned chunk_list_size = chunks_to_send.size();
    MPI_Request new_request;


    request->request_mpi.push_back(new_request);
    ierr = MPI_Isend(&chunk_list_size, 1, MPI_UNSIGNED, target_machine, machine_id * SEND_MACHINE + target_machine * RECV_MACHINE + message_tag + number_chunks, MPI_COMM_WORLD, &request->request_mpi.back());
    //IERR CATCH

    request->request_mpi.push_back(new_request);
    ierr = MPI_Isend(&chunks_to_send.front(), chunk_list_size, MPI_INT, target_machine, machine_id * SEND_MACHINE + target_machine * RECV_MACHINE +  message_tag + number_chunks + 1, MPI_COMM_WORLD, &request->request_mpi.back());
    //IERR CATCH

    while (!chunks_to_send.empty())
    {
        int chunk_id = chunks_to_send.back();
        chunks_to_send.pop_back();

        std::copy(data + chunk_id * chunk_size,
                  data + (chunk_id + 1)*chunk_size,
                  previous_data + chunk_id * chunk_size);
        request->request_mpi.push_back(new_request);
        ierr = MPI_Isend(data + chunk_id * chunk_size, chunk_size, convert_type(get_abstraction_data_type<data_type>()), target_machine, machine_id * SEND_MACHINE + target_machine * RECV_MACHINE + message_tag + chunk_id, MPI_COMM_WORLD, &request->request_mpi.back());
    }
}

template <typename data_type> void data_barrier<data_type>::send_data(distCL_request *request, int target_machine)
{
    std::vector<int> chunks_to_send;
    for (int chunk_id = 0; chunk_id < number_chunks; ++chunk_id)
    {
        chunks_to_send.push_back(chunk_id);
    }

    send_data(&request, chunks_to_send, target_machine);
}

template <typename data_type> void receive_thread(data_type *data, int message_tag, int number_chunks, int chunk_size, int source_machine, int machine_id)
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
}

template <typename data_type> void data_barrier<data_type>::receive_data(distCL_request *request, int source_machine)
{
    request->request_thread.push_back(std::thread(receive_thread<data_type>, data, message_tag, number_chunks, chunk_size, source_machine, machine_id));
}


template <typename data_type> void data_barrier<data_type>::sync_barrier(distCL_request *request)
{
    if (my_id_in_list(shared_machine_list, machine_id))
    {
        std::vector<int> chunks_changed;
        for (int chunk_id = 0; chunk_id < number_chunks; ++chunk_id)
        {
            bool chunk_changed = false;
            for (int offset = 0; offset < chunk_size; ++offset)
            {
                if (data[chunk_id * chunk_size + offset] != previous_data[chunk_id * chunk_size + offset])
                {
                    chunk_changed = true;
                }
            }
            if (chunk_changed)
            {
                chunks_changed.push_back(chunk_id);
            }
        }
        for (auto i = begin(shared_machine_list); i < end(shared_machine_list); ++i)
        {
            if (*i != machine_id)
            {
                send_data(request, chunks_changed, *i);
            }
        }
        for (auto i = begin(shared_machine_list); i < end(shared_machine_list); ++i)
        {
            if (*i != machine_id)
            {
                receive_data(request, *i);
            }
        }
    }
}

