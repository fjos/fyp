#include <mpi.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <thread>
#include <functional>
#include "mpi_data_conversion.h"

#define SEND_MACHINE 1000000
#define RECV_MACHINE 100000

// using namespace std;
typedef struct distCL_request
{
    std::vector<std::thread> request_thread;
    std::vector<MPI_Request> request_mpi;
} distCL_request;

#define init_list std::initializer_list<int>
template <typename data_type> class data_barrier
{
public:
    int data_size;
    int chunk_size;
    int number_chunks;
    init_list target_machines;
    int message_tag;
    int machine_id;
    data_type *data;
    data_type *previous_data;

    data_barrier() {};
    data_barrier(const init_list &target_machines, int size_x, int granularity, int tag_value, int id);
    virtual ~data_barrier();

    void sync_barrier();
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

private:
    void send_data(distCL_request *request, std::vector<int> chunks_to_send, int target_machine);
    void receive_data(distCL_request *request, int source_machine);
    void receive_data(int source_machine);
};


template <typename data_type> data_barrier<data_type>::data_barrier(const std::initializer_list<int> &machines, int size_x, int granularity, int tag_value, int id)
{
    data_size = size_x;
    number_chunks = size_x / granularity;
    chunk_size = granularity;
    message_tag = tag_value;
    machine_id = id;
    target_machines = machines;
    data = new data_type[size_x];
    previous_data = new data_type[size_x];
}

template <typename data_type> data_barrier<data_type>::~data_barrier()
{
    delete []data;
    delete []previous_data;
}

bool my_id_in_list(const std::initializer_list<int> &target_machines, int my_id)
{
    for (auto i = begin(target_machines); i < end(target_machines); ++i)
    {
        if (*i == my_id)
        {
            return true;
        }
    }

    return false;
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

template <typename data_type> void data_barrier<data_type>::receive_data(int source_machine)
{
    distCL_request request;
    request.request_thread.push_back(std::thread(receive_thread<data_type>, data, message_tag, number_chunks, chunk_size, source_machine));
    while (!request.request_thread.empty())
    {
        request.request_thread.back().join();
        request.request_thread.pop_back();
    }
}


template <typename data_type> void data_barrier<data_type>::sync_barrier()
{
    if (my_id_in_list(target_machines, machine_id))
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
        distCL_request send_request;
        for (auto i = begin(target_machines); i < end(target_machines); ++i)
        {
            if (*i != machine_id)
            {
                send_data(&send_request, chunks_changed, *i);
            }
        }

        distCL_request request;
        for (auto i = begin(target_machines); i < end(target_machines); ++i)
        {
            if (*i != machine_id)
            {
                receive_data(&request, *i);
            }
        }

        while (!send_request.request_mpi.empty())
        {
            MPI_Wait(&send_request.request_mpi.back(), MPI_STATUS_IGNORE);
            send_request.request_mpi.pop_back();
        }
        while (!request.request_thread.empty())
        {
            request.request_thread.back().join();
            request.request_thread.pop_back();
        }
    }
}

template <typename data_type> void barrier(data_barrier<data_type> barrier)
{
    barrier.sync_barrier();
}

class distCL
{
public:
    int my_id, root_process, num_processes;
    MPI_Status status;
    int tag_value;

    distCL();
    distCL(int *argc, char ***argv);
    ~distCL();

    void distCL_Init(int *argc, char ***argv);
    void distCL_Finalize();
    // template <typename data_type> data_barrier<data_type> create_barrier(const std::initializer_list<int> &target_machines, int size_x, int granularity);
    template<typename data_type> auto create_barrier(const std::initializer_list<int> &target_machines, int size_x, int granularity) -> data_barrier<data_type>
    {
        int temp_tag_value = tag_value;
        tag_value += size_x / granularity + 2;
        if (my_id_in_list(target_machines, my_id))
        {
            return data_barrier<data_type>(target_machines, size_x, granularity, temp_tag_value, my_id);
        }
        return data_barrier<data_type>();

    }

};

distCL::distCL()
{
}

distCL::distCL(int *argc, char ***argv)
{
    distCL_Init(argc, argv);
}

distCL::~distCL()
{

}

void distCL::distCL_Init(int *argc, char ***argv)
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

void distCL::distCL_Finalize()
{
    int finalized, ierr;
    MPI_Finalized(&finalized);
    if (!finalized)
    {
        ierr = MPI_Finalize();
    }
}
