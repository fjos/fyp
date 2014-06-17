input metauml;
beginfig(1);
    Class.A("data_barrier")
           ("data_size: int",
            "chunk_size: int",
            "number_chunks: int",
            "init_list: shared_machine_list",
            "message_tag: int",
            "machine_id: int",
            "data: data_type *",
            "previous_data: data_type *",
            "context: cl_context",)
            ("data_barrier(machine: const init_list &, size_x: int, granularity: int, tag_value: int, id: int, input_context: cl_context):",
            "send_data(send_event: std::shared_ptr<cl_event>, target_machine: int, offset: int, chunks_sent: int): void",
            "send_data(send_event: std::shared_ptr<cl_event>, target_machine: int, chunks_to_send: std::vector<int>): void",
            "send_data(lock: std::shared_ptr<cl_event>, send_event: std::shared_ptr<cl_event>, target_machine: int, offset: int, chunks_sent: int): void",
            "send_data(lock: std::shared_ptr<cl_event>, send_event: std::shared_ptr<cl_event>, target_machine: int, chunks_to_send: std::vector<int>): void",
            "receive_data(recv_event: std::shared_ptr<cl_event>, int source_machine): void",
            );
    drawObjects(A);
endfig;
end
