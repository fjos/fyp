input metauml;
beginfig(1);
	Class.A("distCL_event")
	       ("+datatype: int",
	        "+send_event: std::vector<MPI_Request>",
		"+recv_event: std::vector<std::thread>",
		"+opencl_event: cl_event")();
	drawObjects(A);
endfig;
end
