input metauml;
beginfig(1);
	Class.A("distCL_event")
	       ("+size: int",
	        "+event: std::shared_ptr<cl_event>*",,)();
	drawObjects(A);
endfig;
end
