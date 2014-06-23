# Based heavily on code by Dr. D. B. Thomas provided for his course High Performance Computing for Engineers at Imperial College London.
# All code provided was written by Dr. D.B. Thomas, apart from the distributed step function.

# Makefile for posix and gcc

# Note on old compilers  *cough*  DoC  *cough* you might need -std=c++0x instead
CPPFLAGS = -I include -Wall -std=c++11

# Turn on optimisations
CPPFLAGS += -O2

#Turn on threads
CPPFLAGS += -pthread

CXX = mpic++

# Include directory
INC_DIR = include/
CPPFLAGS += -I $(INC_DIR)

# Uncomment for required flags
# OS X
CLFLAGS = -framework OpenCL
# Linux
# CLFLAGS = -lOpenCL

# implementations
HEAT_CORE_OBJS = src/heat.o
HEAT_OBJS = $(HEAT_CORE_OBJS)

generate_world: bin/make_world
	./bin/make_world 10 0.1 > world

render_world: generate_world bin/step_world_distributed bin/render_world
	cat world | mpirun -n 2 ./bin/step_world_distributed 0.1 1000  | ./bin/render_world > rendered_world


all: bin/make_world \
	 bin/step_world \
 	 bin/render_world \
 	 bin/step_world_distributed

bin/make_world: src/make_world.cpp $(HEAT_OBJS)
	-mkdir -p bin
	$(CXX) $(CPPFLAGS) $^ -o $@

bin/step_world: src/step_world.cpp $(HEAT_OBJS)
	-mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CLFLAGS) $^ -o $@

bin/render_world: src/render_world.cpp $(HEAT_OBJS)
	-mkdir -p bin
	$(CXX) $(CPPFLAGS) $^ -o $@

bin/step_world_distributed: src/step_world_distributed.cpp $(HEAT_OBJS)
	-mkdir -p bin
	$(CXX) $(CPPFLAGS) $(CLFLAGS) $^ -o $@


# bin/step_world_nonsplit: src/step_world_nonsplit.cpp $(HEAT_OBJS)
# 	-mkdir -p bin
# 	$(CXX) $(CPPFLAGS) $(CLFLAGS) $^ -o $@


clean:
	-rm src/*.o
	-rm -r bin
