CPPFLAGS = -I src -Wall -std=c++11
CPPFLAGS += -O2 -pthread
CLFLAGS += -framework OpenCL

# OBJECTS = src/distCL_data_barrier.o

bin/nonblocking_communication: src/samples/nonblocking_communication/main.cpp $(OBJECTS)
	-mkdir -p bin
	-rm bin/nonblocking_communication
	mpic++ $(CPPFLAGS) $(CLFLAGS) $^ -o $@

nonblocking_communication: bin/nonblocking_communication
	mpirun -v -n 3 ./bin/nonblocking_communication

all: bin/test

bin/test: src/main.cpp $(OBJECTS)
	-mkdir -p bin
	mpic++ $(CPPFLAGS) $(CLFLAGS) $^ -o $@

clean_test: clean bin/test
	mpirun -v -n 3 ./bin/test

clean:
	-rm -r bin
	-rm src/*.o