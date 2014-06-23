Distributed OpenCL for Networked GPUs
===
distributedCL has been tested on OS X 10.9.3, as well as Ubuntu 12.04
LTS.

It has been tested using OpenCL v1.2, and OpenMPI v1.8.1.

OpenCL
------

The first requirement to running distributedCL is
to have OpenCL available upon all machines you wish to target. There are
many guides online for setting up OpenCL, but a good one for linux is
available at Andreas Klöckner’s wiki
<http://wiki.tiker.net/OpenCLHowTo>.

OpenMPI
-------

The next is to ensure all machines have OpenMPI
installed. You’ll compile and run the program as a typical MPI program.
All the information needed to prepare OpenMPI on your machines can be
found at <http://www.open-mpi.org/>.

Samples
-------
Some samples are included in the repository under src/samples

To run them, they must first be compiled with with both OpenMPI and OpenCL, then run as followed

1. **nonblocking\_communication:** This must be compiled with MPI and run with at least three processes. A sample command to run it locally would be "mpirun -n 3 ./bin/nonblocking_communication". See code for citations.

2. **heat\_simulation:** Based on the code provided by Dr. D.B. Thomas for his High Performance Computing course. The step world function has been modified to support distributed computing. More information can be found in the makefile included in this sample.

Coding with distributedCL
-------------------------

After including “distributedCL.h”,
OpenCL code can be almost directly transcribed to distributedCL code,
with a few caveats.

1.  A distributedCL program must always start with the construction of
    an instance of the distributedCL class (at least before any other
    distributedCL functions can be called)

2.  A distributedCL program must always terminate with a call to
    distributedCL::Finalize

3.  All distCL\_events that have been passed into a function must have
    distributedCL:: WaitForEvents called on them prior to
    distributedCL::Finalize

4.  All distributedCL functions return errors as a cl\_int, OpenCL
    functions that ordinarily return alternatives instead take a
    reference to that object as the first parameter

5.  In place of void pointers for data in read and write commands, you
    must pass a pointer to a data\_barrier

6.  Functions that involve cross process communications allow for two
    events (send and receive) as opposed to the standard one

Other than that, any CL function needed is provided by the distributedCL
class. For the most part, apart from some of the above mentioned cases,
it’s enough to initialize distributedCL as dCL and replace the ‘cl’ at
the beginning of any function call with ‘dCL.’ and add a final parameter
‘’. However, not all functions have currently been implemented and some
will not work.
