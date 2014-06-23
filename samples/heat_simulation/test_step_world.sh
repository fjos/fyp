#!/bin/bash
make generate_world
diff \
<( cat world  | ./bin/step_world 0.1 1000) \
<( cat world | mpirun -n 2 ./$1 0.1 1000 )

