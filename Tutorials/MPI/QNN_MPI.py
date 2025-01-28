# mpirun -np 3 python QNN_MPI.py

import cudaq
from cudaq import spin
import numpy as np
import timeit

cudaq.mpi.initialize()

np.random.seed(1)

cudaq.set_target("nvidia")
target = cudaq.get_target()
print(f"My rank {cudaq.mpi.rank()} of {cudaq.mpi.num_ranks()}")

qubit_count = 24
sample_count = 900

ham = spin.z(0)

parameters = np.random.default_rng(13).uniform(low=0,high=1,size=(sample_count, qubit_count))

print('Parameter shape: ', parameters.shape)

@cudaq.kernel
def kernel_rx(theta:list[float]):
    qubits = cudaq.qvector(qubit_count)

    for i in range(qubit_count):
        rx(theta[i], qubits)

# split per node
split_params = np.split(parameters, cudaq.mpi.num_ranks())
my_rank_params = split_params[cudaq.mpi.rank()]

print('We have', parameters.shape[0],
      'parameter sets which we would like to execute')

print('We have', my_rank_params.shape[0],
      'parameter sets on this rank', cudaq.mpi.rank())

print('Number of param sets on this rank:', len(my_rank_params))
print('Shape of each parameter set after splitting:', my_rank_params.shape[0])

start_time = timeit.default_timer()
#result=[]
#for i in range(len(my_rank_params)):
erg=cudaq.observe(kernel_rx, ham, my_rank_params)
result=np.array([r.expectation() for r in erg])

end_time = timeit.default_timer()

print(f'Elapsed time (s) is {end_time-start_time}, rank {cudaq.mpi.rank()}')
print(f"My rank has {len(result)} results")
total_results = cudaq.mpi.all_gather(len(result)*cudaq.mpi.num_ranks(), result)
print(f"My rank has {len(total_results)} results after all gather")

cudaq.mpi.finalize()