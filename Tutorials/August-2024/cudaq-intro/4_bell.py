#Sampling Bell state example

import cudaq

cudaq.set_target('nvidia')

@cudaq.kernel
def bell():
    qubits=cudaq.qvector(2)

    h(qubits[0])
    x.ctrl(qubits[0], qubits[1])

    mz(qubits)

print(cudaq.draw(bell))
# Sample the state generated by bell
# shots_count: the number of kernel executions. Default is 1000
counts = cudaq.sample(bell, shots_count=10000) 

# Print to standard out
print(counts)

# Fine-grained access to the bits and counts 
for bits, count in counts.items():
    print('Observed: {}, {}'.format(bits, count))