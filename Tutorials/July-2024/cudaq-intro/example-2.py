# Multi-qubit example

import cudaq

cudaq.set_target('nvidia')

@cudaq.kernel
def second_kernel(N:int):
    qubits=cudaq.qvector(N)

    h(qubits[0])
    x.ctrl(qubits[0],qubits[1])
    x.ctrl(qubits[0],qubits[2])
    x(qubits)

    mz(qubits)

print(cudaq.draw(second_kernel,3))