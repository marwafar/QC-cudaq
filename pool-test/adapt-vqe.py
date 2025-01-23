import cudaq, cudaq_solvers as solvers
import uccsd_pool

#geometry = [('H', (0., 0., 0.)), ('H', (0., 0., .7474))]
geometry = [('N', (0.0, 0.0, 0.5600)), ('N', (0.0, 0.0, -0.5600))]
molecule = solvers.create_molecule(geometry,
                                   'sto-3g',
                                   0,
                                   0,
                                   nele_cas=2,
                                   norb_cas=2,
                                   verbose=True)
nelectrons=molecule.n_electrons
n_qubits = molecule.n_orbitals * 2

ham=molecule.hamiltonian

spin=0
pools=uccsd_pool.get_pool_operators(nelectrons, spin, n_qubits)

@cudaq.kernel
def initState(q: cudaq.qview):
    for i in range(nelectrons):
        x(q[i])

# Run ADAPT-VQE
energy, thetas, ops = solvers.adapt_vqe(initState, molecule.hamiltonian,
                                        pools)

# Print the result.
print("<H> = ", energy)