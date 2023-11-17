import argparse
import numpy as np
from numpy import linalg as LA
import scipy

from pyscf import gto, scf, mp, mcscf, fci, cc
from pyscf import ao2mo
from pyscf.tools import molden
from functools import reduce

from openfermion import generate_hamiltonian
from openfermion.transforms import jordan_wigner

import cudaq
from cudaq import spin

from typing import List, Tuple
#############################
# Parser
#############################
# Create the parser
parser=argparse.ArgumentParser()

# Add arguments
parser.add_argument('xyz', help="xyz file", type=str)
parser.add_argument('c', help="charge of the system", type=int)
parser.add_argument('s', help="no. of unpaired electrons (2 *s)", type=int)
parser.add_argument('basis', help="The basis set", type=str)

#Parse the argument
args=parser.parse_args()
filename=args.xyz.split('.')[0]

##################################################
# Function to initialize cudaq UCCSD parameters
# from classical CCSD.
###################################################

def init_param_CCSD(qubits_num,nele_cas,t1,t2):
    
    sz=np.empty(qubits_num)

    for i in range(qubits_num):
        if i%2 == 0:
            sz[i]=0.5
        else:
            sz[i]=-0.5

# thetas for single excitation
    thetas_1=[]
# theta for double excitation
    thetas_2=[]

    tot_params=0
    nmo_occ=nele_cas//2

    for p_occ in range(nele_cas):
        for r_vir in range(nele_cas,qubits_num):
            if (sz[r_vir]-sz[p_occ]==0):
                #print(p_occ,r_vir)
                #print(p_occ//2,r_vir//2-nmo_occ)
                thetas_1.append(t1[p_occ//2,r_vir//2-nmo_occ])
                tot_params+=1

    #print('thetas_1= ',thetas_1,'\n')

    for p_occ in range(nele_cas-1):
        for q_occ in range(p_occ+1,nele_cas):
            for r_vir in range(nele_cas,qubits_num-1):
                for s_vir in range(r_vir+1,qubits_num):
                    if (sz[r_vir]+sz[s_vir]-sz[p_occ]-sz[q_occ])==0:
                        #print(p_occ,q_occ,r_vir,s_vir)
                        #print(p_occ//2,q_occ//2,r_vir//2-nmo_occ,s_vir//2-nmo_occ)
                        thetas_2.append(t2[p_occ//2,q_occ//2,r_vir//2-nmo_occ,s_vir//2-nmo_occ])
                        tot_params+=1

    #print('thetas_2= ',thetas_2,'\n')

    # Check that total number of parameters match the parameter_count above 
    #print('total parameters=', tot_params, '\n')

    init_params=np.concatenate((thetas_2,thetas_1), axis=0)
    #print('init_param= ',init_params,'\n')

    return init_params,tot_params


#############################
## Beginning of simulation
#############################

################################
# Initialize the molecule
################################
mol=gto.M(
    atom=args.xyz,
    spin=args.s,
    charge=args.c,
    basis=args.basis,
    output=filename+'.out',
    verbose=4
)

###################################

## 1- Classical preprocessing

print('\n')
print('Beginning of classical preprocessing', '\n')
print ('Energies from classical simulations','\n')

##################################
# Mean field (HF)
##################################
myhf=scf.RHF(mol)
myhf.max_cycle=100
myhf.kernel()

#nuclear_repulsion = myhf.energy_nuc()

nelec = mol.nelectron
print('Total number of electrons= ', nelec, '\n')
norb = myhf.mo_coeff.shape[1]
print('Total number of orbitals= ', norb, '\n')

print('RHF energy= ', myhf.e_tot, '\n')

######################################
# CCSD
######################################

mycc=cc.CCSD(myhf).run()
print('Total CCSD energy= ', mycc.e_tot, '\n')

######################################
# FCI
######################################

myfci=fci.FCI(myhf)
result= myfci.kernel()
print('FCI energy= ', result[0], '\n')

###################################################
## Compute electron integrals within a chosen
## MO basis (HF basis here)
###################################################

# Compute the 1e integral in atomic orbital then convert to HF basis
h1e_ao = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
## Ways to convert from ao to mo
#h1e=np.einsum('pi,pq,qj->ij', myhf.mo_coeff, h1e_ao, myhf.mo_coeff)
#h1e=reduce(np.dot, (myhf.mo_coeff.T, h1e_ao, myhf.mo_coeff))
h1e=reduce(np.dot, (myhf.mo_coeff.conj().T, h1e_ao, myhf.mo_coeff))

# Compute the 2e integrals then convert to HF basis
h2e_ao = mol.intor("int2e_sph", aosym='1')
h2e=ao2mo.incore.full(h2e_ao, myhf.mo_coeff)

# Reorder the chemist notation (pq|rs) ERI h_pqrs to h_prqs
# to "generate_hamiltonian" in openfermion 
h2e=h2e.transpose(0,2,3,1)

nuclear_repulsion = myhf.energy_nuc()

print('h1e_shape ', h1e.shape, '\n')
print('h2e_shape ', h2e.shape, '\n')

###################################################
## Generate the spin operator Hmiltoian for cudaq
####################################################

mol_ham=generate_hamiltonian(h1e,h2e,nuclear_repulsion)

ham_operator = jordan_wigner(mol_ham)

#spin_ham= get_cudaq_Hamiltonian(ham_operator)
spin_ham=cudaq.SpinOperator(ham_operator)

'''
#########################################
# Altherantive way to generate the spin 
# operator Hamiltonian using the built in
# cudaq and openfermion function.
#########################################

geometry = [('Li', (0., 0., 0.)), ('H', (0., 0., 1.5))]

molecule, data = cudaq.chemistry.create_molecular_hamiltonian(geometry, 'sto-3g', 1, 0)

print('Total number of orbitals (of)', data.n_orbitals)
print('Total number of electrons (of)', data.n_electrons)
'''
###############################
## 2- Quantum computing

print('Beginning of quantum computing simulation','\n')

##################

# Define the total number of qubits.

qubits_num=2*norb

### Backend configuration (target=default,nvidia,nvidia-mgpu).
cudaq.set_target("nvidia")

##############################
### Program construction
################################

# Define a quantum kernel function for
# a parametrized quantum circuit.

kernel, params=cudaq.make_kernel(list)

qubits = kernel.qalloc(qubits_num)

##########################################################
## Initialize the qubits to the reference state (here is HF).
## Occupied orbitals are "1" and virtual orbitals are "0"
###########################################################
for i in range(nelec):
    kernel.x(qubits[i])

###########################
##  Ansatz: UCCSD
###########################

cudaq.kernels.uccsd(kernel, qubits, params, nelec, qubits_num)
parameter_count = cudaq.kernels.uccsd_num_parameters(nelec,qubits_num)

# Check that total parameters match with tot_param below
print('param_count from cudaq= ,', parameter_count, '\n')

# Initialize the UCCSD from the classical CCSD results
init_params,tot_params=init_param_CCSD(qubits_num,nelec,mycc.t1,mycc.t2)
print('param_count from my code= ,', tot_params, '\n')

# Initialize (comment if use CCSD params)
#init_params=np.zeros(tot_params)

####################################################
## Optimizer for the parameterized quantum circuit
#####################################################
#optimizer= cudaq.optimizers.GradientDescent()
#optimizer = cudaq.optimizers.LBFGS()
optimizer = cudaq.optimizers.COBYLA()

optimizer.initial_parameters=init_params
#optimizer.max_iterations=200000

#print(optimizer.initial_parameters)

'''
######################################
## Alternative way for optimization
## Using Scipy.optimizer
######################################

# For gradient based optimizer
gradient = cudaq.gradients.CentralDifference()

def objective_function(parameter_vector: List[float], \
                       gradient=gradient, hamiltonian=spin_ham, kernel=kernel):

    get_result = lambda parameter_vector: cudaq.observe\
        (kernel, hamiltonian, parameter_vector).expectation_z()
    
    cost = get_result(parameter_vector)
    print(f"<H> = {cost}")
    #gradient_vector = gradient.compute(parameter_vector, get_result,cost)

    return cost
    #return cost, gradient_vector


#result_vqe=scipy.optimize.minimize(objective_function,init_params,method='L-BFGS-B', jac=True)
result_vqe=scipy.optimize.minimize(objective_function,init_params,method='L-BFGS-B', jac='3-point')


print('Optimizer exited successfully: ',result_vqe.success)
print(result_vqe.message)
print('Cudaq VQE-UCCSD energy (pyscf)= ', result_vqe.fun)
#print(result.x)

'''
# Uncomment when using Cudaq optimizer
###############################
## Kernel execution: vqe
###############################
print('VQE energy: ')

#energy_of,data=cudaq.vqe(kernel,molecule,optimizer,tot_params)
#print('Cudaq CCSD energy (cudaq-openfermion)= ', energy_of)

energy_py,data=cudaq.vqe(kernel,spin_ham,optimizer,tot_params)
print('Cudaq CCSD energy (pyscf)= ', energy_py)

########################
# Collect final result
#########################
print('\n')
print('Final result: ')
print('RHF energy= ', myhf.e_tot)
print('CCSD energy= ', mycc.e_tot)
print('FCI energy= ', result[0])
#print('Cudaq VQE-UCCSD energy (pyscf)= ', result_vqe.fun)
#print('Cudaq VQE-UCCSD energy (cudaq-openfermion)= ', energy_of)
print('Cudaq VQE-UCCSD energy (pyscf)= ', energy_py)
print('Total number of qubits: ', qubits_num)
