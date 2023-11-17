import argparse
import numpy as np
from numpy import linalg as LA
import scipy

from pyscf import gto, scf, mp, mcscf, fci, cc
from pyscf import ao2mo
from pyscf.tools import molden
from functools import reduce

import openfermion
import openfermionpyscf
from openfermion import generate_hamiltonian
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.chem import MolecularData

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
print('RHF energy= ', myhf.e_tot, '\n')

#nuclear_repulsion = myhf.energy_nuc()
nelec = mol.nelectron
print('Total number of electrons= ', nelec, '\n')
norb = myhf.mo_coeff.shape[1]
print('Total number of orbitals= ', norb, '\n')

###################################
# MP2
##################################
mymp=mp.MP2(myhf)
mp_ecorr, mp_t2=mymp.kernel()

print('MP2 corre_e= ', mp_ecorr, '\n')
print('Total MP2 energy= ', mymp.e_tot, '\n')

############################################
# Compute natural orbitals and 
# natural orbital occupation number (NOON) 
# to inspect the active space.
##############################################

dm1=mymp.make_rdm1()
noon, U= LA.eigh(dm1)
noon= np.flip(noon)
natorbs=np.dot(myhf.mo_coeff[:,:],U)
natorbs=np.fliplr(natorbs)

# Export the natural orbitals as molden file for visualization
#molden.from_mo(mol, filename+'_MP2.molden', natorbs)

print('Natural orbital occupation number from MP2')
print (noon, '\n')

#############
# Alternative way to compute nat. orb.
#############

#noons, natorbs = mcscf.addons.make_natural_orbitals(mymp)
#print('Natural orbital occupation number from MP2')
#print(noons, '\n')

##########################################
## Define your active space based on NOON
#########################################
# Choose the active space based on the NOON inspection above.

norb_cas, nele_cas = (8,8)

print('active space[orbital,elec]= ', norb_cas,nele_cas, '\n')

################################
## CASCI
################################

# Using natural orbitals tocompute CASCI (mo_coeff are natural orbitals)

mycasci = mcscf.CASCI(myhf, norb_cas, nele_cas)
mycasci.kernel(natorbs)

print('Tota CASCI energy (with nat orb)= ', mycasci.e_tot, '\n')


#### Without natural orbitals (mo_coeff are not natural orbitals)

mycasci_mo = mcscf.CASCI(myhf, norb_cas, nele_cas)
mycasci_mo.kernel()

print('Tota CASCI energy (without nat orb)= ', mycasci_mo.e_tot, '\n')

#print('No. core orbitals= ',mycasci.ncore, '\n')

##################################
## CCSD
##################################

# Define the frozen orbitals
frozen=[]
frozen+=[y for y in range(0,mycasci.ncore)]
frozen+=[y for y in range(mycasci.ncore+norb_cas, len(mycasci.mo_coeff))]
#print('Frozen orbitals= ',frozen, '\n')

## CCSD for the active space (mo_coeff are nat orb)
mycc=cc.CCSD(myhf,frozen=frozen, mo_coeff=natorbs).run()

print('Total CCSD energy for H_act (with nat orb)= ', mycc.e_tot, '\n')

#print('t1_ccsd= ', mycc.t1.shape, '\n')
#print('t2_ccsd= ', mycc.t2.shape, '\n')

# CCAS without nat. orb.
mycc_mo=cc.CCSD(myhf,frozen=frozen).run()
print('Total CCSD energy for H_act (without nat orb)= ', mycc_mo.e_tot, '\n')

######################################
## CASSCF
######################################

mycas = mcscf.CASSCF(myhf, norb_cas, nele_cas)
mycas.kernel(natorbs)

print('Total CASSCF enery= ', mycas.e_tot, '\n')

#################################
## FCI 
#################################
# Here we use the generated active space Hamiltonian 
# from CASSCF to compute the FCI.

h1e_cas, ecore = mycas.get_h1eff()
h2e_cas = mycas.get_h2eff()

e_fci, fcivec = fci.direct_spin1.kernel(h1e_cas,\
        h2e_cas,norb_cas, nele_cas, ecore=ecore)

print('Total energy from FCI for H_act (casscf)= ', e_fci, '\n')

###################################################
## Compute electron integrals within a chosen
## MO basis 
###################################################
'''
# CASCI MO basis
h1e_cas,ecore=mycasci_mo.get_h1eff()
h2e_cas=mycasci_mo.get_h2eff()
h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
# Reorder the chemist notation (pq|rs) ERI h_pqrs to h_prqs
# to "generate_hamiltonian" in openfermion 
tbi = np.asarray(h2e_cas.transpose(0,2,3,1), order='C')
'''

'''
# CASCI nat orb
h1e_cas,ecore=mycasci.get_h1eff()
h2e_cas=mycasci.get_h2eff()
h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
# Reorder the chemist notation (pq|rs) ERI h_pqrs to h_prqs
# to "generate_hamiltonian" in openfermion 
tbi = np.asarray(h2e_cas.transpose(0,2,3,1), order='C')
'''


# CASSCF
h1e_cas, ecore = mycas.get_h1eff()
h2e_cas = mycas.get_h2eff()
h2e_cas = ao2mo.restore('1', h2e_cas, norb_cas)
# Reorder the chemist notation (pq|rs) ERI h_pqrs to h_prqs
# to "generate_hamiltonian" in openfermion 
tbi = np.asarray(h2e_cas.transpose(0,2,3,1), order='C')


print('h1e_shape ', h1e_cas.shape, '\n')
print('h2e_shape ', h2e_cas.shape, '\n')

########################################################
## Generate the active space spin operator Hamiltonian 
########################################################
as_ham = generate_hamiltonian(h1e_cas,tbi,ecore)

ham_operator = jordan_wigner(as_ham)

spin_ham=cudaq.SpinOperator(ham_operator)
'''
#########################################
# Altherantive way to generate the spin 
# operator Hamiltonian using openfermion.
#########################################
geometry=[('N', (0., 0., 0.)), ('N', (0., 0., 1.2))]
basis='631g'
multiplicity=args.s+1
charge=args.c

#molecule = MolecularData(args.xyz, args.basis, args.s+1,args.c)
molecule = openfermionpyscf.run_pyscf\
    (openfermion.MolecularData(geometry, basis, multiplicity,charge))

molecular_hamiltonian = molecule.get_molecular_hamiltonian(\
    occupied_indices=range(mycasci.ncore), active_indices=range(mycasci.ncore,mycasci.ncore+norb_cas))

fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

spin_ham_of=cudaq.SpinOperator(qubit_hamiltonian)
'''
##########################################################
## 2- Quantum computing

print('Beginning of quantum computing simulation','\n')

###########################################################
# Define the total number of qubits.
# Total number of qubits equal the 
#total spin molecular/natural orbitals

qubits_num=2*norb_cas

### Backend configuration (target=default,nvidia,nvidia-mgpu).
cudaq.set_target("nvidia")

###############################
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
for i in range(nele_cas):
    kernel.x(qubits[i])

###########################
##  Ansatz: UCCSD
###########################

cudaq.kernels.uccsd(kernel, qubits, params, nele_cas, qubits_num)
parameter_count = cudaq.kernels.uccsd_num_parameters(nele_cas,qubits_num)

# Check that total parameters match with tot_param below
print('param_count from cudaq= ,', parameter_count, '\n')

# Initialize the UCCSD from the classical CCSD results
init_params,tot_params=init_param_CCSD(qubits_num,nele_cas,mycc.t1,mycc.t2)
print('param_count from my code= ,', tot_params, '\n')

#####################################################
## Optimizer for the parameterized quantum circuit
#####################################################
#optimizer= cudaq.optimizers.GradientDescent()
#optimizer = cudaq.optimizers.LBFGS()
optimizer = cudaq.optimizers.COBYLA()

optimizer.initial_parameters=init_params
#optimizer.max_iterations=100000

#print(optimizer.initial_parameters)
'''
######################################
## Alternative way for optimization
## Using Scipy.optimizer
######################################

# For gradient based optimizer
gradient = cudaq.gradients.CentralDifference()
#gradient = cudaq.gradients.ForwardDifference()
#gradient = cudaq.gradients.ParameterShift()

def objective_function(parameter_vector: List[float], \
                       gradient=gradient, hamiltonian=spin_ham, kernel=kernel):


    get_result = lambda parameter_vector: cudaq.observe\
        (kernel, hamiltonian, parameter_vector).expectation_z()
    
    cost = get_result(parameter_vector)
    print(f"<H> = {cost}")
    #gradient_vector = gradient.compute(parameter_vector, get_result,cost)
    

    return cost
    #return cost, gradient_vector

#result_vqe=scipy.optimize.minimize(objective_function,init_params,method='L-BFGS-B', jac=True, tol=1e-7)
result_vqe=scipy.optimize.minimize(objective_function,init_params,method='L-BFGS-B', jac='3-point')

print('Optimizer exited successfully: ',result_vqe.success)
print(result_vqe.message)
print('Cudaq VQE-UCCSD energy (pyscf)= ', result_vqe.fun)
#print(result.x)

'''
###############################
## Kernel execution: vqe
###############################
print('VQE')

energy_py,data=cudaq.vqe(kernel,spin_ham,optimizer,tot_params)
print('Cudaq VQE-UCCSD energy (pyscf)= ', energy_py)

#energy_of,data=cudaq.vqe(kernel,spin_ham_of,optimizer,tot_params)
#print('Cudaq VQE-UCCSD energy (of)= ', energy_of)

##################################
# Collect final result
##################################
print('\n')
print('Final result: ')
print('RHF energy= ', myhf.e_tot)
print('Total CCSD energy for H_act (with nat orb)= ', mycc.e_tot)
print('Total CCSD energy for H_act (without nat orb)= ', mycc_mo.e_tot)
print('Tota CASCI energy (with nat orb)= ', mycasci.e_tot)
print('Tota CASCI energy (without nat orb)= ', mycasci_mo.e_tot)
print('Total CASSCF enery= ', mycas.e_tot)
print('Total energy from FCI for H_act (casscf)= ', e_fci)
print('Cudaq VQE-UCCSD energy (pyscf)= ', energy_py)
#print('Cudaq VQE-UCCSD energy (openfermion)= ', energy_of)
print('Total number of qubits: ', qubits_num)
#print('Cudaq VQE-UCCSD energy (pyscf)= ', result_vqe.fun)
