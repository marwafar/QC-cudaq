import cudaq, cudaq_solvers as solvers
import uccsd_pool

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

# Check pool operator
print('check pool operators')
new_pool=[]
coef_pool=[]
for i in range(len(pools)):
    op_i=pools[i]
    temp_op=[]
    temp_coef=[]
    op_i.for_each_term(lambda term: temp_op.append(term.to_string(False)))
    op_i.for_each_term(lambda term: temp_coef.append(term.get_coefficient()))
    new_pool.append(temp_op)
    coef_pool.append(temp_coef)
print(new_pool)
print(coef_pool)


def commutator(pools, ham):
    com_op=[]
    
    for i in range(len(pools)):
        op=pools[i]
        com_op.append(ham*op-op*ham)
         
        #test    
        #x=ham*op-op*ham
        #print(x)
        #or
        #test=[]
        #x.for_each_term(lambda term: test.append(term.to_string(False)))
        #print(test)
    return com_op
        
grad_op=commutator(pools, ham)

print('Test commutator operator')

for op in grad_op:
    print(op)

    

