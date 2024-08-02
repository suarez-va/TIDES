import numpy as np

# Additinal term arising from the moving nuclei in the classical path approximation to be added to the fock matrix

def get_omega(rt_cpa):
    mol = rt_cpa._scf.mol
    Rdot = rt_cpa.nuc.vel
    X = rt_cpa.orth
    print(X)
    Xinv = np.linalg.inv(X)
    dS = -mol.intor('int1e_ipovlp', comp=3)

    Omega = np.zeros(X.shape, dtype = complex)
    RdSX = np.zeros(X.shape)
    aoslices = mol.aoslice_by_atom()
    for i in range(mol.natm):
        p0, p1 = aoslices[i,2:]
        RdSX += np.einsum('x,xij,ik->jk', Rdot[i], dS[:,p0:p1,:], X[p0:p1,:])
    Omega += np.matmul(RdSX, Xinv)
    return Omega

