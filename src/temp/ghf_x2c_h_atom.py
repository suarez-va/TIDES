from pyscf import gto, scf, dft


mol = gto.M(
	verbose = 0,
	atom='H 0 0 0',
	basis='STO-3G',
    spin = 1)

mf = scf.GHF(mol).x2c()
#mf = dft.GKS(mol).x2c()
mf.kernel()
print(mf.e_tot)
print(mf.get_occ())
print(mf.get_ovlp())
mf.dip_moment()
