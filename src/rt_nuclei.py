import numpy as np
from pyscf import gto, dft, scf
from basis_utils import read_mol, write_mol

'''
Nuclear Object for Real-Time Ehrenfest
'''

class NUC:
    def __init__(self, mol):
        self.nnuc = len(mol._atom)
        basis, labels, pos = read_mol(mol)
        self.basis = basis
        self.labels = labels 
        self.pos = np.array(pos)
        self.mass = 1836 * np.array([np.array(mol.atom_mass_list())[i] * np.ones(3) for i in range(self.nnuc)])
        self.vel = np.zeros((self.nnuc, 3))
        self.force = np.zeros((self.nnuc, 3))

    def get_mol(self):
        return write_mol(self.basis, self.labels, self.pos)

    def get_ke(self):
        return 0.5 * self.mass * self.vel**2

    def sample_vel(self, beta):
        #np.random.seed(47)
        self.vel = np.random.normal(scale = 1. / np.sqrt(beta * self.mass))

    # Position full step
    def get_pos(self, timestep):
        self.pos +=  self.vel * timestep

    # Velocity full step
    def get_vel(self, timestep):
        self.vel += self.force / self.mass * timestep

