import numpy as np
from pyscf import gto, dft, scf

'''
Nuclear object for real-time SCF
'''

class NUC:
    def __init__(self, mol):
        self.nnuc = len(mol._atom)
        self.basis = mol.basis
        self.labels = np.array([mol._atom[i][0] for i in range(self.nnuc)])
        self.mass = 1836 * np.array([np.array(mol.atom_mass_list())[i] * np.ones(3) for i in range(self.nnuc)])
        self.pos = np.array([mol._atom[i][1] for i in range(self.nnuc)])
        self.vel = np.zeros((self.nnuc, 3))
        self.force = np.zeros((self.nnuc, 3))

    def get_mol(self):
        atom_str = '\n '
        for i in range(self.nnuc):
            atom_str += self.labels[i]
            atom_str += '    '
            atom_str += str(self.pos[i][0])
            atom_str += '    '
            atom_str += str(self.pos[i][1])
            atom_str += '    '
            atom_str += str(self.pos[i][2])
            atom_str += '\n '
        new_mol = gto.Mole(atom = atom_str, unit = 'Bohr', basis = self.basis)
        new_mol.build()
        return new_mol

    def get_ke(self):
        return np.sum(0.5 * self.mass * self.vel * self.vel)

    def sample_vel(self, beta):
        np.random.seed(47)
        self.vel = np.random.normal(scale = 1. / np.sqrt(beta * self.mass))

    # Position full step
    def get_pos(self, timestep):
        self.pos +=  self.vel * timestep

    # Velocity full step
    def get_vel(self, timestep):
        self.vel += self.force / self.mass * timestep

