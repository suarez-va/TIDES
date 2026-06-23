import numpy as np
import os
import sys
from scipy.linalg import inv
from pyscf import mcscf, fci
from pyscf.scf import addons
from pyscf.lib import logger
from tides.rt_casprop_np import propagate
from tides import rt_observables
from tides import fci_mod as fci_mod
from tides.rt_utils import restart_from_chkfile

'''
RT-RAS-SCF is untested
Does not project out virtual space
'''

class RT_CAS_RAS:
    '''
    opt: String
        opt='CASCI' for TD-CAS/RAS-CI calculation
        opt='CASSCF' for TD-CAS/RAS-SCF calculation
    ras: pyscf mcscf CASSCF/CASCI object, solved for its t=0 state
    timestep: Float, as defined for the given propogation method
    max_time: Float, total time to run the dynamics for
    outputName: String, name of output file used to check for numerical stability
    corrDenName: String, name of output file that shows each AO occupation at each time step
    reg: Float, minimum allowed eigenvalue of matrices before inversion can take place
        Only actually effects CAS/RAS SCF
    opts: Array of integers indicating allowed excitations (1 allows -S, 2 allows -D, etc.). Full CAS calculation is opts left as default.
    filename: I don't use this, left to keep consistent with rt_scf class
    h1e: 1 e Hamiltonian in AO basis at initial time. Use Chemists' notation.
    h2e: 2 e Hamiltonian in AO basis at initial time. Use Chemists' notation.
    prop: Propogation method in rt_integrators to use.
        rk4cr: Fourth-order runge-kutta integrator
        vv: Second-order symplectic split operator integrator. Only implemented for CASCI, not CASSCF
    frequency: How many time steps you want between prints to output files
    mo_to_ao: MO to AO transformation matrix (MO coefficient matrix). Sort columns (core,active,virtual).
    orth: Orthogonal AO coefficient matrix
    chkfile: I don't use this, left to keep consistent with rt_scf class
    verbose: I don't use this, left to keep consistent with rt_scf class
    ovlp: AO overlap matrix
    '''
    def __init__(self, opt,ras, timestep, max_time, outputName, corrDenName, reg=1e-5, opts = None, filename=None, h1e=None, h2e=None, prop=None, frequency=1, mo_to_ao=None, orth=None, chkfile=None, verbose=3, ovlp=None):
        self.timestep = timestep
        self.frequency = frequency
        self.max_time = max_time
        self._scf = ras
        self.outName = outputName
        self.corName = corrDenName
        self.ep = reg
        
        # CASCI or CASSCF
        self._castype = opt

        # Artifact from projected
        self.numP = self._scf.mol.nao

        # Number of spin up electrons
        self.neleca = self._scf.nelecas[0] + self._scf.ncore

        # Number of spin down electrons
        self.nelecb = self._scf.nelecas[1] + self._scf.ncore

        self.verbose = verbose

        # The time-dependent portion of the Hamiltonian will be stored here as a function, if it exists
        self._potential = []

        self.labels = [self._scf.mol._atom[idx][0] for idx, _ in enumerate(self._scf.mol._atom)]

        # Get Hamiltonians in AO basis, MO/AO basis transformation matrix, AO overlap matrix, and/or AO/OAO basis transformation matrix from cas object if no custom ones are given
        if h1e is None:
            self._h1e_AO_0 = self._scf.get_hcore()
        else:
            self._h1e_AO_0 = h1e
        if h2e is None:
            self._h2e_AO = self._scf.mol.intor('int2e')
        else:
            self._h2e_AO = h2e
        if mo_to_ao is None:
            self.mo_to_ao = self._scf.mo_coeff
        else:
            self.mo_to_ao = mo_to_ao
        if ovlp is None:
            self.ovlp = self._scf.mol.intor('int1e_ovlp')
        else:
            self.ovlp = ovlp
        if orth is None:
            self.orth = addons.canonical_orth_(self.ovlp)
        else:
            self.orth = orth

        if prop is None: prop = 'rk4cr'
        self.prop = prop

        # See vv in rt_integrators or rt_integrators_np
        if prop == 'vv':
            self.pMinusHalf = 0
            self.pDotH = 0
            self.firstStep = True
            if self._castype == 'CASSCF':
                print('TD-_-SCF isnt applicable to velocity verlet')
                sys.exit()
                # Will replace with error when error class gets implemented

        # One electron Hamiltonian at the current time
        self._h1e_AO = np.copy(self._h1e_AO_0)

        # Number of atomic orbitals
        self.no = len(self._h1e_AO)

        self.mo_to_ao = np.copy(mo_to_ao)

        # AO to MO transformation matrix
        self.ao_to_mo = self.get_ao_to_mo()

        self.orth = orth

        if filename is None:
            self._log = logger.Logger(verbose=self.verbose)
        else:
            self._fh = open(filename, 'a') # Temporarily making _fh append to file
            self._log = logger.Logger(self._fh, verbose=self.verbose)

        self.den_ao = self.get_den_ao()

        if len(np.shape(self.den_ao)) == 3:
            self.nmat = 2
        else:
            self.nmat = 1

        # Restart from chkfile, or create a chkfile
        # If restarting from chkfile, self.den_ao will be rewritten
        # I do not use this functionality and have not tested it
        self.chkfile = chkfile
        if chkfile is not None:
            if os.path.exists(self.chkfile):
                restart_from_chkfile(self)
                self.den_ao = mcscf.make_rdm1(self._scf)
            else:
                self.current_time = 0
        else:
            self.current_time = 0

        self.ras = False
        # self.ind gives CI vector indices that the input ras scheme allows to be nonzero
        # self.ras being True indicates RAS CI/SCF
        # self.ras being False indicates CAS CI/SCF
        if opts is not None:
            self.opts = opts
            self.ras = True
            indA = np.array(fci.cistring.make_strings(range(self._scf.ncas),self._scf.nelecas[0]),dtype=np.uint8)
            rowOccs = np.unpackbits(indA[:,np.newaxis],axis=1,bitorder='little',count=self._scf.ncas)
            nA = np.sum(rowOccs[0])
            p2A = np.delete(rowOccs,np.s_[0:nA],axis=1)
            indB = np.array(fci.cistring.make_strings(range(self._scf.ncas),self._scf.nelecas[1]),dtype=np.uint8)
            colOccs = np.unpackbits(indB[:,np.newaxis],axis=1,bitorder='little',count=self._scf.ncas)
            nB = np.sum(colOccs[0])
            p2B = np.delete(colOccs,np.s_[0:nB],axis=1)
            goodInd = np.zeros((len(p2A),len(p2B)))
            for i in range(len(goodInd)):
                for j in range(len(goodInd[0])):
                    el = np.sum(p2A[i]) + np.sum(p2B[j])
                    if el == 0 or el in self.opts:
                        goodInd[i,j] = 1
            self.ind = goodInd

        self._t0 = self.current_time

        rt_observables._init_observables(self)

    # I don't use this, left to keep consistent with rt_scf class
    def istype(self, type_code):
        if isinstance(type_code, type):
            return isinstance(self, type_code)

        return any(type_code == t.__name__ for t in self.__class__.__mro__)
    
    # Increment current time by an amount determined by the propogation method
    def update_time(self):
        if self.prop == 'rk4cr' or self.prop == 'vv':
            self.current_time += (self.timestep/2)
        else:
            self.current_time += self.timestep

    # Get AO to MO transformation matrix from MO to AO transformation matrix
    def get_ao_to_mo(self):
        return self.mo_to_ao.conj().T
    
    # Get MO to AO transformation matrix from AO to MO transformation matrix
    def get_mo_to_ao(self):
        return self.ao_to_mo.conj().T

    # Transform 1e Hamiltonian from AO to MO basis
    def get_h1e_mo(self):
        rawOut = np.matmul(self.mo_to_ao.conj().T,np.matmul(self._h1e_AO,self.mo_to_ao)).astype(np.complex128)
        return rawOut

    # Transform 1e Hamiltonian from AO to OAO basis
    # Also, updates time-dependent Hamiltonian
    def get_h1e_orth(self):
        if self._potential: self.apply_potential()
        return np.matmul(self.orth.conj().T,np.matmul(self._h1e_AO,self.orth)).astype(np.complex128)

    # Transform 2e Hamiltonian from AO to MO basis
    def get_h2e_mo(self):
        mat1 = np.einsum('ap,pqrs,qb',self.mo_to_ao.conj().T,self._h2e_AO,self.mo_to_ao).astype(np.complex128)
        return np.einsum('cr,abrs,sd',self.mo_to_ao.conj().T,mat1,self.mo_to_ao).astype(np.complex128)
    
    # Transform 2e Hamiltonian from AO to OAO basis
    def get_h2e_orth(self):
        mat1 = np.einsum('ap,pqrs,qb',self.orth.conj().T,self._h2e_AO,self.orth).astype(np.complex128)
        return np.einsum('cr,abrs,sd',self.orth.conj().T,mat1,self.orth).astype(np.complex128)
    
    # Transform a single-particle orbital from MO to AO basis
    def rotate_mo_to_ao(self,coeff_mo):
        return np.matmul(self.mo_to_ao,coeff_mo)
    
    # Transform a single-particle orbital from AO to OAO basis
    def rotate_ao_to_orth(self, coeff_ao):
        return np.matmul(inv(self.orth), coeff_ao)
    
    # Gets density matrix in AO basis from the current ci vector and AO to MO transformation matrix
    def get_den_ao(self):
        corr1RDMcas, corr2RDMcas = fci_mod.get_corr12RDM(self._scf.ci, self._scf.ncas, self._scf.nelecas)
        self.casrdm1 = np.copy(corr1RDMcas) # Stores active space 1rdm in MO basis
        self.casrdm2 = np.copy(corr2RDMcas) # Stores active space 2rdm in MO basis
        corr1RDMmo = np.zeros((self.numP,self.numP)).astype(np.complex128)
        for a in range(self._scf.ncore):
            corr1RDMmo[a][a] = 2
        for a in range(self._scf.ncas):
            for b in range(self._scf.ncas):
                corr1RDMmo[a+self._scf.ncore][b+self._scf.ncore] = corr1RDMcas[a][b]
        return(np.matmul(self.ao_to_mo.conj().T,np.matmul(corr1RDMmo,self.ao_to_mo)))
    
    # MO to OAO transformation matrix
    def get_mo_to_orth(self):
        return(self.orth.conj().T @ self.mo_to_ao)
    
    # Expresses mo'th molecular orbital in OAO basis
    def mo_unitvec_to_orth(self,mo):
        return self.mo_to_orth[:,mo]
    
    # Adds time-dependent term to the list of time-dependent Hamiltonian terms
    def add_potential(self, *args):
        for v_ext in args:
            self._potential.append(v_ext)

    # Updates h1e in AO basis to reflect Hamiltonian at the current time
    # Note that self._potential must be a list containing a single time-dependent term
    def apply_potential(self):
        for v_ext in self._potential:
            self._h1e_AO = self._h1e_AO_0 + v_ext.calculate_potential(self)
    
    # Returns X[u,q] for the X-matrix expressed in the MO basis, with active orbital q and core orbital u
    # Corresponds to eq 36 of Phys. Rev. A 88, 023402
    # In group notes, corresponds to Xqu on page 9 of td_cas/td_casscf_notes
    def getQU(self,qn,un,dBarInv):
        toReturn = self._h1e_mo[un][qn+self._scf.ncore]
        for k in range(self._scf.ncas):
            kMO = k+self._scf.ncore
            prefac = dBarInv[qn,k]
            toAdd = 0
            for v in range(self._scf.ncore):
                toAdd = toAdd + 2*((2*self._h2e_mo[v][v][un][kMO])-self._h2e_mo[v][un][v][kMO])
                for l in range(self._scf.ncas):
                    lMO = l+self._scf.ncore
                    toAdd = toAdd - (self.casrdm1[k][l]*((2*self._h2e_mo[v][v][un][lMO])-self._h2e_mo[lMO][v][un][v]))
            for l in range(self._scf.ncas):
                for m in range(self._scf.ncas):
                    lMO = l+self._scf.ncore
                    mMO = m+self._scf.ncore
                    toAdd = toAdd + (self.casrdm1[m][l]*((2*self._h2e_mo[lMO][mMO][un][kMO])-self._h2e_mo[lMO][un][mMO][kMO]))
                    for j in range(self._scf.ncas):
                        jMO = j+self._scf.ncore
                        toAdd = toAdd - (self._h2e_mo[jMO][lMO][un][mMO]*self.casrdm2[j][l][m][k])
            toReturn = toReturn + (prefac*toAdd)
        return toReturn
    
    # Returns X[u,a] for the X-matrix expressed in the MO basis, with virtual orbital a and core orbital u
    # Corresponds to eq 34 of Phys. Rev. A 88, 023402
    # In group notes, corresponds to Xau on page 9 of td_cas/td_casscf_notes
    def getAU(self,an,un):
        toReturn = self._h1e_mo[un][an]
        for v in range(self._scf.ncore):
            toReturn = toReturn + (2*self._h2e_mo[v][v][un][an]) - self._h2e_mo[v][un][v][an]
        for k in range(self._scf.ncas):
            kMO = k + self._scf.ncore
            for l in range(self._scf.ncas):
                lMO = l + self._scf.ncore
                toReturn = toReturn + (self.casrdm1[k][l]*(self._h2e_mo[lMO][kMO][un][an]-(self._h2e_mo[lMO][un][kMO][an]/2)))
        return toReturn
    
    # Returns X[q,a] for the X-matrix expressed in the MO basis, with active orbital q and virtual orbital a
    # Corresponds to eq 35 of Phys. Rev. A 88, 023402
    # In group notes, corresponds to Xaq on page 9 of td_cas/td_casscf_notes
    def getAQ(self,an,qn,dInv):
        qMO = qn + self._scf.ncore
        toReturn = self._h1e_mo[qMO][an]
        for u in range(self._scf.ncore):
            toReturn = toReturn + (((2*self._h2e_mo[u][u][qMO][an]) - self._h2e_mo[u][qMO][u][an]))
        for l in range(self._scf.ncas):
            for k in range(self._scf.ncas):
                kMO = k + self._scf.ncore
                for j in range(self._scf.ncas):
                    jMO = j + self._scf.ncore
                    for m in range(self._scf.ncas):
                        mMO = m + self._scf.ncore
                        toReturn = toReturn + (self._h2e_mo[jMO][kMO][mMO][an]*self.casrdm2[l][m][j][k]*dInv[l][qn])
        return toReturn
    
    def get_x(self):
        toReturn = np.zeros((self.numP,self.numP),dtype=np.complex128)
        if self._castype == 'CASCI':
            return toReturn
        else:
            # Regularization of active-space RDM and "active-space-rdm-bar" (See eq 36 in Phys. Rev. A 88, 023402)
            eigvalrdm, eigvecrdm = np.linalg.eig(self.casrdm1)
            for eigInd in range(len(eigvalrdm)):
                if eigvalrdm[eigInd] < self.ep:
                    eigvalrdm[eigInd] = self.ep
            eigMatRdm = np.diag(eigvalrdm)
            self.casrdm1 = eigvecrdm @ eigMatRdm @ inv(eigvecrdm)
            twoEye = 2*np.eye(self._scf.ncas)
            dBar = twoEye - self.casrdm1
            eigvalBar, eigvecBar = np.linalg.eig(dBar)
            for eigInd in range(len(eigvalBar)):
                if eigvalBar[eigInd] < self.ep:
                    eigvalBar[eigInd] = self.ep
            eigMatBar = np.diag(eigvalBar)
            dBarMat = eigvecBar @ eigMatBar @ inv(eigvecBar)
            self.casrdm1 = twoEye - dBarMat
            dBarInv = inv(dBarMat)
            dInv = inv(self.casrdm1)

            # Get values of X with getQU, getAU, and getAQ
            numA = self.numP - self._scf.ncore - self._scf.ncas
            for u in range(self._scf.ncore):
                for q in range(self._scf.ncas):
                    qu = self.getQU(q,u,dBarInv)
                    toReturn[q+self._scf.ncore,u] = qu.conjugate()
                    toReturn[u,q+self._scf.ncore] = np.copy(qu)
                for a in range(numA):
                    aMO = a + self._scf.ncore + self._scf.ncas
                    au = self.getAU(aMO,u)
                    toReturn[aMO,u] = au.conjugate()
                    toReturn[u,aMO] = np.copy(au)
            for a in range(numA):
                aMO = a + self._scf.ncore + self._scf.ncas
                for q in range(self._scf.ncas):
                    aq = self.getAQ(aMO,q,dInv)
                    qMO = q + self._scf.ncore
                    toReturn[aMO,qMO] = aq.conjugate()
                    toReturn[qMO,aMO] = np.copy(aq)
            return toReturn
    
    # Returns active space core shift and 1e and 2e Hamiltonians
    # Analogous to final result on page 12 of group's dmet/dmet_original_jctc notes (cannot find equation in J. Chem. Theory Comput. 2013, 9, 3, 1428–1432)
    def get_embH(self,x):
        h1 = self._h1e_mo - x
        Econst = 0.0
        for e in range(self._scf.ncore):
            Econst = Econst + (2*h1[e,e])
            for f in range(self._scf.ncore):
                Econst = Econst + (2*self._h2e_mo[e][e][f][f]) - self._h2e_mo[e][f][f][e]
        h1Mat = np.zeros((self._scf.ncas,self. _scf.ncas),dtype=np.complex128)
        for a in range(self._scf.ncas):
            for b in range(self._scf.ncas):
                aMO = a+self._scf.ncore
                bMO = b+self._scf.ncore
                h1Mat[a,b] = h1[aMO,bMO]
                for e in range(self._scf.ncore):
                    h1Mat[a,b] = h1Mat[a,b] + (2*self._h2e_mo[bMO][aMO][e][e]) - self._h2e_mo[bMO][e][e][aMO]
        h2Mat = np.zeros((self._scf.ncas,self. _scf.ncas,self._scf.ncas,self. _scf.ncas),dtype=np.complex128)
        for a in range(self._scf.ncas):
            aMO = a+self._scf.ncore
            for b in range(self._scf.ncas):
                bMO = b+self._scf.ncore
                for c in range(self._scf.ncas):
                    cMO = c+self._scf.ncore
                    for d in range(self._scf.ncas):
                        dMO = d+self._scf.ncore
                        h2Mat[a][b][c][d] = self._h2e_mo[aMO][bMO][cMO][dMO]
        return Econst,h1Mat,h2Mat

    # Begin time propogation. I don't use mo_coeff_print, left to keep consistent with rt_scf class
    def kernel(self, mo_coeff_print=None):
        try:
            propagate(self, mo_coeff_print)
        except Exception:
            raise
        finally:
            if np.isclose(self.current_time,self.max_time):  # So calculation terminates once max_time is reached after restarts
                self._log.note('Done')
            else:
                self._log.note('Propogation Stopped Early')
            if hasattr(self,'fh'):
                self.fh.close()
            if hasattr(self,'_xyz_fh'):
                # This is only important for unfrozen nuclei, printing .xyz files
                # Putting this here anyways for RT_Ehrenfest and other future derived classes
                self._xyz_fh.close()

        return self
