import numpy as np

'''
Real-time SCF output file
'''

def create_output_file(rt_mf):
    output_main = open(F'{rt_mf.filename}.txt', 'w')

    output_main.write('{0: >10}'.format('Time'))
    output_main.write('{0: >33}'.format('Total charge'))
    output_main.write('{0: >15}'.format('Energy'))
    if not isinstance(rt_mf.fragments, list):
        for fragment in range(len(rt_mf.fragments)):
            output_main.write('{0: >25}'.format('Fragment' + str(fragment)))

    output_main.write(F'\n')
    output_main.close()

    moocc_file = open(F'{rt_mf.filename}' + '_moocc.txt','w')
    moocc_file.close()

    dipole_file = open(F'{rt_mf.filename}' + '_dipole.txt','w')
    dipole_file.close()
    if rt_mf.mag:
        mag_file = open(F'{rt_mf.filename}' + '_magnetization.txt','w')
        mag_file.write('{0: >5}'.format('Time'))
        mag_file.write('{0: >25}'.format('Mag x'))
        mag_file.write('{0: >35}'.format('Mag y'))
        mag_file.write('{0: >45}'.format('Mag z'))
        mag_file.write(F'\n')
        mag_file.close()


def update_output_file(rt_mf, t, ener_tot, dipole, den_mo, charge, mag=None):
    output_main = open(F'{rt_mf.filename}.txt', 'a')
    output_main.write(F'{t:20.8e} \t {np.real(charge[0]):20.8f} \t {ener_tot:20.8e}')
    if not isinstance(rt_mf.fragments, list):
        for fragment in range(len(rt_mf.fragments)):
            output_main.write(F'\t {np.real(charge[fragment+1]):20.8f}')

    output_main.write(F'\n')
    output_main.close()

    moocc_file = open(F'{rt_mf.filename}' + '_moocc.txt', 'a')
    moocc_file.write(F'{t:20.8e} {str(np.diagonal(den_mo))[1:-1]} \t \n')
    moocc_file.close()

    dipole_file = open(F'{rt_mf.filename}' + '_dipole.txt', 'a')
    dipole_file.write(F'{t:20.8e} \t {str(dipole)} \t \n')
    dipole_file.close()

    if rt_mf.mag:
        mag_file = open(F'{rt_mf.filename}' + '_magnetization.txt','a')
        mag_file.write(F'{t:20.8e} \t {np.real(mag[0]):20.8e} \t {np.real(mag[1]):20.8e} \t {np.real(mag[2]):20.8e} \n')
        mag_file.close()
