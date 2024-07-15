import numpy as np

'''
Real-Time SCF Cleanup
'''

def finalize(rt_mf):
    rt_mf.time = np.array(rt_mf.time)
    for key, print_value in rt_mf.observables.items(): 
        if print_value:
            if key != 'mo_occ' and not rt_mf.fragments:
                setattr(rt_mf, key, np.array(getattr(rt_mf, key))[:,0,:])
            else:
                setattr(rt_mf, key, np.array(getattr(rt_mf, key)))
    rt_mf.log.note("Propagation Finished")
    if hasattr(rt_mf, 'fh'):
        rt_mf.fh.close()
