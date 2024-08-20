import numpy as np

'''
Real-Time SCF Cleanup
'''

def finalize(rt_mf):
    rt_mf.log.note("Propagation Finished")
    if hasattr(rt_mf, 'fh'):
        rt_mf.fh.close()
