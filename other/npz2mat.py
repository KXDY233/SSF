from scipy.io import savemat
import numpy as np
import glob
import os
npzFiles = glob.glob("../data/*.npz")
for f in npzFiles:
    fm = os.path.splitext(f)[0]+'.mat'
    d = np.load(f,allow_pickle=True)
    savemat(fm, d)
    print('generated ', fm, 'from', f)