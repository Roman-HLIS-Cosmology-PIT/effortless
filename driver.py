import glob

import numpy as np
from astropy.io import fits

from effortless.psfutil import PSFModel
from effortless.io_general import InSlice, OutSlice


psf_in = np.zeros((PSFModel.NTOT, PSFModel.NTOT))
with fits.open("../test_imcom_stips/psf_WFI_2.3_F158_wfi01.fits") as hdul:
    psf_in[8:-7, 8:-7] = hdul[0].data.mean(axis=0)
psfmodel = PSFModel(psf_in)
inslices = [InSlice(name, psfmodel) for name
            in glob.glob("../test_imcom_stips/sim_*.fits")]

outcrval = np.array([201.80002222222, -41.799977777778])
outwcs = OutSlice.get_outwcs(outcrval)
outslice = OutSlice(outwcs, inslices, timing=True)
outslice(filename="../test_imcom_stips/test_imcom_stips_new.fits", timing=True)

with fits.open("../test_imcom_stips/test_imcom_stips.fits") as a, \
    fits.open("../test_imcom_stips/test_imcom_stips_new.fits") as b:
    fd = fits.FITSDiff(a, b)
    print('identical:', fd.identical)
