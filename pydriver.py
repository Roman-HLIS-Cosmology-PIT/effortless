import warnings; warnings.filterwarnings("ignore")
# Omit AsdfConversionWarning and AsdfPackageVersionWarning.

import numpy as np

from effortless.io_pyimcom import EConfig, PyOutSlice
# from effortless.psfutil import SubSlice


cfg = EConfig("../pyimcom_hack_2025Aug4/config_test25-F.json")

# Use local input files instead those on OSC scratch disk.
cfg.obsfile = "../S25-RUN/Roman_WAS_obseq_11_1_23.fits"
cfg.inpath = "../S25-RUN/L2"
cfg.inpsf_path = "../S25-RUN/psf6"

# Use science images, injected stars, and simulated noise fields.
cfg.extrainput = [None, "gsstar14", "whitenoise10", "1fnoise9"]

# Use local output directory; turn off cache and PSF split.
cfg.outstem = "../S25-RUN/test25_F"
cfg.tempfile = None  # "S25-RUN/tmp/pyimcomrun_2J"
cfg.inlayercache = None  # "S25-RUN/cache/in_F"
cfg.psfsplit = ""

# Only coadd 4 postage stamps (to see if the code works).
# cfg.stoptile = 4
cfg.stoptile = np.inf

cfg.configure_effortless()
PyOutSlice.SAVE_ALL = True
# PyOutSlice.SIGMA *= 1.2
# SubSlice.ACCEPT = 16
outslice = PyOutSlice(cfg, timing=True)
