import sys
import numpy as np
from effortless.io_pyimcom import EConfig, PyOutSlice


# Usage: python run_anlsim.py <band> <nomask> <this_sub>
# The band should be one of "Y106", "J129", "H158", and "F184".
# The nomask flag should be 0 or 1 (to be converted to a boolean).
# The this_sub index should be a non-negative integer less than BLOCK**2.
band, nomask, this_sub = sys.argv[1], bool(int(sys.argv[2])), int(sys.argv[3])

# Effortless settings that are not in PyIMCOM configuration files.
bl_circ_dict = {"Y106": (22+0.5) * 2.0**0.5, "J129": (22+0.5) * 2.0**0.5,
                "H158": (20+0.5) * 2.0**0.5, "F184": (18+0.5) * 2.0**0.5}
bl_inner_dict = {"Y106": 0, "J129": 0, "H158": 22, "F184": 18}


cfg = EConfig(f"../pyimcom/configs/paper4_configs/{band}_Chol_benchmark.json")

# Use PAS2733 input files instead of those on PCON0003, because PyIMCOM wants
# the input image directory to be called "simple" instead of "simple_model".
cfg.inpath = "/fs/scratch/PAS2733/anlsim"
cfg.inpsf_path = "/fs/scratch/PAS2733/anlsim/psf"

# Use science images, injected stars, and simulated noise fields.
cfg.extrainput = [None, "gsstar14", "whitenoise10", "1fnoise9"]
cfg.cr_mask_rate *= 1.2  # To address the lack of "labnoise".

# Use (pre-made) output directories for Effortless tests.
cfg.outstem = f"/fs/scratch/PAS2733/paper4_effortless/paper4_{band}_" +\
    ("nomask" if nomask else "effortless") + f"/paper4_{band}"
if cfg.stoptile == 0: cfg.stoptile = np.inf

cfg.configure_effortless(bl_circ=bl_circ_dict[band],
                         bl_inner=bl_inner_dict[band], nomask=nomask)
PyOutSlice.SAVE_ALL = True
outslice = PyOutSlice(cfg, this_sub, timing=True)
