# Script to run Effortless tests with IMCOM Paper IV input data.

import os
import sys

# Usage: python run_anlsim.py <band> <nomask> <this_sub>
# The band should be one of "Y106", "J129", "H158", and "F184".
# The nomask flag should be 0 or 1 (to be converted to a boolean).
band, nomask = sys.argv[1], bool(int(sys.argv[2]))
outstem = f"/fs/scratch/PAS2733/paper4_effortless/paper4_{band}_" +\
    ("nomask" if nomask else "effortless") + f"/paper4_{band}"

NB, p = 36, 691
for k in range(16):
    this_sub = k*p%NB**2
    ibx, iby = divmod(this_sub, NB)

    command = f"python run_anlsim.py {band} {int(nomask)} {this_sub}"
    os.system(f"{command} >> {outstem}_{ibx:02d}_{iby:02d}.out")
