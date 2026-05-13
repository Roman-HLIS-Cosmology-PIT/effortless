"""Utilities for analyzing results.

Classes
-------
MockImage : Mock image for analyzing gsstar14.

"""

import sys; sys.path.append("..")  # To import PyIMCOM.
from copy import deepcopy

import numpy as np
from astropy.io import fits

from pyimcom.analysis import OutImage, StarsAnal


class MockImage:
    """Mock image for analyzing gsstar14.

    Methods
    -------
    __init__ : Initialize the mock image.
    __call__ : Analyze the given gsstar14.

    """

    def __init__(self, outim: OutImage):
        """Initialize the mock image.

        Parameters
        ----------
        outim : OutImage
            The output image to be analyzed.

        """

        self.cfg = deepcopy(outim.cfg)
        self.cfg.extrainput = [None, "gsstar14"]
        self.cfg.outmaps = ""

        f = outim.hdu_list  # Shortcut.
        self.hdu_list = fits.HDUList([
            fits.PrimaryHDU(np.zeros((1, 2) + (self.cfg.NsideP,)*2), \
                header=f[0].header), f["INWEIGHT"], f["FIDELITY"]])

    def __call__(self, gsstar14: np.array) -> np.array:
        """Analyze the given gsstar14.

        Parameters
        ----------
        gsstar14 : np.array, shape: `(NsideP, NsideP)`
            The gsstar14 layer to be analyzed.

        Returns
        -------
        np.array, shape: `(npix, len(ColDescr)-6)`
            The analysis results.

        """

        assert gsstar14.shape == (self.cfg.NsideP,)*2
        self.hdu_list[0].data[0, 1] = gsstar14
        anal = StarsAnal(self); anal()
        sub_cat = anal.sub_cat[:, :-6].copy(); del anal
        return sub_cat
