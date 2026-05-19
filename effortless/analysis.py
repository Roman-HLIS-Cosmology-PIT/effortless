"""Utilities for analyzing results.

Classes
-------
MockImage : Mock image for analyzing gsstar14.
StarsCal : Utilities for analyzing gsstar14.

"""

import os
import sys; sys.path.append("..")  # To import PyIMCOM v1.0.3.
from copy import deepcopy
from time import perf_counter

import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy import units as u
from sklearn.linear_model import TheilSenRegressor

from pyimcom.config import Settings as Stn, Config
from pyimcom.analysis import OutImage, ColDescr, StarsAnal


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
        sub_cat = anal.sub_cat.copy(); del anal
        return sub_cat


class StarsCal:
    """Utilities for analyzing gsstar14.

    Attributes
    ----------
    DISTTHR : float, default: 8.0
        Distance threshold for selecting stars without bad pixels nearby.
    KMAX : int, default: 2
        Maximum order of subpixel features for post-measurement calibration.
    QTY_LIST : list[str], default: ["AMPLITUDE", "OFFSET_X", "OFFSET_Y", "WIDTH",
                                    "SHAPE_G1", "SHAPE_G2", "M42_REAL", "M42_IMAG"]
        List of stellar quantities to be measured and calibrated.

    Static Methods
    --------------
    get_outstem : User-provided function to get the output filename stem.
    get_X : Get subpixel features for post-measurement calibration.

    Methods
    -------
    __init__ : Initialize the analysis for a given output image.
    parse_config : Parse the PyIMCOM configuration for the analysis.
    read_outim : Read the PyIMCOM output image and prepare for analysis.

    __call__ : Run the analysis and calibration.
    build_starcat : Build the star catalog for the given output image.
    get_inwcs : Read the FITS file to get the WCS of an input image.
    update_subcat : Update the star catalog for each input image.

    calibr_starcat : Calibrate the star catalog using subpixel features.
    save_results : Save the analysis and calibration results to a .npz file.
    load_results : Load the analysis and calibration results from a .npz file.

    """

    DISTTHR = 8.0  # Distance threshold in output pixels.
    KMAX = 2  # Maximum order for post-measurement calibration.
    QTY_LIST = ["AMPLITUDE", "OFFSET_X", "OFFSET_Y", "WIDTH",
                "SHAPE_G1", "SHAPE_G2", "M42_REAL", "M42_IMAG"]

    @staticmethod
    def get_outstem(band: str, nomask: bool) -> str:
        """User-provided function to get the output filename stem.

        Parameters
        ----------
        band : str
            Filter band of output images.
        nomask : bool
            Whether to use no-mask version.

        Returns
        -------
        str
            The output filename stem.

        """

        return band + "_" + ("nomask" if nomask else "effortless")

    def __init__(self, cfg: Config, ibx: int, iby: int,
                 overwrite: bool = False, i_gsstar14: int = 1) -> None:
        """Initialize the analysis for a given output image.

        Parameters
        ----------
        cfg : Config
            PyIMCOM configuration for the analysis.
        ibx, iby : int, int
            Indices of the output image.
        overwrite : bool, default: False
            Whether to overwrite existing results.
        i_gsstar14 : int, default: 1
            Index of the gsstar14 layer in Effortless outputs.
            This can be different from the index in PyIMCOM outputs.

        """

        self.cfg = cfg
        self.ibx, self.iby = ibx, iby

        self.parse_config()
        if not overwrite: self.load_results()
        self.read_outim()

        self(overwrite, i_gsstar14)

    def parse_config(self) -> None:
        """Parse the PyIMCOM configuration for the analysis.

        Attributes
        ----------
        band : str
            Filter band of output images.
        scale_ratio : float
            Ratio of between input and output pixel scales.
        std_values : dict[str, float]
            Dictionary of standard values for amplitude and width.

        """

        self.band = Stn.RomanFilters[self.cfg.use_filter]
        self.scale_ratio = Stn.pixscale_native / (self.cfg.dtheta * u.degree.to(u.rad))
        self.std_values = {"AMPLITUDE": self.scale_ratio**2,
                           "WIDTH": self.scale_ratio * self.cfg.sigmatarget}

    def read_outim(self) -> None:
        """Read the PyIMCOM output image and prepare for analysis.

        Attributes
        ----------
        outwcs : wcs.WCS
            WCS of the output image.
        mockim : MockImage
            Mock image based on PyIMCOM output.
        cat_py : np.array, shape: `(npix, len(ColDescr))`
            Star catalog from analyzing PyIMCOM output.
        nstar : int
            Number of stars in the output image.

        """

        outim = OutImage(self.cfg.outstem +\
            f"_{self.ibx:02d}_{self.iby:02d}.fits", cfg=self.cfg)
        outim._load_or_save_hdu_list()
        self.outwcs = wcs.WCS(outim.hdu_list[0].header, naxis=2)

        self.mockim = MockImage(outim)
        if not hasattr(self, "cat_py"):
            self.cat_py = self.mockim(outim.hdu_list[0].data[0, \
                self.cfg.extrainput.index("gsstar14")])
        self.nstar = self.cat_py.shape[0]

    def __call__(self, overwrite: bool = False, i_gsstar14: int = 1) -> None:
        """Run the analysis and calibration.

        Parameters
        ----------
        overwrite : bool, default: False
            Whether to overwrite existing results.
        i_gsstar14 : int, default: 1
            Index of the gsstar14 layer in Effortless outputs.
            This can be different from the index in PyIMCOM outputs.

        """

        if (not overwrite) and all(hasattr(self, attr) \
            for attr in ["cat_nm", "cat_el", "cal_nm", "cal_el"]):
            return

        for nomask in [True, False]:
            self.build_starcat(nomask, i_gsstar14)
            print()
        self.calibr_starcat()
        self.save_results()

    def build_starcat(self, nomask: bool, i_gsstar14: int = 1) -> None:
        """Build the star catalog for the given output image.

        Parameters
        ----------
        nomask : bool
            Whether to build the star catalog for no-mask version.
        i_gsstar14 : int, default: 1
            Index of the gsstar14 layer in Effortless outputs.
            This can be different from the index in PyIMCOM outputs.

        Attributes
        ----------
        cat_nm or cat_el : np.array, shape: `(ninslice, nstar, 12)`
            Star catalog for a version of the Effortless output image.

        """

        with fits.open(self.get_outstem(self.band, nomask) +\
            f"_{self.ibx:02d}_{self.iby:02d}.fits") as f:

            # PyIMCOM may miss input images with very small overlaps.
            if not hasattr(self, "obslist"):
                self.obslist = [(rec["obsid"], rec["sca"]) for rec in
                                f["INDATA"].data if rec["valid"]]
            if not hasattr(self, "ninslice"):
                self.ninslice = len(self.obslist)

            # 4 for `update_subcat`, 8 for `QTY_LIST`.
            starcat = np.zeros((self.ninslice, self.nstar, 12))
            for i, idsca in enumerate(self.obslist):
                starcat[i] = self.mockim(f[0].data[i_gsstar14, i])[:, :12]
                self.update_subcat(starcat[i], f["MASK"].data[i], self.get_inwcs(idsca))

        if nomask: self.cat_nm = starcat
        else: self.cat_el = starcat

    def get_inwcs(self, idsca: tuple[int, int]) -> wcs.WCS:
        """Read the FITS file to get the WCS of an input image.

        Parameters
        ----------
        idsca : tuple[int, int]
            Observation ID and SCA number.

        Returns
        -------
        wcs.WCS
            The WCS of the input image.

        """

        with fits.open(self.cfg.inpath + "/simple/Roman_WAS_simple_model" +\
            f"_{self.band}_{idsca[0]}_{idsca[1]}.fits") as f:
            return wcs.WCS(f[0].header)

    def update_subcat(self, subcat: np.array, mask: np.array, inwcs: wcs.WCS) -> np.array:
        """Update the star catalog for each input image.

        Parameters
        ----------
        subcat : np.array, shape: `(nstar, 12)`
            Star catalog for the input image to be updated.
        mask : np.array, shape: `(NsideP, NsideP)`
            The output pixel mask for the input image.
        inwcs : wcs.WCS
            The WCS of the input image.

        """

        bd = StarsAnal.bd  # Shortcut.
        sigma = self.cfg.sigmatarget * self.scale_ratio  # In output pixels.
        ra, dec, x_pos, y_pos = subcat[:, :4].copy().T

        for i, (x, y) in enumerate(zip(x_pos, y_pos)):
            xi = np.rint(x).astype(np.int16)
            yi = np.rint(y).astype(np.int16)
            s_ = np.s_[yi+1-bd:yi+bd, xi+1-bd:xi+bd]

            y_bad, x_bad = np.where(1-mask[s_])
            dist_bad = np.hypot(y_bad-(bd-1)-(y-yi), x_bad-(bd-1)-(x-xi))
            if dist_bad.size == 0:
                subcat[i, 0] = np.inf  # Minimum distance to bad pixels.
                subcat[i, 1] = 0.0  # Sum of Gaussian weights of bad pixels.
            else:
                subcat[i, 0] = dist_bad.min()
                subcat[i, 1] = np.sum(np.exp(-0.5 * np.square(
                    dist_bad / sigma)) / (2.0*np.pi * sigma**2))

        # Switch to positions in the input image.
        subcat[:, 2:4] = inwcs.all_world2pix(np.array([ra, dec]).T, 0)

    @staticmethod
    def get_X(pix_frac: np.array, kmax: int) -> np.array:
        """Get subpixel features for post-measurement calibration.

        Parameters
        ----------
        pix_frac : np.array, shape: `(nstar, 2)`
            Fractional part of star positions in the input image.
        kmax : int
            Maximum order of subpixel features to be returned.

        Returns
        -------
        np.array, shape: `(nstar, 4*kmax)`
            Subpixel features for post-measurement calibration.

        """

        return np.column_stack([trig(2.0*np.pi * k*coord) for k in range(1, kmax+1)
                                for coord in pix_frac.T for trig in (np.cos, np.sin)])

    def calibr_starcat(self, niter: int = 4, timing: bool = True) -> None:
        """Calibrate the star catalog using subpixel features.

        Parameters
        ----------
        niter : int, default: 4
            Number of iterations for astrometry calibration.
        timing : bool, default: True
            Whether to print timing information during calibration.

        Attributes
        ----------
        cal_nm and cal_el : np.array, shape: `(ninslice, KMAX, nstar, 8)`
            Calibrated star catalog for both versions of the Effortless output image.

        """

        xy_true = self.cat_py[:, 2:4]  # X_POS and Y_POS.
        shape = (self.ninslice, StarsCal.KMAX, self.nstar, 8)
        self.cal_nm = np.zeros(shape)
        self.cal_el = np.zeros(shape)

        if timing: tstart = perf_counter()
        for i, idsca in enumerate(self.obslist):
            if timing: print(f"Calibrating inslice {idsca}",
                             f"@ t = {perf_counter() - tstart:.6f} s")
            inwcs = self.get_inwcs(idsca)
            # Select stars without bad pixels nearby for calibration.
            mask_nm = self.cat_nm[i, :, 0] >= StarsCal.DISTTHR
            if not np.any(mask_nm): continue
            pix_frac, pix_int = np.modf(self.cat_nm[i, mask_nm, 2:4])

            for k in range(1, StarsCal.KMAX+1):
                # Train linear models based on no-mask version of the star catalog.
                X_nm = StarsCal.get_X(pix_frac, k)
                models = {}
                for qty in StarsCal.QTY_LIST:
                    y_nm = self.cat_nm[i, mask_nm, ColDescr[qty].value]
                    models[qty] = TheilSenRegressor(random_state=0).fit(X_nm, y_nm)

                for nomask in [True, False]:
                    if nomask: subcat, subcal = self.cat_nm[i], self.cal_nm[i, k-1]
                    else: subcat, subcal = self.cat_el[i], self.cal_el[i, k-1]

                    xy_meas = xy_true + subcat[:, 5:7]  # OFFSET_X and OFFSET_Y.
                    # Iteratively calibrate star positions.
                    xy_meas_ = xy_meas.copy()
                    for _ in range(niter+1):
                        pix_frac_, pix_int_ = np.modf(inwcs.all_world2pix(
                            self.outwcs.all_pix2world(xy_meas_, 0), 0))
                        X_ = StarsCal.get_X(pix_frac_, k)
                        if _ < niter:
                            for j, qty in enumerate(["OFFSET_X", "OFFSET_Y"]):
                                xy_meas_[:, j] = xy_meas[:, j] - models[qty].predict(X_)

                    # Use measured positions to calibrate all quantities.
                    for j, qty in enumerate(StarsCal.QTY_LIST):
                        subcal[:, j] = subcat[:, ColDescr[qty].value] -\
                            models[qty].predict(X_) + self.std_values.get(qty, 0.0)
        if timing: print("Finished calibrating inslices",
                         f"@ t = {perf_counter() - tstart:.6f} s", end="\n\n")

    def save_results(self) -> None:
        """Save the analysis and calibration results to a .npz file.

        """

        npzfile = self.get_outstem(self.band, False) +\
            f"_{self.ibx:02d}_{self.iby:02d}.npz"
        with open(npzfile, "wb") as f:
            np.savez(f, cat_py=self.cat_py,
                     cat_nm=self.cat_nm, cat_el=self.cat_el,
                     cal_nm=self.cal_nm, cal_el=self.cal_el)

    def load_results(self) -> None:
        """Load the analysis and calibration results from a .npz file.

        Attributes
        ----------
        cat_py : np.array, shape: `(npix, len(ColDescr))`
            Star catalog from analyzing PyIMCOM output.
        cat_nm and cat_el : np.array, shape: `(ninslice, nstar, 12)`
            Star catalogs for both versions of the Effortless output image.
        cal_nm and cal_el : np.array, shape: `(ninslice, KMAX, nstar, 8)`
            Calibrated star catalogs for both versions of the Effortless output image.

        """

        npzfile = self.get_outstem(self.band, False) +\
            f"_{self.ibx:02d}_{self.iby:02d}.npz"
        if not os.path.exists(npzfile): return

        with open(npzfile, "rb") as f:
            data = np.load(f)
            self.cat_py = data["cat_py"]
            self.cat_nm, self.cat_el = data["cat_nm"], data["cat_el"]
            self.cal_nm, self.cal_el = data["cal_nm"], data["cal_el"]
