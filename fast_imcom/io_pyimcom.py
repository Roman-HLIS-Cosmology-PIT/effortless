"""I/O interface for PyIMCOM."""

import sys; sys.path.append("..")

import numpy as np
from astropy.io import fits

from pyimcom.config import Settings as Stn, Config
from pyimcom.coadd import InImage, Block
from pyimcom.layer import get_all_data, Mask
from .psfutil import PSFModel
from .io_general import InSlice, OutSlice


class FConfig(Config):

    def configure_fast_imcom(self) -> None:
        self()  # Calculate or update derived quantities.

        assert self.inpsf_format == "L2_2506", \
            'INPSF: Fast IMCOM only supports format "L2_2506".'
        PSFModel.NPIX = 128
        PSFModel.SAMP = 6
        PSFModel.NTOT = PSFModel.NPIX * PSFModel.SAMP
        PSFModel.YXCTR = (PSFModel.NTOT-1) / 2
        PSFModel.BL_CIRC = 57

        InSlice.NLAYER = self.n_inframe
        assert self.pad_sides in ["all", "none"], \
            'PADSIDES: Fast IMCOM only supports "all" or "none".'
        OutSlice.NSUB, OutSlice.NPIX_SUB, OutSlice.CDELT =\
            self.n1P//2, self.n2*2, self.dtheta
        OutSlice.NPIX_TOT = OutSlice.NSUB * OutSlice.NPIX_SUB

        assert self.n_out == 1, "NOUT: Fast IMCOM only supports 1."
        assert self.outpsf == "GAUSSIAN", \
            'OUTPSF: Fast IMCOM only supports "GAUSSIAN".'
        OutSlice.SIGMA = self.sigmatarget
        OutSlice.SAVE_ALL = False


class PyPSFModel(PSFModel):

    def __call__(self, x: float = -np.inf, y: float = -np.inf) -> np.ndarray:
        lpoly = InImage.LPolyArr(1, (x-2043.5)/2044.0, (y-2043.5)/2044.0)
        # pixels are in C/Python convention since pixloc was set this way
        return np.einsum("a,aij->ij", lpoly, self.psfdata)
        # Not calling InImage.smooth_and_pad because of PSFModel.pixelate_psf.


class PyInSlice(InSlice):

    def __init__(self, blk: Block, idsca: tuple[int, int],
                 loaddata: bool = True, paddata: bool = False) -> None:
        self.blk, self.idsca = blk, idsca
        self.inimage = InImage(blk, idsca)
        cfg = self.inimage.blk.cfg  # Shortcut.
        with fits.open(cfg.inpsf_path + "/" + InImage.psf_filename(
            cfg.inpsf_format, idsca[0])) as f:
            psfmodel = PyPSFModel(f[idsca[1]].data)
        super().__init__(self.inimage.infile, psfmodel, loaddata, paddata)

    def load_data_and_mask(self) -> None:
        self.wcs = self.inimage.inwcs.obj
        self.scale = Stn.pixscale_native / Stn.degree

        print("input image", self.inimage.idsca)
        get_all_data(self.inimage)
        self.data = self.inimage.indata
        print()

        cfg = self.blk.cfg  # Shortcut.
        assert cfg.permanent_mask is None and cfg.cr_mask_rate == 0.0
        # Temporarily exclude `L2_2506` input masks
        self.mask = np.ones((InSlice.NSIDE, InSlice.NSIDE), dtype=bool)
        self.data[0] *= Mask.load_mask_from_maskfile(cfg, self.blk.obsdata, self.idsca)
        del self.blk, self.inimage


class PyOutSlice(OutSlice):

    def __init__(self, cfg: FConfig = None, this_sub: int = 0,
                 timing: bool = False, run_coadd: bool = True) -> None:
        self.cfg = cfg if cfg is not None else FConfig()
        self.this_sub = this_sub
        self.blk = Block(self.cfg, this_sub, run_coadd=False)
        self.blk.parse_config()
        # self.process_input_images()
        # Hardcode list of input images to save time on tests.
        self.blk.obslist = [(np.int64(1507), 7), (np.int64(1508), 7), (np.int64(1509), 7),
                            (np.int64(14748), 10), (np.int64(14749), 10), (np.int64(14753), 12)]

        print("Reading input data ... ")
        assert cfg.permanent_mask is None and cfg.cr_mask_rate == 0.0
        print("No permanent mask")
        print()

        inslices = [PyInSlice(self.blk, idsca) for idsca in self.blk.obslist]
        super().__init__(self.blk.outwcs, inslices, timing)
        del self.blk

        ibx, iby = divmod(self.this_sub, self.cfg.nblock)
        self.filename = f"{self.cfg.outstem}_{ibx:02d}_{iby:02d}.fits"
        if run_coadd: self(self.filename, timing, min(np.inf, (self.cfg.stoptile+3)//4))

    def process_input_images(self) -> None:
        search_radius = Stn.sca_sidelength / np.sqrt(2.0) / Stn.degree \
                      + self.cfg.NsideP * self.cfg.dtheta / np.sqrt(2.0)
        self.blk._get_obs_cover(search_radius)
        print(len(self.blk.obslist), "observations within range ({:7.5f} deg)".format(search_radius),
              "filter =", self.cfg.use_filter, "({:s})".format(Stn.RomanFilters[self.cfg.use_filter]))

        self.blk.inimages = [InImage(self.blk, idsca) for idsca in self.blk.obslist]
        any_exists = False
        print("The observations -->")
        print("  OBSID SCA  RAWFI    DECWFI   PA     RASCA   DECSCA       FILE (x=missing)")
        for idsca, inimage in zip(self.blk.obslist, self.blk.inimages):
            cpos = "                 "
            if inimage.exists_:
                any_exists = True
                cpos_coord = inimage.inwcs.all_pix2world([[Stn.sca_ctrpix, Stn.sca_ctrpix]], 0)[0]
                cpos = "{:8.4f} {:8.4f}".format(cpos_coord[0], cpos_coord[1])
            print("{:7d} {:2d} {:8.4f} {:8.4f} {:6.2f} {:s} {:s} {:s}".format(
                idsca[0], idsca[1], self.blk.obsdata["ra"][idsca[0]], self.blk.obsdata["dec"][idsca[0]],
                self.blk.obsdata["pa"][idsca[0]], cpos, " " if inimage.exists_ else "x", inimage.infile))
        print()
        assert any_exists, "No candidate observations found to stack. Exiting now."

        # remove nonexistent input images
        self.blk.obslist = [self.blk.obslist[i] for i, inimage
                            in enumerate(self.blk.inimages) if inimage.exists_]
        self.blk.inimages = [inimage for inimage in self.blk.inimages if inimage.exists_]
        self.blk.n_inimage = len(self.blk.inimages)
