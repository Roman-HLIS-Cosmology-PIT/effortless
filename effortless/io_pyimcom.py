"""PyIMCOM I/O interface.

Classes
-------
EConfig : Configuration class for Effortless.
PyPSFModel : PSF model class for PyIMCOM.
PyInSlice : Input slice class for PyIMCOM.
PyOutSlice : Output slice class for PyIMCOM.

"""

import sys; sys.path.append("..")  # To import PyIMCOM.

import numpy as np
from astropy.io import fits

from pyimcom.config import Settings as Stn, Config
from pyimcom.coadd import InImage, Block
from pyimcom.layer import get_all_data, Mask
from .psfutil import PSFModel
from .io_general import InSlice, OutSlice


class EConfig(Config):
    """Configuration class for Effortless.

    Attributes
    ----------
    INPSF_NPIX : dict, default: {"L2_2506": 128, "anlsim": 32}
        Mapping from input PSF format to input PSF size in native pixels.

    Methods
    -------
    configure_effortless : Configure Effortless using PyIMCOM settings.

    """

    INPSF_NPIX = {"L2_2506": 128, "anlsim": 32}

    def configure_effortless(self, bl_circ: int = 57) -> None:
        """Configure Effortless using PyIMCOM settings.

        Parameters
        ----------
        bl_circ : int, default: 57
            Circular bandlimit in Fourier space.

        Returns
        -------
        None

        """

        self()  # Calculate or update derived quantities.

        PSFModel.NPIX = EConfig.INPSF_NPIX.get(self.inpsf_format)
        assert PSFModel.NPIX is not None, \
            f"INPSF: Effortless only supports formats in {list(EConfig.INPSF_NPIX)}."
        PSFModel.SAMP = self.inpsf_oversamp
        PSFModel.NTOT = PSFModel.NPIX * PSFModel.SAMP
        PSFModel.YXCTR = (PSFModel.NTOT-1) / 2
        PSFModel.BL_CIRC = bl_circ

        InSlice.NLAYER = self.n_inframe
        OutSlice.NSUB, OutSlice.NPIX_SUB, OutSlice.CDELT =\
            self.n1P//2, self.n2*2, self.dtheta
        assert self.pad_sides in ["all", "none"], \
            'PADSIDES: Effortless only supports "all" and "none".'
        if self.pad_sides == "none": OutSlice.NSUB = self.n1//2
        OutSlice.NPIX_TOT = OutSlice.NSUB * OutSlice.NPIX_SUB

        assert self.n_out == 1, "NOUT: Effortless only supports 1."
        assert self.outpsf == "GAUSSIAN", \
            'OUTPSF: Effortless only supports "GAUSSIAN".'
        OutSlice.SIGMA = self.sigmatarget
        OutSlice.SAVE_ALL = False


class PyPSFModel(PSFModel):
    """PSF model class for PyIMCOM.
    
    Methods
    -------
    __call__ : Return the (input) PSF at given coordinates.

    """

    def __call__(self, x: float = -np.inf, y: float = -np.inf) -> np.array:
        """Return the (input) PSF at given coordinates.

        Parameters
        ----------
        x, y : float, default: -np.inf
            Coordinates in the input pixel plane.

        Returns
        -------
        np.array
            PSF array at the given coordinates.
            shape : `(NTOT, NTOT)`, dtype : ``float``

        """

        lpoly = InImage.LPolyArr(1, (x-2043.5)/2044.0, (y-2043.5)/2044.0)
        # pixels are in C/Python convention since pixloc was set this way
        return np.einsum("a,aij->ij", lpoly, self.psfdata)
        # Not calling InImage.smooth_and_pad because of PSFModel.pixelate_psf.


class PyInSlice(InSlice):
    """Input slice class for PyIMCOM.
    
    Methods
    -------
    __init__ : Initialize the input image slice.
    load_data_and_mask : Load the input data and mask.

    """

    def __init__(self, blk: Block, idsca: tuple[int, int],
                 loaddata: bool = True, paddata: bool = False) -> None:
        """Initialize the input image slice.

        Parameters
        ----------
        blk : Block
            Block object from PyIMCOM.
        idsca : tuple[int, int]
            Observation ID and SCA number.
        loaddata : bool, default: True
            Whether to load the input data.
        paddata : bool, default: False
            Whether to pad the input data.

        Attributes
        ----------
        inimage : InImage
            InImage object from PyIMCOM.

        """

        self.blk, self.idsca = blk, idsca
        self.inimage = InImage(blk, idsca)
        cfg = self.inimage.blk.cfg  # Shortcut.
        with fits.open(cfg.inpsf_path + "/" + InImage.psf_filename(
            cfg.inpsf_format, idsca[0])) as f:
            psfmodel = PyPSFModel(f[idsca[1]].data)
        super().__init__(self.inimage.infile, psfmodel, loaddata, paddata)

    def load_data_and_mask(self) -> None:
        """Load the input data and mask.

        Attributes
        ----------
        wcs : wcs.WCS
            WCS object for the input slice.
        scale : float
            Pixel scale in degrees.
        data : np.array
            Input data array.
            shape : `(NLAYER, NSIDE, NSIDE)`, dtype : ``float``
        mask : np.array
            Input mask array.
            shape : `(NSIDE, NSIDE)`, dtype : ``bool``

        """

        print("input image", self.inimage.idsca)
        self.wcs = self.inimage.inwcs.obj
        self.scale = Stn.pixscale_native / Stn.degree

        # Load masks here.
        if self.blk.pmask is not None:
            self.mask = self.blk.pmask[self.idsca[1]-1]
        else:
            self.mask = np.ones((Stn.sca_nside, Stn.sca_nside), dtype=bool)

        get_all_data(self.inimage)  # shape : (n_inframe, Stn.sca_nside, Stn.sca_nside)
        self.data = self.inimage.indata

        cr_mask = Mask.load_cr_mask(self.inimage)
        if cr_mask is not None:
            self.mask = np.logical_and(self.mask, cr_mask)
        del cr_mask, self.inimage

        # Extract mask from file.
        self.mask &= Mask.load_mask_from_maskfile(self.blk.cfg, self.blk.obsdata, self.idsca)
        # Temporarily exclude `L2_2506` input masks.
        # self.data[0] *= Mask.load_mask_from_maskfile(self.blk.cfg, self.blk.obsdata, self.idsca)
        del self.blk


class PyOutSlice(OutSlice):
    """Output slice class for PyIMCOM.

    Methods
    -------
    __init__ : Initialize the output image slice.
    process_input_images : Process input images from PyIMCOM.

    """

    def __init__(self, cfg: EConfig = None, this_sub: int = 0,
                 timing: bool = False, run_coadd: bool = True) -> None:
        """Initialize the output image slice.

        Parameters
        ----------
        cfg : EConfig, default: None
            Configuration object for Effortless.
            If None, a default EConfig will be created.
        this_sub : int, default: 0
            Block index within the Mosaic.
        timing : bool, default: False
            Whether to print timing information.
        run_coadd : bool, default: True
            Whether to run coaddition after initialization.

        Attributes
        ----------
        blk : Block
            Block object from PyIMCOM.

        """

        self.cfg = cfg if cfg is not None else EConfig()
        self.this_sub = this_sub
        self.blk = Block(self.cfg, this_sub, run_coadd=False)
        self.blk.parse_config()
        self.process_input_images()

        inslices = [PyInSlice(self.blk, idsca) for idsca in self.blk.obslist]
        if self.cfg.pad_sides == "none": 
            self.blk.outwcs.wcs.crpix -= self.cfg.postage_pad * self.cfg.n2
        super().__init__(self.blk.outwcs, inslices, timing)
        del self.blk

        ibx, iby = divmod(self.this_sub, self.cfg.nblock)
        self.filename = f"{self.cfg.outstem}_{ibx:02d}_{iby:02d}.fits"
        if run_coadd: self(self.filename, timing, min(np.inf, (self.cfg.stoptile+3)//4))

    def process_input_images(self) -> None:
        """Process input images (from PyIMCOM).

        Returns
        -------
        None

        """

        bypass = True  # Use hardcoded list of input images to save time in tests.
        if bypass:
            self.blk.obslist = [(np.int64(1507), 7), (np.int64(1508), 7), (np.int64(1509), 7),
                                (np.int64(14748), 10), (np.int64(14749), 10), (np.int64(14753), 12)]
            self.blk.pmask = Mask.load_permanent_mask(self.blk)
            return

        # Now figure out which observations we need.
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

        print("Reading input data ... ")
        self.blk.pmask = Mask.load_permanent_mask(self.blk)
        print()

        # Remove nonexistent input images.
        self.blk.obslist = [self.blk.obslist[i] for i, inimage
                            in enumerate(self.blk.inimages) if inimage.exists_]
        self.blk.inimages = [inimage for inimage in self.blk.inimages if inimage.exists_]
        self.blk.n_inimage = len(self.blk.inimages)

    def customize_hdulist(self, hdulist: list[fits.hdu]) -> None:
        """Add config and inlist to the HDU list.

        Parameters
        ----------
        hdulist : list[fits.hdu]
            List of HDU objects to be written to the FITS file.

        Returns
        -------
        None

        """

        config_hdu = fits.TableHDU.from_columns(
            [fits.Column(name="text", array=self.cfg.to_file(None).splitlines(), format="A512", ascii=True)])
        config_hdu.header["EXTNAME"] = "CONFIG"

        inlist_hdu = fits.BinTableHDU.from_columns([
            fits.Column(name="obsid", array=np.array([obs[0] for obs in self.obslist]), format="J"),
            fits.Column(name="sca",   array=np.array([obs[1] for obs in self.obslist]), format="I"),
            fits.Column(name="ra",    array=np.array([self.obsdata["ra" ][obs[0]] for obs in self.obslist]), format="D", unit="degree"),
            fits.Column(name="dec",   array=np.array([self.obsdata["dec"][obs[0]] for obs in self.obslist]), format="D", unit="degree"),
            fits.Column(name="pa",    array=np.array([self.obsdata["pa" ][obs[0]] for obs in self.obslist]), format="D", unit="degree"),
            fits.Column(name="valid", array=np.array([inimage.exists_ for inimage in self.inimages]), format="L")
        ])
        inlist_hdu.header["EXTNAME"] = "INDATA"

        hdulist[1:2] = [config_hdu, inlist_hdu]
