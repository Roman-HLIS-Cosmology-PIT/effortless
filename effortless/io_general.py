"""General I/O interface.

Classes
-------
InSlice : Input image slice, like InImage in PyIMCOM.
OutSlice : Output image slice, like Block in PyIMCOM.

"""

from time import perf_counter

import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy import units as u

from .psfutil import PSFModel, SubSlice


class InSlice:
    """Input image slice, like InImage in PyIMCOM.

    Attributes
    ----------
    NSIDE : int, default: 4088
        Original input slice size in pixels.
    NLAYER : int, default: 1
        Number of input layers.

    Methods
    -------
    __init__ : Initialize the input image slice.
    load_data_and_mask : Load the input data and mask.
    load_wcs : Load the WCS for the input slice.
    pad_data_and_mask : Pad the input data and mask.

    outpix2world2inpix : Convert output to input pixel coordinates.
    inpix2world2outpix : Convert input to output pixel coordinates.
    assess_overlap : Assess the overlap between the input and output slices.
    propagate_mask : Propagate the input pixel mask to the output slice.

    get_psf : Get the PSF at given input pixel coordinates.
    get_data_and_mask : Get the data and mask for specified region.

    """

    NSIDE = 4088  # Original input slice size in pixels.
    NLAYER = 1  # Number of input layers.

    def __init__(self, filename: str, psfmodel: PSFModel = None,
                 loaddata: bool = False, paddata: bool = False) -> None:
        """Initialize the input image slice.

        Parameters
        ----------
        filename : str
            Input image file name.
        psfmodel : PSFModel, default: None
            PSF model for this input image.
        loaddata : bool, default: False
            Whether to load the input data and mask.
        paddata : bool, default: False
            Whether to pad the input data and mask.

        Attributes
        ----------
        inxy_min : np.array
            Minimum input pixel coordinates of the slice.
            shape : `(2,)`, dtype : ``int``

        """

        self.filename = filename
        self.psfmodel = psfmodel

        # Minimum input pixel coordinates.
        self.inxy_min = np.zeros(2, dtype=int)
        if loaddata:
            self.load_data_and_mask()
            if paddata: self.pad_data_and_mask()
        else:
            self.load_wcs()
            self.data = self.mask = None

    def load_wcs(self) -> None:
        """Load the WCS for the input slice.

        Attributes
        ----------
        wcs : wcs.WCS
            WCS object for the input slice.
        scale : float
            Pixel scale in degrees.

        """

        with fits.open(self.filename) as f:
            self.wcs = wcs.WCS(f["WFI01"].header)
        self.scale = np.abs(self.wcs.wcs.cd[0, 0])  # Pixel scale in degrees.

    def load_data_and_mask(self) -> None:
        """Load the input data and mask.

        Attributes
        ----------
        data : np.array
            Input data array.
            shape : `(NLAYER, NSIDE, NSIDE)`, dtype : ``float``
        mask : np.array
            Input mask array.
            shape : `(NSIDE, NSIDE)`, dtype : ``bool``

        """

        self.data = np.zeros((InSlice.NLAYER,) + (InSlice.NSIDE,)*2, dtype=np.float32)
        with fits.open(self.filename) as f:
            self.wcs = wcs.WCS(f["WFI01"].header)
            self.data[0] = f["WFI01"].data.astype(np.float32)
        self.scale = np.abs(self.wcs.wcs.cd[0, 0])  # Pixel scale in degrees.
        self.mask = np.ones((InSlice.NSIDE, InSlice.NSIDE), dtype=bool)

    def pad_data_and_mask(self) -> None:
        """Pad the input data and mask.

        Returns
        -------
        None

        """

        ACCEPT = SubSlice.ACCEPT  # Shortcut.
        self.data = np.pad(self.data, ((0,)*2,) + ((ACCEPT,)*2,)*2,
                           mode="constant", constant_values=0)
        self.mask = np.pad(self.mask, ACCEPT, mode="constant", constant_values=False)

        self.NSIDE = InSlice.NSIDE + ACCEPT*2
        self.inxy_min -= ACCEPT

    def outpix2world2inpix(self, outxys: np.array) -> np.array:
        """Convert output to input pixel coordinates.

        Parameters
        ----------
        outxys : np.array
            Output pixel coordinates.
            shape : `(N, 2)`, dtype : ``float``

        Returns
        -------
        np.array
            Input pixel coordinates.
            shape : `(N, 2)`, dtype : ``float``

        """

        return self.wcs.all_world2pix(self.outslice.wcs.\
            all_pix2world(outxys, 0), 0) - self.inxy_min

    def inpix2world2outpix(self, inxys: np.array) -> np.array:
        """Convert input to output pixel coordinates.

        Parameters
        ----------
        inxys : np.array
            Input pixel coordinates.
            shape : `(N, 2)`, dtype : ``float``

        Returns
        -------
        np.array
            Output pixel coordinates.
            shape : `(N, 2)`, dtype : ``float``

        """

        return self.outslice.wcs.all_world2pix(self.wcs.\
            all_pix2world(inxys + self.inxy_min, 0), 0)

    def assess_overlap(self, shrink: bool = True) -> None:
        """Assess the overlap between the input and output slices.

        Parameters
        ----------
        shrink : bool, default: True
            Whether to shrink the input slice to the relevant region.

        Attributes
        ----------
        mask_out : np.array
            Mask for output pixels.
            shape : `(NPIX_TOT, NPIX_TOT)`, dtype : ``bool``
        is_relevant : bool
            Whether the input slice overlaps with the output slice at all.

        """

        ACCEPT = SubSlice.ACCEPT  # Shortcuts.
        NSUB, NPIX_SUB = OutSlice.NSUB, OutSlice.NPIX_SUB
        # Sparse grid of output pixels in output and input coordinates.
        outxys_sp = np.moveaxis(np.array(np.meshgrid(
            np.arange(NSUB+1)*NPIX_SUB, np.arange(NSUB+1)*NPIX_SUB)), 0, -1).reshape(-1, 2)
        inxys_sp = self.outpix2world2inpix(outxys_sp)
        if shrink:
            inxy_min = np.array([self.NSIDE-1]*2, dtype=int)
            inxy_max = np.zeros(2, dtype=int)

        subsize = NPIX_SUB * self.outslice.wcs.wcs.cdelt[1] /\
            (0.11 * u.arcsec.to("degree"))  # Subslice size in input pixels.
        mask_sp = np.all((inxys_sp >=             -subsize) &
                         (inxys_sp <  self.NSIDE+1+subsize), axis=1).reshape(NSUB+1, NSUB+1)
        self.mask_out = np.zeros((OutSlice.NPIX_TOT,)*2, dtype=bool)  # Mask for output pixels.

        for X in range(NSUB):
            for Y in range(NSUB):
                if not np.any(mask_sp[Y:min(Y+2, NSUB+1), X:min(X+2, NSUB+1)]): continue

                # Fine grid of output pixels in output and input coordinates.
                outxys = np.moveaxis(np.array(np.meshgrid(
                    np.arange(NPIX_SUB) + X*NPIX_SUB,
                    np.arange(NPIX_SUB) + Y*NPIX_SUB)), 0, -1).reshape(-1, 2)
                inxys = self.outpix2world2inpix(outxys)

                mask = np.all((inxys >=         -1+ACCEPT) &
                              (inxys <  self.NSIDE-ACCEPT), axis=1)
                self.mask_out[Y*NPIX_SUB:(Y+1)*NPIX_SUB,
                              X*NPIX_SUB:(X+1)*NPIX_SUB] = mask.reshape(NPIX_SUB, NPIX_SUB)
                if not np.any(mask): continue

                if shrink:
                    inxy_min = np.min([inxy_min, np.floor(np.min(
                        inxys[mask], axis=0)).astype(int)], axis=0)
                    inxy_max = np.max([inxy_max, np.ceil (np.max(
                        inxys[mask], axis=0)).astype(int)], axis=0)

        self.is_relevant = np.any(self.mask_out)
        if not self.is_relevant: return
        self.load_data_and_mask()

        if shrink:
            inxy_min -= ACCEPT-1; inxy_max += ACCEPT-1
            self.inxy_min = inxy_min
            self.data = self.data[:, inxy_min[1]:inxy_max[1]+1, inxy_min[0]:inxy_max[0]+1].copy()
            self.mask = self.mask[   inxy_min[1]:inxy_max[1]+1, inxy_min[0]:inxy_max[0]+1].copy()

    def propagate_mask(self) -> None:
        """Propagate the input pixel mask to the output slice.

        Returns
        -------
        None

        """

        if not self.is_relevant: return

        REJECT = SubSlice.REJECT  # Shortcuts.
        NPIX_TOT = OutSlice.NPIX_TOT
        # Circular region to be masked around each masked input pixel.
        dys = np.arange(-REJECT, REJECT)
        dxs = (((REJECT-0.5)**2 - (dys+(dys<0))**2) ** 0.5).astype(int)

        # Masked input pixels in output coordinates.
        bad_outxys = self.inpix2world2outpix(
            np.flip(np.where(1-self.mask), axis=0).T).astype(int) + 1
        bad_outxys = bad_outxys[(np.min(bad_outxys, axis=1) >           -REJECT)]
        bad_outxys = bad_outxys[(np.max(bad_outxys, axis=1) < NPIX_TOT-1+REJECT)]

        for bad_x, bad_y in bad_outxys:
            for dy, dx in zip(dys, dxs):
                y = bad_y + dy
                if y < 0 or y >= NPIX_TOT: continue
                self.mask_out[y, max(bad_x-dx, 0):
                    max(min(bad_x+dx, NPIX_TOT), 0)] = False

    def get_psf(self, x: float = -np.inf, y: float = -np.inf) -> np.array:
        """Get the PSF at given input pixel coordinates.

        Parameters
        ----------
        x, y : float, float, default: -np.inf, -np.inf
            Input pixel coordinates.

        Returns
        -------
        np.array
            PSF array at the given coordinates.
            shape : `(NTOT, NTOT)`, dtype : ``float``

        """

        return self.psfmodel(x + self.inxy_min[0], y + self.inxy_min[1])

    def get_data_and_mask(self, x_min: int, x_max: int,
                          y_min: int, y_max: int) -> tuple[np.array, np.array]:
        """Get the data and mask for specified region.

        Parameters
        ----------
        x_min, x_max, y_min, y_max : int, int, int, int
            Minimum and maximum input pixel coordinates of the region.

        Returns
        -------
        tuple[np.array, np.array]
            Data and mask arrays for the specified region.
            data shape : `(NLAYER, y_max-y_min+1, x_max-x_min+1)`, dtype : ``float``
            mask shape : `(y_max-y_min+1, x_max-x_min+1)`, dtype : ``bool``

        """

        return (self.data[:, y_min:y_max+1, x_min:x_max+1].copy(),
                self.mask[   y_min:y_max+1, x_min:x_max+1].copy())


class OutSlice:
    """Output image slice, like Block in PyIMCOM.

    Attributes
    ----------
    NSUB : int, default: 73
        Output slice size in subslices, similar to n1//2 in PyIMCOM.
    NPIX_SUB : int, default: 56
        Subslice size in pixels, similar to n2*2 in PyIMCOM.
    NPIX_TOT : int, default: NSUB*NPIX_SUB
        Output slice size in pixels.
    CDELT : float, default: 1.5277777777777777e-05
        Output pixel scale in degrees.
    SIGMA : float, default: 1.401
        Target output PSF width in native pixels.
    SAVE_ALL : bool, default: True
        Whether to save individual regridded images.

    Class Methods
    -------------
    get_outwcs : Generate a WCS for an output slice.

    Methods
    -------
    __init__ : Initialize the output image slice.
    __call__ : Process the output image slice.
    writeto : Write the output image slice to a FITS file.
    customize_hdulist : Customize the HDU list before writing to a FITS file.

    """

    NSUB = 73  # Output slice size in subslices, similar to n1//2 in PyIMCOM.
    NPIX_SUB = 56  # Subslice size in pixels, similar to n2*2 in PyIMCOM.
    NPIX_TOT = NSUB * NPIX_SUB  # Output slice size in pixels.
    CDELT = 0.11 * u.arcsec.to("degree") / 2.0  # Output pixel scale in degrees.
    SIGMA = PSFModel.SIGMA["H158"] * 1.5  # Target output PSF width in native pixels.
    SAVE_ALL = True  # Whether to save individual regridded images.

    @classmethod
    def get_outwcs(cls, outcrval: np.array, outcrpix: list[float, float] = None,
                   outcdelt: list[float, float] = None) -> wcs.WCS:
        """Generate a WCS for an output slice.

        Parameters
        ----------
        outcrval : np.array
            World coordinates at the reference pixel.
            shape : `(2,)`, dtype : ``float``
        outcrpix : list[float, float], default: None
            Pixel coordinates of the reference pixel.
            If None, set to the center of the output slice.
        outcdelt : list[float, float], default: None
            Pixel scale in degrees. If None, set to [CDELT, -CDELT].

        Returns
        -------
        wcs.WCS
            WCS object for the output slice.

        """

        outwcs = wcs.WCS(naxis=2)
        outwcs.wcs.ctype = ["RA---STG", "DEC--STG"]
        outwcs.wcs.crval = outcrval
        outwcs.wcs.crpix = outcrpix if outcrpix is not None else [cls.NPIX_TOT/2]*2
        outwcs.wcs.cdelt = outcdelt if outcdelt is not None else [-cls.CDELT, cls.CDELT]
        return outwcs

    def __init__(self, wcs: wcs.WCS, inslices: list[InSlice], timing: bool = False) -> None:
        """Initialize the output image slice.

        Parameters
        ----------
        wcs : wcs.WCS
            WCS object for the output slice.
        inslices : list[InSlice]
            List of candidate input image slices for the output slice.
        timing : bool, default: False
            Whether to print timing information during initialization.

        Attributes
        ----------
        scale : float
            Pixel scale in degrees.
        ninslice : int
            Number of relevant input slices.
        data : np.array
            Output data array.
            shape : `(NLAYER, NPIX_TOT, NPIX_TOT)`, dtype : ``float`` or
                    `(NLAYER, ninslice, NPIX_TOT, NPIX_TOT)`, dtype : ``float``

        """

        self.wcs = wcs
        self.scale = np.abs(wcs.wcs.cdelt[0])  # Pixel scale in degrees.
        self.inslices = inslices

        if timing: tstart = perf_counter()
        for inslice in self.inslices:
            if timing: print(f"Assessing inslice {inslice.filename}",
                             f"@ t = {perf_counter() - tstart:.6f} s")
            inslice.outslice = self
            inslice.assess_overlap()
            inslice.propagate_mask()
        if timing: print("Finished assessing inslices",
                         f"@ t = {perf_counter() - tstart:.6f} s", end="\n\n")

        self.inslices = [inslice for inslice in self.inslices if inslice.is_relevant]
        self.ninslice = len(self.inslices)
        self.data = np.zeros((InSlice.NLAYER,) + ((self.ninslice,) if self.SAVE_ALL else ()) +
                             (self.NPIX_TOT,)*2, dtype=np.float32)

    def __call__(self, filename: str = None, timing: bool = False, stop: int = np.inf) -> None:
        """Process the output image slice.

        Parameters
        ----------
        filename : str, default: None
            Output file name. If None, do not write to a file.
        timing : bool, default: False
            Whether to print timing information during processing.
        stop : int, default: np.inf
            Maximum number of subslices to process. Useful for testing.

        Attributes
        ----------
        mask : np.array
            Combination of masks from input slices.
            shape : `(ninslice, NPIX_TOT, NPIX_TOT)`, dtype : ``bool``

        """

        if timing: tstart = perf_counter()
        for X in range(self.NSUB):
            if stop > 0 and timing:
                print(f"Processing subslices ({X}, *)",
                      f"@ t = {perf_counter() - tstart:.6f} s")
            for Y in range(self.NSUB):
                if stop > 0:
                    SubSlice(self, X, Y)(sigma=self.SIGMA)
                stop -= 1
        if timing: print("Finished processing subslices",
                         f"@ t = {perf_counter() - tstart:.6f} s", end="\n\n")

        self.mask = np.stack([inslice.mask_out for inslice in self.inslices])
        if not self.SAVE_ALL: self.data /= self.mask.sum(axis=0)
        if InSlice.NLAYER == 1: self.data = self.data[0]
        if filename is not None: self.writeto(filename)

    def writeto(self, filename: str, overwrite: bool = True) -> None:
        """Write the output image slice to a FITS file.

        Parameters
        ----------
        filename : str
            Output file name.
        overwrite : bool, default: True
            Whether to overwrite the file if it already exists.

        Returns
        -------
        None

        """

        my_header = self.wcs.to_header()
        datahdu = fits.PrimaryHDU(self.data, header=my_header)
        inputhdu = fits.TableHDU.from_columns([fits.Column(name="filename", \
            array=[inslice.filename for inslice in self.inslices], format="A512", ascii=True)])
        inputhdu.name = "INPUT"
        maskhdu = fits.ImageHDU(self.mask.astype(np.uint8), header=my_header, name="MASK")

        hdulist = [datahdu, inputhdu, maskhdu]
        self.customize_hdulist(hdulist)
        fits.HDUList(hdulist).writeto(filename, overwrite=overwrite)

    def customize_hdulist(self, hdulist: list[fits.hdu]) -> None:
        """Customize the HDU list before writing to a FITS file.

        This is a hook for derived classes. The base class does nothing.

        Parameters
        ----------
        hdulist : list[fits.hdu]
            List of HDU objects to be written to the FITS file.

        Returns
        -------
        None

        """

        pass
