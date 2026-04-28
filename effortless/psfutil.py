"""Utilities for PSF manipulation.

Classes
-------
PSFModel : Base class for PSF models.
SubSlice : Subslice of the output slice.

"""

import numpy as np
from astropy.wcs.utils import local_partial_pixel_derivatives

from .routine import bandlimited_rfft2, bandlimited_irfft2
from .routine import apply_mask_threshold, compute_weights
from .routine import adjust_weights, apply_weights


class PSFModel:
    """Base class for PSF models.

    Attributes
    ----------
    NPIX : int, default: 48
        PSF array size in native pixels.
    SAMP : int, default: 4
        Oversampling rate of PSF arrays.
    NTOT : int, default: 192
        PSF array size in oversampled pixels.
    YXCTR : float, default: 96.0
        PSF array center in oversampled pixels.
    BL_CIRC : int, default: 33
        Circular bandlimit in Fourier space.

    SIGMA_TO_FWHM : float, default: 2.3548200460338346
        Conversion factor from sigma to FWHM for Gaussian PSFs.
    SIGMA : dict, default: {"Y106": 0.850, "J129": 0.894, "H158": 0.939,
                           "F184": 0.983, "K213": 1.028}
        Dictionary of default sigma values for different filters.

    Class Methods
    -------------
    psf_gaussian : Generate a 2D Gaussian PSF.
    pixelate_psf : Pixelate a 2D (input) PSF.
    get_weight_field : Compute the weight field based on given PSFs.

    Methods
    -------
    __init__ : Initialize the (input) PSF model with given PSF data.
    __call__ : Return the (input) PSF at given coordinates.

    """

    NPIX = 48  # PSF array size in native pixels.
    SAMP = 4  # Oversampling rate of PSF arrays.
    NTOT = NPIX * SAMP  # PSF array size in oversampled pixels.
    YXCTR = NTOT / 2  # PSF array center in oversampled pixels.
    BL_CIRC = 33  # Circular bandlimit in Fourier space.

    SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))  # For Gaussian PSFs.
    SIGMA = {
        "Y106": 2.0 / SIGMA_TO_FWHM,
        "J129": 2.1 / SIGMA_TO_FWHM,
        "H158": 2.2 / SIGMA_TO_FWHM,
        "F184": 2.3 / SIGMA_TO_FWHM,
        "K213": 2.4 / SIGMA_TO_FWHM,
    }  # From pyimcom/configs/production_configs_spring2024.

    @classmethod
    def psf_gaussian(cls, sigma: float, dout_din: np.array = np.diag(np.ones(2))) -> np.array:
        """Generate a 2D Gaussian PSF.

        Parameters
        ----------
        sigma : float
            Sigma of the Gaussian PSF in native pixels.
        dout_din : np.array, shape: `(2, 2)`, dtype: ``float``,
                   default: np.diag(np.ones(2))
            Distortion matrix for converting input to output pixel coordinates.

        Returns
        -------
        np.array, shape: `(NTOT, NTOT)`, dtype: ``float``
            2D Gaussian PSF array, normalized in the output pixel plane.

        """

        xy = np.flip(np.mgrid[-cls.YXCTR:cls.NTOT-1-cls.YXCTR:cls.NTOT*1j,
                              -cls.YXCTR:cls.NTOT-1-cls.YXCTR:cls.NTOT*1j], axis=0)
        invSigma = dout_din.T @ np.diag(np.ones(2) / (sigma*cls.SAMP)**2) @ dout_din
        return np.exp(-0.5 * np.einsum("lyx,lr,ryx->yx", xy, invSigma, xy))\
                / (2.0*np.pi * (sigma*cls.SAMP)**2)  # Normalized in the output pixel plane.

    @classmethod
    def pixelate_psf(cls, psf: np.array) -> np.array:
        """Pixelate a 2D (input) PSF.

        Parameters
        ----------
        psf : np.array, shape: `(NTOT, NTOT)`, dtype: ``float``
            2D PSF array in the input pixel plane.

        Returns
        -------
        np.array, shape: `(NTOT, NTOT)`, dtype: ``float``
            Pixelated PSF array in the input pixel plane.

        """

        k = np.linspace(0, 1, cls.NTOT, endpoint=False)
        k[-(cls.NTOT//2):] -= 1; k *= cls.SAMP
        return np.fft.irfft2(np.fft.rfft2(psf) *\
            np.sinc(k[None, :cls.NTOT//2+1]) * np.sinc(k[:, None]), s=(cls.NTOT, cls.NTOT))

    @classmethod
    def get_weight_field(cls, psf_in: np.array, psf_out: np.array, wd: int = 0) -> np.array:
        """Compute the weight field based on given PSFs.

        Parameters
        ----------
        psf_in : np.array, shape: `(NTOT, NTOT)`, dtype: ``float``
            Input PSF in the input pixel plane.
        psf_out : np.array, shape: `(NTOT, NTOT)`, dtype: ``float``
            Target output PSF in the input pixel plane.
        wd : int, default: 0
            Half window size in real space. If 0, no window is applied.

        Returns
        -------
        np.array, shape: `(NTOT, NTOT)` or `(wd*2, wd*2)`, dtype: ``float``
            Weight field in the input pixel plane.

        """

        bl = (cls.BL_CIRC+0.5) * 2.0**0.5; bl_int = int(bl)
        psf_inp = cls.pixelate_psf(psf_in)
        psf_inp_tbl = bandlimited_rfft2(psf_inp[None], bl_int)[0]
        psf_out_tbl = bandlimited_rfft2(psf_out[None], bl_int)[0]

        weight_tbl = psf_out_tbl / psf_inp_tbl
        # Apply circular bandlimit.
        for du in range(bl_int+1):
            dv = int((bl**2 - du**2)**0.5)
            if dv == bl_int: continue
            weight_tbl[dv+1:bl_int*2-dv, du] = 0

        return np.fft.ifftshift(bandlimited_irfft2(
            weight_tbl[None], cls.NTOT, cls.NTOT, wd))[0] * cls.SAMP**2

    def __init__(self, psfdata: np.array) -> None:
        """Initialize the (input) PSF model with given PSF data.

        Parameters
        ----------
        psfdata : np.array, shape: `(NTOT, NTOT)` or as needed, dtype: ``float``
            PSF data array.

        Returns
        -------
        None

        """

        self.psfdata = psfdata

    def __call__(self, x: float = -np.inf, y: float = -np.inf) -> np.array:
        """Return the (input) PSF at given coordinates.

        Parameters
        ----------
        x, y : float, default: -np.inf
            Coordinates in the input pixel plane. Ignored in the base class.

        Returns
        -------
        np.array, shape: `(NTOT, NTOT)`, dtype: ``float``
            PSF array at the given coordinates.
            In the base class, this is just the input PSF data.

        """

        assert self.psfdata.ndim == 2, "PSFModel: The base class only supports 2D data."
        return self.psfdata


class SubSlice:
    """Subslice of the output slice.

    Attributes
    ----------
    ACCEPT : int, default: 8
        Acceptance radius in native pixels for selecting input pixels.
    REJECT : int, default: 8
        Rejection radius in output pixels for masking output pixels.
    MASK_THR : int, default: 32
        Threshold for number of masked input pixels to mask an output pixel.
    NDIFF : int, default: 5
        Number of iterations for weight diffusion.
    RENORM : bool, default: False
        Whether to renormalize weights after adjustments.

    Static Methods
    --------------
    get_dworld_dpixel : Compute the Jacobian matrix of world coordinates
                        with respect to pixel coordinates at given coordinates.

    Methods
    -------
    __init__ : Initialize the subslice with given output slice and coordinates.
    __call__ : Process the subslice with given target output PSF width.

    """

    ACCEPT = 8  # Acceptance radius in native pixels.
    REJECT = 8  # Rejection radius in output pixels.
    MASK_THR = 32  # Threshold for number of masked input pixels.
    NDIFF = 5  # Number of iterations for weight diffusion.
    RENORM = False  # Whether to renormalize weights after adjustments.

    @staticmethod
    def get_dworld_dpixel(slice, x: float, y: float) -> np.array:
        """Compute the Jacobian matrix of world coordinates
        with respect to pixel coordinates at given coordinates.

        Parameters
        ----------
        slice : "InSlice" or "OutSlice"
            Slice object containing the WCS information.
        x, y : float, float
            Coordinates in the given pixel plane.

        Returns
        -------
        np.array, shape: `(2, 2)`, dtype: ``float``
            Jacobian matrix of world coordinates with respect to pixel coordinates.

        """

        # Shift by 0.5 to get the central differences.
        return local_partial_pixel_derivatives(
            slice.wcs, x-0.5, y-0.5) / slice.scale

    def __init__(self, outslice, X: int, Y: int) -> None:
        """Initialize the subslice with given output slice and coordinates.

        Parameters
        ----------
        outslice : "OutSlice"
            Output slice object.
        X, Y : int
            Indices of the subslice in the output slice.

        Attributes
        ----------
        outxys : np.array, shape: `(NPIX_SUB**2, 2)`, dtype: ``float``
            Output pixel coordinates in the output pixel plane.
        ctr : np.array, shape: `(2,)`, dtype: ``float``
            Center of the subslice in the output pixel plane.

        """

        self.outslice = outslice
        self.X, self.Y = X, Y
        NPIX_SUB = self.outslice.NPIX_SUB  # Shortcut.

        # Output pixel coordinates in the output pixel plane.
        self.outxys = np.moveaxis(np.array(np.meshgrid(
            np.arange(NPIX_SUB) + X*NPIX_SUB,
            np.arange(NPIX_SUB) + Y*NPIX_SUB)), 0, -1).reshape(-1, 2)
        self.ctr = np.array([(X+0.5)*NPIX_SUB-0.5, (Y+0.5)*NPIX_SUB-0.5])

    def __call__(self, sigma: float) -> None:
        """Process the subslice with given target output PSF width.

        Parameters
        ----------
        sigma : float
            Target output PSF width in native pixels.

        Returns
        -------
        None

        """

        NPIX_SUB = self.outslice.NPIX_SUB  # Shortcut.

        for i_sl, inslice in enumerate(self.outslice.inslices):
            mask_out = inslice.mask_out[self.Y*NPIX_SUB:(self.Y+1)*NPIX_SUB,
                                        self.X*NPIX_SUB:(self.X+1)*NPIX_SUB].copy()
            if not np.any(mask_out): continue

            # Get PSFs and weight field at the center of the subslice.
            ctr_in = inslice.outpix2world2inpix(self.ctr[None])[0]
            psf_in = inslice.get_psf(*ctr_in)
            psf_out = PSFModel.psf_gaussian(sigma, dout_din=np.linalg.inv(
                SubSlice.get_dworld_dpixel(self.outslice, *self.ctr)) @\
                SubSlice.get_dworld_dpixel(inslice, *ctr_in))
            wd = self.ACCEPT * PSFModel.SAMP + 6
            # In principle we should flip the input PSF, but
            # in practice it is easier to flip the weight field.
            weight = PSFModel.get_weight_field(psf_in, psf_out, wd)[:0:-1, :0:-1].copy()

            # Convert output to input pixel coordinates.
            inxys = inslice.outpix2world2inpix(self.outxys)
            inxys_frac, inxys_int = np.modf(inxys); inxys_int = inxys_int.astype(int) + 1
            x_min, y_min = np.min(inxys_int[mask_out.ravel()], axis=0) - self.ACCEPT
            x_max, y_max = np.max(inxys_int[mask_out.ravel()], axis=0) + self.ACCEPT
            indata, inmask = inslice.get_data_and_mask(x_min, x_max, y_min, y_max)
            inxys_int -= np.array([x_min, y_min])

            # Apply threshold for number of masked input pixels.
            apply_mask_threshold(mask_out.ravel(), inmask, inxys_int, self.MASK_THR)
            inslice.mask_out[self.Y*NPIX_SUB:(self.Y+1)*NPIX_SUB,
                             self.X*NPIX_SUB:(self.X+1)*NPIX_SUB] = mask_out
            if not np.any(mask_out): continue

            # Compute and adjust reconstruction weights for input pixels.
            weights = np.zeros((NPIX_SUB**2, self.ACCEPT*2, self.ACCEPT*2))
            compute_weights(weights, mask_out.ravel(), weight, inxys_frac,
                            wd-1, PSFModel.SAMP, self.ACCEPT)
            adjust_weights(weights, mask_out.ravel(), inmask, inxys_int,
                           self.ACCEPT, self.NDIFF, self.RENORM)

            # Apply reconstruction weights to input data to get output data.
            outdata = np.zeros((inslice.NLAYER, NPIX_SUB, NPIX_SUB))
            apply_weights(weights, mask_out.ravel(), outdata.reshape(inslice.NLAYER, -1),
                          indata, inxys_int, self.ACCEPT)
            if self.outslice.SAVE_ALL:
                self.outslice.data[:, i_sl, self.Y*NPIX_SUB:(self.Y+1)*NPIX_SUB,
                                            self.X*NPIX_SUB:(self.X+1)*NPIX_SUB] = outdata
            else:
                self.outslice.data[:, self.Y*NPIX_SUB:(self.Y+1)*NPIX_SUB,
                                      self.X*NPIX_SUB:(self.X+1)*NPIX_SUB] += outdata
