"""Obsolete docstring, to be updated.

PSF utilities in two dimensions.

Functions
---------
psf_gaussian : Generate a 2D Gaussian PSF.
pixelate_psf : Pixelate a 2D (input) PSF.

"""

import numpy as np
from astropy.wcs.utils import local_partial_pixel_derivatives

from .routine import bandlimited_rfft2, bandlimited_irfft2
from .routine import apply_mask_threshold, compute_weights
from .routine import adjust_weights, apply_weights


class PSFModel:

    NPIX = 48  # PSF array size in native pixels.
    SAMP = 4  # Oversampling rate of PSF arrays.
    NTOT = NPIX * SAMP  # PSF array size in oversampled pixels.
    YXCTR = NTOT / 2  # PSF array center in oversampled pixels.
    BL_CIRC = 33  # Circular bandlimit in Fourier space.

    SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))
    SIGMA = {
        "Y106": 2.0 / SIGMA_TO_FWHM,
        "J129": 2.1 / SIGMA_TO_FWHM,
        "H158": 2.2 / SIGMA_TO_FWHM,
        "F184": 2.3 / SIGMA_TO_FWHM,
        "K213": 2.4 / SIGMA_TO_FWHM,
    }

    @classmethod
    def psf_gaussian(cls, sigma: float, dout_din: np.ndarray = np.diag(np.ones(2))) -> np.ndarray:
        """Generate a 2D Gaussian PSF."""
        xy = np.flip(np.mgrid[-cls.YXCTR:cls.NTOT-1-cls.YXCTR:cls.NTOT*1j,
                              -cls.YXCTR:cls.NTOT-1-cls.YXCTR:cls.NTOT*1j], axis=0)
        invSigma = dout_din.T @ np.diag(np.ones(2) / (sigma*cls.SAMP)**2) @ dout_din
        return np.exp(-0.5 * np.einsum("lyx,lr,ryx->yx", xy, invSigma, xy))\
                / (2.0*np.pi * (sigma*cls.SAMP)**2)  # Normalized in the output pixel plane.

    @classmethod
    def pixelate_psf(cls, psf: np.ndarray) -> np.ndarray:
        """Pixelate a 2D (input) PSF."""
        k = np.linspace(0, 1, cls.NTOT, endpoint=False)
        k[-(cls.NTOT//2):] -= 1; k *= cls.SAMP
        return np.fft.irfft2(np.fft.rfft2(psf) *\
            np.sinc(k[None, :cls.NTOT//2+1]) * np.sinc(k[:, None]), s=(cls.NTOT, cls.NTOT))

    @classmethod
    def get_weight_field(cls, psf_in: np.ndarray, psf_out: np.ndarray) -> np.ndarray:
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
            weight_tbl[None], cls.NTOT, cls.NTOT))[0] * cls.SAMP**2

    def __init__(self, psfdata: np.ndarray) -> None:
        self.psfdata = psfdata

    def __call__(self, x: float = -np.inf, y: float = -np.inf) -> np.ndarray:
        assert self.psfdata.ndim == 2, "PSFModel: The base class only supports 2D data."
        return self.psfdata


class SubSlice:

    ACCEPT = 8  # Acceptance radius in native pixels.
    REJECT = 8  # Rejection radius in output pixels.
    MASK_THR = 32  # Threshold for number of masked input pixels.
    NDIFF = 5  # Number of iterations for weight diffusion.
    RENORM = False  # Whether to renormalize weights after masking.

    @staticmethod
    def get_dworld_dpixel(slice, x: float, y: float) -> np.ndarray:
        return local_partial_pixel_derivatives(
            slice.wcs, x-0.5, y-0.5) / slice.scale

    def __init__(self, outslice, X: int, Y: int) -> None:
        self.outslice = outslice
        self.X, self.Y = X, Y
        NPIX_SUB = self.outslice.NPIX_SUB  # Shortcut.
        self.outxys = np.moveaxis(np.array(np.meshgrid(
            np.arange(NPIX_SUB) + X*NPIX_SUB,
            np.arange(NPIX_SUB) + Y*NPIX_SUB)), 0, -1).reshape(-1, 2)
        self.ctr = np.array([(X+0.5)*NPIX_SUB-0.5, (Y+0.5)*NPIX_SUB-0.5])

    def __call__(self, sigma: float) -> None:
        NPIX_SUB = self.outslice.NPIX_SUB  # Shortcut.

        for i_sl, inslice in enumerate(self.outslice.inslices):
            mask_out = inslice.mask_out[self.Y*NPIX_SUB:(self.Y+1)*NPIX_SUB,
                                        self.X*NPIX_SUB:(self.X+1)*NPIX_SUB].copy()
            if not np.any(mask_out): continue

            ctr_in = inslice.outpix2world2inpix(self.outslice.wcs, self.ctr[None])[0]
            psf_in = inslice.get_psf(*ctr_in)
            psf_out = PSFModel.psf_gaussian(sigma, dout_din=np.linalg.inv(
                SubSlice.get_dworld_dpixel(self.outslice, *self.ctr)) @\
                SubSlice.get_dworld_dpixel(inslice, *ctr_in))
            weight = PSFModel.get_weight_field(psf_in, psf_out)
            weight[1:, 1:] = np.flip(weight[1:, 1:])  # For convolution.

            inxys = inslice.outpix2world2inpix(self.outslice.wcs, self.outxys)
            inxys_frac, inxys_int = np.modf(inxys); inxys_int = inxys_int.astype(int) + 1
            x_min, y_min = np.min(inxys_int[mask_out.ravel()], axis=0) - self.ACCEPT
            x_max, y_max = np.max(inxys_int[mask_out.ravel()], axis=0) + self.ACCEPT
            indata, inmask = inslice.get_data_and_mask(x_min, x_max, y_min, y_max)
            inxys_int -= np.array([x_min, y_min])

            apply_mask_threshold(mask_out.ravel(), inmask, inxys_int, self.MASK_THR)
            inslice.mask_out[self.Y*NPIX_SUB:(self.Y+1)*NPIX_SUB,
                             self.X*NPIX_SUB:(self.X+1)*NPIX_SUB] = mask_out
            if not np.any(mask_out): continue
            weights = np.zeros((NPIX_SUB**2, self.ACCEPT*2, self.ACCEPT*2))
            compute_weights(weights, mask_out.ravel(), weight, inxys_frac,
                            PSFModel.NTOT/2, PSFModel.SAMP, self.ACCEPT)
            adjust_weights(weights, mask_out.ravel(), inmask, inxys_int,
                           self.ACCEPT, self.NDIFF, self.RENORM)

            outdata = np.zeros((inslice.NLAYER, NPIX_SUB, NPIX_SUB))
            apply_weights(weights, mask_out.ravel(), outdata.reshape(inslice.NLAYER, -1),
                          indata, inxys_int, self.ACCEPT)
            if self.outslice.SAVE_ALL:
                self.outslice.data[:, i_sl, self.Y*NPIX_SUB:(self.Y+1)*NPIX_SUB,
                                            self.X*NPIX_SUB:(self.X+1)*NPIX_SUB] = outdata
            else:
                self.outslice.data[:, self.Y*NPIX_SUB:(self.Y+1)*NPIX_SUB,
                                      self.X*NPIX_SUB:(self.X+1)*NPIX_SUB] += outdata
