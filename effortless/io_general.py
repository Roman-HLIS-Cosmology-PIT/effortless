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

    NSIDE = 4088
    NLAYER = 1  # Number of input layers.

    def __init__(self, filename: str, psfmodel: PSFModel = None,
                 loaddata: bool = True, paddata: bool = False) -> None:
        self.filename = filename
        self.psfmodel = psfmodel

        self.inxy_min = np.zeros(2, dtype=int)
        if loaddata:
            self.load_data_and_mask()
            if paddata: self.pad_data_and_mask()
        else:
            self.wcs = self.scale = None
            self.data = self.mask = None

    def load_data_and_mask(self) -> None:
        self.data = np.zeros((InSlice.NLAYER, InSlice.NSIDE, InSlice.NSIDE), dtype=np.float32)
        with fits.open(self.filename) as f:
            self.wcs = wcs.WCS(f["WFI01"].header)
            self.data[0] = f["WFI01"].data.astype(np.float32)
        self.scale = np.abs(self.wcs.wcs.cd[0, 0])  # Pixel scale in degrees.
        self.mask = np.ones((InSlice.NSIDE, InSlice.NSIDE), dtype=bool)

    def pad_data_and_mask(self) -> None:
        ACCEPT = SubSlice.ACCEPT  # Shortcut.
        self.data = np.pad(self.data, ((0,)*2,) + ((ACCEPT,)*2,)*2,
                           mode="constant", constant_values=0)
        self.mask = np.pad(self.mask, ACCEPT, mode="constant", constant_values=False)

        self.NSIDE = InSlice.NSIDE + ACCEPT*2
        self.inxy_min -= ACCEPT

    def outpix2world2inpix(self, outwcs: wcs.WCS, outxys: np.ndarray) -> np.ndarray:
        return self.wcs.all_world2pix(outwcs.all_pix2world(outxys, 0), 0) - self.inxy_min

    def inpix2world2outpix(self, outwcs: wcs.WCS, inxys: np.ndarray) -> np.ndarray:
        return outwcs.all_world2pix(self.wcs.all_pix2world(inxys + self.inxy_min, 0), 0)

    def assess_overlap(self, shrink: bool = True) -> None:
        ACCEPT = SubSlice.ACCEPT  # Shortcuts.
        NSUB, NPIX_SUB = OutSlice.NSUB, OutSlice.NPIX_SUB
        outxys_sp = np.moveaxis(np.array(np.meshgrid(
            np.arange(NSUB+1)*NPIX_SUB, np.arange(NSUB+1)*NPIX_SUB)), 0, -1).reshape(-1, 2)
        inxys_sp = self.outpix2world2inpix(self.outslice.wcs, outxys_sp)
        if shrink:
            inxy_min = np.array([self.NSIDE-1]*2, dtype=int)
            inxy_max = np.zeros(2, dtype=int)

        subsize = NPIX_SUB * self.outslice.wcs.wcs.cdelt[1] /\
            (0.11 * u.arcsec.to("degree"))  # Subslice size in input pixels.
        mask_sp = np.all((inxys_sp >=             -subsize) &
                         (inxys_sp <  self.NSIDE+1+subsize), axis=1).reshape(NSUB+1, NSUB+1)
        self.mask_out = np.zeros((OutSlice.NPIX_TOT, OutSlice.NPIX_TOT), dtype=bool)

        for X in range(NSUB):
            for Y in range(NSUB):
                if not np.any(mask_sp[Y:min(Y+2, NSUB+1), X:min(X+2, NSUB+1)]): continue

                outxys = np.moveaxis(np.array(np.meshgrid(
                    np.arange(NPIX_SUB) + X*NPIX_SUB,
                    np.arange(NPIX_SUB) + Y*NPIX_SUB)), 0, -1).reshape(-1, 2)
                inxys = self.outpix2world2inpix(self.outslice.wcs, outxys)

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

        if shrink:
            inxy_min -= ACCEPT-1; inxy_max += ACCEPT-1
            self.inxy_min = inxy_min
            self.data = self.data[:, inxy_min[1]:inxy_max[1]+1, inxy_min[0]:inxy_max[0]+1].copy()
            self.mask = self.mask[   inxy_min[1]:inxy_max[1]+1, inxy_min[0]:inxy_max[0]+1].copy()

    def propagate_mask(self, outwcs: wcs.WCS) -> None:
        REJECT = SubSlice.REJECT  # Shortcuts.
        NPIX_TOT = OutSlice.NPIX_TOT
        dys = np.arange(-REJECT, REJECT)
        dxs = (((REJECT-0.5)**2 - (dys+(dys<0))**2) ** 0.5).astype(int)

        bad_outxys = self.inpix2world2outpix(outwcs, \
            np.flip(np.where(1-self.mask), axis=0).T).astype(int) + 1
        bad_outxys = bad_outxys[(np.min(bad_outxys, axis=1) >           -REJECT)]
        bad_outxys = bad_outxys[(np.max(bad_outxys, axis=1) < NPIX_TOT-1+REJECT)]

        for bad_x, bad_y in bad_outxys:
            for dy, dx in zip(dys, dxs):
                y = bad_y + dy
                if y < 0 or y >= NPIX_TOT: continue
                self.mask_out[y, max(bad_x-dx, 0):
                    max(min(bad_x+dx, NPIX_TOT), 0)] = False

    def get_psf(self, x: float = -np.inf, y: float = -np.inf) -> np.ndarray:
        return self.psfmodel(x + self.inxy_min[0], y + self.inxy_min[1])

    def get_data_and_mask(self, x_min, x_max, y_min, y_max) -> tuple[np.ndarray, np.ndarray]:
        return (self.data[:, y_min:y_max+1, x_min:x_max+1].copy(),
                self.mask[   y_min:y_max+1, x_min:x_max+1].copy())


class OutSlice:

    NSUB = 73  # Output slice size in subslices, similar to n1 in PyIMCOM.
    NPIX_SUB = 56  # Subslice size in pixels, similar to n2 in PyIMCOM.
    NPIX_TOT = NSUB * NPIX_SUB  # Output slice size in pixels.
    CDELT = 0.11 * u.arcsec.to("degree") / 2.0  # Output pixel scale in degrees.
    SIGMA = PSFModel.SIGMA["H158"] * 1.5  # Target output PSF width in native pixels.
    SAVE_ALL = True  # Whether to save individual regridded images.

    @classmethod
    def get_outwcs(cls, outcrval: np.ndarray, outcrpix: list[float, float] = None,
                   outcdelt: list[float, float] = None) -> wcs.WCS:
        outwcs = wcs.WCS(naxis=2)
        outwcs.wcs.ctype = ["RA---STG", "DEC--STG"]
        outwcs.wcs.crval = outcrval
        outwcs.wcs.crpix = outcrpix if outcrpix is not None else [cls.NPIX_TOT/2]*2
        outwcs.wcs.cdelt = outcdelt if outcdelt is not None else [-cls.CDELT, cls.CDELT]
        return outwcs

    def __init__(self, wcs: wcs.WCS, inslices: list[InSlice], timing: bool = False) -> None:
        self.wcs = wcs
        self.scale = np.abs(wcs.wcs.cdelt[0])  # Pixel scale in degrees.
        self.inslices = inslices

        if timing: tstart = perf_counter()
        for inslice in self.inslices:
            if timing: print(f"Assessing inslice {inslice.filename}",
                             f"@ t = {perf_counter() - tstart:.6f} s")
            inslice.outslice = self
            inslice.assess_overlap()
            inslice.propagate_mask(self.wcs)
        if timing: print("Finished assessing inslices",
                         f"@ t = {perf_counter() - tstart:.6f} s", end="\n\n")

        self.inslices = [inslice for inslice in self.inslices if inslice.is_relevant]
        self.ninslice = len(self.inslices)
        if self.SAVE_ALL:
            self.data = np.zeros((InSlice.NLAYER, self.ninslice,
                                  self.NPIX_TOT, self.NPIX_TOT), dtype=np.float32)
        else:
            self.data = np.zeros((InSlice.NLAYER, self.NPIX_TOT, self.NPIX_TOT), dtype=np.float32)

    def __call__(self, filename: str = None, timing: bool = False, stop: int = np.inf) -> None:
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

    def writeto(self, filename: str) -> None:
        datahdu = fits.PrimaryHDU(self.data, header=self.wcs.to_header())
        inputhdu = fits.TableHDU.from_columns([fits.Column(name="filename", \
            array=[inslice.filename for inslice in self.inslices], format="A512", ascii=True)])
        inputhdu.name = "INPUT"
        maskhdu = fits.ImageHDU(self.mask.astype(np.uint8), name="MASK")
        fits.HDUList([datahdu, inputhdu, maskhdu]).writeto(filename, overwrite=True)
