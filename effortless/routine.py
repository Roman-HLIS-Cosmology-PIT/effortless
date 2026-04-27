'''Obsolete docstring, to be updated.

Numba version of pyimcom_croutines.c.
Slightly slower than C, used when furry-parakeet is not installed.

Functions
---------
iD5512C_getw : Interpolation code written by Python.
gridD5512C : iD5512C for rectangular grid.

'''

import numpy as np
from numba import njit


def bandlimited_rfft2(arr: np.array, bl: int) -> np.array:
    """Bandlimited forward real FFT in 2D.

    Parameters
    ----------
    arr : np.array, shape : (nf, ny, nx), dtype : real
        Array of nf functions of shape (ny, nx) to be FFT'ed.
    bl : int
        The bandlimit. Only modes between +/-bl will be saved.

    Returns
    -------
    np.array, shape : (nf, bl*2, bl+1), dtype : complex
        Array of nf sets of forward real FFT results.

    """

    rft = np.fft.fft(np.fft.rfft(arr)[:, :, :bl+1], axis=-2)
    return np.concatenate([rft[:, :bl, :], rft[:, -bl:, :]], axis=-2)


def bandlimited_irfft2(rft: np.array, ny: int, nx: int) -> np.array:
    """Bandlimited inverse real FFT in 2D.

    Parameters
    ----------
    rft : np.array, shape : (nf, bl*2, bl+1), dtype : complex
        Array of nf sets of forward real FFT results.
    ny, nx : int, int
        Shape of functions to be recovered via inverse FFT.

    Returns
    -------
    np.array : shape : (nf, ny, nx), dtype : real
        Array of nf functions of shape (ny, nx) from inverse FFT.

    """

    nf, bl_times2, bl_plus1 = rft.shape; bl = bl_plus1 - 1
    return np.fft.irfft(np.concatenate([np.fft.ifft(np.concatenate(
        [rft[:, :bl, :], np.zeros((nf, ny-bl_times2, bl_plus1), dtype=complex),
         rft[:, -bl:, :]], axis=-2), axis=-2),
         np.zeros((nf, ny, nx//2-bl_plus1))], axis=-1), n=nx)


@njit
def iD5512C_getw(w: np.array, fh: float) -> None:
    '''Interpolation code written by Python.

    Parameters
    ----------
    w : np.array, shape : (10,)
        Interpolation weights in one direction.
    fh : float
        'xfh' and 'yfh' with 1/2 subtracted.

    Returns
    -------
    None

    '''

    fh2 = fh * fh
    e_ =  (((+1.651881673372979740E-05*fh2 - 3.145538007199505447E-04)*fh2 +
          1.793518183780194427E-03)*fh2 - 2.904014557029917318E-03)*fh2 + 6.187591260980151433E-04
    o_ = ((((-3.486978652054735998E-06*fh2 + 6.753750285320532433E-05)*fh2 -
          3.871378836550175566E-04)*fh2 + 6.279918076641771273E-04)*fh2 - 1.338434614116611838E-04)*fh
    w[0] = e_ + o_
    w[9] = e_ - o_
    e_ =  (((-1.146756217210629335E-04*fh2 + 2.883845374976550142E-03)*fh2 -
          1.857047531896089884E-02)*fh2 + 3.147734488597204311E-02)*fh2 - 6.753293626461192439E-03
    o_ = ((((+3.121412120355294799E-05*fh2 - 8.040343683015897672E-04)*fh2 +
          5.209574765466357636E-03)*fh2 - 8.847326408846412429E-03)*fh2 + 1.898674086370833597E-03)*fh
    w[1] = e_ + o_
    w[8] = e_ - o_
    e_ =  (((+3.256838096371517067E-04*fh2 - 9.702063770653997568E-03)*fh2 +
          8.678848026470635524E-02)*fh2 - 1.659182651092198924E-01)*fh2 + 3.620560878249733799E-02
    o_ = ((((-1.243658986204533102E-04*fh2 + 3.804930695189636097E-03)*fh2 -
          3.434861846914529643E-02)*fh2 + 6.581033749134083954E-02)*fh2 - 1.436476114189205733E-02)*fh
    w[2] = e_ + o_
    w[7] = e_ - o_
    e_ =  (((-4.541830837949564726E-04*fh2 + 1.494862093737218955E-02)*fh2 -
          1.668775957435094937E-01)*fh2 + 5.879306056792649171E-01)*fh2 - 1.367845996704077915E-01
    o_ = ((((+2.894406669584551734E-04*fh2 - 9.794291009695265532E-03)*fh2 +
          1.104231510875857830E-01)*fh2 - 3.906954914039130755E-01)*fh2 + 9.092432925988773451E-02)*fh
    w[3] = e_ + o_
    w[6] = e_ - o_
    e_ =  (((+2.266560930061513573E-04*fh2 - 7.815848920941316502E-03)*fh2 +
          9.686607348538181506E-02)*fh2 - 4.505856722239036105E-01)*fh2 + 6.067135256905490381E-01
    o_ = ((((-4.336085507644610966E-04*fh2 + 1.537862263741893339E-02)*fh2 -
          1.925091434770601628E-01)*fh2 + 8.993141455798455697E-01)*fh2 - 1.213035309579723942E+00)*fh
    w[4] = e_ + o_
    w[5] = e_ - o_


@njit
def reggridD5512C(infunc: np.array, x0: float, y0: float, SAMP: int,
                  ACCEPT: int, out_arr: np.array, circ_cut: bool = False) -> None:
    wx_ar = np.zeros((10,))
    wy_ar = np.zeros((10,))
    x0i = int(x0)
    y0i = int(y0)
    iD5512C_getw(wx_ar, x0-x0i-0.5)
    iD5512C_getw(wy_ar, y0-y0i-0.5)

    ACCEPT2 = ACCEPT*2
    xmin = x0i - ACCEPT*SAMP
    ymin = y0i - ACCEPT*SAMP
    cut = 0

    interp_vstrip = np.zeros((10+(ACCEPT2-1-cut)*SAMP,))
    for ix in range(ACCEPT2):
        xi = xmin + ix*SAMP
        if circ_cut:
            cut = (ACCEPT-1) - int(((ACCEPT-0.5)**2 -\
                (ix-(ACCEPT-(ix<ACCEPT)))**2) ** 0.5)

        for i in range(cut*SAMP, 10+(ACCEPT2-1)*SAMP):
            interp_vstrip[i] = 0.0
            for j in range(10):
                interp_vstrip[i] += wx_ar[j] * infunc[ymin-4+i, xi-4+j]

        for iy in range(cut, ACCEPT2-cut):
            out_arr[iy, ix] = 0.0
            for i in range(10):
                out_arr[iy, ix] += interp_vstrip[iy*SAMP+i] * wy_ar[i]


@njit
def apply_mask_threshold(mask_out: np.ndarray, inmask: np.ndarray,
                         inxys_int: np.ndarray, MASK_THR: int, ACCEPT_: int = 8) -> None:
    for i in range(mask_out.shape[0]):
        if not mask_out[i]: continue

        inmask_i = inmask[inxys_int[i, 1]-ACCEPT_:inxys_int[i, 1]+ACCEPT_,
                          inxys_int[i, 0]-ACCEPT_:inxys_int[i, 0]+ACCEPT_]
        if np.sum(1-inmask_i) > MASK_THR:
            mask_out[i] = False


@njit
def compute_weights(weights: np.ndarray, mask_out: np.ndarray, weight: np.ndarray,
                    inxys_frac: np.ndarray, YXCTR: float, SAMP: int, ACCEPT: int) -> None:
    for i in range(mask_out.shape[0]):
        if not mask_out[i]: continue

        x0, y0 = YXCTR + (1-inxys_frac[i])*SAMP
        reggridD5512C(weight, x0, y0, SAMP, ACCEPT, weights[i])


@njit
def adjust_weights(weights: np.ndarray, mask_out: np.ndarray, inmask: np.ndarray,
                   inxys_int: np.ndarray, ACCEPT: int, NDIFF: int = 5,
                   RENORM: bool = False) -> None:
    for i in range(mask_out.shape[0]):
        if not mask_out[i]: continue

        inmask_i = inmask[inxys_int[i, 1]-ACCEPT:inxys_int[i, 1]+ACCEPT,
                          inxys_int[i, 0]-ACCEPT:inxys_int[i, 0]+ACCEPT]
        if NDIFF > 0:
            bad_ys, bad_xs = np.where(1-inmask_i)
            max_ = ACCEPT * 2 - 1
            weights_i = weights[i]

        for _ in range(NDIFF):
            for j in range(bad_ys.shape[0]):
                bad_y, bad_x = bad_ys[j], bad_xs[j]
                quarter = weights_i[bad_y, bad_x] / 4
                if quarter == 0.0: continue

                weights_i[bad_y, bad_x] = 0.0
                if bad_y >    0: weights_i[bad_y-1, bad_x] += quarter
                if bad_y < max_: weights_i[bad_y+1, bad_x] += quarter
                if bad_x >    0: weights_i[bad_y, bad_x-1] += quarter
                if bad_x < max_: weights_i[bad_y, bad_x+1] += quarter

        if RENORM:
            loss_frac = np.sum(weights[i] * (1-inmask_i)) / np.sum(weights[i])
            if loss_frac == 0.0: continue
        weights[i] *= inmask_i
        if RENORM:
            weights[i] *= 1 / (1 - loss_frac)


@njit
def apply_weights(weights: np.ndarray, mask_out: np.ndarray, outdata: np.ndarray,
                  indata: np.ndarray, inxys_int: np.ndarray, ACCEPT: int) -> None:
    for i in range(mask_out.shape[0]):
        if not mask_out[i]: continue

        for j in range(outdata.shape[0]):
            outdata[j, i] += np.sum(weights[i] *\
                indata[j, inxys_int[i, 1]-ACCEPT:inxys_int[i, 1]+ACCEPT,
                          inxys_int[i, 0]-ACCEPT:inxys_int[i, 0]+ACCEPT])
