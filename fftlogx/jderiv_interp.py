import numpy as np
from scipy.interpolate import interp1d

from .fftlogx import fftlog
from .lib_wrapper import BesselIntType

from typing import Tuple


class BesselDerivInterp:
    def __init__(self, ells_expand: np.ndarray, jsqr_ells: Tuple[np.ndarray, np.ndarray],
                 ells: np.ndarray, cross_lm1_lp1_ells: Tuple[np.ndarray, np.ndarray], cross_lp1_lm1_ells: Tuple[np.ndarray, np.ndarray]):
        self.jsqr_interp = self.get_interp_ells(ells_expand, jsqr_ells)
        self.cross_lm1_lp1_interp = self.get_interp_ells(ells, cross_lm1_lp1_ells)
        self.cross_lp1_lm1_interp = self.get_interp_ells(ells, cross_lp1_lm1_ells)

    def get_interp_ells(self, ells: np.ndarray, value_ells: np.ndarray):
        return {l: interp1d(np.log(value_ells[0][i]), value_ells[1][i], bounds_error=False) for i, l in enumerate(ells)}

    def djl_mult(self, l, jsqr_lm1, jsqr_lp1, cross_lm1_lp1, cross_lp1_lm1):
        return (l**2 * jsqr_lm1 + (l+1)**2 * jsqr_lp1 - l*(l+1)*(cross_lm1_lp1+cross_lp1_lm1)) / (2*l+1)**2

    def __call__(self, chi, ells):
        log_chi = np.log(chi)

        cl = [self.djl_mult(
                l, self.jsqr_interp[l-1](log_chi), self.jsqr_interp[l+1](log_chi),
                self.cross_lm1_lp1_interp[l](log_chi), self.cross_lp1_lm1_interp[l](log_chi)) for l in ells]
        return np.array(cl)


class BesselDerivInterpEq(BesselDerivInterp):
    def __init__(self, ells_expand: np.ndarray, jsqr_ells: np.ndarray, ells: np.ndarray, cross_ells: np.ndarray):
        super().__init__(ells_expand, jsqr_ells, ells, cross_ells, cross_ells)

    def djl_mult(self, l, jsqr_lm1, jsqr_lp1, cross_lm1_lp1, cross_lp1_lm1):
        return (jsqr_lm1 + jsqr_lp1 - cross_lm1_lp1 - cross_lp1_lm1) / (2*l+1)**2

    def __call__(self, chi, ells):
        log_chi = np.log(chi)

        cl = []
        for l in ells:
            cross = self.cross_lm1_lp1_interp[l](log_chi)
            cl.append(self.djl_mult(
                l, self.jsqr_interp[l-1](log_chi), self.jsqr_interp[l+1](log_chi),
                cross, cross))
        return np.array(cl)


def djl_djl_neq(fftlog_integrator: fftlog, y_ratio: float, ell_start: int, ell_end: int):
    """
    Calculate F(y) = \int_0^\infty dx / x * f(x) * j'_\ell(xy) * j'_\ell(beta*xy)
    where y_ratio is beta.
    """
    ell_expanded = np.arange(ell_start-1, ell_end+2)
    jsqr = fftlog_integrator.jl1_jl2_neq(ell_expanded, ell_expanded, y_ratio)

    ells = np.arange(ell_start, ell_end+1)
    cross_lm1_lp1 = fftlog_integrator.jl1_jl2_neq(ells-1, ells+1, y_ratio)
    cross_lp1_lm1 = fftlog_integrator.jl1_jl2_neq(ells+1, ells-1, y_ratio)

    return BesselDerivInterp(
        ell_expanded, jsqr, ells, cross_lm1_lp1, cross_lp1_lm1)


def djl_djl_eq(fftlog_integrator: fftlog, ell_start: int, ell_end: int):
    """
    Calculate F(y) = \int_0^\infty dx / x * f(x) * (j'_\ell(xy))^2
    """
    ell_expanded = np.arange(ell_start-1, ell_end+2)
    jsqr_chi, jsqr_int = fftlog_integrator.fftlog_jsqr_ells(ell_expanded)

    ells = np.arange(ell_start, ell_end+1)
    cross_lm1_lp1 = 2*ells*(ells+1)*\
        fftlog_integrator._fftlog_ells(ells, BesselIntType.J_LP1_J_LM1_EQ, ells2=ells)

    return BesselDerivInterpEq(
        ell_expanded, (jsqr_chi, jsqr_int), ells, cross_lm1_lp1)
