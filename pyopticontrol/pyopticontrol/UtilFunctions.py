import numpy as np
import casadi as cs
from numpy import float64
from numpy.typing import NDArray
from casadi import SX
from typing import Union, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

__all__ = ["matmul3d", "repmat", "max_smooth", "min_smooth", "FB", "csFB", "logpen", "IndirectOCP", "CSIndirectOCP",
           "DirectOCP", "OptiSol"]

np.seterr(over="ignore")


@dataclass
class OptiSol:
    time: Optional[NDArray[float64]] = None
    ode_sol: Optional[NDArray[float64]] = None
    alg_sol: Optional[NDArray[float64]] = None
    alg_sol_mid: Optional[NDArray[float64]] = None
    params: Optional[NDArray[float64]] = None


class IndirectOCP(ABC):

    @abstractmethod
    def ode(self, time: NDArray[float64], ode_sol: NDArray[float64], alg_sol: NDArray[float64]
            ) -> NDArray[float64]:
        pass

    @abstractmethod
    def odejac(self, time: NDArray[float64], ode_sol: NDArray[float64], alg_sol: NDArray[float64]
               ) -> tuple[NDArray[float64], NDArray[float64]]:
        pass

    @abstractmethod
    def algeq(self, time: NDArray[float64], ode_sol: NDArray[float64], alg_sol: NDArray[float64]) -> NDArray[float64]:
        pass

    @abstractmethod
    def algjac(self, time: NDArray[float64], ode_sol: NDArray[float64], alg_sol: NDArray[float64]
               ) -> tuple[NDArray[float64], NDArray[float64]]:
        pass

    @abstractmethod
    def twobc(self, ode_sol_init: NDArray[float64], ode_sol_final: NDArray[float64], alg_sol_init: NDArray[float64],
              alg_sol_final: NDArray[float64], p: Optional[NDArray[float64]] = None) -> NDArray[float64]:
        pass

    @abstractmethod
    def bcjac(self, ode_sol_init: NDArray[float64], ode_sol_final: NDArray[float64], alg_sol_init: NDArray[float64],
              alg_sol_final: NDArray[float64], p: Optional[NDArray[float64]] = None
              ) -> (
            Union[
                tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64]],
                tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64]]
            ]):
        pass


class CSIndirectOCP(ABC):
    @abstractmethod
    def ode(self, time: SX, ode_sol: SX, alg_sol: SX) -> SX:
        pass

    @abstractmethod
    def algeq(self, time: SX, ode_sol: SX, alg_sol: SX) -> SX:
        pass

    @abstractmethod
    def twobc(self, ode_sol_init: SX, ode_sol_final: SX, alg_sol_init: SX,
              alg_sol_final: SX, p: Optional[SX] = None) -> SX:
        pass


class DirectOCP(ABC):
    @abstractmethod
    def ode(self, time: SX, ode_sol: SX, alg_sol: SX) -> SX:
        pass

    @abstractmethod
    def running_cost(self, time: SX, ode_sol: SX, alg_sol: SX) -> SX:
        pass

    @abstractmethod
    def twobc(self, ode_sol_init: SX, ode_sol_final: SX, alg_sol_init: SX,
              alg_sol_final: SX, p: Optional[SX] = None) -> SX:
        pass


def FB(x, y, eps, dx=0, dy=0):
    """
    :param x: scalar positive variable
    :param y: scalar negative variable
    :param eps: scalar positive penalty parameter
    :param dx: order of derivative wrt x (0, 1 value)
    :param dy: order of derivative wrt y (0 or 1 value)
    :return: Ficher Burmeister complementarity function
    """
    if dx == 0 and dy == 0:
        return x - y - np.sqrt(x ** 2 + y ** 2 + 2. * eps)
    if dx == 1 and dy == 0:
        return 1. - x / np.sqrt(x ** 2 + y ** 2 + 2. * eps)
    if dx == 0 and dy == 1:
        return -1. - y / np.sqrt(x ** 2 + y ** 2 + 2. * eps)
    raise Exception("Only first order derivatives are defined for Fisher-Burmeister complementarity function")


def csFB(x, y, eps):
    return x - y - cs.sqrt(x ** 2 + y ** 2 + 2. * eps)


def max_smooth(x, y, mu=1e-6, d1=0, d2=0):
    """
    This functions returns the mu-smooth maximum of (x, y, dx=0, dy=0). If x and y are 1d-arrays, the function returns a 1d-array z s.t. z[i] = max_smooth(x[i], y[i])

    :param x: NumPy array
    :param y: NumPy array
    :param mu: regularisation parameter of |.| ~ sqrt(. ** 2 + mu)
    :param d1: order of x-derivative of max_smooth
    :param d2: order of y-derivative of max_smooth
    :return: (d/dx) ** d1 * (d/dy) ** d2 * (x + y + sqrt((x-y) ** 2 + mu))
    """
    if d1 == 0 and d2 == 0:
        return .5 * (x + y + abs_smooth(x - y, mu=mu))
    elif d1 == 1 and d2 == 0:
        return .5 * (1 + abs_smooth(x - y, mu=mu, d=1))
    elif d1 == 0 and d2 == 1:
        return .5 * (1 - abs_smooth(x - y, mu=mu, d=1))
    elif d1 == 1 and d2 == 1:
        return - .5 * abs_smooth(x - y, mu=mu, d=2)
    elif d1 == 0 and d2 == 2:
        return .5 * abs_smooth(x - y, mu=mu, d=2)
    elif d1 == 2 and d2 == 0:
        return .5 * abs_smooth(x - y, mu=mu, d=2)
    else:
        raise Exception('max_smooth derivatives must be such that 0 <= dx + dy <= 2')


def min_smooth(x, y, mu=1e-6, d1=0, d2=0):
    """
    This functions returns the mu-smooth maximum of (x, y, dx=0, dy=0). If x and y are 1d-arrays, the function returns a 1d-array z s.t. z[i] = max_smooth(x[i], y[i])

    :param x: NumPy array
    :param y: NumPy array
    :param mu: regularisation parameter of |.| ~ sqrt(. ** 2 + mu)
    :param d1: order of x-derivative of max_smooth
    :param d2: order of y-derivative of max_smooth
    :return: (d/dx) ** d1 * (d/dy) ** d2 * (x + y - sqrt((x-y) ** 2 + mu))
    """
    if d1 == 0 and d2 == 0:
        return .5 * (x + y - abs_smooth(x - y, mu=mu))
    elif d1 == 1 and d2 == 0:
        return .5 * (1 - abs_smooth(x - y, mu=mu, d=1))
    elif d1 == 0 and d2 == 1:
        return .5 * (1 + abs_smooth(x - y, mu=mu, d=1))
    elif d1 == 1 and d2 == 1:
        return .5 * abs_smooth(x - y, mu=mu, d=2)
    elif d1 == 0 and d2 == 2:
        return - .5 * abs_smooth(x - y, mu=mu, d=2)
    elif d1 == 2 and d2 == 0:
        return - .5 * abs_smooth(x - y, mu=mu, d=2)
    else:
        raise Exception('max_smooth derivatives must be such that 0 <= dx + dy <= 2')


def abs_smooth(x, mu=1e-6, d=0):
    """
    This functions computes the smooth absolute value of x

    :param x: NumPy array
    :param mu: regularisation parameter of |.| ~ sqrt(. ** 2 + mu)
    :param d: order of x-derivative of abs_smooth
    :return: (d/dx) ** d * sqrt(x ** 2 + mu)
    """
    if d == 0:
        return np.sqrt(x ** 2 + mu)
    if d == 1:
        return x / np.sqrt(x ** 2 + mu)
    if d == 2:
        x2pmu = x ** 2 + mu
        return mu / (np.sqrt(x2pmu) * x2pmu)
    raise Exception('abs_smooth derivative must be in {0, 1, 2}')


def repmat(a, rep_dim):
    """
    This function allows to replicated a 2D-matrix A along first, second and optionally third dimension.
    :param a: Matrix to be replicated
    :param rep_dim: tuple of integer (d0, d1, [d2]) giving the number of times matrix a is replicated along each dimension
    :return: NumPy array of replicated a matrix
    """
    if len(rep_dim) < 2:
        raise Exception("Repmat needs at least 2 dimensions")
    if len(rep_dim) == 2:
        return np.tile(a, rep_dim)
    if len(rep_dim) == 3:
        d0, d1, d2 = rep_dim
        ad0, ad1 = a.shape
        return np.reshape(np.tile(np.tile(a, (d0, d1)), (1, d2)), (ad0*d0, ad1*d1, d2), order="F")


def matmul3d(a, b):
    """
    3D multiplication of matrices a, b
    """
    if len(a.shape) == 2 and len(b.shape) == 3:
        return np.einsum('ij,jlk->ilk', a, b)
    elif len(a.shape) == 3 and len(b.shape) == 3:
        return np.einsum('ijk,jlk->ilk', a, b)
    elif len(a.shape) == 3 and len(b.shape) == 2:
        return np.einsum('ijk,jl->ilk', a, b)
    else:
        raise Exception("not a 3D matrix product")


def logpen(x, d=0):
    if isinstance(x, np.ndarray):
        g = np.where(x < 0., 0., x)
        gplus = g > 0.
        if d == 0:
            g[gplus] = - np.log(g[gplus])
            return g
        if d == 1:
            g[gplus] = - 1. / g[gplus]
            return g
        if d == 2:
            g[gplus] = 1. / g[gplus] ** 2
            return g
        raise Exception("gamma derivative must be in {0, 1, 2}")
    else:
        if d == 0:
            g = cs.if_else(x <= 0., 0., - cs.log(x))
            return g
        if d == 1:
            g = cs.if_else(x <= 0., 0., - 1. / x)
            return g
        if d == 2:
            g = cs.if_else(x <= 0., 0., 1. / x ** 2)
            return g
        raise Exception("gamma derivative must be in {0, 1, 2}")

