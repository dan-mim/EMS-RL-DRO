import numpy as np
from .UtilFunctions import matmul3d, IndirectOCP
from scipy import sparse
from scipy.sparse.linalg import spsolve as scipy_spsolve
from pypardiso import spsolve as pardiso_spsolve
import sys
from numpy.typing import NDArray
from numpy import float64
from typing import Optional


class NLSEInfos:
    """
    The NLSEInfos class contains the information of the DAE's solving and has the following attributes
    :param success: Boolean indicating if the problems is successfully solved
    :param iter: number of Newton iterations
    :param ode_residual: Residual error on for the discretization of the ODEs
    :param ode_tol: Error tolerance (mesh dependent) of the ODEs
    :param ae_residual: Residual error for the Algebraic Equations
    :param ae_tol: Error tolerance of the AEs
    :param bc_residual: Residual error for the boundary conditions
    :param bc_tol: Error tolerance for the BCs
    """
    def __init__(self,
                 success: bool,
                 iter: int,
                 ode_residual: float,
                 ode_tol: float,
                 ae_residual: float,
                 ae_tol: float,
                 bc_residual: float,
                 bc_tol: float):
        self.success = success
        self.iter = iter
        self.ode_residual = ode_residual
        self.ode_tol = ode_tol
        self.ae_residual = ae_residual
        self.ae_tol = ae_tol
        self.bc_residual = bc_residual
        self.bc_tol = bc_tol

    def __str__(self) -> str:
        string_nlse = (
                "NSLE infos : \n"
                + "    succes = " + str(self.success) + "\n"
                + "    iter = " + str(self.iter) + "\n"
                + "    ode_residual = " + str(self.ode_residual) + " for ode_tol = " + str(self.ode_tol) + "\n"
                + "    ae_residual = " + str(self.ae_residual) + " for ae_tol = " + str(self.ae_tol) + "\n"
                + "    bc_residual = " + str(self.bc_residual) + " for bc_tol = " + str(self.bc_tol)
        )
        return string_nlse


def approx_res_colocation(
        time: NDArray[float64],
        ode_sol: NDArray[float64],
        alg_sol: NDArray[float64],
        ocp: IndirectOCP) -> tuple[NDArray[float64], NDArray[float64]]:
    """
    This function returns the non-linear equations corresponding to the ODE's discretization and the AEs
    and the ODE's RHS evaluated at collocation point.
    """
    h = np.diff(time)  # vector containing the length of every time step of the time-grid
    alg_mid = .5 * (alg_sol[:, 1:] + alg_sol[:, :-1])
    tmid = (time[:-1] + time[1:]) / 2.  # Time collocation point
    rhs = ocp.ode(time, ode_sol, alg_sol)  # Evaluation of the differential equations
    ode_mid = (ode_sol[:, :-1] + ode_sol[:, 1:]) / 2. - (rhs[:, 1:] - rhs[:, :-1]) * h / 8.  # State colocation point
    rhsmid = ocp.ode(tmid, ode_mid, alg_mid)  # ODEs at colocation point

    alg = ocp.algeq(time, ode_sol, alg_sol)  # AEs at time t

    # Computation of the boundary condition
    bound_const = ocp.twobc(ode_sol[:, 0], ode_sol[:, -1], alg_sol[:, 0], alg_sol[:, -1])
    odes = ode_sol[:, 1:] - ode_sol[:, :-1] - (rhs[:, 1:] + 4. * rhsmid + rhs[:, :-1]) * h / 6.

    size_block_ode_alg = (time.size - 1) * (ode_sol.shape[0] + alg_sol.shape[0])
    res = np.empty((ode_sol.size + alg_sol.size,), dtype=float64)
    res[:ode_sol.shape[0]] = bound_const
    res[ode_sol.shape[0]: ode_sol.shape[0] + size_block_ode_alg] = np.vstack((odes, alg[:, :-1])).ravel(order="F")
    res[ode_sol.shape[0] + size_block_ode_alg:] = alg[:, -1]

    return res, rhsmid


def approx_jac_res_colocation(
        time: NDArray[float64],
        ode_sol: NDArray[float64],
        alg_sol: NDArray[float64],
        ocp: IndirectOCP,
        id3d: NDArray[float64],
        rowis: NDArray[int],
        colis: NDArray[int],
        shape_jac: tuple[int, int]) -> sparse.csc_matrix:
    values_jacobian = approx_jac_res_values(time, ode_sol, alg_sol, ocp, id3d)

    non_zeros_indices = np.nonzero(values_jacobian)

    return sparse.csc_matrix(
        (values_jacobian[non_zeros_indices], (rowis[non_zeros_indices], colis[non_zeros_indices])), shape_jac
    )


def approx_jac_res_values(
        time: NDArray[float64],
        ode_sol: NDArray[float64],
        alg_sol: NDArray[float64],
        ocp: IndirectOCP,
        id3d: NDArray[float64]) -> NDArray[float64]:

    N = len(time) - 1  # number of time step - 1
    h = np.diff(time)  # vector of length of every time step
    ne, na = ode_sol.shape[0], alg_sol.shape[0]
    tmid = time[:-1] + h / 2.  # vector containing the midpoints of the time grid
    alg_mid = .5 * (alg_sol[:, 1:] + alg_sol[:, :-1])
    h3d = h.reshape((1, 1, N))
    h3d6 = h3d / 6.
    h3d8 = h3d / 8.
    rhs = ocp.ode(time, ode_sol, alg_sol)  # ODEs at time t
    ode_mid = (ode_sol[:, :-1] + ode_sol[:, 1:]) / 2. - h / 8. * (rhs[:, 1:] - rhs[:, :-1])

    # Calling AEs jacobian
    gx, gz = ocp.algjac(time, ode_sol, alg_sol)  # evaluate the jacobian of the algebraic equations
    # Calling ODEs jacobian
    fx, fz = ocp.odejac(time, ode_sol, alg_sol)  # evaluate the jacobian of the ODEs
    fxmid, fzmid = ocp.odejac(tmid, ode_mid, alg_mid)  # evaluate the ODEs jacobian at midpoints

    block_ode_alg_wrt_k = np.zeros((ne + na, ne + na, N))

    block_ode_alg_wrt_k[:ne, :ne, :] = (
            - id3d - h3d6 * (4. * matmul3d(fxmid, id3d / 2. + h3d8 * fx[:, :, :-1]) + fx[:, :, :-1])
    )

    block_ode_alg_wrt_k[:ne, ne:ne + na, :] = (
            - h3d6 * (4. * matmul3d(fxmid, h3d8 * fz[:, :, :-1]) + 2. * fzmid + fz[:, :, :-1])
    )

    block_ode_alg_wrt_k[ne:, :ne, :] = gx[:, :, :-1]

    block_ode_alg_wrt_k[ne:, ne: ne + na, :] = gz[:, :, :-1]

    block_ode_wrt_kp1 = np.zeros((ne, ne + na, N))
    block_ode_wrt_kp1[:, :ne, :] = (
            id3d - h3d6 * (4. * matmul3d(fxmid, id3d / 2. - h3d8 * fx[:, :, 1:]) + fx[:, :, 1:])
    )

    block_ode_wrt_kp1[:, ne:, :] = (
            - h3d6 * (4. * matmul3d(fxmid, - h3d8 * fz[:, :, 1:]) + 2. * fzmid + fz[:, :, 1:])
    )

    # Computing the Hessian of the boundary conditions
    jac_bc_x0, jac_bc_xend, jac_bc_z0, jac_bc_zend = ocp.bcjac(ode_sol[:, 0], ode_sol[:, -1], alg_sol[:, 0],
                                                               alg_sol[:, -1])

    size_bc_jac = jac_bc_x0.size + jac_bc_xend.size + jac_bc_z0.size + jac_bc_zend.size
    vals = np.empty(
        (
            size_bc_jac + block_ode_alg_wrt_k.size + block_ode_wrt_kp1.size + gx[:, :, -1].size + gz[:, :, -1].size,),
        dtype=np.float64
    )
    vals[:size_bc_jac] = np.hstack((jac_bc_x0, jac_bc_z0, jac_bc_xend, jac_bc_zend)).ravel(order="F")
    vals[size_bc_jac: size_bc_jac + block_ode_alg_wrt_k.size] = block_ode_alg_wrt_k.ravel(order="F")
    vals[size_bc_jac + block_ode_alg_wrt_k.size: size_bc_jac + block_ode_alg_wrt_k.size + block_ode_wrt_kp1.size] = (
        block_ode_wrt_kp1.ravel(order="F")
    )
    vals[size_bc_jac + block_ode_alg_wrt_k.size + block_ode_wrt_kp1.size:] = (
        np.hstack((gx[:, :, -1], gz[:, :, -1])).ravel(order="F")
    )
    return vals


def approx_solve_newton(
        time: NDArray[float64],
        ode_sol: NDArray[float64],
        alg_sol: NDArray[float64],
        ocp: IndirectOCP,
        id3d: NDArray[float64],
        rowis: NDArray[np.int64],
        colis: NDArray[np.int64],
        shape_jac: tuple[int, int],
        res_odeis: NDArray[np.int64],
        res_algis: NDArray[np.int64],
        res_tol: Optional[float] = 1e-3,
        max_iter: Optional[int] = 100,
        display: Optional[int] = 0,
        linear_solver: Optional[int] = 0,
        atol: Optional[float] = 1e-9,
        coeff_damping: Optional[float] = 2.,
        max_probes: Optional[int] = 8,
        odes_control: Optional[bool] = True) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], NLSEInfos]:
    success = False
    niter = 0
    alpha = -1.
    h = np.diff(time)
    tol_odes = 2 / 3 * h * 5e-2 * res_tol
    while not success and niter < max_iter:
        res, rhsmid = approx_res_colocation(time, ode_sol, alg_sol, ocp)
        jac = approx_jac_res_colocation(time, ode_sol, alg_sol, ocp, id3d, rowis, colis, shape_jac)

        if display == 2:
            print('        * Newton Iteration # ' + str(niter))
            print('        * Initial error = ', np.max(np.abs(res)))
        if linear_solver == 0:
            direction = scipy_spsolve(jac, res)
        else:
            direction = pardiso_spsolve(jac, res)
        ode_sol_old, alg_sol_old, alpha_old = ode_sol, alg_sol, alpha

        ode_sol, alg_sol, cost, alpha, rhsmid, res, armflag = approx_armijo(
            time, ode_sol, alg_sol, ocp, direction, res, rhsmid, coeff_damping, max_probes
        )

        abs_res = np.abs(res)
        max_res = np.max(abs_res)

        if odes_control:
            tol = (tol_odes * (1. + np.abs(rhsmid))).ravel(order="F")
            success = (
                    np.all(abs_res[res_odeis] <= tol) and np.all(abs_res[res_algis] <= atol)
                    and np.all(abs_res[:ode_sol.shape[0]] <= atol)
            )
        else:
            tol = atol
            success = np.all(abs_res <= atol)

        if success and display == 2:
            print('           Success damped Newton step = ' + str(max_res) + ' alpha = ' + str(
                alpha) + '; iter = ' + str(niter))
            sys.stdout.flush()
        elif display == 2:
            print(
                '                  Newton step = ' + str(max_res) + ' alpha = ' + str(alpha) +
                '; iter = ' + str(niter))
            sys.stdout.flush()

        if np.any(np.isnan(res)) or np.any(np.isinf(res)):
            nlse_infos = NLSEInfos(success, niter, None, tol, None, atol, None, atol)
            return ode_sol, alg_sol, rhsmid, nlse_infos

        if np.linalg.norm(ode_sol - ode_sol_old) == 0. and np.linalg.norm(
                alg_sol - alg_sol_old) == 0. and alpha == alpha_old:
            break

        niter += 1

    ode_res = np.max(abs_res[res_odeis])
    ae_res = np.max(abs_res[res_algis])
    bc_res = np.max(abs_res[:ode_sol.shape[0]])
    nlse_infos = NLSEInfos(success, niter, ode_res, max(tol_odes), ae_res, atol, bc_res, atol)
    return ode_sol, alg_sol, rhsmid, nlse_infos


def approx_armijo(time, ode_sol_0, alg_sol_0, OCP, direction, res0, rhsmid0, coeff_damping, max_probes):
    iarm = 0
    sigma1 = 1. / coeff_damping
    alpha = 1e-4
    armflag = True
    lbd, lbdm, lbdc = 1., 1., 1.
    dxp, dz = get_ode_ae_from_sol(direction, ode_sol_0.shape[0], alg_sol_0.shape[0])
    xpt = ode_sol_0 - lbd * dxp
    zt = alg_sol_0 - lbd * dz
    rest, rhsmidt = approx_res_colocation(time, xpt, zt, OCP)
    nft, nf0 = np.linalg.norm(rest), np.linalg.norm(res0)
    ff0, ffc = nf0 * nf0, nft * nft
    ffm = ffc
    best_xp, best_z, best_cost, best_lbd, best_rhsmid, best_res = ode_sol_0, alg_sol_0, ff0, lbd, rhsmid0, res0
    if ffc < best_cost:
        best_xp, best_z, best_cost, best_lbd, best_rhsmid, best_res = xpt, zt, ffc, lbd, rhsmidt, rest
    while nft >= (1. - alpha * lbd) * nf0:
        if iarm == 0 or np.isinf(ffm) or np.isinf(ffc) or np.isnan(ffm) or np.isnan(ffc):
            lbd = sigma1 * lbd
        else:
            lbd = approx_parab3p(lbdc, lbdm, ff0, ffc, ffm)
        xpt = ode_sol_0 - lbd * dxp
        zt = alg_sol_0 - lbd * dz
        lbdm = lbdc
        lbdc = lbd
        rest, rhsmidt = approx_res_colocation(time, xpt, zt, OCP)
        nft = np.linalg.norm(rest)
        ffm = ffc
        ffc = nft * nft
        if ffc < best_cost:
            best_xp, best_z, best_cost, best_lbd, best_rhsmid, best_res = xpt, zt, ffc, lbd, rhsmidt, rest
        iarm += 1
        if iarm > max_probes:
            armflag = False
            return best_xp, best_z, best_cost, best_lbd, best_rhsmid, best_res, armflag
    return xpt, zt, .5 * ffc, lbd, rhsmidt, rest, armflag


def approx_parab3p(lbdc, lbdm, ff0, ffc, ffm):
    sigma0, sigma1 = .1, .5
    c2 = lbdm * (ffc - ff0) - lbdc * (ffm - ff0)
    if c2 >= 0.:
        return sigma1 * lbdc
    c1 = lbdc * lbdc * (ffm - ff0) - lbdm * lbdm * (ffc - ff0)
    lbdp = -c1 * .5 / c2
    if lbdp < sigma0 * lbdc:
        lbdp = sigma0 * lbdc
    if lbdp > sigma1 * lbdc:
        lbdp = sigma1 * lbdc
    return lbdp


def _get_sol_from_ode_ae(ode_sol, alg_sol):
    return np.vstack((ode_sol, alg_sol)).ravel(order="F")


def get_ode_ae_from_sol(x, n_odes, n_aes):
    nrow = n_odes + n_aes
    nt = x.size // nrow
    ode_alg_sol = x.reshape((nrow, nt), order="F")
    ode_sol = ode_alg_sol[:n_odes, :]
    alg_sol = ode_alg_sol[n_odes:, :]
    return ode_sol, alg_sol


"""
def approx_jac_res_colocation_deprecated(
        time: NDArray[float64],
        ode_sol: NDArray[float64],
        alg_sol: NDArray[float64],
        ocp: IndirectOCP,
        id3d: NDArray[float64],
        rowis: NDArray[int],
        colis: NDArray[int],
        shape_jac: tuple[int, int]) -> sparse.csc_matrix:
    values_jacobian = approx_jac_res_values(time, ode_sol, alg_sol, ocp, id3d)
    non_zeros_indices = np.nonzero(values_jacobian)
    red_values, red_rows, red_cols = (values_jacobian[non_zeros_indices],
                                      rowis[non_zeros_indices],
                                      colis[non_zeros_indices]
                                      )

    return sparse.csc_matrix((red_values, (red_rows, red_cols)), shape_jac)


def approx_solve_newton_deprecated(
        time: NDArray[float64],
        ode_sol: NDArray[float64],
        alg_sol: NDArray[float64],
        ocp: IndirectOCP,
        id3d: NDArray[float64],
        rowis: NDArray[np.int64],
        colis: NDArray[np.int64],
        shape_jac: tuple[int, int],
        res_odeis: NDArray[np.int64],
        res_algis: NDArray[np.int64],
        res_tol: Optional[float] = 1e-3,
        max_iter: Optional[int] = 100,
        display: Optional[int] = 0,
        linear_solver: Optional[int] = 0,
        atol: Optional[float] = 1e-9,
        coeff_damping: Optional[float] = 2.,
        max_probes: Optional[int] = 8,
        odes_control: Optional[bool] = True) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], NLSEInfos]:
    success = False
    niter = 0
    alpha = -1.
    h = np.diff(time)
    tol_odes = 2 / 3 * h * 5e-2 * res_tol
    while not success and niter < max_iter:
        res, rhsmid = approx_res_colocation(time, ode_sol, alg_sol, ocp)
        jac = approx_jac_res_colocation(time, ode_sol, alg_sol, ocp, id3d, rowis, colis, shape_jac)
        if display == 2:
            print('        * Newton Iteration # ' + str(niter))
            print('        * Initial error = ', np.max(np.abs(res)))
        if linear_solver == 0:
            direction = scipy_spsolve(jac, res)
        else:
            direction = pardiso_spsolve(jac, res)
        ode_sol_old, alg_sol_old, alpha_old = ode_sol, alg_sol, alpha

        ode_sol, alg_sol, cost, alpha, rhsmid, res, armflag = approx_armijo(
            time, ode_sol, alg_sol, ocp, direction, res, rhsmid, coeff_damping, max_probes
        )

        abs_res = np.abs(res)
        max_res = np.max(abs_res)

        if odes_control:
            tol = (tol_odes * (1. + np.abs(rhsmid))).ravel(order="F")
            success = (
                    np.all(abs_res[res_odeis] <= tol) and np.all(abs_res[res_algis] <= atol)
                    and np.all(abs_res[:ode_sol.shape[0]] <= atol)
            )
        else:
            tol = atol
            success = np.all(abs_res <= atol)

        if success and display == 2:
            print('           Success damped Newton step = ' + str(max_res) + ' alpha = ' + str(
                alpha) + '; iter = ' + str(niter))
            sys.stdout.flush()
        elif display == 2:
            print(
                '                  Newton step = ' + str(max_res) + ' alpha = ' + str(alpha) +
                '; iter = ' + str(niter))
            sys.stdout.flush()

        if np.any(np.isnan(res)) or np.any(np.isinf(res)):
            nlse_infos = NLSEInfos(success, niter, None, tol, None, atol, None, atol)
            return ode_sol, alg_sol, rhsmid, nlse_infos

        if np.linalg.norm(ode_sol - ode_sol_old) == 0. and np.linalg.norm(
                alg_sol - alg_sol_old) == 0. and alpha == alpha_old:
            break

        niter += 1

    ode_res = np.max(abs_res[res_odeis])
    ae_res = np.max(abs_res[res_algis])
    bc_res = np.max(abs_res[:ode_sol.shape[0]])
    nlse_infos = NLSEInfos(success, niter, ode_res, max(tol_odes), ae_res, atol, bc_res, atol)
    return ode_sol, alg_sol, rhsmid, nlse_infos


def approx_row_col_jac_indices_deprecated(
        time: NDArray[float64],
        n_odes: int,
        n_aes: int) -> tuple[NDArray[int], NDArray[int], tuple[int, int], NDArray[float64], NDArray[int], NDArray[int]]:
    N = len(time)
    # indices for bcjac
    row_bcjac = np.tile(np.arange(n_odes), 2 * (n_odes + n_aes))
    col_bcjac = (np.tile(np.repeat(np.arange(n_odes + n_aes), n_odes), 2)
                 + np.repeat(np.array([0, (N - 1) * (n_odes + n_aes)]), n_odes * (n_odes + n_aes)))

    # indices for block ode alg
    row_block_ode_alg = (
            n_odes + np.tile(np.tile(np.arange(n_odes + n_aes), 2 * n_odes + 2 * n_aes), N - 1)
            + np.repeat(np.arange(N - 1) * (n_odes + n_aes), 2 * (n_odes + n_aes) ** 2)
    )
    col_block_ode_alg = (
            np.tile(np.repeat(np.arange(2 * n_odes + 2 * n_aes), n_odes + n_aes), N - 1)
            + np.repeat(np.arange(N - 1) * (n_odes + n_aes), 2 * (n_odes + n_aes) ** 2)
    )

    range_eqxend = (n_odes + (N - 1) * (n_odes + n_aes), N * (n_odes + n_aes))
    row_algend = np.tile(np.arange(range_eqxend[0], range_eqxend[1]), n_odes + n_aes)
    col_algend = np.repeat(np.arange((N - 1) * (n_odes + n_aes), N * (n_odes + n_aes)), n_aes)
    rowis = np.concatenate((row_bcjac, row_block_ode_alg, row_algend))
    colis = np.concatenate((col_bcjac, col_block_ode_alg, col_algend))

    shape_jac = (
        (N - 1) * (n_odes + n_aes) + n_odes + n_aes, (N - 1) * (n_odes + n_aes) + n_odes + n_aes
    )

    id3d = repmat(np.eye(n_odes), (1, 1, N - 1))

    res_odeis = np.tile(np.arange(n_odes), N - 1) + np.repeat(np.arange(N - 1) * (n_odes + n_aes), n_odes) + n_odes

    res_algis = np.concatenate(
        (2 * n_odes + np.tile(np.arange(n_aes), N - 1) + np.repeat(np.arange(N - 1) * (n_odes + n_aes), n_aes),
         np.arange(n_odes + (N - 1) * (n_odes + n_aes), n_odes + (N - 1) * (n_odes + n_aes) + n_aes)
         )
    )
    return rowis, colis, shape_jac, id3d, res_odeis, res_algis


def approx_jac_res_values_deprecated(
        time: NDArray[float64],
        ode_sol: NDArray[float64],
        alg_sol: NDArray[float64],
        ocp: IndirectOCP,
        id3d: NDArray[float64]) -> NDArray[float64]:

    N = len(time) - 1  # number of time step - 1
    h = np.diff(time)  # vector of length of every time step
    ne, na = ode_sol.shape[0], alg_sol.shape[0]
    tmid = time[:-1] + h / 2.  # vector containing the midpoints of the time grid
    alg_mid = .5 * (alg_sol[:, 1:] + alg_sol[:, :-1])
    h3d = h.reshape((1, 1, N))
    h3d6 = h3d / 6.
    h3d8 = h3d / 8.
    rhs = ocp.ode(time, ode_sol, alg_sol)  # ODEs at time t
    ode_mid = (ode_sol[:, :-1] + ode_sol[:, 1:]) / 2. - h / 8. * (rhs[:, 1:] - rhs[:, :-1])

    # Calling AEs jacobian
    gx, gz = ocp.algjac(time, ode_sol, alg_sol)  # evaluate the jacobian of the algebraic equations
    # Calling ODEs jacobian
    fx, fz = ocp.odejac(time, ode_sol, alg_sol)  # evaluate the jacobian of the ODEs
    fxmid, fzmid = ocp.odejac(tmid, ode_mid, alg_mid)  # evaluate the ODEs jacobian at midpoints

    block_ode_alg = np.zeros((ne + na, 2 * ne + 2 * na, N))

    block_ode_alg[:ne, :ne, :] = (
            - id3d - h3d6 * (4. * matmul3d(fxmid, id3d / 2. + h3d8 * fx[:, :, :-1]) + fx[:, :, :-1])
    )

    block_ode_alg[:ne, ne:ne + na, :] = (
            - h3d6 * (4. * matmul3d(fxmid, h3d8 * fz[:, :, :-1]) + 2. * fzmid + fz[:, :, :-1])
    )

    block_ode_alg[:ne, ne + na: 2 * ne + na, :] = (
            id3d - h3d6 * (4. * matmul3d(fxmid, id3d / 2. - h3d8 * fx[:, :, 1:]) + fx[:, :, 1:])
    )

    block_ode_alg[:ne, 2 * ne + na:, :] = (
            - h3d6 * (4. * matmul3d(fxmid, - h3d8 * fz[:, :, 1:]) + 2. * fzmid + fz[:, :, 1:])
    )

    block_ode_alg[ne:, :ne, :] = gx[:, :, :-1]

    block_ode_alg[ne:, ne: ne + na, :] = gz[:, :, :-1]

    # Computing the Hessian of the boundary conditions
    jac_bc_x0, jac_bc_xend, jac_bc_z0, jac_bc_zend = ocp.bcjac(ode_sol[:, 0], ode_sol[:, -1], alg_sol[:, 0],
                                                               alg_sol[:, -1])
    vals = np.concatenate((
        np.hstack((jac_bc_x0, jac_bc_z0, jac_bc_xend, jac_bc_zend)).ravel(order="F"),
        block_ode_alg.ravel(order="F"),
        np.hstack((gx[:, :, -1], gz[:, :, -1])).ravel(order="F")
    ))
    return vals
"""