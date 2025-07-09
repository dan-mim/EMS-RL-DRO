import numpy as np
from .UtilFunctions import repmat, matmul3d, IndirectOCP
from scipy import sparse
from scipy.sparse.linalg import spsolve as scipy_spsolve
from pypardiso import spsolve as pardiso_spsolve
import sys
from numpy.typing import NDArray
from typing import Optional
from numpy import float64


class NLSEInfos:
    def __init__(self, success, iter, ode_residual, ode_tol, ae_residual, ae_tol, bc_residual, bc_tol, jac_reuse=0):
        self.success = success
        self.iter = iter
        self.ode_residual = ode_residual
        self.ode_tol = ode_tol
        self.ae_residual = ae_residual
        self.ae_tol = ae_tol
        self.bc_residual = bc_residual
        self.bc_tol = bc_tol
        self.jac_reuse = jac_reuse

    def __str__(self):
        string_nlse = ("NSLE infos : \n"
                  + "    succes = " + str(self.success) + "\n"
                  + "    iter = " + str(self.iter) + "\n"
                  + "    ode_residual = " + str(self.ode_residual) + " for ode_tol = " + str(self.ode_tol) + "\n"
                  + "    ae_residual = " + str(self.ae_residual) + " for ae_tol = " + str(self.ae_tol) + "\n"
                  + "    bc_residual = " + str(self.bc_residual) + " for bc_tol = " + str(self.bc_tol) + "\n"
                       + "    jac reuse number = " + str(self.jac_reuse))
        return string_nlse


def res_colocation(time: NDArray[float64],
                   ode_sol: NDArray[float64],
                   alg_sol: NDArray[float64],
                   alg_sol_mid: NDArray[float64],
                   params: NDArray[float64],
                   ocp: IndirectOCP) -> tuple[NDArray[float64], NDArray[float64]]:
    h = np.diff(time)  # vector containing the length of every time step of the time-grid

    tmid = (time[:-1] + time[1:]) / 2.  # Time collocation point
    rhs = ocp.ode(time, ode_sol, alg_sol)  # Evaluation of the diffential equations
    xmid = (ode_sol[:, :-1] + ode_sol[:, 1:]) / 2. - (rhs[:, 1:] - rhs[:, :-1]) * h / 8.  # State colocation point
    rhsmid = ocp.ode(tmid, xmid, alg_sol_mid)  # ODEs at colocation point

    alg = ocp.algeq(time, ode_sol, alg_sol)  # AEs at time t
    algmid = ocp.algeq(tmid, xmid, alg_sol_mid)

    # Computation of the boundary condition
    bound_const = ocp.twobc(ode_sol[:, 0], ode_sol[:, -1], alg_sol[:, 0], alg_sol[:, -1], params)
    odes = ode_sol[:, 1:] - ode_sol[:, :-1] - (rhs[:, 1:] + 4. * rhsmid + rhs[:, :-1]) * h / 6.
    # Concatenation

    res = np.empty((bound_const.size + odes.size + alg.size + algmid.size,), dtype=np.float64)
    res[:bound_const.size] = bound_const
    indice_ode_alg = bound_const.size + (ode_sol.shape[0] + 2 * alg_sol.shape[0]) * (len(time) - 1)
    res[bound_const.size: indice_ode_alg] = np.vstack((odes, alg[:, :-1], algmid)).ravel(order="F")
    res[indice_ode_alg:] = alg[:, -1]
    return res, rhsmid


def solve_newton_params(
        time: NDArray[float64],
        ode_sol: NDArray[float64],
        alg_sol: NDArray[float64],
        alg_sol_mid: NDArray[float64],
        params: NDArray[float64],
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
        odes_control: Optional[bool] = True
) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NDArray[float64], NLSEInfos]:
        success = False
        iter = 0
        alpha = -1.
        h = np.diff(time)
        tol_odes = 2 / 3 * h * 5e-2 * res_tol
        while not success and iter < max_iter:
            res, rhsmid = res_colocation(time, ode_sol, alg_sol, alg_sol_mid, params, ocp)
            jac = jac_res_colocation(time, ode_sol, alg_sol, alg_sol_mid, params, ocp, id3d, rowis, colis, shape_jac)

            if display == 2:
                print('        * Newton Iteration # ' + str(iter))
                print('        * Initial error = ' + str(np.max(np.abs(res))))

            if linear_solver == 0:
                direction = scipy_spsolve(jac, res)
            else:
                direction = pardiso_spsolve(jac, res)

            xp_old, z_old, zmid_old, params_old, alpha_old = ode_sol, alg_sol, alg_sol_mid, params, alpha

            ode_sol, alg_sol, alg_sol_mid, params, cost, alpha, rhsmid, res, armflag = armijo(
                time, ode_sol, alg_sol, alg_sol_mid, params, ocp, direction, res, rhsmid, coeff_damping, max_probes
            )

            abs_res = np.abs(res)
            max_res = np.max(abs_res)

            if odes_control:
                tol = (tol_odes * (1. + np.abs(rhsmid))).ravel(order="F")
                success = (np.all(abs_res[res_odeis] <= tol) and np.all(abs_res[res_algis] <= atol)
                           and np.all(abs_res[:ode_sol.shape[0]] <= atol))
            else:
                tol = atol
                success = np.all(abs_res <= atol)

            if success and display == 2:
                print('           Success damped Newton step = ' + str(max_res) + ' alpha = ' + str(
                    alpha) + '; iter = ' + str(iter))
                sys.stdout.flush()
            elif display == 2:
                print(
                    '                  Newton step = ' + str(max_res) + ' alpha = ' + str(alpha) +
                    '; iter = ' + str(iter))
                sys.stdout.flush()
            if np.any(np.isnan(res)) or np.any(np.isinf(res)):
                nlse_infos = NLSEInfos(success, iter, None, tol, None, atol, None, atol)
                return ode_sol, alg_sol, alg_sol_mid, params, rhsmid, nlse_infos
            if (np.linalg.norm(ode_sol - xp_old) == 0. and np.linalg.norm(alg_sol - z_old) == 0.
                    and np.linalg.norm(alg_sol_mid - zmid_old) == 0. and np.linalg.norm(params_old - params)
                    and alpha == alpha_old):
                break

            iter += 1
        ode_res = np.max(abs_res[res_odeis])
        ae_res = np.max(abs_res[res_algis])
        bc_res = np.max(abs_res[:ode_sol.shape[0] + params.size])
        nlse_infos = NLSEInfos(success, iter, ode_res, max(tol_odes), ae_res, atol, bc_res, atol)
        return ode_sol, alg_sol, alg_sol_mid, params, rhsmid, nlse_infos


def armijo(
        time: NDArray[float64],
        ode_sol_0: NDArray[float64],
        alg_sol_0: NDArray[float64],
        alg_sol_mid_0: NDArray[float64],
        params_0: NDArray[float64],
        OCP: IndirectOCP,
        direction: NDArray[float64],
        res0: NDArray[float64],
        rhsmid0: NDArray[float64],
        coeff_damping: float,
        max_probes: int):
    iarm = 0
    sigma1 = 1. / coeff_damping
    alpha = 1e-4
    armflag = True
    lbd, lbdm, lbdc = 1., 1., 1.
    d_ode_sol, d_alg_sol, d_alg_sol_mid, d_params = get_solution_from_X(
        direction, ode_sol_0.shape[0], alg_sol_0.shape[0], params_0.size
    )
    ode_sol_t = ode_sol_0 - lbd * d_ode_sol
    alg_sol_t = alg_sol_0 - lbd * d_alg_sol
    alg_sol_mid_t = alg_sol_mid_0 - lbd * d_alg_sol_mid
    params_t = params_0 - lbd * d_params
    rest, rhsmidt = res_colocation(time, ode_sol_t, alg_sol_t, alg_sol_mid_t, params_t, OCP)
    nft, nf0 = np.linalg.norm(rest), np.linalg.norm(res0)
    ff0, ffc = nf0 * nf0, nft * nft
    ffm = ffc
    best_ode_sol, best_alg_sol, best_alg_sol_mid, best_params, best_cost, best_lbd, best_rhsmid, best_res = (
        ode_sol_0, alg_sol_0, alg_sol_mid_0, params_0, ff0, lbd, rhsmid0, res0
    )
    if ffc < best_cost:
        best_ode_sol, best_alg_sol, best_alg_sol_mid, best_params, best_cost, best_lbd, best_rhsmid, best_res = (
            ode_sol_t, alg_sol_t, alg_sol_mid_t, params_t, ffc, lbd, rhsmidt, rest
        )
    while nft >= (1. - alpha * lbd) * nf0:
        if iarm == 0 or np.isinf(ffm) or np.isinf(ffc) or np.isnan(ffm) or np.isnan(ffc):
            lbd = sigma1 * lbd
        else:
            lbd = parab3p(lbdc, lbdm, ff0, ffc, ffm)
        ode_sol_t = ode_sol_0 - lbd * d_ode_sol
        alg_sol_t = alg_sol_0 - lbd * d_alg_sol
        alg_sol_mid_t = alg_sol_mid_0 - lbd * d_alg_sol_mid
        params_t = params_0 - lbd * d_params
        lbdm = lbdc
        lbdc = lbd
        rest, rhsmidt = res_colocation(time, ode_sol_t, alg_sol_t, alg_sol_mid_t, params_t, OCP)
        nft = np.linalg.norm(rest)
        ffm = ffc
        ffc = nft * nft
        if ffc < best_cost:
            best_ode_sol, best_alg_sol, best_alg_sol_mid, best_params, best_cost, best_lbd, best_rhsmid, best_res = (
                ode_sol_t, alg_sol_t, alg_sol_mid_t, params_t, ffc, lbd, rhsmidt, rest
            )
        iarm += 1
        if iarm > max_probes:
            armflag = False
            return best_ode_sol, best_alg_sol, best_alg_sol_mid, best_params, best_cost, best_lbd, best_rhsmid, best_res, armflag
    return ode_sol_t, alg_sol_t, alg_sol_mid_t, params_t, .5 * ffc, lbd, rhsmidt, rest, armflag


def parab3p(lbdc, lbdm, ff0, ffc, ffm):
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


def _get_X_from_solution(ode_sol, alg_sol, alg_sol_mid, params):
    xvec = np.empty((ode_sol.size + alg_sol.size, + alg_sol_mid.size + params.size,), dtype=float64)

    len_time = ode_sol.shape[1]

    indice = (len_time - 1) * (ode_sol.shape[0] + 2 * alg_sol.shape[0])

    xvec[:indice] = np.vstack(
        (ode_sol[:, :-1], alg_sol[:, :-1], alg_sol_mid)).ravel(order='F')

    xvec[indice:] = np.concatenate((ode_sol[:, -1], alg_sol[:, -1], params))
    return xvec


def get_solution_from_X(x, n_ode, n_ae, n_params):
    len_time = (x.size - n_params + n_ae) // (n_ode + 2 * n_ae)
    ode_sol, alg_sol = np.empty((n_ode, len_time), dtype=float64), np.empty((n_ae, len_time), dtype=float64)
    indice = (len_time - 1) * (n_ode + 2 * n_ae)
    ode_alg_alg_mid = x[: indice].reshape((n_ode + 2 * n_ae, len_time - 1), order="F")
    ode_sol[:, :-1] = ode_alg_alg_mid[:n_ode, :]
    alg_sol[:, :-1] = ode_alg_alg_mid[n_ode: n_ode + n_ae, :]
    alg_sol_mid = ode_alg_alg_mid[n_ode + n_ae:]
    ode_sol[:, -1] = x[indice: indice + n_ode]
    alg_sol[:, -1] = x[indice + n_ode: indice + n_ode + n_ae]
    params = x[indice + n_ode + n_ae:]

    return ode_sol, alg_sol, alg_sol_mid, params


def jac_res_colocation(time, ode_sol, alg_sol, alg_sol_mid, params, ocp, id3d, rowis, colis, shape_jac):
    values = jac_res_values(time, ode_sol, alg_sol, alg_sol_mid, params, ocp, id3d)
    non_zeros_indices = np.nonzero(values)
    jac = sparse.csc_matrix((values[non_zeros_indices], (rowis[non_zeros_indices], colis[non_zeros_indices])), shape_jac)
    return jac


def jac_res_values(time, ode_sol, alg_sol, alg_sol_mid, params, ocp, id3d):
    N = len(time) - 1  # number of time step - 1
    h = np.diff(time)  # vector of length of every time step
    n_ode, n_ae = ode_sol.shape[0], alg_sol.shape[0]
    tmid = time[:-1] + h / 2.  # vector containing the midpoints of the time grid
    h3d = h.reshape((1, 1, N))
    h3d6 = h3d / 6.
    h3d8 = h3d / 8.
    rhs = ocp.ode(time, ode_sol, alg_sol)  # ODEs at time t
    xmid = (ode_sol[:, :-1] + ode_sol[:, 1:]) / 2. - h / 8. * (rhs[:, 1:] - rhs[:, :-1])

    # Calling AEs jacobian
    gx, gz = ocp.algjac(time, ode_sol, alg_sol)  # evaluate the jacobian of the algebraic equations
    gxmid, gzmid = ocp.algjac(tmid, xmid, alg_sol_mid)
    # Calling ODEs jacobian
    fx, fz = ocp.odejac(time, ode_sol, alg_sol)  # evaluate the jacobian of the ODEs
    fxmid, fzmid = ocp.odejac(tmid, xmid, alg_sol_mid)  # evaluate the ODEs jacobian at midpoints

    dxmid_dxk = id3d / 2. + h3d8 * fx[:, :, :-1]
    dxmid_dzk = h3d8 * fz[:, :, :-1]
    dxmid_dxkp1 = id3d / 2. - h3d8 * fx[:, :, 1:]
    dxmid_dzkp1 = - h3d8 * fz[:, :, 1:]

    block_ode_alg_algmid = np.zeros((n_ode + 2 * n_ae, 2 * n_ode + 3 * n_ae, N))
    # derivative rhs wrt xk
    block_ode_alg_algmid[:n_ode, :n_ode, :] = - id3d - h3d6 * (4. * matmul3d(fxmid, dxmid_dxk) + fx[:, :, :-1])
    # derivative rhs wrt zk
    block_ode_alg_algmid[:n_ode, n_ode:n_ode + n_ae, :] = - h3d6 * (4. * matmul3d(fxmid, dxmid_dzk) + fz[:, :, :-1])
    # derivative rhs wrt alg_sol_mid
    block_ode_alg_algmid[:n_ode, n_ode + n_ae:n_ode + 2 * n_ae, :] = - h3d6 * 4. * fzmid
    # derivative rhs wrt xk+1
    block_ode_alg_algmid[:n_ode, n_ode + 2 * n_ae: 2 * (n_ode + n_ae), :] = (
            id3d - h3d6 * (4. * matmul3d(fxmid, dxmid_dxkp1) + fx[:, :, 1:])
    )
    # derivative rhs wrt zk+1
    block_ode_alg_algmid[:n_ode, 2 * (n_ode + n_ae):, :] = - h3d6 * (4. * matmul3d(fxmid, dxmid_dzkp1) + fz[:, :, 1:])

    # derivative alg wrt xk
    block_ode_alg_algmid[n_ode:n_ode + n_ae, :n_ode, :] = gx[:, :, :-1]
    # derivative alg wrt zk
    block_ode_alg_algmid[n_ode:n_ode + n_ae, n_ode: n_ode + n_ae, :] = gz[:, :, :-1]

    # derivative algmid wrt xk
    block_ode_alg_algmid[n_ode + n_ae:, :n_ode, :] = matmul3d(gxmid, dxmid_dxk)
    # derivative algmid wrt zk
    block_ode_alg_algmid[n_ode + n_ae:, n_ode: n_ode + n_ae, :] = matmul3d(gxmid, dxmid_dzk)
    # derivative algmid wrt alg_sol_mid
    block_ode_alg_algmid[n_ode + n_ae:, n_ode + n_ae: n_ode + 2 * n_ae, :] = gzmid
    # derivative algmid wrt xk+1
    block_ode_alg_algmid[n_ode + n_ae:, n_ode + 2 * n_ae: 2 * (n_ode + n_ae), :] = matmul3d(gxmid, dxmid_dxkp1)
    # derivative algmid wrt zk+1
    block_ode_alg_algmid[n_ode + n_ae:, 2 * (n_ode + n_ae):, :] = matmul3d(gxmid, dxmid_dzkp1)

    # Computing the Hessian of the boundary conditions
    jac_bc_x0, jac_bc_xend, jac_bc_z0, jac_bc_zend, jac_bc_params = (
        ocp.bcjac(ode_sol[:, 0], ode_sol[:, -1], alg_sol[:, 0], alg_sol[:, -1], params)
    )

    # Gathering values in a numpy array
    size_jac = jac_bc_x0.size + jac_bc_xend.size + jac_bc_z0.size + jac_bc_zend.size + jac_bc_params.size
    vals = np.empty((size_jac + block_ode_alg_algmid.size + gx[:, :, -1].size + gz[:, :, -1].size,),
                    dtype=float64
                    )
    vals[: size_jac] = np.hstack((jac_bc_x0, jac_bc_z0, jac_bc_xend, jac_bc_zend, jac_bc_params)).ravel(order="F")
    vals[size_jac: size_jac + block_ode_alg_algmid.size] = block_ode_alg_algmid.ravel(order="F")
    vals[size_jac + block_ode_alg_algmid.size:] = np.hstack((gx[:, :, -1], gz[:, :, -1])).ravel(order="F")
    return vals


def row_col_jac_indices(time: NDArray[float64], n_ode: int, n_ae: int, n_params: Optional[int] = 0):
    N = len(time)

    # indices for bcjac
    row_bcjac = np.tile(np.arange(n_ode + n_params), 2 * (n_ode + n_ae) + n_params)

    col_bcjac = np.concatenate((
        np.repeat(np.arange(n_ode + n_ae), n_ode + n_params),
        np.repeat(np.arange((N - 1) * (n_ode + 2 * n_ae), N * n_ode + (2 * N - 1) * n_ae + n_params), n_ode + n_params)
    ))

    offset_row = n_ode + n_params

    # indices for block_ode_alg_algmid
    row_block_ode_alg_algmid = (
            offset_row + np.tile(np.tile(np.arange(n_ode + 2 * n_ae), 2 * n_ode + 3 * n_ae), N - 1)
            + np.repeat(np.arange(N - 1) * (n_ode + 2 * n_ae), (n_ode + 2 * n_ae) * (2 * n_ode + 3 * n_ae))
    )

    col_block_ode_alg_algmid = (
            np.tile(np.repeat(np.arange(2 * n_ode + 3 * n_ae), n_ode + 2 * n_ae), N - 1)
            + np.repeat(np.arange(N - 1) * (n_ode + 2 * n_ae), (n_ode + 2 * n_ae) * (2 * n_ode + 3 * n_ae))
    )

    range_rows_alg_end = (n_ode + n_params + (N - 1) * (n_ode + 2 * n_ae),
                          n_ode + n_params + (N - 1) * (n_ode + 2 * n_ae) + n_ae
                          )
    row_algend = np.tile(np.arange(range_rows_alg_end[0], range_rows_alg_end[1]), n_ode + n_ae)
    col_algend = np.repeat(np.arange((N - 1) * (n_ode + 2 * n_ae), (N - 1) * (n_ode + 2 * n_ae) + n_ode + n_ae), n_ae)
    rowis = np.concatenate((row_bcjac, row_block_ode_alg_algmid, row_algend))
    colis = np.concatenate((col_bcjac, col_block_ode_alg_algmid, col_algend))

    shape_jac = ((N - 1) * (n_ode + 2 * n_ae) + n_ode + n_ae + n_params,
                 (N - 1) * (n_ode + 2 * n_ae) + n_ode + n_ae + n_params)

    id3d = repmat(np.eye(n_ode), (1, 1, N - 1))

    res_odeis = np.tile(np.arange(n_ode), N - 1) + np.repeat(np.arange(N - 1) * (n_ode + 2 * n_ae), n_ode) + offset_row

    res_algis = np.concatenate(
        (n_ode + offset_row + np.tile(np.arange(2 * n_ae), N - 1) + np.repeat(np.arange(N - 1) * (n_ode + 2 * n_ae), 2 * n_ae),
         np.arange(offset_row + (N - 1) * (n_ode + 2 * n_ae), n_ode + (N - 1) * (n_ode + 2 * n_ae) + n_ae))
    )
    return rowis, colis, shape_jac, id3d, res_odeis, res_algis

