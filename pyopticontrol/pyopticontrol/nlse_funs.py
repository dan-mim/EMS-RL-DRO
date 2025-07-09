import numpy as np
from .UtilFunctions import repmat, matmul3d
from scipy import sparse
from scipy.sparse.linalg import spsolve as scipy_spsolve
from pypardiso import spsolve as pardiso_spsolve
from scipy.interpolate import interp1d
import sys


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


def res_colocation(time, ode_sol, alg_sol, alg_sol_mid, ocp):
    h = np.diff(time)  # vector containing the length of every time step of the time-grid

    tmid = (time[:-1] + time[1:]) / 2.  # Time collocation point
    rhs = ocp.ode(time, ode_sol, alg_sol)  # Evaluation of the diffential equations
    ode_sol_mid = (ode_sol[:, :-1] + ode_sol[:, 1:]) / 2. - (rhs[:, 1:] - rhs[:, :-1]) * h / 8.  # State colocation point
    rhsmid = ocp.ode(tmid, ode_sol_mid, alg_sol_mid)  # ODEs at colocation point

    alg = ocp.algeq(time, ode_sol, alg_sol)  # AEs at time t
    algmid = ocp.algeq(tmid, ode_sol_mid, alg_sol_mid)

    # Computation of the boundary condition
    bound_const = ocp.twobc(ode_sol[:, 0], ode_sol[:, -1], alg_sol[:, 0], alg_sol[:, -1])
    odes = ode_sol[:, 1:] - ode_sol[:, :-1] - (rhs[:, 1:] + 4. * rhsmid + rhs[:, :-1]) * h / 6.
    # Concatenation

    size_block_ode_alg_alg_mid = (time.size - 1) * (ode_sol.shape[0] + 2 * alg_sol.shape[0])
    res = np.empty((bound_const.size + size_block_ode_alg_alg_mid + alg_sol.shape[0],), dtype=np.float64)
    res[:bound_const.size] = bound_const
    res[bound_const.size: bound_const.size + size_block_ode_alg_alg_mid] = (
        np.vstack((odes, alg[:, :-1], algmid)).ravel(order="F")
    )
    res[bound_const.size + size_block_ode_alg_alg_mid:] = alg[:, -1]
    return res, rhsmid


def solve_newton(
        time, xp, z, zmid, ocp, Inn, rowis, colis, shape_jac, res_odeis, res_algis, res_tol=1e-3, max_iter=100,
        display=0, linear_solver=0, atol=1e-9, coeff_damping=2., max_probes=8, odes_control=True
):
        success = False
        iter = 0
        alpha = -1.
        h = np.diff(time)
        tol_odes = 2 / 3 * h * 5e-2 * res_tol
        while not success and iter < max_iter:
            res, rhsmid = res_colocation(time, xp, z, zmid, ocp)
            jac = jac_res_colocation(time, xp, z, zmid, ocp, Inn, rowis, colis, shape_jac)

            if display == 2:
                print('        * Newton Iteration # ' + str(iter))
                print('        * Initial error = ' + str(np.max(np.abs(res))))

            if linear_solver == 0:
                direction = scipy_spsolve(jac, res)
            else:
                direction = pardiso_spsolve(jac, res)

            xp_old, z_old, zmid_old, alpha_old = xp, z, zmid, alpha

            xp, z, zmid, cost, alpha, rhsmid, res, armflag = armijo(
                time, xp, z, zmid, ocp, direction, res, rhsmid, coeff_damping, max_probes
            )

            abs_res = np.abs(res)
            max_res = np.max(abs_res)

            if odes_control:
                tol = (tol_odes * (1. + np.abs(rhsmid))).ravel(order="F")
                success = (np.all(abs_res[res_odeis] <= tol) and np.all(abs_res[res_algis] <= atol)
                           and np.all(abs_res[:xp.shape[0]] <= atol))
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
                return xp, z, zmid, rhsmid, nlse_infos
            if (np.linalg.norm(xp - xp_old) == 0. and np.linalg.norm(z - z_old) == 0.
                    and np.linalg.norm(zmid - zmid_old) == 0. and alpha == alpha_old):
                break

            iter += 1
        ode_res = np.max(abs_res[res_odeis])
        ae_res = np.max(abs_res[res_algis])
        bc_res = np.max(abs_res[:xp.shape[0]])
        nlse_infos = NLSEInfos(success, iter, ode_res, max(tol_odes), ae_res, atol, bc_res, atol)
        return xp, z, zmid, rhsmid, nlse_infos


def armijo(time, xp0, z0, zmid0, OCP, direction, res0, rhsmid0, coeff_damping, max_probes):
    iarm = 0
    sigma1 = 1. / coeff_damping
    alpha = 1e-4
    armflag = True
    lbd, lbdm, lbdc = 1., 1., 1.
    dxp, dz, dzmid = get_solution_from_X(direction, xp0.shape[0], z0.shape[0])
    xpt = xp0 - lbd * dxp
    zt = z0 - lbd * dz
    zmidt = zmid0 - lbd * dzmid
    rest, rhsmidt = res_colocation(time, xpt, zt, zmidt, OCP)
    nft, nf0 = np.linalg.norm(rest), np.linalg.norm(res0)
    ff0, ffc = nf0 * nf0, nft * nft
    ffm = ffc
    best_xp, best_z, best_zmid, best_cost, best_lbd, best_rhsmid, best_res = xp0, z0, zmid0, ff0, lbd, rhsmid0, res0
    if ffc < best_cost:
        best_xp, best_z, best_zmid, best_cost, best_lbd, best_rhsmid, best_res = xpt, zt, zmidt, ffc, lbd, rhsmidt, rest
    while nft >= (1. - alpha * lbd) * nf0:
        if iarm == 0 or np.isinf(ffm) or np.isinf(ffc) or np.isnan(ffm) or np.isnan(ffc):
            lbd = sigma1 * lbd
        else:
            lbd = parab3p(lbdc, lbdm, ff0, ffc, ffm)
        xpt = xp0 - lbd * dxp
        zt = z0 - lbd * dz
        zmidt = zmid0 - lbd * dzmid
        lbdm = lbdc
        lbdc = lbd
        rest, rhsmidt = res_colocation(time, xpt, zt, zmidt, OCP)
        nft = np.linalg.norm(rest)
        ffm = ffc
        ffc = nft * nft
        if ffc < best_cost:
            best_xp, best_z, best_zmid, best_cost, best_lbd, best_rhsmid, best_res = xpt, zt, zmidt, ffc, lbd, rhsmidt, rest
        iarm += 1
        if iarm > max_probes:
            armflag = False
            return best_xp, best_z, best_zmid, best_cost, best_lbd, best_rhsmid, best_res, armflag
    return xpt, zt, zmidt, .5 * ffc, lbd, rhsmidt, rest, armflag


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


def _get_X_from_solution(xp, z, zmid):
    return np.concatenate((np.concatenate([xp[:, :-1], z[:, :-1], zmid], axis=0).ravel(order='F'),
                           xp[:, -1], z[:, -1]))


def get_solution_from_X(x, ne, na):
    nt = (x.size - ne - na) // (ne + 2 * na)
    xzzmid = x[:nt * (ne + 2 * na)].reshape((ne + 2 * na, nt), order="F")
    xp = np.zeros((ne, nt + 1))
    xp[:, :-1] = xzzmid[:ne, :]
    xp[:, -1] = x[nt*(ne + 2 * na):nt*(ne + 2 * na) + ne]
    z = np.zeros((na, nt+1))
    z[:, :-1] = xzzmid[ne:ne+na, :]
    z[:, -1] = x[nt * (ne + 2 * na)+ne:]
    zmid = xzzmid[ne+na:, :]
    return xp, z, zmid


def jac_res_colocation(time, xp, z, zmid, ocp, id3d, rowis, colis, shape_jac):
    values = jac_res_values(time, xp, z, zmid, ocp, id3d)
    non_zeros_indices = np.nonzero(values)
    jac = sparse.csc_matrix((values[non_zeros_indices], (rowis[non_zeros_indices], colis[non_zeros_indices])), shape_jac)
    return jac


def jac_res_values(time, xp, z, zmid, ocp, id3d):
    N = len(time) - 1  # number of time step - 1
    h = np.diff(time)  # vector of length of every time step
    n_ode, n_ae = xp.shape[0], z.shape[0]
    tmid = time[:-1] + h / 2.  # vector containing the midpoints of the time grid
    h3d = h.reshape((1, 1, N))
    h3d6 = h3d / 6.
    h3d8 = h3d / 8.
    rhs = ocp.ode(time, xp, z)  # ODEs at time t
    xmid = (xp[:, :-1] + xp[:, 1:]) / 2. - h / 8. * (rhs[:, 1:] - rhs[:, :-1])

    # Calling AEs jacobian
    gx, gz = ocp.algjac(time, xp, z)  # evaluate the jacobian of the algebraic equations
    gxmid, gzmid = ocp.algjac(tmid, xmid, zmid)
    # Calling ODEs jacobian
    fx, fz = ocp.odejac(time, xp, z)  # evaluate the jacobian of the ODEs
    fxmid, fzmid = ocp.odejac(tmid, xmid, zmid)  # evaluate the ODEs jacobian at midpoints

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
    jac_bc_x0, jac_bc_xend, jac_bc_z0, jac_bc_zend = ocp.bcjac(xp[:, 0], xp[:, -1], z[:, 0], z[:, -1])

    # Gathering values in a numpy array
    size_bc_jac = jac_bc_x0.size + jac_bc_xend.size + jac_bc_z0.size + jac_bc_zend.size
    vals = np.empty((size_bc_jac + block_ode_alg_algmid.size + gx[:, :, -1].size + gz[:, :, -1].size,),
                    dtype=np.float64)
    vals[:size_bc_jac] = np.hstack((jac_bc_x0, jac_bc_z0, jac_bc_xend, jac_bc_zend)).ravel(order="F")
    vals[size_bc_jac: size_bc_jac + block_ode_alg_algmid.size] = block_ode_alg_algmid.ravel(order="F")
    vals[size_bc_jac + block_ode_alg_algmid.size:] = np.hstack((gx[:, :, -1], gz[:, :, -1])).ravel(order="F")

    return vals


def estimate_rms(time, xp, z, zmid, ocp, atol=1e-9, restol=1e-3):
    h = np.diff(time)
    tmid = time[:-1] + h / 2.
    if zmid is not None:
        aggregated_z = np.concatenate(
            (np.reshape(np.vstack((z[:, :-1], zmid)), (z.shape[0], 2 * (len(time)-1)), order="F"),
             z[:, -1:]), axis=1)
        aggregated_time = np.sort(np.concatenate((time, tmid)))
        fun_interp_z = interp1d(x=aggregated_time, y=aggregated_z)
    else:
        zmid = .5 * (z[:, :-1] + z[:, 1:])
        fun_interp_z = interp1d(x=time, y=z)
    threshold = atol / restol
    lob4 = (1. + np.sqrt(3. / 7.)) / 2.
    lob2 = (1. - np.sqrt(3. / 7.)) / 2.
    lobw24 = 49. / 90.
    lobw3 = 32. / 45.
    rhs = ocp.ode(time, xp, z)

    xpmid = (xp[:, :-1] + xp[:, 1:]) / 2. - (rhs[:, 1:] - rhs[:, :-1]) * h / 8.
    rhsmid = ocp.ode(tmid, xpmid, zmid)
    colloc_res = xp[:, 1:] - xp[:, :-1] - (rhs[:, 1:] + 4. * rhsmid + rhs[:, :-1]) * h / 6.

    hscale = 1.5 / h
    temp = colloc_res * hscale / np.fmax(np.abs(rhsmid), threshold)
    res = lobw3 * np.sum(temp ** 2, axis=0)

    # Lobatto 2 points
    tlob = time[:-1] + lob2 * h
    xplob, derxp_lob = interp_hermite(h, xp, rhs, lob2)
    zlob = fun_interp_z(tlob)
    rhslob = ocp.ode(tlob, xplob, zlob)
    temp = (derxp_lob - rhslob) / np.fmax(np.abs(rhslob), threshold)
    res += lobw24 * np.sum(temp ** 2, axis=0)

    # Lobatto 4 points
    tlob = time[:-1] + lob4 * h
    xplob, derxp_lob = interp_hermite(h, xp, rhs, lob4)
    zlob = fun_interp_z(tlob)
    rhslob = ocp.ode(tlob, xplob, zlob)
    temp = (derxp_lob - rhslob) / np.fmax(np.abs(rhslob), threshold)
    res += lobw24 * np.sum(temp ** 2, axis=0)

    return np.sqrt(np.abs(h/2.) * res), fun_interp_z


def interp_hermite(h, xp, rhs, lob):
    scal = 1. / h
    slope = (xp[:, 1:] - xp[:, :-1]) * scal
    c = (3. * slope - 2. * rhs[:, :-1] - rhs[:, 1:]) * scal
    d = (rhs[:, :-1] + rhs[:, 1:] - 2. * slope) * scal ** 2

    scal = lob * h
    d *= scal
    xplob = ((d + c) * scal + rhs[:, :-1]) * scal + xp[:, :-1]
    derxp_lob = (3. * d + 2. * c) * scal + rhs[:, :-1]
    return xplob, derxp_lob


def create_new_xp_z_zmid(time, xp, z, fun_interp_z, residuals, ocp, restol=1e-3, coeff_reduce_mesh=.5, nmax=10000,
                         authorize_reduction=True):
    n = xp.shape[0]
    T = len(time)
    new_T = T + np.sum(np.where(residuals > restol, 1, 0)) + np.sum(np.where(residuals > 100. * restol, 1, 0))
    new_time = np.zeros((new_T,))
    new_time[0] = time[0]
    new_xp = np.zeros((n, new_T))
    rhs = ocp.ode(time, xp, z)
    ti = 0
    nti = 0
    new_xp[:, 0] = xp[:, 0]
    h = np.diff(time)
    while ti <= T-2:
        if residuals[ti] > restol:
            if residuals[ti] > 100. * restol:
                ni = 2
            else:
                ni = 1
            hi = h[ti] / (ni + 1)
            inds = np.arange(1, ni + 1)
            new_time[nti+1: nti + ni+1] = new_time[nti] + hi * inds
            xinterp = ntrp3h(new_time[nti: nti+ni], time[ti], xp[:, ti],
                             time[ti+1], xp[:, ti+1], rhs[:, ti], rhs[:, ti+1], ni)
            new_xp[:, nti+1:nti+ni+1] = xinterp
            nti += ni
        elif authorize_reduction and ti <= T-4 and max(residuals[ti:ti+3]) < restol * coeff_reduce_mesh:
            hnew = (time[ti+3] - time[ti]) / 2.
            pred_res = residuals[ti] / (h[ti] / hnew) ** 3.5
            pred_res = max(pred_res, residuals[ti+1] / ((time[ti+2] - time[ti+1]) / hnew) ** 3.5)
            pred_res = max(pred_res, residuals[ti+2] / ((time[ti+3] - time[ti+2]) / hnew) ** 3.5)
            if pred_res < restol * coeff_reduce_mesh:
                new_time[nti + 1] = new_time[nti] + hnew
                xinterp = ntrp3h(new_time[nti + 1], time[ti], xp[:, ti], time[ti + 3], xp[:, ti + 3], rhs[:, ti],
                                 rhs[:, ti + 3], 1)
                new_xp[:, nti + 1] = xinterp[:, 0]
                nti += 1
                ti += 2
        new_time[nti + 1] = time[ti + 1]
        new_xp[:, nti + 1] = xp[:, ti + 1]
        nti += 1
        ti += 1
    time = new_time[:nti+1]
    xp = new_xp[:, :nti+1]
    z = fun_interp_z(time)
    tmid = time[:-1] + np.diff(time) / 2.
    zmid = fun_interp_z(tmid)
    too_much_nodes = len(time) > nmax
    return time, xp, z, zmid, too_much_nodes


def ntrp3h(newtime, tk, xk, tkp1, xkp1, rhsk, rhskp1, ni):
    h = tkp1 - tk
    slope = (xkp1 - xk) / h
    c = 3. * slope - 2. * rhsk - rhskp1
    d = rhsk + rhskp1 - 2. * slope
    s = (newtime - tk) / h
    s2 = s ** 2
    s3 = s * s2
    xinterp = np.zeros((len(xk), ni))
    if ni == 1:
        xinterp[:, 0] = xk + h * (d * s3 + c * s2 + rhsk * s)
    else:
        for col in range(ni):
            xinterp[:, col] = xk + h * (d * s3[col] + c * s2[col] + rhsk * s[col])
    return xinterp
