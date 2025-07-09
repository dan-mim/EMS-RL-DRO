import numpy as np
from .approximate_nlse_funs import approx_solve_newton
from .approximate_nlse_funs_params import approx_solve_newton_params, approx_row_col_jac_indices
from .nlse_funs_params import solve_newton_params, row_col_jac_indices
from .nlse_funs import solve_newton, create_new_xp_z_zmid, estimate_rms, NLSEInfos
from .UtilFunctions import IndirectOCP, OptiSol
import sys
from numpy.typing import NDArray
from numpy import float64
from typing import Union
from functools import partial
__all__ = ["BVPDAE"]

class DAEInfos:
    """
    This class contains the BVPDAE's numerical solving informations
    :param success: Boolean indicating if the BVPDAE is solved with required discretization residual error
    :param rms_residual: numpy array with ODEs discretization residual error
    """
    def __init__(self, success: bool, rms_residual: float):
        self.success = success
        self.rms_residual = rms_residual


class Infos:
    """
    This class contains the BVPDAE's numerical solving informations
    :param success: Boolean indicating if the problems is successfully solved
    :param NLSE_infos: class whose attributes gather informations on the Non Linear Equations Solver
    :param DAE_infos: classe whose attributes gather informations on the mesh refinement procedure
    """
    def __init__(self, success: bool, NLSE_infos: NLSEInfos, DAE_infos: Union[DAEInfos, None]):

        self.success = success
        self.NLSE_infos = NLSE_infos
        self.DAE_infos = DAE_infos

    def __str__(self):
        print("##########################")
        print(" BVPDAE Infos")
        print("   Global Success = ", self.success)
        print("          ")
        print(" Non Linear Equations Solver infos")
        print(self.NLSE_infos)
        print("  ")


class BVPDAE:
    """
This is a Two Point Boundary Value Problem solver consisting in ODEs coupled with DAEs. The parameterization of the solver is done through an options dictionnary given at instanciation containing the following items
    Attributes:
    -----------
    - **display**:
        Verbosity of the algorithm (ranging from 0 to 2). Default is 0
    - **check_jacobian**
        Boolean when True checks provided Jacobian. Default is False
    - **exact_prb**:
        Boolean when True algebraic variables are computed exactly at collocation points. If False, alg_sol_mid is interpolated.
    - **control_odes_error**:
        Boolean when True time mesh is adapted to control residual error. Default is False
    - **max_mesh_point**:
        Maximum length of time array. Default is 1e5
    - **res_tol**:
        residual relative error on ODEs. Default is 1e-3
    - **no_mesh_reduction**:
        Boolean when True mesh modification can only add points., Default is False
    - **threshold_mesh_reduction**:
        Real value in (0,1] such that mesh points are removed if ODE's residual error is <= threshold_mesh_reduction * res_tol on three consecutive time interval.
    - **max_NLSE_iter**:
        Maximum number of iteration for solving the NLSE. Default is 100 in case of mesh refinement 500 otherwise
    - **max_rms_control_iter**:
        Maximum number of mesh modification iterations. Default is 1000
    - **newton_tol**:
        relative tolerance of Newton scheme. Default is 1e-8
    - **abs_tol**:
        residual absolute error on ODEs. Default is 1e-7
    - **coeff_damping**:
        Damping coefficient of damping Newton-Raphson scheme. Default is 2.
    - **linear_solver**:
        Linear solver to be chosen among lu or umfpack. Default is umfpack
    - **newton_tol**:
        relative tolerance of Newton scheme. Default is 1e-6
    - **coeff_damping**:
        Damping coefficient of damping Newton scheme. Default is 2.
    - **max_probes**:
        Maximum number of damping operations for armijo step selection. Default is 6
    """
    def __init__(self, **kwargs):
        self.display = kwargs.get("display", 0)
        self.check_jacobian = kwargs.get("check_jacobian", False)
        self.exact_prb = kwargs.get("exact_prb", False)

        self.control_odes_error = kwargs.get("control_odes_error", False)
        self.max_mesh_point = kwargs.get("max_mesh_point", 100000)
        self.res_tol = kwargs.get("res_tol", 1e-3)
        self.no_mesh_reduction = kwargs.get("no_mesh_reduction", False)

        if self.no_mesh_reduction:
            self.max_NLSE_iter = kwargs.get("max_NLSE_iter", 500)
        else:
            self.max_NLSE_iter = kwargs.get("max_NLSE_iter", 100)

        self.threshold_mesh_reduction = min(1., kwargs.get("threshold_mesh_reduction", .1))
        self.max_rms_control_iter = kwargs.get("max_rms_control_iter", 1000)

        self.newton_tol = kwargs.get("newton_tol", 1e-8)
        self.abs_tol = kwargs.get("abs_tol", 1e-7)
        self.coeff_damping = kwargs.get("coeff_damping", 2.)
        self.max_probes = kwargs.get("max_probes", 6)

        linear_solver = kwargs.get("linear_solver", "scipy")
        if linear_solver not in ["pardiso", "scipy"]:
            linear_solver = "scipy"
        if linear_solver == "scipy":
            self.linear_solver = 0
        else:
            self.linear_solver = 1

    def solve(self, optisol: OptiSol, ocp: IndirectOCP) -> tuple[OptiSol, Infos]:
        """
        Solve the Optimal Control Problem from OCP initializing the algorithm with an OptiSol object containing initial
        solutions of the OCP.

        :param optisol: OptiSol object
        :param ocp: IndirectOCP
        :return: OptiSol , Infos.
        """
        time, ode_sol, alg_sol, alg_sol_mid, params = (
            optisol.time, optisol.ode_sol, optisol.alg_sol, optisol.alg_sol_mid, optisol.params
        )
        if self.check_jacobian:
            self.test_jacobian(time, ode_sol, alg_sol, params, ocp)

        if not self.exact_prb:
            return self._solve_approximate_problem(time, ode_sol, alg_sol, params, ocp)
        else:
            return self._solve(time, ode_sol, alg_sol, alg_sol_mid, params, ocp)

    def _solve_approximate_problem(self,
                                   time: NDArray[float64],
                                   ode_sol: NDArray[float64],
                                   alg_sol: NDArray[float64],
                                   params: Union[NDArray[float64], None],
                                   ocp: IndirectOCP) -> tuple[OptiSol, Infos]:
        """
        Solve the Optimal Control Problem from OCP initializing the algorithm with (time, ode_sol, alg_sol, params) values

        :param time: 1d-array containing time stamps.
        :param ode_sol:  2d-array containing the solutions of ODEs (x(t), p(t))
        :param alg_sol: : 2d-array containing the solutions of AEs
        :params params: 1d-array containing the additional variables to compute
        :param ocp: IndirectOCP
        :return: OptiSol , Infos.
        """

        include_params = params is not None

        dae_infos = None
        n_ode, n_ae = ode_sol.shape[0], alg_sol.shape[0]
        n_params = params.size if include_params else 0
        rowis, colis, shape_jac, Inn, res_odeis, res_algis = approx_row_col_jac_indices(time, n_ode, n_ae, n_params)

        rms_control_iter = 0
        # Begining of the iterative solving of the TPBVP
        while rms_control_iter < self.max_rms_control_iter:
            # Computing the solution of grad_dae = 0 through a Newton scheme
            ode_sol_new, alg_sol_new, params_new, rhs_mid_new, nlse_infos = self._approx_newton_raphson(
                time, ode_sol, alg_sol, params, ocp, Inn, rowis, colis, shape_jac, res_odeis, res_algis
            )

            if nlse_infos.ode_residual is None or nlse_infos.ae_residual is None or nlse_infos.bc_residual is None:
                infos = Infos(False, nlse_infos, dae_infos)
                break

            ode_sol, alg_sol, rhs_mid = ode_sol_new, alg_sol_new, rhs_mid_new
            params = params_new if include_params else None

            if not self.control_odes_error:
                infos = Infos(nlse_infos.success, nlse_infos, dae_infos)
                break

            rms_res, fun_interp_z = estimate_rms(
                time, ode_sol, alg_sol, None, ocp, atol=self.abs_tol, restol=self.res_tol
            )
            max_rms_res = np.max(rms_res)

            if np.isnan(max_rms_res):
                dae_infos = DAEInfos(False, rms_res)
                infos = Infos(False, nlse_infos, dae_infos)
                break

            self._print('     # Residual error = ' + str(max_rms_res) + ' with N = ' + str(len(time)), 1)

            if max_rms_res < self.res_tol:
                dae_infos = DAEInfos(True, rms_res)
                infos = Infos(True, nlse_infos, dae_infos)
                break

            new_time, new_ode_sol, new_alg_sol, new_alg_sol_mid, too_much_nodes = create_new_xp_z_zmid(
                time, ode_sol, alg_sol, fun_interp_z, rms_res, ocp, restol=self.res_tol,
                coeff_reduce_mesh=self.threshold_mesh_reduction, nmax=self.max_mesh_point,
                authorize_reduction=not self.no_mesh_reduction
            )

            if too_much_nodes:
                dae_infos = DAEInfos(False, rms_res)
                infos = Infos(False, nlse_infos, dae_infos)
                break

            time, ode_sol, alg_sol, alg_sol_mid = new_time, new_ode_sol, new_alg_sol, new_alg_sol_mid

            rowis, colis, shape_jac, Inn, res_odeis, res_algis = approx_row_col_jac_indices(time, n_ode, n_ae, n_params)

            rms_control_iter += 1

        self._print("Solving Complete", 1)

        optisol = OptiSol(time=time, ode_sol=ode_sol, alg_sol=alg_sol, params=params)
        return optisol, infos

    def _approx_newton_raphson(self,
                               time: NDArray[float64],
                               ode_sol: NDArray[float64],
                               alg_sol: NDArray[float64],
                               params: Union[NDArray[float64], None],
                               ocp: IndirectOCP,
                               Inn: NDArray[float64],
                               rowis: NDArray[np.int64],
                               colis: NDArray[np.int64],
                               shape_jac: tuple[int, int],
                               res_odeis: NDArray[np.int64],
                               res_algis: NDArray[np.int64]) -> (
            tuple[NDArray[float64], NDArray[float64], Union[NDArray[float64], None], NDArray[float64], NLSEInfos]):

        if params is None:
            ode_sol_new, alg_sol_new, rhs_mid_new, nlse_infos = approx_solve_newton(
                time, ode_sol, alg_sol, ocp, Inn, rowis, colis, shape_jac, res_odeis, res_algis,
                res_tol=self.res_tol, max_iter=self.max_NLSE_iter, display=self.display,
                linear_solver=self.linear_solver, atol=self.newton_tol, coeff_damping=self.coeff_damping,
                max_probes=self.max_probes, odes_control=self.control_odes_error)
            params_new = None
        else:
            ode_sol_new, alg_sol_new, params_new, rhs_mid_new, nlse_infos = approx_solve_newton_params(
                time, ode_sol, alg_sol, params, ocp, Inn, rowis, colis, shape_jac, res_odeis, res_algis,
                res_tol=self.res_tol, max_iter=self.max_NLSE_iter, display=self.display,
                linear_solver=self.linear_solver, atol=self.newton_tol, coeff_damping=self.coeff_damping,
                max_probes=self.max_probes, odes_control=self.control_odes_error)
        return ode_sol_new, alg_sol_new, params_new, rhs_mid_new, nlse_infos

    def _solve(self,
               time: NDArray[float64],
               ode_sol: NDArray[float64],
               alg_sol: NDArray[float64],
               alg_sol_mid: NDArray[float64],
               params: Union[NDArray[float64], None],
               ocp: IndirectOCP) -> tuple[OptiSol, Infos]:
        """
        Solve the Optimal Control Problem from OCP initializing the algorithm with (time, ode_sol, z) values

        :param time: 1d-array containing time stamps.
        :param ode_sol:  2d-array containing the solutions of ODEs (x(t), p(t))
        :param alg_sol: : 2d-array containing the solutions of AEs
        :param ocp: Object representing an indirect optimal control problem
        :return: time, ode_sol, z, done.
        """

        include_params = params is not None

        dae_infos = None
        n_ode, n_ae = ode_sol.shape[0], alg_sol.shape[0]
        n_params = params.size if include_params else 0

        rowis, colis, shape_jac, Inn, res_odeis, res_algis = row_col_jac_indices(time, n_ode, n_ae, n_params)

        if alg_sol_mid is None:
            alg_sol_mid = .5 * (alg_sol[:, :-1] + alg_sol[:, 1:])

        rms_control_iter = 0
        # Begining of the iterative solving of the TPBVP

        while rms_control_iter < self.max_rms_control_iter:

            new_ode_sol, new_alg_sol, new_alg_sol_mid, new_params, new_rhs_mid, nlse_infos = self._newton_raphson(
                time, ode_sol, alg_sol, alg_sol_mid, params, ocp, Inn, rowis, colis, shape_jac, res_odeis,
                res_algis)

            if nlse_infos.ode_residual is None or nlse_infos.ae_residual is None or nlse_infos.bc_residual is None:
                infos = Infos(False, nlse_infos, dae_infos)
                break

            ode_sol, alg_sol, alg_sol_mid, rhsmid = new_ode_sol, new_alg_sol, new_alg_sol_mid, new_rhs_mid
            params = new_params if include_params else None

            if not self.control_odes_error:
                infos = Infos(nlse_infos.success, nlse_infos, dae_infos)
                break

            rms_res, fun_interp_z = estimate_rms(
                time, ode_sol, alg_sol, alg_sol_mid, ocp, atol=self.abs_tol, restol=self.res_tol
            )
            max_rms_res = np.max(rms_res)

            if np.isnan(max_rms_res):
                dae_infos = DAEInfos(False, rms_res)
                infos = Infos(False, nlse_infos, dae_infos)
                break

            self._print('     # Residual error = ' + str(max_rms_res) + ' with N = ' + str(len(time)), 1)

            if max_rms_res < self.res_tol:
                dae_infos = DAEInfos(True, rms_res)
                infos = Infos(True, nlse_infos, dae_infos)
                break

            new_time, new_ode_sol, new_alg_sol, new_alg_sol_mid, too_much_nodes = create_new_xp_z_zmid(
                time, ode_sol, alg_sol, fun_interp_z, rms_res, ocp, restol=self.res_tol,
                coeff_reduce_mesh=self.threshold_mesh_reduction, nmax=self.max_mesh_point,
                authorize_reduction=not self.no_mesh_reduction
            )
            if too_much_nodes:
                dae_infos = DAEInfos(False, rms_res)
                infos = Infos(False, nlse_infos, dae_infos)
                break

            time, ode_sol, alg_sol, alg_sol_mid = new_time, new_ode_sol, new_alg_sol, new_alg_sol_mid

            rowis, colis, shape_jac, Inn, res_odeis, res_algis = row_col_jac_indices(time, n_ode, n_ae, n_params)

            rms_control_iter += 1

        self._print("Solving Complete", 1)
        optisol = OptiSol(time=time, ode_sol=ode_sol, alg_sol=alg_sol, alg_sol_mid=alg_sol_mid, params=params)
        return optisol, infos

    def _newton_raphson(self,
                        time: NDArray[float64],
                        ode_sol: NDArray[float64],
                        alg_sol: NDArray[float64],
                        alg_sol_mid: NDArray[float64],
                        params: Union[NDArray[float64], None],
                        ocp: IndirectOCP,
                        Inn: NDArray[float64],
                        rowis: NDArray[np.int64],
                        colis: NDArray[np.int64],
                        shape_jac: tuple[int, int],
                        res_odeis: NDArray[np.int64],
                        res_algis: NDArray[np.int64]) -> (
            tuple[NDArray[float64], NDArray[float64], NDArray[float64], Union[NDArray[float64], None], NDArray[float64],
            NLSEInfos]):

        if params is None:
            ode_sol_new, alg_sol_new, alg_sol_mid_new, rhs_mid_new, nlse_infos = solve_newton(
                time, ode_sol, alg_sol, alg_sol_mid, ocp, Inn, rowis, colis, shape_jac, res_odeis, res_algis,
                res_tol=self.res_tol, max_iter=self.max_NLSE_iter, display=self.display,
                linear_solver=self.linear_solver, atol=self.newton_tol, coeff_damping=self.coeff_damping,
                max_probes=self.max_probes, odes_control=self.control_odes_error)
            params_new = None
        else:
            ode_sol_new, alg_sol_new, alg_sol_mid_new, params_new, rhs_mid_new, nlse_infos = solve_newton_params(
                time, ode_sol, alg_sol, alg_sol_mid, params, ocp, Inn, rowis, colis, shape_jac, res_odeis, res_algis,
                res_tol=self.res_tol, max_iter=self.max_NLSE_iter, display=self.display,
                linear_solver=self.linear_solver, atol=self.newton_tol, coeff_damping=self.coeff_damping,
                max_probes=self.max_probes, odes_control=self.control_odes_error)
        return ode_sol_new, alg_sol_new, alg_sol_mid_new, params_new, rhs_mid_new, nlse_infos

    def _print(self, msg, lvl):
        if lvl >= self.display:
            if self.display >= 1:
                print(msg)
                if self.display >= 2:
                    print('     ')
            sys.stdout.flush()

    def test_jacobian(self, time, ode_sol, alg_sol, params, ocp):
        include_params = params is not None
        N = len(time)
        ind = int(np.floor(np.random.rand(1) * N))
        ind2 = int(np.floor(np.random.rand(1) * N))

        t = np.array([time[ind]])
        xp1 = ode_sol[:, ind].reshape((ode_sol.shape[0], 1), order='F')
        z1 = alg_sol[:, ind].reshape((alg_sol.shape[0], 1), order='F')
        xp2 = ode_sol[:, ind2].reshape((ode_sol.shape[0], 1), order="F")
        z2 = alg_sol[:, ind].reshape((alg_sol.shape[0], 1), order='F')
        dx = 1e-4 * np.random.rand(ode_sol.shape[0], 1)
        dz = 1e-4 * np.random.rand(alg_sol.shape[0], 1)
        if include_params:
            dparams = np.random.rand(params.size) * 1e-4

        # erreur eq ode
        fx = ocp.ode(t, xp1, z1)
        fx_dx = ocp.ode(t, xp1 + dx, z1)
        fx_dz = ocp.ode(t, xp1, z1 + dz)
        jac_ode_x, jac_ode_z = ocp.odejac(t, xp1, z1)
        erreur_ode_dx = fx_dx - fx - np.matmul(jac_ode_x[:, :, 0], dx)
        erreur_ode_dz = fx_dz - fx - np.matmul(jac_ode_z[:, :, 0], dz)
        print('Error odejac w.r.t. x = ', erreur_ode_dx)
        print('Error odejac w.r.t. z = ', erreur_ode_dz)

        # erreur eq alg
        gx = ocp.algeq(t, xp1, z1)
        gx_dx = ocp.algeq(t, xp1 + dx, z1)
        gx_dz = ocp.algeq(t, xp1, z1 + dz)
        jac_alg_x, jac_alg_z = ocp.algjac(t, xp1, z1)
        erreur_alg_dx = gx_dx - gx - np.matmul(jac_alg_x[:, :, 0], dx)
        erreur_alg_dz = gx_dz - gx - np.matmul(jac_alg_z[:, :, 0], dz)
        print('Error algjac w.r.t. x = ', erreur_alg_dx)
        print('Error algjac w.r.t. z = ', erreur_alg_dz)

        # erreur twobc
        if not include_params:
            bc = ocp.twobc(xp1[:, 0], xp2[:, 0], z1[:, 0], z2[:, 0])
            bc_x0 = ocp.twobc(xp1[:, 0] + dx[:, 0], xp2[:, 0], z1[:, 0], z2[:, 0])
            bc_xT = ocp.twobc(xp1[:, 0], xp2[:, 0] + dx[:, 0], z1[:, 0], z2[:, 0])
            bc_z0 = ocp.twobc(xp1[:, 0], xp2[:, 0], z1[:, 0] + dz[:, 0], z2[:, 0])
            bc_zT = ocp.twobc(xp1[:, 0], xp2[:, 0], z1[:, 0], z2[:, 0] + dz[:, 0])
            jac_twobc_x0, jac_twobc_xT, jac_twobc_z0, jac_twobc_zT = ocp.bcjac(xp1[:, 0], xp2[:, 0], z1[:, 0], z2[:, 0])
        else:
            bc = ocp.twobc(xp1[:, 0], xp2[:, 0], z1[:, 0], z2[:, 0], params)
            bc_x0 = ocp.twobc(xp1[:, 0] + dx[:, 0], xp2[:, 0], z1[:, 0], z2[:, 0], params)
            bc_xT = ocp.twobc(xp1[:, 0], xp2[:, 0] + dx[:, 0], z1[:, 0], z2[:, 0], params)
            bc_z0 = ocp.twobc(xp1[:, 0], xp2[:, 0], z1[:, 0] + dz[:, 0], z2[:, 0], params)
            bc_zT = ocp.twobc(xp1[:, 0], xp2[:, 0], z1[:, 0], z2[:, 0] + dz[:, 0], params)
            bc_params = ocp.twobc(xp1[:, 0], xp2[:, 0], z1[:, 0], z2[:, 0], params + dparams)
            jac_twobc_x0, jac_twobc_xT, jac_twobc_z0, jac_twobc_zT, jac_twobc_params = ocp.bcjac(
                xp1[:, 0], xp2[:, 0], z1[:, 0], z2[:, 0], params
            )

        erreur_twobc_x0 = bc_x0 - bc - np.matmul(jac_twobc_x0, dx[:, 0])
        erreur_twobc_xT = bc_xT - bc - np.matmul(jac_twobc_xT, dx[:, 0])
        erreur_twobc_z0 = bc_z0 - bc - np.matmul(jac_twobc_z0, dz[:, 0])
        erreur_twobc_zT = bc_zT - bc - np.matmul(jac_twobc_zT, dz[:, 0])
        print('Error twobc jac w.r.t. x0 = ', erreur_twobc_x0)
        print('Error twobc jac w.r.t. xT = ', erreur_twobc_xT)
        print('Error twobc jac w.r.t. z0 = ', erreur_twobc_z0)
        print('Error twobc jac w.r.t. zT = ', erreur_twobc_zT)
        if include_params:
            erreur_twobc_params = bc_params - bc - np.matmul(jac_twobc_params, dparams)
            print('Error twobc jac w.r.t. params = ', erreur_twobc_params)
        if np.linalg.norm(erreur_ode_dx) / np.linalg.norm(dx) >= .01:
            raise Exception("ODEs Jacobian with w.r.t. ode_sol seems false")
        if np.linalg.norm(erreur_ode_dz) / np.linalg.norm(dz) >= .01:
            raise Exception("ODEs Jacobian with w.r.t. z seems false")

        if np.linalg.norm(erreur_alg_dx) / np.linalg.norm(dx) >= .01:
            raise Exception("AEs Jacobian with w.r.t. ode_sol seems false")
        if np.linalg.norm(erreur_alg_dz) / np.linalg.norm(dz) >= .01:
            raise Exception("AEs Jacobian with w.r.t. z seems false")

        if np.linalg.norm(erreur_twobc_x0) / np.linalg.norm(dx) >= .01:
            raise Exception("BCs Jacobian with w.r.t. xp0 seems false")
        if np.linalg.norm(erreur_twobc_xT) / np.linalg.norm(dx) >= .01:
            raise Exception("BCs Jacobian with w.r.t. xpT seems false")
        if np.linalg.norm(erreur_twobc_z0) / np.linalg.norm(dz) >= .01:
            raise Exception("BCs Jacobian with w.r.t. z0 seems false")
        if np.linalg.norm(erreur_twobc_zT) / np.linalg.norm(dz) >= .01:
            raise Exception("BCs Jacobian with w.r.t. zT seems false")
        if include_params:
            if np.linalg.norm(erreur_twobc_params) / np.linalg.norm(dparams) >= .01:
                raise Exception("BCs Jacobian with w.r.t. params seems false")

