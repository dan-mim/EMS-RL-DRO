from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from pyopticontrol import FB, max_smooth, min_smooth
from pystocoptim.SDAPFuns import SDAPSolver, SOCSDAP
from pyreductree import reduce_scenarios_forward_selection
import pickle
from numpy.typing import NDArray
from numpy import float64

class MyOCPStoc(SOCSDAP):

    def __init__(self, time, price_buy, price_sell, dim_control, dim_state, eps_ref):
        super().__init__(dim_control=dim_control, dim_state=dim_state, eps_ref=eps_ref)
        self.eps_ref = eps_ref
        self.dim_control = dim_control
        self.dim_state = dim_state
        self.q, self.rho, self.pmax = 13., .97, 8.
        self.qinit, self.qfinal = 6.5, 6.5
        self.time = time
        self.normalization_price = np.max(price_buy)
        self.fun_price_buy = interp1d(self.time, price_buy / self.normalization_price)
        self.fun_price_sell = interp1d(self.time, price_sell / self.normalization_price)
        self.norm_power = self.pmax
        self.norm_capa = self.q
        self.smooth_param = 1e-5

    def ode(self, time, ode_sol, alg_sol):
        pbat, lqp, lqm, lpcmax, lpdmax = alg_sol
        dxpdt = np.zeros_like(ode_sol)
        dxpdt[0] = pbat * self.norm_power / self.norm_capa
        dxpdt[1] = - lqp + lqm
        return dxpdt

    def odejac(self, time, ode_sol, alg_sol):
        fxp = np.zeros((ode_sol.shape[0], ode_sol.shape[0], ode_sol.shape[1]))
        fz = np.zeros((ode_sol.shape[0], alg_sol.shape[0], ode_sol.shape[1]))
        fz[0, 0] = self.norm_power / self.norm_capa
        fz[1, 1] = -1.
        fz[1, 2] = 1.
        return fxp, fz

    def algeq(self, time, ode_sol, alg_sol):
        pbat, lqp, lqm, lpcmax, lpdmax = alg_sol
        scenario = self.fun_scenario(time)
        pcompteur = (scenario / self.norm_power + self.smooth_pos(pbat) / self.rho
                     + self.smooth_neg(pbat) * self.rho)
        dcompteur_dpbat = self.smooth_pos(pbat, d=1) / self.rho + self.smooth_neg(pbat, d=1) * self.rho

        dcost_dcompteur = (self.fun_price_buy(time) * self.smooth_pos(pcompteur, d=1)
                           + self.fun_price_sell(time) * self.smooth_neg(pcompteur, d=1))
        dcost_dpbat = dcost_dcompteur * dcompteur_dpbat
        g = np.zeros_like(alg_sol)
        g[0] = dcost_dpbat + ode_sol[1] + lpcmax - lpdmax
        g[1] = FB(lqp, ode_sol[0] - self.q / self.norm_capa, self.eps)
        g[2] = FB(lqm, - ode_sol[0], self.eps)
        g[3] = FB(lpcmax, pbat - self.rho * self.pmax / self.norm_power, self.eps)
        g[4] = FB(lpdmax, - self.pmax / self.norm_power - pbat, self.eps)
        return g

    def algjac(self, time, xp, z):
        pbat, lqp, lqm, lpcmax, lpdmax = z
        scenario = self.fun_scenario(time)
        pcompteur = (scenario / self.norm_power + self.smooth_pos(pbat) / self.rho
                     + self.smooth_neg(pbat) * self.rho)
        dcompteur_dpbat = self.smooth_pos(pbat, d=1) / self.rho + self.smooth_neg(pbat, d=1) * self.rho
        d2compteur_d2pbat = self.smooth_pos(pbat, d=2) / self.rho + self.smooth_neg(pbat, d=2) * self.rho
        dcost_dcompteur = (self.fun_price_buy(time) * self.smooth_pos(pcompteur, d=1)
                           + self.fun_price_sell(time) * self.smooth_neg(pcompteur, d=1))
        d2cost_d2compteur = (self.fun_price_buy(time) * self.smooth_pos(pcompteur, d=2)
                             + self.fun_price_sell(time) * self.smooth_neg(pcompteur, d=2))
        d2cost_d2pbat = d2cost_d2compteur * (dcompteur_dpbat ** 2) + dcost_dcompteur * d2compteur_d2pbat

        dgxp = np.zeros((z.shape[0], xp.shape[0], z.shape[1]))
        dgxp[0, 1] = 1.
        dgxp[1, 0] = FB(lqp, xp[0] - self.q / self.norm_capa, self.eps, dy=1)
        dgxp[2, 0] = - FB(lqm, - xp[0], self.eps, dy=1)

        dgz = np.zeros((z.shape[0], z.shape[0], z.shape[1]))
        dgz[0, 0] = d2cost_d2pbat
        dgz[0, 3] = 1.
        dgz[0, 4] = - 1.
        dgz[1, 1] = FB(lqp, xp[0] - self.q / self.norm_capa, self.eps, dx=1)
        dgz[2, 2] = FB(lqm, - xp[0], self.eps, dx=1)
        dgz[3, 0] = FB(lpcmax, pbat - self.rho * self.pmax / self.norm_power, self.eps, dy=1)
        dgz[3, 3] = FB(lpcmax, pbat - self.rho * self.pmax / self.norm_power, self.eps, dx=1)
        dgz[4, 0] = - FB(lpdmax, - self.pmax / self.norm_power - pbat, self.eps, dy=1)
        dgz[4, 4] = FB(lpdmax, - self.pmax / self.norm_power - pbat, self.eps, dx=1)
        return dgxp, dgz

    def twobc(self, ode_sol_init: NDArray[float64], ode_sol_final: NDArray[float64], alg_sol_init: NDArray[float64],
              alg_sol_final: NDArray[float64], p=None) -> NDArray[float64]:

        return np.array([ode_sol_init[0] - self.qinit / self.norm_capa,
                         ode_sol_final[0] - self.qfinal / self.norm_capa])

    def bcjac(self, ode_sol_init, ode_sol_final, alg_sol_init, alg_sol_final, p=None):
        bcx0 = np.zeros((ode_sol_init.size, ode_sol_init.size))
        bcx0[0, 0] = 1.
        bcxT = np.zeros((ode_sol_init.size, ode_sol_init.size))
        bcxT[1, 0] = 1.
        bcz0 = np.zeros((ode_sol_init.size, alg_sol_init.size))
        bczT = bcz0.copy()
        return bcx0, bcxT, bcz0, bczT

    def smooth_pos(self, x, d=0):
        return max_smooth(x, 0., d1=d, mu=self.smooth_param)

    def smooth_neg(self, x, d=0):
        return min_smooth(x, 0., d1=d, mu=self.smooth_param)

    def eval_fs(self, time, ode_sol, alg_sol):
        pbat, lqp, lqm, lpcmax, lpdmax = alg_sol
        scenario = self.fun_scenario(time)[0]
        pcompteur = (scenario / self.norm_power + self.smooth_pos(pbat) / self.rho
                     + self.smooth_neg(pbat) * self.rho)
        running_cost = (self.fun_price_buy(time) * self.smooth_pos(pcompteur)
                        + self.fun_price_sell(time) * self.smooth_neg(pcompteur))
        cost = np.sum(np.diff(time) * (running_cost[:-1] + running_cost[1:]) / 2)
        return cost


class MyOCPprojEpiSmart(SOCSDAP):

    def __init__(self, time, price_buy, price_sell, dim_control, dim_state, eps_ref):
        super().__init__(dim_control=dim_control, dim_state=dim_state, eps_ref=eps_ref)
        # recall :
        # Proj_{\epi(F_s)}((fun_v_proj, rho_proj))
        # := \inf { \| P_b - fun_v_proj \|_{L^2([0,T])}^2 + |rho_proj - r|^2 : (P_b, r) \in \epi(F_s)}.
        self.time = time
        self.q, self.rho, self.pmax = 13., .97, 8.
        self.normalization_price = np.max(price_buy)
        self.fun_price_buy = interp1d(self.time, price_buy / self.normalization_price)
        self.fun_price_sell = interp1d(self.time, price_sell / self.normalization_price)

        self.qinit, self.qfinal = 6.5, 6.5
        self.norm_power = self.pmax
        self.norm_capa = self.q
        self.smooth_param = 1e-5

    def initialize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        ode_sol : (Q, q, r, p1, p2, p3) := (x1, x2, x3, p1, p2, p3) # solution of the differential equations
        alg_sol : (Pb:=pbat, mu1, mu2, mu3, mu4, mu5) := (u, mu1, mu2, mu3, mu4, mu5) # solution of the algebrical equations

        """
        time, ode_sol, alg_sol = self.time, np.zeros((4, self.time.size)), np.zeros((5, self.time.size))
        params = np.zeros((2,))
        ode_sol[0] = self.qinit / self.norm_capa
        ode_sol[2] = 1.
        ode_sol[5] = 1.
        # alg_sol[0] = np.ones(self.time.size) / self.time.size
        return time, ode_sol, alg_sol, params

    def eval_fs(self, time, ode_sol, alg_sol):
        locate_q = (ode_sol.shape[0] - 2) // 2
        return ode_sol[locate_q, -1]


    def ode(self, time, ode_sol, alg_sol):
        """
        equa diff f1, f2 and f3 and the complementary differential equations: p_dot = dH/dode_sol
        nb: cost := l, pbat := u
        """
        x, q, px, pq = ode_sol
        pbat, mqp, mqm, mup, mum = alg_sol
        scenario = self.fun_scenario(time)
        pcompteur = (scenario / self.norm_power + self.smooth_pos(pbat) / self.rho
                     + self.smooth_neg(pbat) * self.rho)
        dxpdt = np.zeros_like(ode_sol)

        dxpdt[0] = pbat * self.norm_power / self.norm_capa
        dxpdt[1] = (
                self.fun_price_buy(time) * self.smooth_pos(pcompteur)
                + self.fun_price_sell(time) * self.smooth_neg(pcompteur)
        )
        dxpdt[2] = - mqp + mqm
        return dxpdt

    def odejac(self, time, ode_sol, alg_sol):
        """
        Jacobians of the differential equations with regards to ode_sol and alg_sol
        fxp = df_i/d_ode_sol_j
        fz = df_i/d_alg_sol_j
        """
        x, q, px, pq = ode_sol
        pbat, mqp, mqm, mup, mum = alg_sol
        scenario = self.fun_scenario(time)

        pcompteur = (
                scenario / self.norm_power + self.smooth_pos(pbat) / self.rho
                + self.smooth_neg(pbat) * self.rho
        )
        dcompteur_dpbat = self.smooth_pos(pbat, d=1) / self.rho + self.smooth_neg(pbat, d=1) * self.rho

        dcost_dcompteur = (self.fun_price_buy(time) * self.smooth_pos(pcompteur, d=1)
                           + self.fun_price_sell(time) * self.smooth_neg(pcompteur, d=1))
        dcost_dpbat = dcost_dcompteur * dcompteur_dpbat

        fxp = np.zeros((ode_sol.shape[0], ode_sol.shape[0], ode_sol.shape[1]))
        fz = np.zeros((ode_sol.shape[0], alg_sol.shape[0], ode_sol.shape[1]))
        fz[0, 0] = self.norm_power / self.norm_capa
        fz[1, 0] = dcost_dpbat
        fz[2, 1] = - 1.
        fz[2, 2] = 1
        return fxp, fz

    def algeq(self, time, ode_sol, alg_sol):
        """
        algebrical equations: dHamiltonien/du and FB(mu, g_inequality_equations)
        """
        x, q, px, pq = ode_sol
        pbat, mqp, mqm, mup, mum = alg_sol
        scenario = self.fun_scenario(time)
        pcompteur = (scenario / self.norm_power + self.smooth_pos(pbat) / self.rho
                     + self.smooth_neg(pbat) * self.rho)
        dcompteur_dpbat = self.smooth_pos(pbat, d=1) / self.rho + self.smooth_neg(pbat, d=1) * self.rho

        dcost_dcompteur = (self.fun_price_buy(time) * self.smooth_pos(pcompteur, d=1)
                           + self.fun_price_sell(time) * self.smooth_neg(pcompteur, d=1))
        dcost_dpbat = dcost_dcompteur * dcompteur_dpbat

        g = np.zeros_like(alg_sol)
        g[0] = pbat - self.fun_v_proj(time) + px * self.norm_power / self.norm_capa + pq * dcost_dpbat + mup - mum
        g[1] = FB(mqp, x - self.q / self.norm_capa, self.eps)
        g[2] = FB(mqm, - x, self.eps)
        g[3] = FB(mup, pbat - self.rho * self.pmax / self.norm_power, self.eps)
        g[4] = FB(mum, - self.pmax / self.norm_power - pbat, self.eps)
        return g

    def algjac(self, time, ode_sol, alg_sol):
        """
        dgxp = dg_i/dode_sol_j
        dgz = dg_i/dalg_sol_j
        """
        x, q, px, pq = ode_sol
        pbat, mqp, mqm, mup, mum = alg_sol

        scenario = self.fun_scenario(time)

        pcompteur = (
                scenario / self.norm_power + self.smooth_pos(pbat) / self.rho + self.smooth_neg(pbat) * self.rho
        )

        dcompteur_dpbat = self.smooth_pos(pbat, d=1) / self.rho + self.smooth_neg(pbat, d=1) * self.rho

        dcost_dcompteur = (
                self.fun_price_buy(time) * self.smooth_pos(pcompteur, d=1)
                + self.fun_price_sell(time) * self.smooth_neg(pcompteur, d=1)
        )

        dcost_dpbat = dcost_dcompteur * dcompteur_dpbat

        d2compteur_d2pbat = self.smooth_pos(pbat, d=2) / self.rho + self.smooth_neg(pbat, d=2) * self.rho

        d2cost_d2compteur = (
                self.fun_price_buy(time) * self.smooth_pos(pcompteur, d=2)
                + self.fun_price_sell(time) * self.smooth_neg(pcompteur, d=2)
        )

        d2cost_d2pbat = d2cost_d2compteur * (dcompteur_dpbat ** 2) + dcost_dcompteur * d2compteur_d2pbat

        dgxp = np.zeros((alg_sol.shape[0], ode_sol.shape[0], alg_sol.shape[1]))
        dgz = np.zeros((alg_sol.shape[0], alg_sol.shape[0], ode_sol.shape[1]))

        #g[0] = pbat - self.fun_v_proj(time) + px * self.norm_power / self.norm_capa + pq * dcost_dpbat + mup - mum
        # dgxp
        dgxp[0, 2] = self.norm_power / self.norm_capa
        dgxp[0, 3] = dcost_dpbat
        # dgz
        dgz[0, 0] = 1. + pq * d2cost_d2pbat
        dgz[0, 3] = 1.
        dgz[0, 4] = - 1.

        #g[1] = FB(mqp, x - self.q / self.norm_capa, self.eps)
        # dgxp
        dgxp[1, 0] = FB(mqp, x - self.q / self.norm_capa, self.eps, dy=1)
        # dgz
        dgz[1, 1] = FB(mqp, x - self.q / self.norm_capa, self.eps, dx=1)

        #g[2] = FB(mqm, - x, self.eps)
        # dgxp
        dgxp[2, 0] = - FB(mqm, - x, self.eps, dy=1)
        # dgz
        dgz[2, 2] = FB(mqm, - x, self.eps, dx=1)

        #g[3] = FB(mup, pbat - self.rho * self.pmax / self.norm_power, self.eps)
        # dgz
        dgz[3, 0] = FB(mup, pbat - self.rho * self.pmax / self.norm_power, self.eps, dy=1)
        dgz[3, 3] = FB(mup, pbat - self.rho * self.pmax / self.norm_power, self.eps, dx=1)

        #g[4] = FB(mum, - self.pmax / self.norm_power - pbat, self.eps)
        # dgz
        dgz[4, 0] = - FB(mum, - self.pmax / self.norm_power - pbat, self.eps, dy=1)
        dgz[4, 4] = FB(mum, - self.pmax / self.norm_power - pbat, self.eps, dx=1)

        return dgxp, dgz

    def twobc(self,
              ode_sol_init: np.ndarray,
              ode_sol_final: np.ndarray,
              alg_sol_init: np.ndarray,
              alg_sol_final: np.ndarray,
              params=None) -> np.ndarray:
        """
        Boundary conditions
        ode_sol_init := X1, x2, x3, p1, p2, p3 evaluated at t=0
        final --> same evaluated at t=T
        """

        eps_bc = min(.01, self.eps)
        lbd, mu_psi = params
        bc0 = ode_sol_init[0] - self.qinit / self.norm_capa
        bc1 = ode_sol_init[1]
        bc2 = ode_sol_final[0] - self.qfinal / self.norm_capa
        bc3 = - ode_sol_final[3] + mu_psi
        bc4 = lbd - self.rho_proj - mu_psi
        bc5 = FB(mu_psi, ode_sol_final[1] - lbd, eps_bc)

        return np.array([bc0, bc1, bc2, bc3, bc4, bc5])

    def bcjac(self, ode_sol_init, ode_sol_final, alg_sol_init, alg_sol_final, params=None):
        """
        Jacobians of the boundary conditions
        bcx0 : jacobian of the boundary condition with respect to ode_sol_init
        bcxt : jacobian of the boundary condition with respect to ode_sol_final
        bcz0 : jacobian of the boundary condition with respect to alg_sol_init
        bczT : jacobian of the boundary condition with respect to alg_sol_final
        """
        eps_bc = min(.01, self.eps)
        lbd, mu_psi = params

        bcx0 = np.zeros((ode_sol_init.size + params.size, ode_sol_init.size))
        bcx0[0, 0] = 1.
        bcx0[1, 1] = 1.

        bcxT = np.zeros((ode_sol_init.size + params.size, ode_sol_init.size))
        bcxT[2, 0] = 1.
        bcxT[3, 3] = -1
        bcxT[5, 1] = FB(mu_psi, ode_sol_final[1] - lbd, eps_bc, dy=1)


        bcz0 = np.zeros((ode_sol_init.size + params.size, alg_sol_init.size))

        bczT = bcz0.copy()

        bcparams = np.zeros((ode_sol_init.size + params.size, params.size))
        bcparams[3, 1] = 1.
        bcparams[4, 0] = 1.
        bcparams[4, 1] = - 1.
        bcparams[5, 0] = - FB(mu_psi, ode_sol_final[1] - lbd, eps_bc, dy=1)
        bcparams[5, 1] = FB(mu_psi, ode_sol_final[1] - lbd, eps_bc, dx=1)
        return bcx0, bcxT, bcz0, bczT, bcparams

    def smooth_pos(self, x, d=0):
        return max_smooth(x, 0., d1=d, mu=self.smooth_param)

    def smooth_neg(self, x, d=0):
        return min_smooth(x, 0., d1=d, mu=self.smooth_param)


def main(savepath_fig='Results'):
    with open("data_scens.pickle", "rb") as fh:
        dscenarios = pickle.load(fh)
    time, scenarios_original_tree, price_buy, price_sell = (
        dscenarios["time"], dscenarios["scenarios"], dscenarios["price_buy"], dscenarios["price_sell"]
    )
    options = dict(exact_prb=False, display=0, check_jacobian=False)
    tol_clust = 1e-2 # 1e-6
    time = (time - time[0]) / 3600.
    ode_sol = np.zeros((2, len(time)))
    ode_sol[0] = .5
    ode_sol_epi = np.zeros((4, len(time)))
    ode_sol_epi[0] = .5
    alg_sol = np.zeros((5, len(time)))
    alg_sol_epi = alg_sol.copy()

    for indice, scen in enumerate(scenarios_original_tree):
        scenarios_original_tree[indice] = scen.reshape((1, len(scen)))

    # OCPs
    socp = MyOCPStoc(
        time=time, price_buy=price_buy, price_sell=price_sell, dim_control=1, dim_state=1, eps_ref=1.
    )

    epi_ocp = MyOCPprojEpiSmart(
        time, price_buy, price_sell, dim_control=1, dim_state=1, eps_ref=1.
    )   #MyOCPprojEpi, OCP_epi_Casadi


    probas_original_tree = [.01] * 100

    ## FFS
    number_selected_scenarios = 25
    scenarios_reduced_tree, probas_reduced_tree = reduce_scenarios_forward_selection(
                                                                                    scenarios_original_tree,
                                                                                    number_selected_scenarios
                                                                                        )

    ## Scenario tree reduction // Kovaevic Pichler
    from pyreductree import KP_reduction, nested_distance, filtration_into_networkx_tree, draw_tree
    from pystocoptim import compute_filtration, retrieve_scenarios_from_filtration_dfs
    original_filtration = compute_filtration(scenarios_original_tree, tol=tol_clust, probas=probas_original_tree)
    reduced_filtration = compute_filtration(scenarios_reduced_tree, tol=tol_clust, probas=probas_reduced_tree)
    # draw_tree(filtration_into_networkx_tree(original_filtration))
    # draw_tree(filtration_into_networkx_tree(reduced_filtration))
    reduced_filtration, distance_nd = KP_reduction(
        original_filtration, reduced_filtration, method='LP', delta=500, itred=3, npool=1, lambda_IBP=100, rho=1000,
        precisionMAM=10 ** -4)
    scenarios_reduced_tree, probas_reduced_tree = retrieve_scenarios_from_filtration_dfs(reduced_filtration)

    # scenarios_reduced_tree, probas_reduced_tree = scenarios_original_tree.copy(), probas_original_tree.copy()

    delay_hazard = time[1] - time[0]
    r_proj_ambiguity_set = 1.
    TOL_control, TOL_sdap = 1e-5, 1e-5
    iteration_max = 1000

    obj_values = []
    nb_scenarios = []
    thetas = [5000]
    for theta in thetas:

        socp_solver = SDAPSolver(socp,
                                 epi_ocp,
                                 options=options,
                                 r_proj_ambiguity_set=r_proj_ambiguity_set,
                                 theta=theta)


        time_opt, controls, states, obj_value, filtration_reduced_tree, reduced_proba, new_scenarios = socp_solver.solve(
            scenarios_reduced_tree.copy(), probas_reduced_tree.copy(), scenarios_original_tree.copy(),
            probas_original_tree.copy(), delay_hazard, time, ode_sol, alg_sol, ode_sol_epi, alg_sol_epi, TOL_control,
            TOL_sdap, iteration_max,tol_clust=tol_clust, parallelize_sdap=1, pool_number=5)

        print("number of scenarios = ", len(controls))
        nb_scenarios.append(len(controls))
        print("reduced_proba = ", reduced_proba)
        obj_values.append(obj_value)
        print("obj value = ", obj_value)

        plt.figure()
        plt.scatter(x=np.arange(len(reduced_proba)), y=reduced_proba)
        title = r'Probability distribution for $\theta=$'+f"{theta}"
        plt.title(title)
        safe_title = title.replace(" ", "_").replace("$", "").replace("\\", "").replace(">", "sup").replace('=', '_')
        plt.savefig(f"{savepath_fig}/{safe_title}.png", dpi=300)

        plt.figure()
        plt.plot(time_opt, socp.fun_price_buy(time_opt) / max(price_buy), label="price")
        for i in range(len(controls)):
            plt.plot(time_opt, controls[i][0], label="control #" + str(i))
        title = r'Controls for $\theta=$'+f"{theta}"
        plt.title(title)
        plt.legend()
        safe_title = title.replace(" ", "_").replace("$", "").replace("\\", "").replace(">", "sup").replace('=', '_')
        plt.savefig(f"{savepath_fig}/{safe_title}.png", dpi=300)

        plt.figure()
        plt.plot(time_opt, socp.fun_price_buy(time_opt) / max(price_buy), label="price")
        for i, conso_prod in enumerate(new_scenarios):
            pbat = controls[i][0]
            conso_prod_compute = conso_prod[0][:len(time)] / socp.norm_power
            pcompteur = conso_prod_compute + np.where(pbat >= 0., pbat / .97, 0.) + np.where(pbat < 0, .97 * pbat, 0.)
            plt.plot(time_opt, pcompteur, label="PCompteur #" + str(i))
        title = r'Pcompteurs for $\theta=$'+f"{theta}"
        plt.title(title)
        plt.legend()
        safe_title = title.replace(" ", "_").replace("$", "").replace("\\", "").replace(">", "sup").replace('=', '_')
        plt.savefig(f"{savepath_fig}/{safe_title}.png", dpi=300)

        plt.figure()
        for i in range(len(controls)):
            plt.plot(time_opt, states[i][0], label="state #" + str(i))
        title = r'States for $\theta=$'+f"{theta}"
        plt.title(title)
        safe_title = title.replace(" ", "_").replace("$", "").replace("\\", "").replace(">", "sup").replace('=', '_')
        plt.savefig(f"{savepath_fig}/{safe_title}.png", dpi=300)


    plt.figure()
    plt.plot(thetas, nb_scenarios, '--o')
    title = r'Evolution of the number of scenarios with >0 probability with $\theta$'
    plt.title(title)
    safe_title = title.replace(" ", "_").replace("$", "").replace("\\", "").replace(">", "sup").replace('=', '_')
    plt.savefig(f"{savepath_fig}/{safe_title}.png", dpi=300)

    plt.figure()
    plt.plot(thetas, obj_values, '--o')
    title = r'Evolution of the objective value with $\theta$ IN SAMPLE'
    plt.title(title)
    safe_title = title.replace(" ", "_").replace("$", "").replace("\\", "").replace(">", "sup").replace('=', '_')
    plt.savefig(f"{savepath_fig}/{safe_title}.png", dpi=300)


    for i,theta in enumerate(thetas):
        print(f'for theta={theta}, objective value = {obj_values[i]}')
    plt.show()

if __name__ == "__main__":
    main(savepath_fig='tests')
