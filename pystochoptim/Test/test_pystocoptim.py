from scipy.interpolate import interp1d
import numpy as np
from pyopticontrol import FB, max_smooth, min_smooth
from pystocoptim.RPHAFuns import SOCRPHA, RPHASolver
import pickle
import matplotlib.pyplot as plt


class MyOCPStoc(SOCRPHA):

    def __init__(self, pha_weight, variance_weight, time, price_buy, price_sell, dim_control, dim_state, eps_ref):
        super().__init__(pha_weight=pha_weight, variance_weight=variance_weight, dim_control=dim_control,
                         dim_state=dim_state, eps_ref=eps_ref)
        self.q, self.rho, self.pmax = 13., .97, 8.
        self.qinit, self.qfinal = 6.5, 6.5
        self.time = time
        self.normalization_price = np.max(price_buy)
        self.fun_price_buy = interp1d(self.time, price_buy / self.normalization_price)
        self.fun_price_sell = interp1d(self.time, price_sell / self.normalization_price)

        self.fun_z_pha = interp1d(self.time, np.zeros_like(self.time))
        self.fun_mult_pha = interp1d(self.time, np.zeros_like(self.time))
        self.norm_power = self.pmax
        self.norm_capa = self.q
        self.smooth_param = 1e-5

    def initialize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time, ode_sol, alg_sol = self.time, np.zeros((2, self.time.size)), np.zeros((5, self.time.size))
        ode_sol[0] = self.qinit / self.norm_capa
        return time, ode_sol, alg_sol

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
        g[0] = dcost_dpbat + ode_sol[1] + self.fun_mult_pha(time) + self.pha_weight * (
                    pbat - self.fun_z_pha(time)) + lpcmax - lpdmax
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
        dgz[0, 0] = d2cost_d2pbat + self.pha_weight
        dgz[0, 3] = 1.
        dgz[0, 4] = - 1.
        dgz[1, 1] = FB(lqp, xp[0] - self.q / self.norm_capa, self.eps, dx=1)
        dgz[2, 2] = FB(lqm, - xp[0], self.eps, dx=1)
        dgz[3, 0] = FB(lpcmax, pbat - self.rho * self.pmax / self.norm_power, self.eps, dy=1)
        dgz[3, 3] = FB(lpcmax, pbat - self.rho * self.pmax / self.norm_power, self.eps, dx=1)
        dgz[4, 0] = - FB(lpdmax, - self.pmax / self.norm_power - pbat, self.eps, dy=1)
        dgz[4, 4] = FB(lpdmax, - self.pmax / self.norm_power - pbat, self.eps, dx=1)
        return dgxp, dgz

    def twobc(self, ode_sol_init: np.ndarray, ode_sol_final: np.ndarray, alg_sol_init: np.ndarray,
              alg_sol_final: np.ndarray, p=None) -> np.ndarray:
        return np.array([ode_sol_init[0] - self.qinit / self.norm_capa, ode_sol_final[0] - self.qfinal / self.norm_capa])

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


def main():
    with open("data_scens.pickle", "rb") as fh:
        dscenarios = pickle.load(fh)
    time, scenarios, price_buy, price_sell = (
        dscenarios["time"], dscenarios["scenarios"], dscenarios["price_buy"], dscenarios["price_sell"]
    )
    options = dict(exact_prb=False, display=0, check_jacobian=False)

    time = (time - time[0]) / 3600.
    ode_sol = np.zeros((2, len(time)))
    ode_sol[0] = 6.5
    alg_sol = np.zeros((5, len(time)))

    for indice, scen in enumerate(scenarios):
        scenarios[indice] = scen.reshape((1, len(scen)))

    socp = MyOCPStoc(pha_weight=1, variance_weight=np.inf, time=time, price_buy=price_buy, price_sell=price_sell,
                     dim_control=1, dim_state=1, eps_ref=1.)

    probas = [.01] * 100

    socp_solver = RPHASolver(socp, options=options)
    time, controls, states, filtration = socp_solver.solve(scenarios, probas, time[1] - time[0], time, ode_sol, alg_sol,
                                                           tol_clust=.1, parallelize_pha=1, pool_number=-1)

    costs = []
    price_buy_compute = price_buy[:len(time)] / max(price_buy)
    price_sell_compute = price_sell[:len(time)] / max(price_buy)
    for i, conso_prod in enumerate(scenarios):
        pbat = controls[i][0]
        conso_prod_compute = conso_prod[0][:len(time)] / socp.norm_power
        pcompteur = conso_prod_compute + np.where(pbat >= 0., pbat / .97, 0.) + np.where(pbat < 0, .97 * pbat, 0.)
        dfacturedt = (price_buy_compute * np.where(pcompteur >= 0., pcompteur, 0.)
                      + price_sell_compute * np.where(pcompteur < 0., pcompteur, 0.))
        facture = np.sum(.5 * (dfacturedt[:-1] + dfacturedt[1:]) * np.diff(time))
        costs.append(facture)

    mean_cost = sum(costs[i] * probas[i] for i in range(len(costs)))
    print("Mean cost = ", mean_cost)

    plt.figure()
    plt.plot(time, price_buy_compute, label="price")
    for i in range(len(controls)):
        plt.plot(time, controls[i][0], label="control #" + str(i))
    plt.title("Controls")
    plt.legend()

    plt.figure()
    for i in range(len(controls)):
        plt.plot(time, states[i][0], label="state #" + str(i))
    plt.title("States")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
