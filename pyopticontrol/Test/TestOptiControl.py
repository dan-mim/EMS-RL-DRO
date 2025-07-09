from pyopticontrol import OptiControl, FB, OptiSol, csFB
import numpy as np
import casadi as cs


def main_direct_casadi():
    ocp = OCPDirectCasadiVanDerPol()
    ocp_params = OCPDirectCasadiVanDerPolParams()
    for rk_method in ["euler", "rk4", "hermite_simpson"]:
        for include_params in [True, False]:
            print("casadi rk_method = ", rk_method)
            print("include_params = ", include_params)
            options = dict(rk_method=rk_method, display=2, print=False, hess_fact_method="umfpack")

            if include_params:
                time, x, u, params = ocp_params.initialize()
                sol = OptiSol(time=time, ode_sol=x, alg_sol=u, params=params)
                ems = OptiControl(ocp_params, options=options)
            else:
                time, x, u = ocp.initialize()
                sol = OptiSol(time=time, ode_sol=x, alg_sol=u)
                ems = OptiControl(ocp, options=options)

            sol, infos = ems.solve(sol)


def main_indirect():
    import timeit
    import os
    file_name = 'log_test.txt'
    if os.path.isfile(file_name):
        os.remove(file_name)
    for include_params in [False, True]:
        for exact_prb in [True, False]:
            for linear_solver in ["scipy", "pardiso"]:
                print("      ")
                print("      ")
                print("      ")
                print("include_params = ", include_params)
                print("exact_prb = ", exact_prb)
                print("linear_solver = " + linear_solver)
                file = open(file_name, "a")
                file.write("   \n")
                file.write("   \n")
                file.write("   \n")
                file.write("include_params = " + str(include_params) + "\n")
                file.write("exact_prb = " + str(exact_prb) + "\n")
                file.write("linear_solver = " + linear_solver + "\n")
                file.close()

                options = dict(
                    control_odes_error=True, display=1, res_tol=1e-3, linear_solver=linear_solver,
                    exact_prb=exact_prb
                )

                t0 = timeit.default_timer()

                if include_params:
                    ocp = OCPIndirectVanDerPolParams()
                    time, ode_sol, alg_sol, params = ocp.initialize()
                    sol = OptiSol(time=time, ode_sol=ode_sol, alg_sol=alg_sol, params=params)
                else:
                    ocp = OCPIndirectVanDerPol()
                    time, ode_sol, alg_sol = ocp.initialize()
                    sol = OptiSol(time=time, ode_sol=ode_sol, alg_sol=alg_sol)

                tol = 1e-7
                convergence = False
                rhoeps = .5
                ems = OptiControl(ocp, options)
                eps = 10.
                ocp.set_eps(eps)

                while not convergence:
                    sol, infos = ems.solve(sol)
                    print("eps = ", eps)
                    print("success = ", infos.success)
                    assert infos.success, "failure optimization"
                    convergence = eps < tol
                    eps *= rhoeps
                    ocp.set_eps(eps)

                t1 = timeit.default_timer()
                print("execution time =", t1 - t0)
                print("   ")
                print("  ")
                file = open(file_name, "a")
                file.write("done = " + str(infos.success) + "\n")
                file.write("execution time = " + str(t1 - t0) + "\n")
                file.close()
    file.close()


def main_indirect_casadi():
    import timeit
    import os
    file_name = 'log_test_casadi.txt'
    if os.path.isfile(file_name):
        os.remove(file_name)
    for include_params in [False, True]:
        for exact_prb in [True, False]:
            for linear_solver in ["scipy", "pardiso"]:

                print("      ")
                print("      ")
                print("      ")
                print("include_params = ", include_params)
                print("exact_prb = ", exact_prb)
                print("linear_solver = " + linear_solver)
                file = open(file_name, "a")
                file.write("   \n")
                file.write("   \n")
                file.write("   \n")
                file.write("include_params = " + str(include_params) + "\n")
                file.write("exact_prb = " + str(exact_prb) + "\n")
                file.write("linear_solver = " + linear_solver + "\n")
                file.close()
                if include_params:
                    ocp = OCPIndirectVanDerPolCasadiParams()
                    time, ode_sol, alg_sol, params = ocp.initialize()
                    bvpsol = OptiSol(time=time, ode_sol=ode_sol, alg_sol=alg_sol, params=params)
                else:
                    ocp = OCPIndirectVanDerPolCasadi()
                    time, ode_sol, alg_sol = ocp.initialize()
                    bvpsol = OptiSol(time=time, ode_sol=ode_sol, alg_sol=alg_sol)
                options = dict(
                    control_odes_error=True, display=1, res_tol=1e-3, linear_solver=linear_solver,
                    exact_prb=exact_prb
                )
                ems = OptiControl(ocp, options)
                t0 = timeit.default_timer()
                tol = 1e-7
                convergence = False
                rhoeps = .5
                eps = 10.
                ocp.set_eps(eps)
                while not convergence:
                    bvpsol, infos = ems.solve(bvpsol)
                    print("eps = ", eps)
                    print("success = ", infos.success)
                    assert infos.success, "failure optimization"
                    convergence = eps < tol
                    eps *= rhoeps
                    ocp.set_eps(eps)

                t1 = timeit.default_timer()
                print("execution time =", t1 - t0)
                print("   ")
                print("  ")
                file = open(file_name, "a")
                file.write("done = " + str(infos.success) + "\n")
                file.write("execution time = " + str(t1 - t0) + "\n")
                file.close()
    file.close()


class OCPDirectCasadiVanDerPolParams():
    def __init__(self):
        self.a = -.4

    def initialize(self):
        time = np.linspace(0., 4., 201)
        x = np.ones((2, len(time)))
        u = np.ones((1, len(time)))
        params = np.zeros((2,))
        return time, x, u, params

    def ode(self, time, x, u):
        f0 = x[1]
        f1 = - x[0] + x[1] * (1. - x[0] ** 2) + u
        return cs.vertcat(f0, f1)

    def running_cost(self, time, x, u):
        return x[0] ** 2 + x[1] ** 2 + u ** 2

    def const_ineq(self, time, x, u):
        c1 = u - 1.
        c2 = - 1 - u
        c3 = self.a - x[1]
        return cs.vertcat(c1, c2, c3)

    def twobc(self, x0, xT, u0, uT, params):
        return cs.vertcat(x0[0] - 1., x0[1] - 1., xT[0], xT[1], -params[0] + params[1] - 1., params[0] + params[1] - 1.)


class OCPDirectCasadiVanDerPol():
    def __init__(self):
        self.a = -.4

    def initialize(self):
        time = np.linspace(0., 4., 201)
        x = np.ones((2, len(time)))
        u = np.ones((1, len(time)))
        return time, x, u

    def ode(self, time, x, u):
        f0 = x[1]
        f1 = - x[0] + x[1] * (1. - x[0] ** 2) + u
        return cs.vertcat(f0, f1)

    def running_cost(self, time, x, u):
        return x[0] ** 2 + x[1] ** 2 + u ** 2

    def const_ineq(self, time, x, u):
        c1 = u - 1.
        c2 = - 1 - u
        c3 = self.a - x[1]
        return cs.vertcat(c1, c2, c3)

    def twobc(self, x0, xT, u0, uT):
        return cs.vertcat(x0[0] - 1., x0[1] - 1., xT[0], xT[1])


class OCPIndirectVanDerPolParams:
    def __init__(self,):
        self.a = -.4
        self.eps = 1.

    def set_eps(self, x):
        self.eps = x

    def initialize(self,):
        time = np.linspace(0., 4., 201)
        xp = np.ones((4, len(time)))
        z = np.ones((4, len(time)))
        params = np.zeros((2,))
        return time, xp, z, params

    def ode(self, time, xp, z):
        x, p = xp[:2], xp[2:]
        u, lup, lum, lx = z
        f = np.zeros_like(xp)
        f[0, :] = x[1, :]
        f[1, :] = - x[0, :] + x[1, :] * (1. - x[0, :] ** 2) + u
        f[2, :] = - 2. * x[0, :] + p[1, :] + 2. * x[0, :] * x[1, :] * p[1, :]
        f[3, :] = - 2. * x[1, :] - p[0, :] - p[1, :] * (1. - x[0] ** 2) + lx
        return f

    def odejac(self, time, xp, z):
        x, p = xp[:2], xp[2:]
        fx = np.zeros((xp.shape[0], xp.shape[0], len(time)))
        fz = np.zeros((xp.shape[0], z.shape[0], len(time)))

        fx[0, 1, :] = 1.
        fx[1, 0, :] = - 1. - 2. * x[1, :] * x[0, :]
        fx[1, 1, :] = 1. - x[0, :] ** 2
        fx[2, 0, :] = -2. + 2. * x[1, :] * p[1, :]
        fx[2, 1, :] = 2. * x[0, :] * p[1, :]
        fx[2, 3, :] = 1. + 2. * x[0, :] * x[1, :]
        fx[3, 0, :] = 2. * x[0, :] * p[1, :]
        fx[3, 1, :] = - 2.
        fx[3, 2, :] = - 1.
        fx[3, 3, :] = - 1. + x[0, :] ** 2

        fz[1, 0, :] = 1.
        fz[3, 3, :] = 1.
        return fx, fz

    def algeq(self, time, xp, z):
        x, p = xp[:2], xp[2:]
        u, lup, lum, lx = z
        g = np.zeros_like(z)
        g[0, :] = 2. * u + p[1, :] + lup - lum
        g[1, :] = FB(lup, u - 1., self.eps)
        g[2, :] = FB(lum, -1. - u, self.eps)
        g[3, :] = FB(lx, self.a - x[1, :], self.eps)
        return g

    def algjac(self, time, xp, z):
        x, p = xp[:2], xp[2:]
        u, lup, lum, lx = z
        gx = np.zeros((z.shape[0], xp.shape[0], len(time)))
        gu = np.zeros((z.shape[0], z.shape[0], len((time))))

        gx[0, 3, :] = 1.
        gx[3, 1, :] = - FB(lx, self.a - x[1, :], self.eps, dy=1)

        gu[0, 0, :] = 2.
        gu[0, 1, :] = 1.
        gu[0, 2, :] = -1.
        gu[1, 0, :] = FB(lup, u - 1., self.eps, dy=1)
        gu[1, 1, :] = FB(lup, u - 1., self.eps, dx=1)
        gu[2, 0, :] = - FB(lum, -1. - u, self.eps, dy=1)
        gu[2, 2, :] = FB(lum, -1. - u, self.eps, dx=1)
        gu[3, 3, :] = FB(lx, self.a - x[1, :], self.eps, dx=1)
        return gx, gu

    def twobc(self, xp0, xpT, z0, zT, params):
        bc = np.zeros((xp0.size + params.size))
        bc[0] = xp0[0] - 1.
        bc[1] = xp0[1] - 1.
        bc[2] = xpT[0]
        bc[3] = xpT[1]
        bc[4] = - params[0] + params[1] - 1.
        bc[5] = params[0] + params[1] - 1.
        return bc

    def bcjac(self, xp0, xpT, z0, zT, params):
        bcx0 = np.zeros((xp0.size + params.size, xp0.size))
        bcxT = bcx0.copy()
        bcz0 = np.zeros((xp0.size + params.size, z0.size))
        bczT = bcz0.copy()
        bcx0[0, 0] = 1.
        bcx0[1, 1] = 1.
        bcxT[2, 0] = 1.
        bcxT[3, 1] = 1.
        bcparams = np.zeros((xp0.size + params.size, params.size))
        bcparams[4, 0] = -1.
        bcparams[4, 1] = 1.
        bcparams[5, 0] = 1.
        bcparams[5, 1] = 1.
        return bcx0, bcxT, bcz0, bczT, bcparams


class OCPIndirectVanDerPol:
    def __init__(self,):
        self.a = - .4
        self.eps = 1.

    def set_eps(self, x):
        self.eps = x

    def initialize(self,):
        time = np.linspace(0., 4., 201)
        xp = np.ones((4, len(time)))
        z = np.ones((4, len(time)))
        return time, xp, z

    def ode(self, time, xp, z):
        x, p = xp[:2], xp[2:]
        u, lup, lum, lx = z
        f = np.zeros_like(xp)
        f[0] = x[1]
        f[1] = - x[0] + x[1] * (1. - x[0] ** 2) + u
        f[2] = - 2. * x[0] + p[1] + 2. * x[0] * x[1] * p[1]
        f[3] = - 2. * x[1] - p[0] - p[1] * (1. - x[0] ** 2) + lx
        return f

    def odejac(self, time, xp, z):
        x, p = xp[:2], xp[2:]
        fx = np.zeros((xp.shape[0], xp.shape[0], len(time)))
        fz = np.zeros((xp.shape[0], z.shape[0], len(time)))

        fx[0, 1] = 1.
        fx[1, 0] = - 1. - 2. * x[1] * x[0]
        fx[1, 1] = 1. - x[0] ** 2
        fx[2, 0] = -2. + 2. * x[1] * p[1]
        fx[2, 1] = 2. * x[0] * p[1]
        fx[2, 3] = 1. + 2. * x[0] * x[1]
        fx[3, 0] = 2. * x[0] * p[1]
        fx[3, 1] = - 2.
        fx[3, 2] = - 1.
        fx[3, 3] = - 1. + x[0] ** 2

        fz[1, 0] = 1.
        fz[3, 3] = 1.
        return fx, fz

    def algeq(self, time, xp, z):
        x, p = xp[:2], xp[2:]
        u, lup, lum, lx = z
        g = np.zeros_like(z)
        g[0] = 2. * u + p[1] + lup - lum
        g[1] = FB(lup, u - 1., self.eps)
        g[2] = FB(lum, -1. - u, self.eps)
        g[3] = FB(lx, self.a - x[1], self.eps)
        return g

    def algjac(self, time, xp, z):
        x, p = xp[:2], xp[2:]
        u, lup, lum, lx = z
        gx = np.zeros((z.shape[0], xp.shape[0], len(time)))
        gu = np.zeros((z.shape[0], z.shape[0], len((time))))

        gx[0, 3] = 1.
        gx[3, 1] = - FB(lx, self.a - x[1], self.eps, dy=1)

        gu[0, 0] = 2.
        gu[0, 1] = 1.
        gu[0, 2] = - 1.
        gu[1, 0] = FB(lup, u - 1., self.eps, dy=1)
        gu[1, 1] = FB(lup, u - 1., self.eps, dx=1)
        gu[2, 0] = - FB(lum, -1. - u, self.eps, dy=1)
        gu[2, 2] = FB(lum, -1. - u, self.eps, dx=1)
        gu[3, 3] = FB(lx, self.a - x[1], self.eps, dx=1)
        return gx, gu

    def twobc(self, xp0, xpT, z0, zT):
        bc = np.zeros((xp0.size,))
        bc[0] = xp0[0] - 1.
        bc[1] = xp0[1] - 1.
        bc[2] = xpT[0]
        bc[3] = xpT[1]
        return bc

    def bcjac(self, xp0, xpT, z0, zT):
        bcx0 = np.zeros((xp0.size, xp0.size))
        bcxT = bcx0.copy()
        bcz0 = np.zeros((xp0.size, z0.size))
        bczT = bcz0.copy()
        bcx0[0, 0] = 1.
        bcx0[1, 1] = 1.
        bcxT[2, 0] = 1.
        bcxT[3, 1] = 1.
        return bcx0, bcxT, bcz0, bczT


class OCPIndirectVanDerPolCasadiParams:
    def __init__(self,):
        self.a = -.4
        self.eps = 1.

    def set_eps(self, eps):
        self.eps = eps

    def initialize(self,):
        time = np.linspace(0., 4., 201)
        xp = np.zeros((4, len(time)))
        z = np.zeros((4, len(time)))
        params = np.zeros((2,))
        return time, xp, z, params

    def ode(self, time, xp, z):
        x0, x1, p0, p1 = xp[0], xp[1], xp[2], xp[3]
        u, lup, lum, lx = z[0], z[1], z[2], z[3]
        f0 = x1
        f1 = - x0 + x1 * (1. - x0 ** 2) + u
        f2 = - 2. * x0 + p1 + 2. * x0 * x1 * p1
        f3 = - 2. * x1 - p0 - p1 * (1. - x0 ** 2) + lx
        df = cs.vertcat(f0, f1, f2, f3)
        return df

    def algeq(self, time, xp, z):
        x0, x1, p0, p1 = xp[0], xp[1], xp[2], xp[3]
        u, lup, lum, lx = z[0], z[1], z[2], z[3]
        g0 = 2. * u + p1 + lup - lum
        g1 = csFB(lup, u - 1., self.eps)
        g2 = csFB(lum, -1. - u, self.eps)
        g3 = csFB(lx, self.a - x1, self.eps)
        return cs.vertcat(g0, g1, g2, g3)

    def twobc(self, xp0, xpT, z0, zT, params):
        bc0 = xp0[0] - 1.
        bc1 = xp0[1] - 1.
        bc2 = xpT[0]
        bc3 = xpT[1]
        bc4 = -params[0] + params[1] - 1.
        bc5 = params[0] + params[1] - 1.
        return cs.vertcat(bc0, bc1, bc2, bc3, bc4, bc5)


class OCPIndirectVanDerPolCasadi:
    def __init__(self,):
        self.a = -.4
        self.eps = 1.

    def set_eps(self, eps):
        self.eps = eps

    def initialize(self,):
        time = np.linspace(0., 4., 201)
        xp = np.zeros((4, len(time)))
        z = np.zeros((4, len(time)))
        return time, xp, z

    def ode(self, time, xp, z):
        x0, x1, p0, p1 = xp[0], xp[1], xp[2], xp[3]
        u, lup, lum, lx = z[0], z[1], z[2], z[3]
        f0 = x1
        f1 = - x0 + x1 * (1. - x0 ** 2) + u
        f2 = - 2. * x0 + p1 + 2. * x0 * x1 * p1
        f3 = - 2. * x1 - p0 - p1 * (1. - x0 ** 2) + lx
        df = cs.vertcat(f0, f1, f2, f3)
        return df

    def algeq(self, time, xp, z):
        x0, x1, p0, p1 = xp[0], xp[1], xp[2], xp[3]
        u, lup, lum, lx = z[0], z[1], z[2], z[3]
        g0 = 2. * u + p1 + lup - lum
        g1 = csFB(lup, u - 1., self.eps)
        g2 = csFB(lum, -1. - u, self.eps)
        g3 = csFB(lx, self.a - x1, self.eps)
        return cs.vertcat(g0, g1, g2, g3)

    def twobc(self, xp0, xpT, z0, zT):
        bc0 = xp0[0] - 1.
        bc1 = xp0[1] - 1.
        bc2 = xpT[0]
        bc3 = xpT[1]
        return cs.vertcat(bc0, bc1, bc2, bc3)


if __name__ == "__main__":

    print("start direct casadi optimization")
    main_direct_casadi()

    print("start indirect optimization")
    main_indirect()

    print("start indirect optimization casadi")
    main_indirect_casadi()