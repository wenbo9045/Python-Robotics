import cvxpy

class SCProblem:
    """
    Defines a standard Successive Convexification problem and
            adds the model specific constraints and objectives.

    :param m: The model object
    :param K: Number of discretization points
    """

    def __init__(self, m, K):
        # Variables:
        self.var = dict()
        self.var['X'] = cvxpy.Variable((m.n_x, K))
        self.var['U'] = cvxpy.Variable((m.n_u, K))
        self.var['sigma'] = cvxpy.Variable(nonneg=True)
        self.var['nu'] = cvxpy.Variable((m.n_x, K - 1))
        self.var['delta_norm'] = cvxpy.Variable(nonneg=True)
        self.var['sigma_norm'] = cvxpy.Variable(nonneg=True)

        # Parameters:
        self.par = dict()
        self.par['A_bar'] = cvxpy.Parameter((m.n_x * m.n_x, K - 1))
        self.par['B_bar'] = cvxpy.Parameter((m.n_x * m.n_u, K - 1))
        self.par['C_bar'] = cvxpy.Parameter((m.n_x * m.n_u, K - 1))
        self.par['S_bar'] = cvxpy.Parameter((m.n_x, K - 1))
        self.par['z_bar'] = cvxpy.Parameter((m.n_x, K - 1))

        self.par['X_last'] = cvxpy.Parameter((m.n_x, K))
        self.par['U_last'] = cvxpy.Parameter((m.n_u, K))
        self.par['sigma_last'] = cvxpy.Parameter(nonneg=True)

        self.par['weight_sigma'] = cvxpy.Parameter(nonneg=True)
        self.par['weight_delta'] = cvxpy.Parameter(nonneg=True)
        self.par['weight_delta_sigma'] = cvxpy.Parameter(nonneg=True)
        self.par['weight_nu'] = cvxpy.Parameter(nonneg=True)

        # Constraints:
        constraints = []

        # Model:
        constraints += m.get_constraints(
            self.var['X'], self.var['U'], self.par['X_last'], self.par['U_last'])

        # Dynamics:
        # x_t+1 = A_*x_t+B_*U_t+C_*U_T+1*S_*sigma+zbar+nu
        constraints += [
            self.var['X'][:, k + 1] ==
            cvxpy.reshape(self.par['A_bar'][:, k], (m.n_x, m.n_x)) *
            self.var['X'][:, k] +
            cvxpy.reshape(self.par['B_bar'][:, k], (m.n_x, m.n_u)) *
            self.var['U'][:, k] +
            cvxpy.reshape(self.par['C_bar'][:, k], (m.n_x, m.n_u)) *
            self.var['U'][:, k + 1] +
            self.par['S_bar'][:, k] * self.var['sigma'] +
            self.par['z_bar'][:, k] +
            self.var['nu'][:, k]
            for k in range(K - 1)
        ]

        # Trust regions:
        dx = cvxpy.sum(cvxpy.square(self.var['X'] - self.par['X_last']), axis=0)
        du = cvxpy.sum(cvxpy.square(self.var['U'] - self.par['U_last']), axis=0)
        ds = self.var['sigma'] - self.par['sigma_last']
        constraints += [cvxpy.norm(dx + du, 1) <= self.var['delta_norm']]
        constraints += [cvxpy.norm(ds, 'inf') <= self.var['sigma_norm']]

        # Flight time positive:
        constraints += [self.var['sigma'] >= 0.1]

        # Objective:
        sc_objective = cvxpy.Minimize(
            self.par['weight_sigma'] * self.var['sigma'] +
            self.par['weight_nu'] * cvxpy.norm(self.var['nu'], 'inf') +
            self.par['weight_delta'] * self.var['delta_norm'] +
            self.par['weight_delta_sigma'] * self.var['sigma_norm']
        )

        objective = sc_objective

        self.prob = cvxpy.Problem(objective, constraints)

    def set_parameters(self, **kwargs):
        """
        All parameters have to be filled before calling solve().
        Takes the following arguments as keywords:

        A_bar
        B_bar
        C_bar
        S_bar
        z_bar
        X_last
        U_last
        sigma_last
        E
        weight_sigma
        weight_nu
        radius_trust_region
        """

        for key in kwargs:
            if key in self.par:
                self.par[key].value = kwargs[key]
            else:
                print('Parameter \'{key}\' does not exist.')

    def get_variable(self, name):
        if name in self.var:
            return self.var[name].value
        else:
            print('Variable \'{name}\' does not exist.')
            return None


    def solve(self, solver = 'ECOS', verbose_solver = False):
        error = False
        try:
            self.prob.solve(verbose=verbose_solver,
                            solver=solver)
        except cvxpy.SolverError:
            error = True

        stats = self.prob.solver_stats

        info = {
            'setup_time': stats.setup_time,
            'solver_time': stats.solve_time,
            'iterations': stats.num_iters,
            'solver_error': error
        }

        return info