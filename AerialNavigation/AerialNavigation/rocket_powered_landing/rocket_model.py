
import numpy as np
import cvxpy

class Rocket_Model_6DoF:
    """
    A 6 degree of freedom rocket landing problem.
    """

    def __init__(self):
        """
        A large r_scale for a small scale problem will
        ead to numerical problems as parameters become excessively small
        and (it seems) precision is lost in the dynamics.
        """
        self.n_x = 14
        self.n_u = 3

        # Mass
        self.m_wet = 3.0  # 30000 kg
        self.m_dry = 2.2  # 22000 kg

        # Flight time guess
        self.t_f_guess = 10.0  # 10 s

        self.w_B_max = np.deg2rad(60)

        # Angles
        max_gimbal = 20
        max_angle = 90
        glidelslope_angle = 20

        self.tan_delta_max = np.tan(np.deg2rad(max_gimbal))
        self.cos_theta_max = np.cos(np.deg2rad(max_angle))
        self.tan_gamma_gs = np.tan(np.deg2rad(glidelslope_angle))

        # Thrust limits
        self.T_max = 5.0
        self.T_min = 0.3

        # Angular moment of inertia
        self.J_B = 1e-2 * np.diag([1., 1., 1.])

        # Gravity
        self.g_I = np.array((-1, 0., 0.))

        # Fuel consumption
        self.alpha_m = 0.01

        # Vector from thrust point to CoM
        self.r_T_B = np.array([-1e-2, 0., 0.])

        self.x_init = self.set_random_initial_state()
        self.x_final = self.set_final_state()

        self.r_scale = np.linalg.norm(self.r_I_init)
        self.m_scale = self.m_wet

    def set_random_initial_state(self):
        self.r_I_init = np.array((0., 0., 0.))
        self.r_I_init[0] = np.random.uniform(3, 4)
        self.r_I_init[1:3] = np.random.uniform(-2, 2, size=2)

        self.v_I_init = np.array((0., 0., 0.))
        self.v_I_init[0] = np.random.uniform(-1, -0.5)
        self.v_I_init[1:3] = np.random.uniform(-0.5, -0.2, size=2) * self.r_I_init[1:3]

        self.q_B_I_init = self.euler_to_quat((0, np.random.uniform(-30, 30), np.random.uniform(-30, 30)))
        self.w_B_init = np.deg2rad((0, np.random.uniform(-20, 20), np.random.uniform(-20, 20)))

        return np.concatenate(((self.m_wet,), self.r_I_init, self.v_I_init, self.q_B_I_init, self.w_B_init))

    def set_final_state(self):
        self.r_I_final = np.array((0., 0., 0.))
        self.v_I_final = np.array((-1e-1, 0., 0.))
        self.q_B_I_final = self.euler_to_quat((0, 0, 0))
        self.w_B_final = np.deg2rad(np.array((0., 0., 0.)))

        return np.concatenate(((self.m_dry,), self.r_I_final, self.v_I_final, self.q_B_I_final, self.w_B_final))

    def f_func(self, x, u):
        m, rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz = \
            x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]
        ux, uy, uz = u[0], u[1], u[2]

        return np.matrix([
            [-0.01 * np.sqrt(ux**2 + uy**2 + uz**2)],
            [vx],
            [vy],
            [vz],
            [(-1.0 * m - ux * (2 * q2**2 + 2 * q3**2 - 1) - 2 * uy
              * (q0 * q3 - q1 * q2) + 2 * uz * (q0 * q2 + q1 * q3)) / m],
            [(2 * ux * (q0 * q3 + q1 * q2) - uy * (2 * q1**2
                                                   + 2 * q3**2 - 1) - 2 * uz * (q0 * q1 - q2 * q3)) / m],
            [(-2 * ux * (q0 * q2 - q1 * q3) + 2 * uy
              * (q0 * q1 + q2 * q3) - uz * (2 * q1**2 + 2 * q2**2 - 1)) / m],
            [-0.5 * q1 * wx - 0.5 * q2 * wy - 0.5 * q3 * wz],
            [0.5 * q0 * wx + 0.5 * q2 * wz - 0.5 * q3 * wy],
            [0.5 * q0 * wy - 0.5 * q1 * wz + 0.5 * q3 * wx],
            [0.5 * q0 * wz + 0.5 * q1 * wy - 0.5 * q2 * wx],
            [0],
            [1.0 * uz],
            [-1.0 * uy]
        ])

    def A_func(self, x, u):
        m, rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz = \
            x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]
        ux, uy, uz = u[0], u[1], u[2]

        return np.matrix([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [(ux * (2 * q2**2 + 2 * q3**2 - 1) + 2 * uy * (q0 * q3 - q1 * q2) - 2 * uz * (q0 * q2 + q1 * q3)) / m**2,
             0, 0, 0, 0, 0, 0, 2 * (q2 * uz - q3 * uy) / m, 2 * (q2 * uy + q3 * uz) / m, 2 * (q0 * uz + q1 * uy - 2 * q2 * ux) / m,
             2 * (-q0 * uy + q1 * uz - 2 * q3 * ux) / m, 0, 0, 0],
            [(-2 * ux * (q0 * q3 + q1 * q2) + uy * (2 * q1**2 + 2 * q3**2 - 1) + 2 * uz * (q0 * q1 - q2 * q3)) / m**2,
             0, 0, 0, 0, 0, 0, 2 * (-q1 * uz + q3 * ux) / m, 2 * (-q0 * uz - 2 * q1 * uy + q2 * ux) / m, 2 * (q1 * ux + q3 * uz) / m,
             2 * (q0 * ux + q2 * uz - 2 * q3 * uy) / m, 0, 0, 0],
            [(2 * ux * (q0 * q2 - q1 * q3) - 2 * uy * (q0 * q1 + q2 * q3) + uz * (2 * q1**2 + 2 * q2**2 - 1)) / m**2,
             0, 0, 0, 0, 0, 0, 2 * (q1 * uy - q2 * ux) / m, 2 * (q0 * uy - 2 * q1 * uz + q3 * ux) / m, 2 * (-q0 * ux - 2 * q2 * uz + q3 * uy) / m,
             2 * (q1 * ux + q2 * uy) / m, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -0.5 * wx, -0.5 * wy,
             - 0.5 * wz, -0.5 * q1, -0.5 * q2, -0.5 * q3],
            [0, 0, 0, 0, 0, 0, 0, 0.5 * wx, 0, 0.5 * wz,
             - 0.5 * wy, 0.5 * q0, -0.5 * q3, 0.5 * q2],
            [0, 0, 0, 0, 0, 0, 0, 0.5 * wy, -0.5 * wz, 0,
             0.5 * wx, 0.5 * q3, 0.5 * q0, -0.5 * q1],
            [0, 0, 0, 0, 0, 0, 0, 0.5 * wz, 0.5 * wy,
             - 0.5 * wx, 0, -0.5 * q2, 0.5 * q1, 0.5 * q0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    def B_func(self, x, u):
        m, rx, ry, rz, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz = \
            x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13]
        ux, uy, uz = u[0], u[1], u[2]

        return np.matrix([
            [-0.01 * ux / np.sqrt(ux**2 + uy**2 + uz**2), -0.01 * uy / np.sqrt(ux ** 2 + uy**2 + uz**2),
             -0.01 * uz / np.sqrt(ux**2 + uy**2 + uz**2)],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [(-2 * q2**2 - 2 * q3**2 + 1) / m, 2 * (-q0 * q3 + q1 * q2) / m, 2 * (q0 * q2 + q1 * q3) / m],
            [2 * (q0 * q3 + q1 * q2) / m, (-2 * q1**2 - 2 * q3**2 + 1) / m, 2 * (-q0 * q1 + q2 * q3) / m],
            [2 * (-q0 * q2 + q1 * q3) / m, 2 * (q0 * q1 + q2 * q3) / m, (-2 * q1**2 - 2 * q2**2 + 1) / m],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1.00000000000000],
            [0, -1.00000000000000, 0]
        ])

    def euler_to_quat(self, a):
        a = np.deg2rad(a)

        cy = np.cos(a[1] * 0.5)
        sy = np.sin(a[1] * 0.5)
        cr = np.cos(a[0] * 0.5)
        sr = np.sin(a[0] * 0.5)
        cp = np.cos(a[2] * 0.5)
        sp = np.sin(a[2] * 0.5)

        q = np.zeros(4)

        q[0] = cy * cr * cp + sy * sr * sp
        q[1] = cy * sr * cp - sy * cr * sp
        q[3] = cy * cr * sp + sy * sr * cp
        q[2] = sy * cr * cp - cy * sr * sp

        return q

    def skew(self, v):
        return np.matrix([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def dir_cosine(self, q):
        return np.matrix([
            [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
            [2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])],
            [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
            ])

    def omega(self, w):
        return np.matrix([
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0],
        ])

    def initialize_trajectory(self, X, U):
        """
        Initialize the trajectory with linear approximation.
        """
        K = X.shape[1]

        for k in range(K):
            alpha1 = (K - k) / K
            alpha2 = k / K

            m_k = (alpha1 * self.x_init[0] + alpha2 * self.x_final[0],)
            r_I_k = alpha1 * self.x_init[1:4] + alpha2 * self.x_final[1:4]
            v_I_k = alpha1 * self.x_init[4:7] + alpha2 * self.x_final[4:7]
            q_B_I_k = np.array([1, 0, 0, 0])
            w_B_k = alpha1 * self.x_init[11:14] + alpha2 * self.x_final[11:14]

            X[:, k] = np.concatenate((m_k, r_I_k, v_I_k, q_B_I_k, w_B_k))
            U[:, k] = m_k * -self.g_I

        return X, U

    def get_constraints(self, X_v, U_v, X_last_p, U_last_p):
        """
        Get model specific constraints.

        :param X_v: cvx variable for current states
        :param U_v: cvx variable for current inputs
        :param X_last_p: cvx parameter for last states
        :param U_last_p: cvx parameter for last inputs
        :return: A list of cvx constraints
        """
        # Boundary conditions:
        constraints = [
            X_v[0, 0] == self.x_init[0],
            X_v[1:4, 0] == self.x_init[1:4],
            X_v[4:7, 0] == self.x_init[4:7],
            # X_v[7:11, 0] == self.x_init[7:11],  # initial orientation is free
            X_v[11:14, 0] == self.x_init[11:14],

            # X_[0, -1] final mass is free
            X_v[1:, -1] == self.x_final[1:],
            U_v[1:3, -1] == 0,
        ]

        constraints += [
            # State constraints:
            X_v[0, :] >= self.m_dry,  # minimum mass
            cvxpy.norm(X_v[2: 4, :], axis=0) <= X_v[1, :] / self.tan_gamma_gs,  # glideslope
            cvxpy.norm(X_v[9:11, :], axis=0) <= np.sqrt((1 - self.cos_theta_max) / 2),  # maximum angle

            # maximum angular velocity
            cvxpy.norm(X_v[11: 14, :], axis=0) <= self.w_B_max,

            # Control constraints:
            cvxpy.norm(U_v[1:3, :], axis=0) <= self.tan_delta_max * U_v[0, :],  # gimbal angle constraint
            cvxpy.norm(U_v, axis=0) <= self.T_max,  # upper thrust constraint
        ]

        # linearized lower thrust constraint
        rhs = [U_last_p[:, k] / cvxpy.norm(U_last_p[:, k]) * U_v[:, k] for k in range(X_v.shape[1])]
        constraints += [self.T_min <= cvxpy.vstack(rhs)]

        return constraints
