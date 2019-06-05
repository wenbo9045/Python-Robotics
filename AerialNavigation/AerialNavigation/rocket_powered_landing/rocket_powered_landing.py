"""

A rocket powered landing with successive convexification

author: Sven Niederberger
        Atsushi Sakai

Ref:
- Python implementation of 'Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time' paper
by Michael Szmuk and Behcet Acıkmese.

- EmbersArc/SuccessiveConvexificationFreeFinalTime: Implementation of "Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time" https://github.com/EmbersArc/SuccessiveConvexificationFreeFinalTime

"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from rocket_model import Rocket_Model_6DoF
from integrator import Integrator
from scproblem import SCProblem

# Trajectory points
K = 50

# Max solver iterations
iterations = 30

# Weight constants
W_SIGMA = 1  # flight time
W_DELTA = 1e-3  # difference in state/input
W_DELTA_SIGMA = 1e-1  # difference in flight time
W_NU = 1e5  # virtual control

def axis3d_equal(X, Y, Z, ax):

    max_range = np.array([X.max() - X.min(), Y.max()
                          - Y.min(), Z.max() - Z.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')


def plot_animation(X, U):  # pragma: no cover

    for k in range(K):
        if k % 10 == 0:
            fig = plt.figure()
            ax = fig.gca(projection='3d')

            ax.plot(X[2, :], X[3, :], X[1, :])  # trajectory
            ax.scatter3D([0.0], [0.0], [0.0], c="r", marker="x")  # target landing point
            axis3d_equal(X[2, :], X[3, :], X[1, :], ax)

            rx, ry, rz = X[1:4, k]
            # vx, vy, vz = X[4:7, k]
            qw, qx, qy, qz = X[7:11, k]

            CBI = np.array([
                [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy + qw * qz),
                 2 * (qx * qz - qw * qy)],
                [2 * (qx * qy - qw * qz), 1 - 2
                 * (qx ** 2 + qz ** 2), 2 * (qy * qz + qw * qx)],
                [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx),
                 1 - 2 * (qx ** 2 + qy ** 2)]
            ])

            Fx, Fy, Fz = np.dot(np.transpose(CBI), U[:, k])
            dx, dy, dz = np.dot(np.transpose(CBI), np.array([1., 0., 0.]))
            # attitude vector
            ax.quiver(ry, rz, rx, dy, dz, dx, length=0.5, linewidth=3.0,
                      arrow_length_ratio=0.0, color='black')

            # thrust vector
            ax.quiver(ry, rz, rx, -Fy, -Fz, -Fx, length=0.1,
                      arrow_length_ratio=0.0, color='red')

            ax.set_title("Rocket powered landing")
            plt.pause(0.5)


def main():
    print("start!!")
    model = Rocket_Model_6DoF()

    # state and input list
    X = np.empty(shape=[model.n_x, K])
    U = np.empty(shape=[model.n_u, K])

    # INITIALIZATION
    sigma = model.t_f_guess
    X, U = model.initialize_trajectory(X, U)

    integrator = Integrator(model, K)
    problem = SCProblem(model, K)

    converged = False
    w_delta = W_DELTA
    for it in range(iterations):
        t0_it = time()
        print('-' * 18 + ' Iteration {str(it + 1).zfill(2)} ' + '-' * 18)

        A_bar, B_bar, C_bar, S_bar, z_bar = integrator.calculate_discretization(X, U, sigma)

        problem.set_parameters(A_bar=A_bar, B_bar=B_bar, C_bar=C_bar, S_bar=S_bar, z_bar=z_bar,
                               X_last=X, U_last=U, sigma_last=sigma,
                               weight_sigma=W_SIGMA, weight_nu=W_NU,
                               weight_delta=w_delta, weight_delta_sigma=W_DELTA_SIGMA)
        problem.solve()

        X = problem.get_variable('X')
        U = problem.get_variable('U')
        sigma = problem.get_variable('sigma')

        delta_norm = problem.get_variable('delta_norm')
        sigma_norm = problem.get_variable('sigma_norm')
        nu_norm = np.linalg.norm(problem.get_variable('nu'), np.inf)

        print('delta_norm', delta_norm)
        print('sigma_norm', sigma_norm)
        print('nu_norm', nu_norm)

        if delta_norm < 1e-3 and sigma_norm < 1e-3 and nu_norm < 1e-7:
            converged = True

        w_delta *= 1.5

        print('Time for iteration', time() - t0_it, 's')

        if converged:
            print('Converged after {it + 1} iterations.')
            break

    plot_animation(X, U)

    print("done!!")


if __name__ == '__main__':
    main()
