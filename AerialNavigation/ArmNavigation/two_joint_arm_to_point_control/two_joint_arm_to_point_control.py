"""
Inverse kinematics of a two-joint arm
Left-click the plot to set the goal position of the end effector

Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai (@Atsushi_twi)

Ref: P. I. Corke, "Robotics, Vision & Control", Springer 2017, ISBN 978-3-319-54413-7 p102
- [Robotics, Vision and Control \| SpringerLink](https://link.springer.com/book/10.1007/978-3-642-20144-8)

"""

import matplotlib.pyplot as plt
import numpy as np

# Similation parameters
Kp = 5
dt = 0.05

# Link lengths
l1 = l2 = 1

# Set initial goal position to the initial end-effector position
x = 0.8
y = 0.8

times = 0

show_animation = True

def two_joint_arm(GOAL_TH=0.0, theta1=0.0, theta2=0.0):
    """
    Computes the inverse kinematics for a planar 2DOF arm
    """
    while True:
        try:
            theta2_goal = np.arccos((x**2 + y**2 - l1**2 - l2**2) / (2 * l1 * l2))
            theta1_goal = np.math.atan2(y, x) - np.math.atan2(l2 * np.sin(theta2_goal), (l1 + l2 * np.cos(theta2_goal)))

            if theta1_goal < 0:
                theta2_goal = -theta2_goal
                theta1_goal = np.math.atan2(y, x) - np.math.atan2(l2 * np.sin(theta2_goal), (l1 + l2 * np.cos(theta2_goal)))

            theta1 = theta1 + Kp * ang_diff(theta1_goal, theta1) * dt
            theta2 = theta2 + Kp * ang_diff(theta2_goal, theta2) * dt
        except ValueError as e:
            print("Unreachable goal")

        wrist = forward_kinematics(theta1, theta2, x, y)

        # check goal
        d2goal = np.math.sqrt((wrist[0] - x)**2 + (wrist[1] - y)**2)

        if abs(d2goal) < GOAL_TH:
            break


def forward_kinematics(theta1, theta2, x, y):
    shoulder = np.array([0, 0])
    elbow = shoulder + np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    wrist = elbow + np.array([l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)])

    global times
    times = times + 1
    if times % 2 == 0:
        plt.cla()

        plt.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], 'k-')
        plt.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], 'k-')

        plt.plot(shoulder[0], shoulder[1], 'ro')
        plt.plot(elbow[0], elbow[1], 'ro')
        plt.plot(wrist[0], wrist[1], 'ro')

        plt.plot([wrist[0], x], [wrist[1], y], 'g--')
        plt.plot(x, y, 'g*')

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        plt.show()
        plt.pause(dt)

    return wrist


def ang_diff(theta1, theta2):
    # Returns the difference between two angles in the range -pi to +pi
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi

def main():  # pragma: no cover
    two_joint_arm(0.01)


if __name__ == "__main__":

    main()
