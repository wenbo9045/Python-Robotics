"""
Inverse kinematics for an n-link arm using the Jacobian inverse method

Author: Daniel Ingram (daniel-s-ingram)
        Atsushi Sakai (@Atsushi_twi)
"""
import numpy as np

from NLinkArm import NLinkArm

# Simulation parameters
Kp = 2
dt = 0.1
N_LINKS = 6
N_ITERATIONS = 1000

DIS_THRE = 0.05

def inverse_kinematics(link_lengths, joint_angles, goal_pos):
    """
    Calculates the inverse kinematics using the Jacobian inverse method.
    """
    print("start iteration.")
    for iteration in range(N_ITERATIONS):
        current_pos = forward_kinematics(link_lengths, joint_angles)
        errors, distance = distance_to_goal(current_pos, goal_pos)
        print("\r第%d次迭代运算！" % iteration, end=" ")
        if distance < DIS_THRE:
            print('\n')
            print("Solution found in %d iterations. " % iteration)
            return joint_angles, True
        J = jacobian_inverse(link_lengths, joint_angles)
        joint_angles = joint_angles + np.matmul(J, errors)
    return joint_angles, False


def forward_kinematics(link_lengths, joint_angles):
    x = y = 0
    for i in range(1, N_LINKS + 1):
        x += link_lengths[i - 1] * np.cos(np.sum(joint_angles[:i]))
        y += link_lengths[i - 1] * np.sin(np.sum(joint_angles[:i]))
    return np.array([x, y]).T


def distance_to_goal(current_pos, goal_pos):
    x_diff = goal_pos[0] - current_pos[0]
    y_diff = goal_pos[1] - current_pos[1]
    return np.array([x_diff, y_diff]).T, np.math.sqrt(x_diff**2 + y_diff**2)


def jacobian_inverse(link_lengths, joint_angles):
    J = np.zeros((2, N_LINKS))
    for i in range(N_LINKS):
        J[0, i] = 0
        J[1, i] = 0
        for j in range(i, N_LINKS):
            J[0, i] -= link_lengths[j] * np.sin(np.sum(joint_angles[:j]))
            J[1, i] += link_lengths[j] * np.cos(np.sum(joint_angles[:j]))

    return np.linalg.pinv(J)


def ang_diff(theta1, theta2):
    """
    Returns the difference between two angles in the range -pi to +pi
    """
    return (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi


def get_random_goal():
    from random import random
    SAREA = N_LINKS
    goal = [SAREA * random() - SAREA / 2.0, SAREA * random() - SAREA / 2.0]
    print("goal position: ({}, {})".format(goal[0], goal[1]))
    return goal


if __name__ == '__main__':

    link_lengths = np.array([1] * N_LINKS)
    joint_angles = np.array([0] * N_LINKS)
    goal_pos = get_random_goal()
    arm = NLinkArm(link_lengths, joint_angles, goal_pos)

    joint_goal_angles, solution_found = inverse_kinematics(link_lengths, joint_angles, goal_pos)
    while True:
        if solution_found:
            joint_angles = joint_angles + Kp * ang_diff(joint_goal_angles, joint_angles) * dt
            arm.update_joints(joint_angles)
            end_effector = arm.end_effector
            errors, distance = distance_to_goal(end_effector, goal_pos)
            if distance < DIS_THRE:
                break
        else:
            print('\n')
            print("Solution can not found!")
            break