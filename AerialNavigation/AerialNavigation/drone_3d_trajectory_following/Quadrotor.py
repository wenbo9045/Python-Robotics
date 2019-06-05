"""
Class for plotting a quadrotor

Author: Daniel Ingram (daniel-s-ingram)
"""

from math import cos, sin
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

times = 0
class Quadrotor():
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=0, yaw=0, size=0.25):
        self.p1 = np.matrix([size / 2, 0, 0, 1]).T
        self.p2 = np.matrix([-size / 2, 0, 0, 1]).T
        self.p3 = np.matrix([0, size / 2, 0, 1]).T
        self.p4 = np.matrix([0, -size / 2, 0, 1]).T

        self.x_data = []
        self.y_data = []
        self.z_data = []

        self.update_pose(x, y, z, roll, pitch, yaw)

    def update_pose(self, x, y, z, roll, pitch, yaw):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.x_data.append(x)
        self.y_data.append(y)
        self.z_data.append(z)
        global times
        times = times + 1
        if times % 40 == 0:
            self.plot()

    def transformation_matrix(self):
        x = self.x
        y = self.y
        z = self.z
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw
        return np.matrix(
            [[cos(yaw) * cos(pitch), -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll), sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll), x],
             [sin(yaw) * cos(pitch), cos(yaw) * cos(roll) + sin(yaw) * sin(pitch)
              * sin(roll), -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll), y],
             [-sin(pitch), cos(pitch) * sin(roll), cos(pitch) * cos(yaw), z]
             ])

    def plot(self):  # pragma: no cover
        T = self.transformation_matrix()

        p1_t = np.array(T * self.p1)
        p2_t = np.array(T * self.p2)
        p3_t = np.array(T * self.p3)
        p4_t = np.array(T * self.p4)

        fig = plt.figure()
        self.ax = fig.gca(projection='3d')

        self.ax.plot([p1_t[0][0], p2_t[0][0], p3_t[0][0], p4_t[0][0]],
                     [p1_t[1][0], p2_t[1][0], p3_t[1][0], p4_t[1][0]],
                     [p1_t[2][0], p2_t[2][0], p3_t[2][0], p4_t[2][0]], 'k.')

        self.ax.plot([p1_t[0][0], p2_t[0][0]],
                     [p1_t[1][0], p2_t[1][0]],
                     [p1_t[2][0], p2_t[2][0]], 'r-')
        self.ax.plot([p3_t[0][0], p4_t[0][0]],
                     [p3_t[1][0], p4_t[1][0]],
                     [p3_t[2][0], p4_t[2][0]], 'r-')

        self.ax.plot(self.x_data, self.y_data, self.z_data, c='r')

        plt.pause(0.001)