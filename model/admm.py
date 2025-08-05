#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2022/7/15 21:55
@desc:
"""
import numpy as np

class proxADMM:
    def __init__(self, G):
        self.D = G.shape[1]
        self.N = G.shape[0]

        self.g0 = np.mean(G, axis=0)
        self.X = np.zeros_like(self.g0)
        self.Z = np.zeros_like(self.g0)
        self.U = np.zeros_like(self.g0)
        self.XBar = np.zeros((self.N, self.D))
        self.UBar = np.zeros((self.N, self.D))

        self.G = G
        self.nu = 1
        self.c = 0.1

    def proxOperater_x(self, z, u, g):
        term1 = z - u + self.nu*g
        conflict = np.dot(g, term1)
        vec_len = np.dot(g, g)
        if conflict >= 0:
            return term1
        else:
            project = term1 - g*(conflict/vec_len)
            return project

    def proxOperater_z(self, v, c, g0):
        term1 = np.power(c,2)*np.dot(g0, g0)
        term2 = v - g0
        term2_len = np.sqrt(np.dot(term2, term2))
        if term1 >= term2_len:
            return v
        else:
            return (term1/term2_len)*term2 + g0

    def step_iterative(self):
        # Solve for x_t+1
        for i in range(0, self.N):
            t = self.proxOperater_x(self.Z, self.UBar[i], self.G[i])
            self.XBar[i] = t

        self.X = np.average(self.XBar, axis=0)

        # Solve for z_t+1
        x_k_plus_1 = self.X
        u_k = np.average(self.UBar, axis=0)
        v = x_k_plus_1+u_k
        self.Z = self.proxOperater_z(v, self.c, self.g0)

        # Solve for u_t+1
        for i in range(0, self.N):
            t = self.UBar[i]
            t = t + self.XBar[i] - self.Z
            self.UBar[i] = t

    def getParams(self):
        return self.X