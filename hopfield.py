#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 03:05:55 2021

@author: gino
"""

import numpy as np

def sign(x):
    sign = np.empty(x.shape,dtype=int)
    sign[x >= 0] = 1
    sign[x < 0] = -1
    return sign
    

def weights(xi):
    """Calcula la matriz de pesos a partir de la matriz de memorias"""
    n, p = xi.shape
    w = np.zeros([n, n], dtype=int)
    for mu in range(p):
        pattern = xi[:, mu]
        w += np.outer(pattern, pattern)
    for i in range(n):
        w[i,i] = 0
    return w

def superposition(s1, s2):
    """Calcula la superposición entre las configuraciones s1 y s2."""
    n = len(s1)
    m = np.dot(s1, s2)
    m = m/n
    return m

def generate_random_array(n, p):
    """Genera p patrones aleatorios de longitud n
linealmente independientes."""
    while True:
        array = np.random.choice([-1,1], size=[n, p])
        #si los patrones son LD regenero xi
        if np.linalg.matrix_rank(array) < p:
            continue
        else:
            break
    return array

def generate_random_pattern(n):
    """Genera un patrón aleatorio de longitud n."""
    s = np.random.choice([-1,1], size=n)
    return s

def is_in(s, xi):
    """Chequea si s coincide con alguna columna de xi."""
    n, p = xi.shape
    is_in=False
    for mu in range(p):
        pattern = xi[:, mu]
        if np.array_equal(s, pattern):
            is_in=True
            return is_in, mu
    return is_in, None

class HopfieldNetwork:
    def __init__(self, n, p):
        self.n = n #numero de neuronas
        self.p = p #numero de patrones
        self.s = np.ones(n, dtype=int)
        self.xi = np.ones([n, p], dtype=int)
        self.w = np.ones([n, n], dtype=int)

    def set_conf(self, s):
        """Coloca la red en la configuración s."""
        s_aux = np.ones(self.n, dtype=int)
        s_aux[:] = s[:]
        self.s = s_aux
        
    def evolve(self, t_max=50, print_arrays=False):
        """Evoluciona la red hasta que llegue al estado estacionario o\
            a t_max=50."""
        s_aux = np.ones(self.n, dtype=int)
        conv = False
        if print_arrays:
            print()
        for t in range(t_max):
            if print_arrays:
                print('    t = {}:    s = {}'.format(t, self.s))
            s_aux[:] = self.s[:]
            self.s = sign(np.matmul(self.w, s_aux))
            if np.array_equal(s_aux, self.s):
                conv = True
                break
        t_conv = t
        return conv, t_conv

    def is_a_memory(self):
        """Chequea si la configuración actual coincide con alguna de las
memorias o sus negativos."""
        s = np.ones(self.n)
        xi = np.ones(self.xi.shape)
        s = self.s
        xi = self.xi
        is_a_memory, mu_mem = is_in(s, xi)
        is_a_negative, mu_neg = is_in(s, -xi)
        if is_a_memory:
            return 'mem', mu_mem
        elif is_a_negative:
            return 'neg', mu_neg
        else:
            return 'no', None