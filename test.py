#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:31:21 2021

@author: gino
"""

import hopfield as hf
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()
print_arrays = False
sync = False
plot_H_vals = True

n = 500
p = 50
T = 0.5
t_max = 100

print('Generando red...')
net = hf.HopfieldNetwork(n, p, sync=sync, T=T)
print('Almacenando memorias aleatorias...')
net.xi = hf.generate_random_array(n, p)
if print_arrays:
    print()
    print('xi ** T = \n{}.'.format(net.xi.transpose()))
    print()
print('Inicializando los pesos...')
net.w = hf.weights(net.xi)
if print_arrays:
    print()
    print('w = \n{}.'.format(net.w))
    print()
print('Colocando la red en una configuración aleatoria...')
pattern = hf.generate_random_pattern(n)
net.set_conf(pattern)
print('Evolucionando...')
conv, t_conv, H_vals = net.evolve(t_max=t_max, print_arrays=print_arrays,\
                                  plot_H_vals=plot_H_vals)
if conv:
    if t_conv==1:
            print('La red ha alcanzado el estado estacionario en 1 \
paso.')
    else:
        print('La red ha alcanzado el estado estacionario en \
{} pasos.'.format(t_conv))
else:
    print('La red no ha convergido en {} pasos'.format(t_max))
is_a_memory, mu_mem = net.is_a_memory()
if is_a_memory == 'mem':
    print('La red ha convergido a la memoria {}.'
          .format(mu_mem + 1))
elif is_a_memory == 'neg':
    print('La red ha convergido al negativo de la memoria {}.'
          .format(mu_mem + 1))
else:
    print('La red no ha convergido a ninguna memoria.')

sup_list = []
for mu in range(net.p):
    mem = np.ones(net.n)
    mem = net.xi[:,mu]
    sup = hf.superposition(net.s, mem)
    sup_list.append(sup)
m = np.abs(np.array(sup_list)).max()

print()
print('m = {}'.format(m))

if plot_H_vals:
    plt.figure()
    plt.scatter(np.arange(t_conv + 1),H_vals)
    plt.xlabel('t')
    plt.ylabel('H')
    plt.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
    plt.title('Evolución de la energía de la red')

print()
print('Tiempo de ejecución: {}s'.format(time.time() - start_time))