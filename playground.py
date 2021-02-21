#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:31:21 2021

@author: gino
"""

import hopfield as hf
import time

start_time = time.time()
print_arrays = False

n = 5
p = 2
t_max = 100

print('Generando red...')
net = hf.HopfieldNetwork(n, p)
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
conv, t_conv = net.evolve(print_arrays=print_arrays, t_max=t_max)
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
print()
print('Tiempo de ejecución: {}s'.format(time.time() - start_time))