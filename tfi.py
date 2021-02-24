#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 20:20:27 2021

@author: gino
"""
import hopfield as hf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()

print_arrays = False
sync = False

n = 500
p_start = 10
p_stop = 100 + p_start
p_step = 10
t_max = 100

parameters = [n, p_start, p_stop, p_step]
parameters_list = np.array([[2**k*i for i in parameters] for k in range(3)])
neurons_list = parameters_list[:,0]
alpha_vals = np.array([p/n for p in range(p_start, p_stop, p_step)])

data = pd.DataFrame()
data['alpha'] = alpha_vals

for k in range(len(parameters_list)):
    n, p_start, p_stop, p_step = parameters_list[k]
    m_mean_vals = []
    for p in range(p_start, p_stop, p_step):
        print('n = {},    p = {}'.format(n, p))      
        print('Generando red...')
        net = hf.HopfieldNetwork(n, p)
        print('Generando patrones aleatorios...')
        net.xi = hf.generate_random_array(n, p)
        print('Inicializando los pesos...')
        net.w = hf.weights(net.xi)
        m_vals = []
        for mu in range(net.p):
            print()
            # pattern = np.ones(net.n, dtype=int)
            # pattern[:] = net.xi[:, mu]
            pattern = hf.generate_random_pattern(n)
            print('Colocando la red en la memoria #{}/{}...'
                  .format(mu+1, net.p))
            net.set_conf(pattern)
            print('Evolucionando...')
            conv, t_conv = net.evolve(print_arrays=print_arrays, t_max=t_max,\
                                      sync=sync)
            if conv:
                if t_conv==1:
                        print('La red ha alcanzado el estado estacionario en \
1 paso.')
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
            m = hf.superposition(net.s, pattern)
            print('m = {}'.format(m))
            m_vals.append(m)
        m_mean = np.array(m_vals).mean()
        m_err = np.array(m_vals).std()/np.sqrt(net.p)
        print()
        print('m_mean = {}'.format(m_mean))
        print()
        m_mean_vals.append(m_mean)
    data[str(neurons_list[k])] = m_mean_vals

data.to_csv('Data/data.csv', float_format='%.3f', index=False)

for k in range(len(parameters_list)):
    label = '{} neuronas'.format(neurons_list[k])
    plt.scatter(alpha_vals, data[str(neurons_list[k])], label=label)
plt.title('Capacidad de almacenamiento de una red de Hopfield.')
plt.axvline(x=0.138, color='r', label='p/n = 0.138')
plt.xlabel(r'$p/n$')
plt.ylabel(r'$\overline{m}$')
plt.legend(loc='lower left')

plt.savefig('Figs/fig.pdf')

print('Execution time: {}s'.format(time.time() - start_time))
