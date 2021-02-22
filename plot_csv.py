#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 10:43:40 2021

@author: gino
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Data/data.csv')
alpha_vals = data['alpha'].to_numpy()
m_mean_list = data.to_numpy()[:,1:4]
neurons = data.columns.to_list()[1:4]

for i in range(len(neurons)):
    m_mean_vals = m_mean_list[:,i]
    label = '{} neuronas.'.format(neurons[i])
    plt.scatter(alpha_vals, m_mean_vals, label=label)

plt.title('Capacidad de almacenamiento de una red de Hopfield.')
plt.axvline(x=0.138, color='r', label='p/n = 0.138')
plt.xlabel(r'$p/n$')
plt.ylabel(r'$\overline{m}$')
plt.legend(loc='lower left')
plt.show()
plt.savefig('Figs/capacity.pdf')
