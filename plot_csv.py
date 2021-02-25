#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 10:43:40 2021

@author: gino
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('Data/data.csv')
alpha_vals = df['alpha'].to_numpy()
m_mean_list = df.to_numpy()[:,1:4]
neurons = df.columns.to_list()[1:4]

for i in range(len(neurons)):
    m_mean_vals = m_mean_list[:,i]
    label = '{} neuronas.'.format(neurons[i])
    plt.scatter(alpha_vals, m_mean_vals, label=label)

plt.axvline(x=0.138, color='r', label='p/n = 0.138')
plt.xticks(np.linspace(0.0,0.2,11))
plt.xlabel(r'$p/n$')
plt.ylabel(r'$\overline{m}$')
plt.legend(loc='lower left')
plt.show()
plt.savefig('Figs/fig.pdf')
