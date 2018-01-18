# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 07:36:29 2017

@author: zhaoy
"""


import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

min_val = 0
max_val = 20
step = 0.1
num_classes = [10, 100, 1000, 10000, 100000]
#num_classes = [10572, 13403, 58207, 78771, 100000]

#norm = np.arange(min_val, max_val, step)
#squared_norm = np.square(norm)
squared_norm = np.arange(min_val, max_val + step, step)

prefixed_prob = 0.9

probs_list = []
for n_cls in num_classes:
    print '\n===> num of classes: ', n_cls
    exp_alpha = np.exp(squared_norm)
    exp_alpha_inv = 1.0 / exp_alpha
    probs = exp_alpha / (exp_alpha + (n_cls - 1) * exp_alpha_inv)
    probs_list.append(probs)

    idx = np.where(squared_norm==1)[0]
    print 'prob=%g, when alpha_list = 1 (no scale after norm)' % probs[idx]

    loss_only_norm = -np.log(probs[idx])
    print 'softmax loss = %g, when alpha_list = 1 (no scale after norm)' % loss_only_norm

    print 'probs.max: ', probs.max()
    print 'probs.min: ', probs.min()
    print 'probs.mean: ', probs.mean()

#    print 'probs: ', probs

    label = "C=%d" % n_cls

    plt.plot(squared_norm, probs, hold=True, label=label)

plt.xlabel('alpha')
plt.ylabel('prob')
plt.title('prob vs. alpha in normface')

plt.xticks(np.arange(min_val, max_val+1, 1))
plt.yticks(np.arange(0, 1 + 0.1, 0.1))

plt.grid(True)
plt.legend()

plt.savefig('normface_softmax_prob_bound.png')

plt.show()


plt.figure()

for n_cls in num_classes:
    n_cls = float(n_cls)
    print '\n===> num of classes: ', n_cls
    probs_inv = 1 + (n_cls-1) * np.exp(-n_cls / (n_cls-1) * squared_norm)
    loss = np.log(probs_inv)

    print 'loss.max: ', loss.max()
    print 'loss.min: ', loss.min()
    print 'loss.mean: ', loss.mean()

    idx = np.where(squared_norm==1)[0]
    print 'index=%g when norm=1' % idx
    print 'softmax loss = %g, when norm = 1 (no scale after norm)' % loss[idx]

    loss_1 = np.log(1 + (n_cls-1) * np.exp(-n_cls / (n_cls-1)))
    print 'softmax loss = %g, when norm = 1 (no scale after norm)' % loss_1

#    print 'loss: ', loss

    label = "C=%d" % n_cls

#    plt.plot(norm, loss, hold=True, label=label)
    plt.plot(squared_norm, loss, hold=True, label=label)

plt.xticks(np.arange(min_val, max_val+1, 1))
plt.yticks(np.arange(0, 12 + 1, 1))
plt.xlabel('Squared Norm l^2')
plt.ylabel('Loss Bound')
plt.title('Loss Bound vs. Squared Norm')

plt.grid(True)
plt.legend()
plt.savefig('normface_softmax_loss_bound.png')
plt.show()