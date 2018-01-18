# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 22:36:52 2017

@author: zhaoy
"""

import numpy as np
import matplotlib.pyplot as plt

min_val = 0
max_val = 20
step = 0.1
num_classes = [10, 100, 1000, 10000, 100000]
#num_classes = [10572, 13403, 58207, 78771, 100000]

alpha_list = np.arange(min_val, max_val+step, step)

probs_list = []

plt.figure()

prefixed_prob = 0.9

for n_cls in num_classes:
    print '\n===> num of classes: ', n_cls
    exp_alpha = np.exp(alpha_list)
    exp_alpha_inv = 1.0 / exp_alpha
    probs = exp_alpha / (exp_alpha + n_cls - 2 + exp_alpha_inv)
    probs_list.append(probs)

    idx = np.where(alpha_list==1)[0]
    print 'prob=%g, when alpha_list = 1 (no scale after norm)' % probs[idx]

    loss_only_norm = -np.log(probs[idx])
    print 'softmax loss = %g, when alpha_list = 1 (no scale after norm)' % loss_only_norm

    print 'probs.max: ', probs.max()
    print 'probs.min: ', probs.min()
    print 'probs.mean: ', probs.mean()

#    print 'probs: ', probs

    label = "C=%d" % n_cls

    plt.plot(alpha_list, probs, hold=True, label=label)

    alpha_lower = np.log(prefixed_prob * (n_cls-2) / (1-prefixed_prob))
    print 'alpha=%g for prefixed prob=%g' % (alpha_lower, prefixed_prob)

plt.xlabel('alpha')
plt.ylabel('prob')
plt.title('prob vs. alpha in L2-softmax')

plt.xticks(np.arange(min_val, max_val+1, 1))
plt.yticks(np.arange(0, 1 + 0.1, 0.1))

plt.grid(True)
plt.legend()

plt.savefig('l2_softmax_prob_bound.png')

plt.show()

plt.figure()

for i in range(len(num_classes)):
    n_cls = num_classes[i]
    probs = probs_list[i]
    print '\n===> num of classes: ', n_cls
    loss = -np.log(probs)

    idx = np.where(alpha_list==1)[0]
    print 'prob=%g, when alpha_list = 1 (no scale after norm)' % loss[idx]

    loss_only_norm = -np.log(loss[idx])
    print 'softmax loss = %g, when alpha_list = 1 (no scale after norm)' % loss_only_norm

    print 'loss.max: ', loss.max()
    print 'loss.min: ', loss.min()
    print 'loss.mean: ', loss.mean()

#    print 'loss: ', loss

    label = "C=%d" % n_cls

    plt.plot(alpha_list, loss, hold=True, label=label)

plt.xlabel('alpha')
plt.ylabel('loss')
plt.title('loss vs. alpha in L2-softmax')

plt.xticks(np.arange(min_val, max_val+1, 1))
plt.yticks(np.arange(0, 12 + 1, 1))

plt.grid(True)
plt.legend()

plt.savefig('l2_softmax_loss_bound.png')

plt.show()