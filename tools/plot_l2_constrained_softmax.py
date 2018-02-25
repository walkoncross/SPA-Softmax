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
#num_classes = [10, 100, 1000, 10000, 100000]
num_classes = [8631, 10572, 13403, 58207, 78771, 85000, 100000]

scale_list = np.arange(min_val, max_val+step, step)
# scale is called 'alpha' in the L2-softmax paper

targe_scales = [12, 16, 32, 64]

probs_list = []

plt.figure()

prefixed_prob = 0.9

for n_cls in num_classes:
    print '\n===> num of classes: ', n_cls
    exp_scale = np.exp(scale_list)
    exp_scale_inv = 1.0 / exp_scale
    probs = exp_scale / (exp_scale + n_cls - 2 + exp_scale_inv)
    probs_list.append(probs)

    idx = np.where(scale_list==1)[0]
    print 'prob=%g, when scale = 1 (no scale after norm)' % probs[idx]

    loss_only_norm = -np.log(probs[idx])
    print 'softmax loss = %g, when scale = 1 (no scale after norm)' % loss_only_norm

    print 'probs.max: ', probs.max()
    print 'probs.min: ', probs.min()
    print 'probs.mean: ', probs.mean()

#    print 'probs: ', probs

    label = "C=%d" % n_cls

    plt.plot(scale_list, probs, hold=True, label=label)

    scale_lower = np.log(prefixed_prob * (n_cls-2) / (1-prefixed_prob))
    print 'scale=%g for prefixed prob=%g' % (scale_lower, prefixed_prob)

plt.xlabel('scale')
plt.ylabel('prob')
plt.title('prob vs. scale in L2-softmax')

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

    print 'loss.max: ', loss.max()
    print 'loss.min: ', loss.min()
    print 'loss.mean: ', loss.mean()

    idx = np.where(scale_list==1)[0]
    print 'softmax loss=%g, when scale = 1 (no scale after norm)' % loss[idx]

    exp_scale2 = np.exp(np.array(targe_scales))
    exp_scale2_inv = 1.0 / exp_scale2
    probs2 = exp_scale2 / (exp_scale2 + n_cls - 2 + exp_scale2_inv)
    loss2 = -np.log(probs2)

    for i, ss in enumerate(targe_scales):
        print 'softmax loss=%g, when scale = %f' % (loss2[i], ss)

#    print 'loss: ', loss

    label = "C=%d" % n_cls

    plt.plot(scale_list, loss, hold=True, label=label)

plt.xlabel('scale')
plt.ylabel('loss')
plt.title('loss vs. scale in L2-softmax')

plt.xticks(np.arange(min_val, max_val+1, 1))
plt.yticks(np.arange(0, 12 + 1, 1))

plt.grid(True)
plt.legend()

plt.savefig('l2_softmax_loss_bound.png')

plt.show()