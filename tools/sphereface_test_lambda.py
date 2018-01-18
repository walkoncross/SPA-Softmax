# -*- coding = utf-8 -*-
"""
Created on Sun Nov 05 07 =52 =30 2017

@author = zhaoy
"""
import numpy as np
import matplotlib.pyplot as plt

base = 1000

#gamma = 0.12
gamma = 0.0024
power = 1
lambda_min = 5
#start_iter = 0
#n_iter = 28000
#iter_step = 100
start_iter = 0
n_iter = 84000
iter_step = 100

##params for sphereface's issue:https://github.com/wy1iu/sphereface/issues/14
## for first train with SINGLE MODE, and then finetune with QUADRAPLE MODE
## I think this setting is meaningless.
#base = 1000
#gamma = 0.5
#power = 1
#lambda_min = 10
#start_iter = 0
#n_iter = 28000
#iter_step = 100

##params for sphereface
#base = 1000
#gamma = 0.12
#power = 1
#lambda_min = 5
#start_iter = 0
#n_iter = 28000
#iter_step = 100
#start_iter = 10000
#n_iter = 18000
#iter_step = 100

##params for Large-margin repo
#base = 1000
#gamma = 0.000025
#power = 35
#start_iter = 0
#lambda_min = 5
##n_iter = 28000
##iter_step = 100
#start_iter = 10000
#n_iter = 18000
#iter_step = 100

iters = np.arange(0, n_iter, iter_step) + start_iter
clip_line = np.ones( iters.shape) * lambda_min

lambdas = np.zeros( iters.shape)
for i in range(len( iters)):
    lambdas[i] = base * ( 1 + gamma * iters[i]) ** (-power)

print 'lambdas.max: ', lambdas.max()
print 'lambdas.min: ', lambdas.min()
print 'lambdas.mean: ', lambdas.mean()

print 'lambdas: ', lambdas

lambdas_clip = np.maximum(lambdas, lambda_min)

print 'lambdas_clip.max: ', lambdas_clip.max()
print 'lambdas_clip.min: ', lambdas_clip.min()
print 'lambdas_clip.mean: ', lambdas_clip.mean()

plt.plot(iters, clip_line, hold=True)
plt.plot(iters, lambdas, hold=True)
plt.plot(iters, lambdas_clip, hold=True)
#plt.plot(iters, lambdas_clip-lambdas, hold=True)


#x=np.linspace(0,np.pi*2,1000)
#
#plt.plot(x,np.sin(x),hold=True)
#plt.plot(x,np.sin(x)**2,hold=True)
plt.show()