# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 22:36:52 2017

@author: zhaoy

This python script is modified from a Matlab script.
Do not mind some strange things.
"""

import numpy as np

import psutil

if psutil.LINUX:
    import matplotlib
    matplotlib.use('agg')

import matplotlib.pyplot as plt


theta = np.linspace(0, 180, 91, endpoint=True) * np.pi / 180.0
cos_theta = np.cos(theta)

zero_array = np.zeros_like(theta)

#----- modified softmax loss functions
# legend_strings = []
figsize = (10, 8)
fig = plt.figure(figsize=figsize)

legend_str = 'softmax(m=1)'
line, = plt.plot(theta, cos_theta,
                 '-b', LineWidth=2, label=legend_str)
print line
ax_hanlders = [line]
legend_strings = [legend_str]

# plt.hold(True)  # deprecated


#----- A-softmax loss functions
Fai = np.zeros((4, theta.shape[0]))

line_styles_0 = ['-r', '-g', '-m']
style_cnt = -1

for m in [2, 3, 4]:
    #     if m==3：
    #         continue
    style_cnt = style_cnt + 1
    for k in range(m):
        idx = np.bitwise_and((theta >= k * np.pi / m),
                             (theta <= (k + 1) * np.pi / m))
#        print idx
        Fai[m - 1][idx] = np.power(-1, k) * np.cos(m * theta[idx]) - 2 * k

    legend_str = 'A-Softmax(m=%d, $\\lambda$=0)' % (m)

    line, = plt.plot(theta, Fai[m - 1, :],
                     line_styles_0[style_cnt], LineWidth=2, label=legend_str)
    ax_hanlders.append(line)
    legend_strings.append(legend_str)

#line_styles_1 = [
#    ['--r', '-+r', '-sr', '-or', '-dr'],
#    ['--g', '-+g', '-sg', '-og', '-dg'],
#    ['--m', '-+m', '-sm', '-om', '-dm']
#]
#
#style_cnt = -1
#
##----- A-softmax loss functions with lambda (weighted A-softmax and softmax)
#
#for m in [2, 3, 4]:
#    style_cnt = style_cnt + 1
##     if m<4：
##         continue
#
#    style_cnt2 = -1
#
#    # for lambda_1 in [1, 2, 5, 10]:
#    for lambda_1 in [5, 10]:
#        style_cnt2 = style_cnt2 + 1
#
#        legend_str = 'A-Softmax(m=%d, $\\lambda$=%d)' % (
#            m, lambda_1)
#
#        line, = plt.plot(theta, (cos_theta * lambda_1 + Fai[m - 1, :]) / (1 + lambda_1),
#                         line_styles_1[style_cnt][style_cnt2],
#                         LineWidth=2,
#                         label=legend_str)
#
#        ax_hanlders.append(line)
#        legend_strings.append(legend_str)

line_styles_2 = [
    '--b', '-+b', '-sb', '-ob', '-db',
    '--k', '-+k', '-sk', '-ok', '-dk',
    '--y', '-+y', '-sy', '-oy', '-dy'
]

style_cnt = -1
# for alpha in [2, 3, 4, 5]:
# for alpha in [2, 3, 4, 5, 6, 7, 8]:
# for alpha in [1.5, 2, 2.5, 3, 3.5, 4]:
for alpha in [2, 3, 4]:
    style_cnt = style_cnt + 1

    legend_str = 'QA-Softmax(%d*cos$\\theta$-%d)' % (alpha, alpha - 1)

    line, = plt.plot(theta,  alpha * cos_theta - alpha + 1,
                    line_styles_2[style_cnt],
                    LineWidth=2, label=legend_str)

    ax_hanlders.append(line)
    legend_strings.append(legend_str)

# plt.hold(False)  # deprecated

plt.legend(tuple(ax_hanlders), tuple(legend_strings), loc='lower left')
# plt.legend(tuple(ax_hanlders), loc='lower left')
plt.grid(True)

plt.savefig('QA-Softmax_vs_A-Softmax.png')
plt.show()
