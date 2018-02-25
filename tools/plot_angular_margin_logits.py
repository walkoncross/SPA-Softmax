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


theta_degree = np.linspace(0, 180, 91, endpoint=True)

theta = theta_degree * np.pi / 180.0
cos_theta = np.cos(theta)

zero_array = np.zeros_like(theta)

#----- modified softmax loss functions
# legend_strings = []
figsize = (10, 8)
fig = plt.figure(figsize=figsize)

legend_str = 'softmax(m=1)'
line, = plt.plot(theta_degree, cos_theta,
                 '-k', LineWidth=2, label=legend_str)
print line
ax_hanlders = [line]
legend_strings = [legend_str]

# plt.hold(True)  # deprecated


#----- logits of A-Softmax loss functions
Fai = np.zeros((4, theta.shape[0]))

line_styles = ['-r', '-g', '-b']
style_idx = -1

for m in [2, 3, 4]:
    #     if m==3：
    #         continue
    style_idx = style_idx + 1
    for k in range(m):
        idx = np.bitwise_and((theta >= k * np.pi / m),
                             (theta <= (k + 1) * np.pi / m))
#        print idx
        Fai[m - 1][idx] = np.power(-1, k) * np.cos(m * theta[idx]) - 2 * k

    legend_str = 'A-Softmax(m=%g)' % (m)

    line, = plt.plot(theta_degree, Fai[m - 1, :],
                     line_styles[style_idx],
                     LineWidth=1, label=legend_str)
    ax_hanlders.append(line)
    legend_strings.append(legend_str)

line_styles = [
    ['--r', '-.r', '-+r', '-or', '-sr'],
    ['--g', '-.g', '-+g', '-og', '-sg'],
    ['--b', '-.b', '-+b', '-ob', '-sb']
]

style_idx = -1

# ----- logits of A-Softmax loss functions with lambda (weighted A-Softmax and softmax)
for m in [2, 3, 4]:
    style_idx += 1
    if m < 4:
        continue

    style_idx2 = -1

    # for lambda_1 in [1, 2, 5, 10]:
    for lambda_1 in [5, 10]:
        style_idx2 += 1

        logits = (cos_theta * lambda_1 + Fai[m - 1, :]) / (1 + lambda_1)

        legend_str = 'A-Softmax(m=%g, $\\lambda$=%g)' % (
            m, lambda_1)

        line, = plt.plot(theta_degree, logits,
                         line_styles[style_idx][style_idx2],
                         LineWidth=1,
                         label=legend_str)

        ax_hanlders.append(line)
        legend_strings.append(legend_str)

# ----- logits of AMSoftmax
line_styles = [
    '-y', '--y', '-.y', '-+y',
]

style_idx = -1

for m in [0.35, 0.4, 0.45, 0.5]:
    style_idx = style_idx + 1

    logits = cos_theta - m

    legend_str = 'AMSoftmax(cos$\\theta$-%g)' % (m)

    line, = plt.plot(theta_degree,  logits,
                     line_styles[style_idx],
                     LineWidth=1, label=legend_str)

    ax_hanlders.append(line)
    legend_strings.append(legend_str)

# ----- logits of ArcFace/InsightFace

line_styles = [
    '-k', '--k', '-.k', '-+k', '-ok', '-sk'
]

style_idx = -1

for m in [0.5, 0.6, 0.7]:
    style_idx = style_idx + 1

    logits = np.cos(theta + m)

    legend_str = 'ArcFace(cos($\\theta$-%g))' % (m)

    line, = plt.plot(theta_degree, logits,
                     line_styles[style_idx],
                     LineWidth=1, label=legend_str)

    ax_hanlders.append(line)
    legend_strings.append(legend_str)

# plt.hold(False)  # deprecated


# ----- logits of SPA-Softmax loss functions by zhaoyafei
line_styles = [
    '-c', '--c', '-.c', '-+c',
    '-m', '--m', '-.m', '-+m',
]

style_idx = -1
# for alpha in [2, 3, 4, 5]:
# for alpha in [2, 3, 4, 5, 6, 7, 8]:
for alpha in [1.5, 2, 2.5, 3, 3.5, 4]:
#for alpha in [2, 3, 4]:
    style_idx = style_idx + 1

    logits = alpha * cos_theta - (alpha - 1)

    legend_str = 'SPA-Softmax(%g*cos$\\theta$-%g)' % (alpha, alpha - 1)

    line, = plt.plot(theta_degree, logits,
                     line_styles[style_idx],
                     LineWidth=1, label=legend_str)

    ax_hanlders.append(line)
    legend_strings.append(legend_str)

# plt.hold(False)  # deprecated
plt.legend(tuple(ax_hanlders), tuple(legend_strings), loc='lower left')
# plt.legend(tuple(ax_hanlders), loc='lower left')
plt.xticks(np.arange(0, 181, 30))
plt.grid(True)

plt.savefig('Angular_Margin_Logits.png')
plt.show()
