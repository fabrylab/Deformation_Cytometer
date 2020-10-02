# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 09:10:28 2020

@author: Ben
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
import pylustrator
pylustrator.start()
# import pylustrator #to alter the appearance of the figure
# pylustrator.start()

# plt.close('all')
# ----------general fonts for plots and figures----------
#----------general fonts for plots and figures----------
font = {'family' : 'sans-serif',
        'sans-serif':['Arial'],
        'weight' : 'normal',
        #'size'   : 20
        }
"""
plt.rc('font', **font)
plt.rc('legend', fontsize=16)
plt.rc('axes', titlesize=20)
"""

C1 = '#1f77b4'  # blue-ish color
C2 = '#ff7f0e'  # orange-ish color

# fig1=plt.figure(1, (9, 4))
border_width = 0.2
ax_size = [0 + border_width, 0 + border_width,
           1 - 2 * border_width, 1 - 2 * border_width]
ax1 = plt.axes(ax_size)
e = 0.75

# draw a circle and an ellipse
T = np.arange(0, 2 * np.pi + 0.000000000000001, np.pi / 100)
y = np.sin(T)
x = np.cos(T)
xe = np.cos(T) + e * y
ax1.plot(x, y, '-', color=C1, linewidth=2)
ax1.plot(xe, y, '-', color=C2, linewidth=2)

# draw arrows
T = [-1, -0.8, -0.6, -0.4, -0.2, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0]
T = np.arcsin(T)
T = np.array(T) + np.pi
# T = np.hstack((T+3*np.pi/2, T+np.pi))
# T = np.arange(0, 2* np.pi+0.000000000000001,np.pi/12)
y = np.sin(T)
x = np.cos(T)
xe = np.cos(T) + e * y
# ax1.plot(x,y,'o', color = C1)
# ax1.plot(xe,y,'o', color = C2)
# y = np.array([-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0])
# x = y*0 -1
dx = e * y
dy = dx * 0
for i in range(len(x)):
    plt.arrow(x[i], y[i], dx[i], dy[i], zorder=3, width=0.035, length_includes_head=True, linewidth=0,
              facecolor=[0.5, .5, 0.5])

# draw parallelogram
x = [-1, -1, 1, 1]
x = np.array(x)
y1 = [-1, 1, 1, 1]
y2 = [-1, -1, -1, 1]
xe = x + np.array([-1, 1, -1, 1]) * e
ax1.fill_between(x, y1, y2, facecolor=C1, alpha=0.25, edgecolor=C1, linewidth=0)
ax1.fill_between(xe, y1, y2, facecolor=C2, alpha=0.25, edgecolor=C2, linewidth=0)

# draw major axis
t0 = 0.5 * (np.pi / 2 - np.arctan(-0.5 * e))
y = np.sin(t0)
x = np.cos(t0) + e * y
ax1.plot([0, x], [0, y], '--', color='black', linewidth=1.5)

# draw minor axis
t0 = 0.5 * (np.pi / 2 - np.arctan(-0.5 * e)) + np.pi / 2
y = np.sin(t0)
x = np.cos(t0) + e * y
ax1.plot([0, x], [0, y], '--', color='black', linewidth=1.5)
# drawradius r0
t0 = 2.7 * np.pi / 2
y = np.sin(t0)
x = np.cos(t0)
ax1.plot([0, x], [0, y], '--', color='black', linewidth=1.5)

# draw anlge theta
t1 = np.pi / 2 - np.arctan(e)
ax1.plot([1, 1 + 0.7 * np.cos(t1)], [0, 0.7 * np.sin(t1)], '-', color='black', linewidth=1)
ax1.plot([1, 1], [0, 0.7], '-', color='black', linewidth=1)
ar = patches.Arc((1, 0), 1.2, 1.2, angle=0, theta1=t1 * 180 / np.pi, theta2=90, linewidth=1, zorder=4)
ax1.add_patch(ar)

# draw arrow for shear deformation and coordinate origin
plt.arrow(1, 1, e, 0, zorder=3, width=0.035, length_includes_head=True, linewidth=0, facecolor=[0, 0, 0])
plt.arrow(0, 0, 0.5, 0, zorder=3, width=0.02, head_width=0.1, length_includes_head=True, linewidth=0,
          facecolor=[0.5, 0.5, 0.5])
plt.text(0.5, 0, '$x$', fontsize=8, color=[0.5, 0.5, 0.5], va='center')
plt.arrow(0, 0, 0, 0.5, zorder=3, width=0.02, head_width=0.1, length_includes_head=True, linewidth=0,
          facecolor=[0.5, 0.5, 0.5])
plt.text(0, 0.5, '$y$', fontsize=8, color=[0.5, 0.5, 0.5], va='bottom', ha='center')

#plt.show()

ax1.axis('equal')
ax1.set_axis_off()

plt.text(0.5, 0.5, '$a$', fontsize=10)
plt.text(-0.45, 0.1, '$b$', fontsize=10)
plt.text(-0.15, -0.5, '$r_0$', fontsize=10)
plt.text(1.02, 0.25, '$\\theta $', fontsize=10)

plt.text(1.1, 1.1, '$\\epsilon r_0 $', fontsize=10)

ax1 = plt.axes([0.5, 0.5, 1, 1])
ax1.axis('equal')
ax1.set_axis_off()

# %%draw rotated ellipse (shifted)
xshift = 3
t = -0.3
T = np.arange(0, 2 * np.pi + 0.000000000000001, np.pi / 100)
y = np.sin(T)
x = np.cos(T) + e * y
xe = xshift + x * np.cos(t) - y * np.sin(t)
ye = x * np.sin(t) + y * np.cos(t)
ax1.plot(xe, ye, '-', color=C2, linewidth=2)  # stress evaded ellipse
ax1.plot(xshift + x, y, '--', color=C2, linewidth=2)  # original elipse

if 0:
    ax1.plot([xshift + 0.8, xshift + 3], [1, 1], '-', color='black', linewidth=1)
    ax1.plot([xshift + 0.6, xshift + 2.5], [np.max(ye), np.max(ye)], '-', color='black', linewidth=2)
    plt.arrow(xshift + 2.8, 0.5, 0, 0.5, zorder=3, width=0.02, head_width=0.15, length_includes_head=True, linewidth=0,
              facecolor=[0, 0, 0])
    plt.arrow(xshift + 2.8, 0.5, 0, -0.5, zorder=3, width=0.02, head_width=0.15, length_includes_head=True, linewidth=0,
              facecolor=[0, 0, 0])
    plt.text(xshift + 2.6, 0.5, '$r_0$', fontsize=10, color=[0, 0, 0], va='center', ha='center', rotation=90)
    plt.arrow(xshift + 2.3, 0.5, 0, np.max(ye) - 0.5, zorder=3, width=0.02, head_width=0.15, length_includes_head=True,
              linewidth=0, facecolor=[0, 0, 0])
    plt.arrow(xshift + 2.3, 0.5, 0, -0.5, zorder=3, width=0.02, head_width=0.15, length_includes_head=True, linewidth=0,
              facecolor=[0, 0, 0])
    plt.text(xshift + 2.05, 0.4, '$r_{red}$', fontsize=10, color=[0, 0, 0], va='center', ha='center', rotation=90)

# draw major axis
t0 = 0.5 * (np.pi / 2 - np.arctan(-0.5 * e))
y = np.sin(t0)
x = np.cos(t0) + e * y
xe = x * np.cos(t) - y * np.sin(t)
ye = x * np.sin(t) + y * np.cos(t)
# ax1.plot([xshift-xe,xshift+xe],[-ye,ye],'--', color = 'black', linewidth = 1)
# draw anlge alpha
t1 = np.pi / 2 - t0 + t
r = 1.7
ax1.plot([xshift, xshift + r * np.cos(t1)], [0, r * np.sin(t1)], '-', color='black', linewidth=1)
ax1.plot([xshift, xshift + r], [0, 0], '-', color='black', linewidth=1)
ar = patches.Arc((xshift, 0), 3.2, 3.2, angle=0, theta1=0, theta2=t1 * 180 / np.pi, linewidth=1, zorder=4)
ax1.add_patch(ar)
plt.text(xshift + 1.1, 0.05, '$\\beta $', fontsize=10)

plt.savefig(r'strain_illustration.png', dpi=600, format='png')
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).axes[0].set_position([0.200000, 0.437500, 0.600000, 0.600000])
plt.figure(1).axes[1].set_position([0.170313, -0.070833, 0.696875, 0.700000])
#% end: automatic generated code from pylustrator
plt.show()
