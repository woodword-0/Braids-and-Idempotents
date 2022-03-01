import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import *
import sympy as sp
from sympy import *
import sympy.matrices.matrices
from sympy.abc import t
import scipy
from scipy.stats import norm
# generate random numbers from N(0,1)
f = 20
X = np.linspace(0,1,100)
# X = 2*X
y = norm.rvs(size=100,loc=0,scale=1)
t = np.linspace(0, (2* np.pi),100)
g = np.cos(t)
x = np.cos(-2*np.pi*f*t)
x = g*x
y = np.sin(-2*np.pi*f*t)
y = g*y
ydist = y-min(y)
r = np.sqrt(x**2 + y**2)
theta = np.arctan(y/x)

t = np.linspace(0, (2* np.pi),100)
rads = 3*rads
X1 = ydist*np.cos(3*rads)
y1 = ydist*np.sin(3*rads)
Xmid = X1/2
Xmid
plt.axes(projection = 'polar')
plt.polar(x,y)
plt.show()

fig, ax = plt.subplots()
ax.plot(x, y, linewidth=2.0)
plt.show()
# ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
#        ylim=(0, 8), yticks=np.arange(1, 8))

# # setting the radius
# r = 2

# # creating an array containing the
# # radian values
# rads = np.arange(0, (2 * np.pi), 0.01)

# # plotting the circle
# for rad in rads:
# 	plt.polar(rad, r, 'g.')



plt.show()

import numpy as np
import matplotlib.pyplot as plt


# setting the axes projection as polar
plt.axes(projection = 'polar')

# setting the radius
r = 2

# creating an array containing the
# radian values
rads = np.arange(0, (2 * np.pi), 0.01)

# plotting the circle
for rad in rads:
	plt.polar(rad, r, 'g.')

# display the Polar plot
plt.show()

# point1  = np.array([1, 1, 0])
point2  = np.array([1, 1, 1])

# normal1 = np.array([0, 0, 1])
normal2 = np.array([0, 0, 21])

# # a1 = normal1[0]
# # b1 = normal1[1]
# # c1 = normal1[2]


# a2 = normal2[0]
# b2 = normal2[1]
# c2 = normal2[2]

# # a plane is a*x+b*y+c*z+d=0
# # [a,b,c] is the normal. Thus, we have to calculate
# # d and we're set
# # d1 = -point.dot(normal1)
# d2 = -point.dot(normal2)

# # create x,y
# # xx, yy = np.meshgrid(range(10), range(10))
# uu, vv = np.meshgrid(range(10), range(10))

# # calculate corresponding z
# # z1 = (-a1 * xx - b1 * yy - d1) * 1. /c1
# z2 = (-a2 * xx - b2 * yy - d2) * 1. /c2
# fig = plt.figure(figsize = (8,8))
# ax = plt.axes(projection='3d')
# ax.grid()

# # plot the surface
# plt3d = plt.figure().gca(projection='3d')
# # plt3d.plot_surface(xx, yy, z1, alpha=0.2)
# plt3d.plot_surface(uu, vv, z2, alpha=0.2)
# ax.set_xlabel('x', labelpad=20)
# ax.set_ylabel('y', labelpad=20)
# ax.set_zlabel('z', labelpad=20)


# # #and i would like to plot this point : 
# # ax.scatter(point2[0] , point2[1] , point2[2],  color='green')

# plt.show()

def planes(p,n):
    fig = plt.figure()
    a = n[0]
    b = n[1]
    c = n[2]
    d = -p.dot(n)
    xx, yy = np.meshgrid(range(-100,100), range(-100,100))
    z = (- a* xx - b* yy - d) * 1. /c
    return [xx,yy,z]

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
P = [np.array([0,0,2*i])for i in range(4)]

for p in P:ax.plot_surface(planes(p,normal2)[0],planes(p,normal2)[1],planes(p,normal2)[2], alpha=0.2)
plt.show()
   # Add an axes
    # ax = fig.add_subplot(111,projection='3d')
    # # plot the surface
    # ax.plot_surface(planes(p,n), alpha=0.2)
# def planes(p,n):
#     ax = fig.add_subplot(111,projection='3d')
#     a = n[0]
#     b = n[1]
#     c = n[2]
#     d = -p.dot(n)
#     xx, yy = np.meshgrid(range(-10,10), range(-10,10))
#     z = (-a * xx - b * yy - d) * 1. /c
#     return ax.plot_surface(xx, yy, z, alpha=0.2)

    
    
    # plot_surface(xx, yy, z, alpha=0.2)

type(planes(p,n)) #.planes(p1,n1)
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
planes(p,n).planes(p1,n1)
plt.show()
planes(p,n)

def planes(p,n):
    fig = plt.figure()
    a = n[0]
    b = n[1]
    c = n[2]
    d = -p.dot(n)
    xx, yy = np.meshgrid(range(-10,10), range(-10,10))
    z = (- a* xx - b* yy - d) * 1. /c
    return [xx,yy,z]
# Add an axes
fig = plt.figure()
P = [np.array([0,0,2*i])for i in range(4)]
ax = fig.add_subplot(111,projection='3d')
n = np.array([0, 0, 1])
planes = [planes(p,n) for p in P]
planes[0][2][0]
points1 = [np.array([0,0,2*i])for i in range(4)]
points2 = [np.array([5 + i/2,0,2*i])for i in range(4)]
points3 = [np.array([-5 - i/2,0,2*i])for i in range(4)]
for p in planes:
    print(len(p))
color = ["red","orange","yellow","green"] #,"blue","black","white"]
# ax.scatter(points1[i],points2[i],points3[i],  color= color[i]) #1f77b4
i=-1
for p in planes:
    i = i+1
    ax.plot_surface(p[0],p[1],p[2],alpha = 0.5)
    ax.scatter(points1[i][0],points1[i][1],points1[i][2],color= color[i]) #1f77b4
    ax.scatter(points2[i][0],points2[i][1],points2[i][2],color= color[i]) #1f77b4
    ax.scatter(points3[i][0],points3[i][1],points3[i][2],color= color[i]) #1f77b4

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)
plt.show()
planes[1][0][0][0]
# x = planes(p,n)[0]
# y = planes(p,n)[1]
# z = planes(p,n)[2]
# x1 = planes(p1,n1)[0]
# y1 = planes(p1,n1)[1]
# z1 = planes(p1,n1)[2]
# ax.plot_surface(x,y,z,alpha=0.2)
# ax.plot_surface(x1,y1,z1,alpha=0.2)

# ax.plot_surface(planes(p,n)[0],planes(p,n)[1],planes(p,n)[2], alpha=0.2)
# ax.plot_surface(planes(p1,n1)[0],planes(p1,n1)[1],planes(p1,n1)[2], alpha=0.2)
# plt.show()




ax = fig.add_subplot(111,projection='3d')
surface = planes(p,n)
ax.planes(p,n) #.planes(p1,n1)


    # Set axes label
    # ax.set_xlabel('x', labelpad=20)
    # ax.set_ylabel('y', labelpad=20)
    # ax.set_zlabel('z', labelpad=20)
    # return ax #plt.show()

braids(3)
# Create the figure
fig = plt.figure()
p  = np.array([1, 1, 1])
p1  = np.array([1, 1, 0])
n = np.array([0, 0, 1])
n1 = np.array([0, 0, 1])
planes(p,n) #.planes(p1,n1)
plt.show()
n = np.array([0, 0, 1])
n1 = np.array([0, 0, 1])

a1 = normal1[0]
b1 = normal1[1]
c1 = normal1[2]
a2 = normal2[0]
b2 = normal2[1]
c2 = normal2[2]

d1 = -point1.dot(normal1)
d2 = -point2.dot(normal2)
# create x,y
xx, yy = np.meshgrid(range(-10,10), range(-10,10))
uu, vv = np.meshgrid(range(-10,10), range(-10,10))

# calculate corresponding z
z1 = (-a1 * xx - b1 * yy - d1) * 1. /c1
z2 = (-a2 * uu - b2 * vv - d2) * 1. /c2


# Add an axes
ax = fig.add_subplot(111,projection='3d')

# plot the surface
ax.plot_surface(xx, yy, z1, alpha=0.2)
ax.plot_surface(uu,vv,z2, alpha=0.7)
# and plot the point 
#z = 0
ax.scatter(-5 , 1, 0,  color='red')
ax.scatter(0 , 0, 0,  color='green')
ax.scatter(5 , 0 , 0,  color='blue')
#z = 1
ax.scatter(-5 , 0 , 1,  color='red')
ax.scatter(0 , 0, 1,  color='green')
ax.scatter(5 , 0 , 1,  color='blue')

 
x = [0,5]
y = [0,0]
z = [0,1] 
x1 = [0,5]
y1 = [0,1]
z1 = [1,0]
# plotting
ax.plot3D(x, y, z)
ax.plot3D(x1, y1, z1)

# tt = np.meshgrid(range(-100,100))
# l = [0,0,0]*tt +(1 - tt)*[0,0,1]
# uu, vv = np.meshgrid(range(-100,100), range(-100,100))

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation

df = y_periodic_embedded
len(df[:,0])
z = np.zeros(42054)
z
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')


sc = ax.scatter([],[],[], c='darkblue', alpha=0.5)

def update(i):
    sc._offsets3d = (df[:i,0], df[:i,1], df[:i,2])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-30,30)
ax.set_ylim(-30,30)
ax.set_zlim(-30,30)

ani = matplotlib.animation.FuncAnimation(fig, update, frames=len(df), interval=70)

plt.tight_layout()
plt.show()
