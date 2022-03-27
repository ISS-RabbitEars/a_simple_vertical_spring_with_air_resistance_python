import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation

def integrate(ic, ti, p):
	y, v = ic
	m, k, yeq, gc, rho, cd, ar = p

	print(ti)

	return [v, AY.subs({M:m, K:k, YEQ:yeq, g:gc, Y:y, Ydot:v, RHO:rho, CD:cd, Ar:ar})]


M, K, YEQ, g, t = sp.symbols('M K YEQ g t')
RHO, CD, Ar = sp.symbols('RHO CD Ar')
Y = dynamicsymbols('Y')

Ydot = Y.diff(t, 1)

T = sp.Rational(1, 2) * M * Ydot**2
V = sp.Rational(1, 2) * K * (Y - YEQ)**2 + M * g * Y

L = T - V

dLdY = L.diff(Y, 1)
dLdYdot = L.diff(Ydot, 1)
ddtdLdYdot = dLdYdot.diff(t, 1)

F = sp.Rational(1, 2) * RHO * sp.sign(Ydot) * Ydot**2 * CD * Ar

dL = ddtdLdYdot - dLdY + F

sol = sp.solve(dL, Y.diff(t, 2))

AY = sp.simplify(sol[0])

#---------------------------------------------------

gc = 9.8
m = 1
k = 10 
yeq = -1
yo = -1
vo = 0
rho = 1.225
cd = 0.47
rad = 0.25
ar = np.pi * rad**2


p = m, k, yeq, gc, rho, cd, ar
ic = yo, vo

tf = 120
nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)

yv = odeint(integrate, ic, ta, args=(p,))

y = yv[:,0]
v = yv[:,1]

ke = np.asarray([T.subs({M:m, Ydot:i}) for i in v])
pe = np.asarray([V.subs({M:m, K:k, YEQ:yeq, g:gc, Y:i}) for i in y])
E = ke + pe

fig, a=plt.subplots()

xline = 0
xmax = xline + 2 * rad
xmin = xline - 2 * rad
ymax = 2 * rad
ymin = min(y) - 2 * rad
nl = int(np.ceil((max(np.abs(y))+rad)/(2*rad)))
xl = np.zeros((nl,nframes))
yl = np.zeros((nl,nframes))
for i in range(nframes):
	l = (np.abs(y[i])/nl)
	yl[0][i] = y[i] + rad + 0.5*l
	for j in range(1,nl):
		yl[j][i] = yl[j-1][i] + l
	for j in range(nl):
		xl[j][i] = xline+((-1)**j)*(np.sqrt(rad**2 - (0.5*l)**2))

def run(frame):
	plt.clf()
	plt.subplot(141)
	circle=plt.Circle((xline,y[frame]),radius=rad,fc='xkcd:red')
	plt.gca().add_patch(circle)
	plt.plot([xline,xl[0][frame]],[y[frame]+rad,yl[0][frame]],'xkcd:cerulean')
	plt.plot([xl[nl-1][frame],xline],[yl[nl-1][frame],rad],'xkcd:cerulean')
	for i in range(nl-1):
		plt.plot([xl[i][frame],xl[i+1][frame]],[yl[i][frame],yl[i+1][frame]],'xkcd:cerulean')
	plt.title("A Simple Vertical\nSpring With Air Resistance")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([xmin,xmax])
	plt.ylim([ymin,ymax])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(1,4,(2,4))
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=0.5)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=0.5)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.0)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')
	ax.yaxis.set_label_position("right")
	ax.yaxis.tick_right()

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('vertical_spring_w_air_resistance.mp4', writer=writervideo)
plt.show()
