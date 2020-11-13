import numpy as np 
from geometry import e

one3rd = 0.3333333333333333
one24th = 0.0416666666666667
one12th = 0.0833333333333333
one30th = 0.0333333333333333
one6th = 0.1666666666666667
two3rd = 0.6666666666666667
four3rd = 1.3333333333333333
eight3rd = 2.666666666666667
PI = 3.1415926535897931

'''
Compute equlibrium distribution function as described in Denniston et. al. (2004)
'''
def calc_f_equilibrium(f, Rho, u, Temp):
	
	#sigma = -Rho * Temp (3x3 uniform matrix defined at each point in grid)
	sigma = np.ones(Rho.shape + (3, 3), dtype=Rho.dtype)
	for i in range(3):
		for j in range(3):
			sigma[..., i, j] = -Rho * Temp
	tr_sigma = -Rho

	u2 = np.einsum('...i,...i', u, u)	# u \cdot u
	ue = np.einsum('ab,cdeb', e, u)		# u \cdot e
	ue2 = ue * ue						# (u \cdot e)^2

	A2 = -one24th * one3rd * tr_sigma
	A1 = -one3rd * one3rd * tr_sigma
	A0 = Rho - 8. * A2 - 6. * A1

	B2 = one24th
	B1 = one3rd
	B0 = 0.

	C2 = -0.5 * one24th
	C1 = -2. * one12th
	C0 = -one3rd

	D2 = 0.0625
	D1 = 0.5
	D0 = 0.
	
	E2 = -sigma
	E2[..., 0, 0] += tr_sigma * one3rd
	E2[..., 1, 1] += tr_sigma * one3rd
	E2[..., 2, 2] += tr_sigma * one3rd
	E2 *= 0.0625
	E1 = 8. * E2
	E0 = np.zeros_like(sigma)

	for i in range(f.shape[0]):
		if i == 0:
			A, B, C, D, E = A0, B0, C0, D0, E0
		elif i <= 6:
			A, B, C, D, E = A1, B1, C1, D1, E2
		else:
			A, B, C, D, E = A2, B2, C2, D2, E2
		
		E = np.einsum('...ab,a,b', E, e[i], e[i])
		#E = 0
		f[i] = A + Rho * (B * ue[i] + C * u2 + D * ue2[i] + E)
	
'''
Compute the matrix W = \grad u or W_{ij} = \partial_i u_j 
'''
def cal_W(W, u, nextf):
	idz_m = nextf[1][1:]
	idz_p = nextf[2][1:]
	idy_m = nextf[3][1:]
	idy_p = nextf[4][1:]
	idx_m = nextf[5][1:]
	idx_p = nextf[6][1:]

	W[..., 0, :] = 0.5 * (u[idz_p] - u[idz_m])	#W_{zi} = \partial_z u_i
	W[..., 1, :] = 0.5 * (u[idy_p] - u[idy_m])	#W_{yi} = \partial_y u_i
	W[..., 2, :] = 0.5 * (u[idx_p] - u[idx_m])	#W_{xi} = \partial_x u_i

'''
Lattice Boltzmann evolution of f
'''

'''
Compute forcing term in LB 

Following Guo et.al. (2002), the correct LB forcing term for a body force F is
p_i = (1 - \frac{1}{2\tau}) w_i [ \frac{e_i - v}{c_s^2} + \frac{e_i \cdot v}{c_s^4} e_i ] \cdot F 

For D2Q9 (done in Guo) we have
w_0 = 4/9
w_1 = 1/9
w_2 = 1/36

Note: In Denniston this forcing term is
p_i = T_s \partial_{\beta} \tau_{\alpha \beta} e_{i \alpha}
where \tau_{\alpha \beta} is the antisymmetric portion of the stress term
'''
def cal_p(p, u, fr, sigma_p=None):
	u[..., 0] = 0
	edotu = np.einsum('ab,cdeb', e, u)
	eminusu = e[:, None, None, None, :] - u

	eminusudotu = np.einsum('...i,...i', eminusu, u[None])

	p[:] = 0

	# Body force due to walls - (wall_force - u)
	p += -fr * np.einsum('...i,...i', eminusu, u[None])
	p += -fr * 3. * edotu * edotu

	# Body force due to elastic + active stress
	if sigma_p is not None:
		sigma_p[..., 0] = 0
		p += np.einsum('...i,...i', eminusu, sigma_p[None])
		p += 3. * edotu * np.einsum('ab,cdeb', e, sigma_p)
	
	print(np.max(p[1]), np.max(p[2]))
	p[0]   *= two3rd
	p[1:7] *= one3rd
	p[7:]  *= one24th

'''
Compute difference with equilibrium density for collision step
'''
def cal_feqf_diff(fin, fout, Rho, u, Temp):
	calc_f_equilibrium(fout, Rho, u, Temp)
	fout *= -1
	fout += fin

def cal_rho(f0, Rho):
	Rho[:] = np.sum(f0, axis=0)	

def cal_u(f0, u, Rho):
	cal_rho(f0, Rho)
	u[:] = np.einsum('abcd,ae', f0, e) / Rho[..., None]

'''
Evolve f using LB

General LB from Wikipedia
1. Collision Step
	f [x, t + d_t] = f[x, t] + 1 / \tau_f (f_eq [x, t] - f[x, t]) + p[x, t]
2. Streaming Step
	f [x + e, t + 1] = f[x, t]


Specific implementation from Denniston
f[x + e dt, t + dt) - f[x, t] = 
	\frac{dt}{2} ( Cf[x, t, f] + Cf[x + e dt, t + dt, f[x + e dt, t + dt]] )
'''
def evol_f(fin, fout, nextf, p, u, Rho, sigma_p, Cf, fr, Temp, itau_f, Q_on, dt=1.):
	print('sigma_z', np.max(sigma_p[..., 0]), np.min(sigma_p[..., 0]))
	print('rho', np.max(Rho), np.min(Rho))
	print('uz', np.max(u[..., 0]), np.min(u[..., 0]), '\n')

	cal_p(p, u, fr, sigma_p)				#Compute and store FIRST order forcing term
	cal_feqf_diff(fin, Cf, Rho, u, Temp)	#Compute and store FIRST order collision function
	print('P', np.max(np.sum(p[7:], axis=0)), np.min(np.sum(p[7:], axis=0)))
	Cf = -itau_f * Cf + p
	for i in range(fout.shape[0]):
		fout[nextf[i]] = fin[i] + dt * Cf[i]#FIRST order approximation of f[x + e dt, t + dt]
		fin[i] += 0.5 * dt * Cf[i]
	
	cal_u(fout, u, Rho)						#FIRST order approximation of Rho, U
	print('RHO', np.max(Rho), np.min(Rho))
	print('Uz', np.max(u[..., 0]), np.min(u[..., 0]), '\n')
	
	cal_p(p, u, fr, sigma_p)				#Compute and store SECOND order forcing term
	cal_feqf_diff(fout, Cf, Rho, u, Temp)	#Compute and store SECOND order collision function
	print('Cf', np.max(Cf), np.min(Cf))

	Cf = -itau_f * Cf + p
	for i in range(fin.shape[0]):
		fin[i] += 0.5 * dt * Cf[nextf[i]]	#SECOND order approximation of f[x + e dt, t + dt]
		fout[nextf[i]] = fin[i]				#Streaming step	
	
	cal_u(fout, u, Rho)						#SECOND order approximation of Rho, U
	print('RHO', np.max(Rho), np.min(Rho))
	print('Uz', np.max(u[..., 0]), np.min(u[..., 0]), '\n')
