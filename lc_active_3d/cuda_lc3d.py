import numpy as np
import matplotlib.pyplot as plt
import itertools
from numba import jit, cuda
import numba
import math
from time import time
from sys import exit
from argparse import ArgumentParser

from lc_3d import display, read_restart

oneThird = 1./3.

@cuda.jit
def init_stream_functions(nextf, e):
	z, y, x = cuda.grid(3)
	if z < nextf.shape[0] and y < nextf.shape[1] and x < nextf.shape[2]:
		for i in range(nextf.shape[3]):
			nextf[z, y, x, i, 0] = (z + e[i, 2]) % nextf.shape[0]
			nextf[z, y, x, i, 1] = (y + e[i, 1]) % nextf.shape[1]
			nextf[z, y, x, i, 2] = (x + e[i, 0]) % nextf.shape[2]
			nextf[z, y, x, i, 3] = i

@cuda.jit
def init_neighbor_list(fd_nbr):
	z, y, x = cuda.grid(3)
	if z < fd_nbr.shape[0] and y < fd_nbr.shape[1] and x < fd_nbr.shape[2]:
		fd_nbr[z, y, x, 0, 0] = z
		fd_nbr[z, y, x, 0, 1] = y
		fd_nbr[z, y, x, 0, 2] = (x - 1) % fd_nbr.shape[2]
		
		fd_nbr[z, y, x, 1, 0] = z
		fd_nbr[z, y, x, 1, 1] = y
		fd_nbr[z, y, x, 1, 2] = (x + 1) % fd_nbr.shape[2]

		fd_nbr[z, y, x, 2, 0] = z
		fd_nbr[z, y, x, 2, 1] = (y - 1) % fd_nbr.shape[1]
		fd_nbr[z, y, x, 2, 2] = x
		
		fd_nbr[z, y, x, 3, 0] = z
		fd_nbr[z, y, x, 3, 1] = (y + 1) % fd_nbr.shape[1]
		fd_nbr[z, y, x, 3, 2] = x
		
		fd_nbr[z, y, x, 4, 0] = (z - 1) % fd_nbr.shape[0]
		fd_nbr[z, y, x, 4, 1] = y
		fd_nbr[z, y, x, 4, 2] = x
		
		fd_nbr[z, y, x, 5, 0] = (z + 1) % fd_nbr.shape[0]
		fd_nbr[z, y, x, 5, 1] = y
		fd_nbr[z, y, x, 5, 2] = x

@cuda.jit
def cal_W(W, U, nextf):
	z, y, x = cuda.grid(3)
	if z < W.shape[0] and y < W.shape[1] and x < W.shape[2]:
		Uxp = U[nextf[z, y, x, 1, 0], nextf[z, y, x, 1, 1], nextf[z, y, x, 1, 2]]
		Uxm = U[nextf[z, y, x, 2, 0], nextf[z, y, x, 2, 1], nextf[z, y, x, 2, 2]]
		Uyp = U[nextf[z, y, x, 3, 0], nextf[z, y, x, 3, 1], nextf[z, y, x, 3, 2]]
		Uym = U[nextf[z, y, x, 4, 0], nextf[z, y, x, 4, 1], nextf[z, y, x, 4, 2]]
		Uzp = U[nextf[z, y, x, 5, 0], nextf[z, y, x, 5, 1], nextf[z, y, x, 5, 2]]
		Uzm = U[nextf[z, y, x, 6, 0], nextf[z, y, x, 6, 1], nextf[z, y, x, 6, 2]]

		W[z, y, x, 0, 0] = 0.5 * (Uxp[0] - Uxm[0])
		W[z, y, x, 0, 1] = 0.5 * (Uxp[1] - Uxm[1])
		W[z, y, x, 0, 2] = 0.5 * (Uxp[2] - Uxm[2])
		W[z, y, x, 1, 0] = 0.5 * (Uyp[0] - Uym[0])
		W[z, y, x, 1, 1] = 0.5 * (Uyp[1] - Uym[1])
		W[z, y, x, 1, 2] = 0.5 * (Uyp[2] - Uym[2])
		W[z, y, x, 2, 0] = 0.5 * (Uzp[0] - Uzm[0])
		W[z, y, x, 2, 1] = 0.5 * (Uzp[1] - Uzm[1])
		W[z, y, x, 2, 2] = 0.5 * (Uzp[2] - Uzm[2])

@cuda.jit
def cal_dQ(Q, U, H, fd_nbr, A_ldg, U_lc, L1, Gamma_rot, oneThirdDelta):
	z, y, x = cuda.grid(3)
	if z < Q.shape[0] and y < Q.shape[1] and x < Q.shape[2]:
		QQij = cuda.local.array((5,), dtype=numba.float64)
		trQQ = Q[z, y, x, 0]**2 + Q[z, y, x, 1]**2 + Q[z, y, x, 2]**2 + Q[z, y, x, 3]**2 + Q[z, y, x, 4]**2
		trQQ = 2 * (trQQ + Q[z, y, x, 0] * Q[z, y, x, 3])

		QQij[0] = Q[z, y, x, 0]**2 + Q[z, y, x, 1]**2 + Q[z, y, x, 2]**2
		QQij[1] = Q[z, y, x, 1] * (Q[z, y, x, 0] + Q[z, y, x, 3]) + Q[z, y, x, 4] * Q[z, y, x, 2]
		QQij[2] = Q[z, y, x, 2] * (-Q[z, y, x, 3]) + Q[z, y, x, 1] * Q[z, y, x, 4]
		QQij[3] = Q[z, y, x, 3]**2 + Q[z, y, x, 1]**2 + Q[z, y, x, 4]**2
		QQij[4] = Q[z, y, x, 4] * (-Q[z, y, x, 0]) + Q[z, y, x, 1] * Q[z, y, x, 2]

		xm = fd_nbr[z, y, x, 0]
		xp = fd_nbr[z, y, x, 1]
		ym = fd_nbr[z, y, x, 2]
		yp = fd_nbr[z, y, x, 3]
		zm = fd_nbr[z, y, x, 4]
		zp = fd_nbr[z, y, x, 5]

		for i in range(Q.shape[3]):
			dxQ = 0.5 * (Q[xp[0], xp[1], xp[2], i] - Q[xm[0], xm[1], xm[2], i])
			dyQ = 0.5 * (Q[yp[0], yp[1], yp[2], i] - Q[ym[0], ym[1], ym[2], i])
			dzQ = 0.5 * (Q[zp[0], zp[1], zp[2], i] - Q[zm[0], zm[1], zm[2], i])

			d2xQ = Q[xp[0], xp[1], xp[2], i] - 2. * Q[z, y, x, i] + Q[xm[0], xm[1], xm[2], i]
			d2yQ = Q[yp[0], yp[1], yp[2], i] - 2. * Q[z, y, x, i] + Q[ym[0], ym[1], ym[2], i]
			d2zQ = Q[zp[0], zp[1], zp[2], i] - 2. * Q[z, y, x, i] + Q[zm[0], zm[1], zm[2], i]

			H[z, y, x, i] = -1*(
				  A_ldg*(1. - oneThird * U_lc) * Q[z, y, x, i] \
				- A_ldg*U_lc*(QQij[i] - trQQ * (Q[z, y, x, i] + oneThirdDelta[i])) \
				- L1 * (d2xQ + d2yQ + d2zQ))
			H[z, y, x, i] = Gamma_rot * H[z, y, x, i] - \
							U[z, y, x, 0] * dxQ - \
							U[z, y, x, 1] * dyQ - \
							U[z, y, x, 2] * dzQ * 0.

@cuda.jit
def evol_Q(Q, H, W, xi, xi1, xi2, qdt):
	z, y, x = cuda.grid(3)
	if z < Q.shape[0] and y < Q.shape[1] and x < Q.shape[2]:
		W1 = cuda.local.array((3, 3), dtype=numba.float64)
		M1 = cuda.local.array((3, 3), dtype=numba.float64)
		S = cuda.local.array((3, 3), dtype=numba.float64)
		#Compute co-rotation term
		trQW = Q[z, y, x, 0] * (W[z, y, x, 0, 0] - W[z, y, x, 2, 2]) + \
			   Q[z, y, x, 3] * (W[z, y, x, 1, 1] - W[z, y, x, 2, 2]) + \
			   Q[z, y, x, 1] * (W[z, y, x, 0, 1] + W[z, y, x, 1, 0]) + \
			   Q[z, y, x, 2] * (W[z, y, x, 0, 2] + W[z, y, x, 2, 0]) + \
			   Q[z, y, x, 4] * (W[z, y, x, 1, 2] + W[z, y, x, 2, 1])
		
		M1[0, 0] = Q[z, y, x, 0] + oneThird
		M1[0, 1] = Q[z, y, x, 1]
		M1[0, 2] = Q[z, y, x, 2]
		M1[1, 0] = Q[z, y, x, 1]
		M1[1, 1] = Q[z, y, x, 3] + oneThird
		M1[1, 2] = Q[z, y, x, 4]
		M1[2, 0] = Q[z, y, x, 2]
		M1[2, 1] = Q[z, y, x, 4]
		M1[2, 2] = 1. - M1[0, 0] - M1[1, 1]

		for i in range(W.shape[-2]):
			for j in range(W.shape[-1]):
				W1[i, j] = xi1 * W[z, y, x, j, i] + xi2 * W[z,y, x, i, j]

		for i in range(S.shape[0]):
			for j in range(W1.shape[0]):
				S[i, j] = -2. * xi * M1[i, j] * trQW
				for k in range(W1.shape[1]):
					S[i, j] += W1[i, k] * M1[k, j] + M1[i, k] * W1[j, k]

		Q[z, y, x, 0] += qdt * (H[z, y, x, 0] + S[0, 0])
		Q[z, y, x, 1] += qdt * (H[z, y, x, 1] + S[0, 1])
		Q[z, y, x, 2] = 0.
		Q[z, y, x, 3] += qdt * (H[z, y, x, 3] + S[1, 1])
		Q[z, y, x, 4] = 0.

@cuda.jit
def cal_stress(sigma_p, sigma_q, Q, U, H, fd_nbr, A_ldg, U_lc, L1, xi, zeta, oneThirdDelta):
	z, y, x = cuda.grid(3)
	if z < sigma_q.shape[0] and y < sigma_q.shape[1] and x < sigma_q.shape[2]:
		M1 = cuda.local.array((3, 3), dtype=numba.float64)
		H1 = cuda.local.array((3, 3), dtype=numba.float64)
		dxQ = cuda.local.array((5,), dtype=numba.float64)
		dyQ = cuda.local.array((5,), dtype=numba.float64)
		dzQ = cuda.local.array((5,), dtype=numba.float64)
		QQij = cuda.local.array((5,), dtype=numba.float64)
		
		#Compute molecular field
		trQQ = Q[z, y, x, 0]**2 + Q[z, y, x, 1]**2 + Q[z, y, x, 2]**2 + Q[z, y, x, 3]**2 + Q[z, y, x, 4]**2
		trQQ = 2 * (trQQ + Q[z, y, x, 0] * Q[z, y, x, 3])

		QQij[0] = Q[z, y, x, 0]**2 + Q[z, y, x, 1]**2 + Q[z, y, x, 2]**2
		QQij[1] = Q[z, y, x, 1] * (Q[z, y, x, 0] + Q[z, y, x, 3]) + Q[z, y, x, 4] * Q[z, y, x, 2]
		QQij[2] = Q[z, y, x, 2] * (-Q[z, y, x, 3]) + Q[z, y, x, 1] * Q[z, y, x, 4]
		QQij[3] = Q[z, y, x, 3]**2 + Q[z, y, x, 1]**2 + Q[z, y, x, 4]**2
		QQij[4] = Q[z, y, x, 4] * (-Q[z, y, x, 0]) + Q[z, y, x, 1] * Q[z, y, x, 2]

		xm = fd_nbr[z, y, x, 0]
		xp = fd_nbr[z, y, x, 1]
		ym = fd_nbr[z, y, x, 2]
		yp = fd_nbr[z, y, x, 3]
		zm = fd_nbr[z, y, x, 4]
		zp = fd_nbr[z, y, x, 5]
		
		summ = 0
		sigma_p[z, y, x, 0] = 0
		sigma_p[z, y, x, 1] = 0
		sigma_p[z, y, x, 2] = 0

		for i in range(Q.shape[3]):
			dxQ[i] = 0.5 * (Q[xp[0], xp[1], xp[2], i] - Q[xm[0], xm[1], xm[2], i])
			dyQ[i] = 0.5 * (Q[yp[0], yp[1], yp[2], i] - Q[ym[0], ym[1], ym[2], i])
			dzQ[i] = 0.5 * (Q[zp[0], zp[1], zp[2], i] - Q[zm[0], zm[1], zm[2], i])
			
			d2xQ = Q[xp[0], xp[1], xp[2], i] - 2. * Q[z, y, x, i] + Q[xm[0], xm[1], xm[2], i]
			d2yQ = Q[yp[0], yp[1], yp[2], i] - 2. * Q[z, y, x, i] + Q[ym[0], ym[1], ym[2], i]
			d2zQ = Q[zp[0], zp[1], zp[2], i] - 2. * Q[z, y, x, i] + Q[zm[0], zm[1], zm[2], i]
			
			H[z, y, x, i] = -1*(
				  A_ldg*(1. - oneThird * U_lc) * Q[z, y, x, i] \
				- A_ldg*U_lc*(QQij[i] - trQQ * (Q[z, y, x, i] + oneThirdDelta[i])) \
				- L1 * (d2xQ + d2yQ + d2zQ))
			
		
			#Compute trace terms of sigma_p while we have derivatives computed
			summ += Q[z, y, x, i] * H[z, y, x, i]
			sigma_p[z, y, x, 0] -= H[z, y, x, i] * dxQ[i]
			sigma_p[z, y, x, 1] -= H[z, y, x, i] * dyQ[i]
			sigma_p[z, y, x, 2] -= H[z, y, x, i] * dzQ[i]

		summ = 2. * summ + Q[z, y, x, 0] * H[z, y, x, 3] + Q[z, y, x, 3] * H[z, y, x, 0]
		sigma_p[z, y, x, 0] = 2. * sigma_p[z, y, x, 0] - H[z, y, x, 0] * dxQ[3] - H[z, y, x, 3] * dxQ[0]
		sigma_p[z, y, x, 1] = 2. * sigma_p[z, y, x, 1] - H[z, y, x, 0] * dyQ[3] - H[z, y, x, 3] * dyQ[0]
		sigma_p[z, y, x, 2] = 2. * sigma_p[z, y, x, 2] - H[z, y, x, 0] * dzQ[3] - H[z, y, x, 3] * dzQ[0]
		
		M1[0, 0] = Q[z, y, x, 0] + oneThird
		M1[0, 1] = Q[z, y, x, 1]
		M1[0, 2] = Q[z, y, x, 2]
		M1[1, 0] = Q[z, y, x, 1]
		M1[1, 1] = Q[z, y, x, 3] + oneThird
		M1[1, 2] = Q[z, y, x, 4]
		M1[2, 0] = Q[z, y, x, 2]
		M1[2, 1] = Q[z, y, x, 4]
		M1[2, 2] = 1. - M1[0, 0] - M1[1, 1]

		H1[0, 0] = H[z, y, x, 0]
		H1[0, 1] = H[z, y, x, 1]
		H1[0, 2] = H[z, y, x, 2]
		H1[1, 0] = H[z, y, x, 1]
		H1[1, 1] = H[z, y, x, 3]
		H1[1, 2] = H[z, y, x, 4]
		H1[2, 0] = H[z, y, x, 2]
		H1[2, 1] = H[z, y, x, 4]
		H1[2, 2] = -H[z, y, x, 0] - H[z, y, x, 3]

		#Compute stress
		for i in range(sigma_q.shape[-2]):
			for j in range(sigma_q.shape[-1]):
				sigma_q[z, y, x, i, j] = 2. * xi * summ * M1[i, j]
				for k in range(sigma_q.shape[-1]):
					sigma_q[z, y, x, i, j] += (1. - xi) * M1[i, k] * H1[k, j]
					sigma_q[z, y, x, i, j] -= (1. + xi) * H1[i, k] * M1[k, j]
				sigma_q[z, y, x, i,j] -= zeta * M1[i, j]

@cuda.jit
def calc_div_stress(sigma_p, sigma_q, nextf):
	z, y, x = cuda.grid(3)
	if z < sigma_p.shape[0] and y < sigma_p.shape[1] and x < sigma_p.shape[2]:
		for j in range(sigma_q.shape[-1]):	#x, y, z
			for i in range(sigma_q.shape[-2]):	#x, y, z
				ip = nextf[z, y, x, 1 + (2 * j)]	#z, y, x, i
				im = nextf[z, y, x, 2 + (2 * j)]	#z, y, x, i
				sigma_p[z, y, x, i] += 0.5 * (sigma_q[ip[0], ip[1], ip[2], i, j] - sigma_q[im[0], im[1], im[2], i, j])


@cuda.jit
def stream(F1, F2, nextf):
	z, y, x = cuda.grid(3)
	if z < F1.shape[0] and y < F1.shape[1] and x < F1.shape[2]:
		for i in range(F1.shape[3]):
			zyxi = nextf[z, y, x, i]
			F2[zyxi[0], zyxi[1], zyxi[2], zyxi[3]] = F1[z, y, x, i]

@cuda.jit
def first_order_evol(F1, F2, U, Rho, nextf, sigma_p, fr, Temp, itau_f, dt, e, weights):
	z, y, x = cuda.grid(3)
	if z < F1.shape[0] and y < F1.shape[1] and x < F1.shape[2]:
		u2 = U[z, y, x, 0]**2 + U[z, y, x, 1]**2 + U[z, y, x, 2]**2
		for i in range(F1.shape[3]):
			#Calculate p - the external forcing term (e.g. elastic stresses)
			edotu = e[i, 0] * U[z, y, x, 0] + e[i, 1] * U[z, y, x, 1] + e[i, 2] * U[z, y, x, 2]
			eminusu_x = e[i, 0] - U[z, y, x, 0]
			eminusu_y = e[i, 1] - U[z, y, x, 1]
			eminusu_z = e[i, 2] - U[z, y, x, 2]
			eminusudotu = eminusu_x * U[z, y, x, 0] + eminusu_y * U[z, y, x, 1] + eminusu_z * U[z, y, x, 2]

			p = -fr * (eminusudotu + 3. * edotu**2) 
			p += eminusu_x * sigma_p[z, y, x, 0] + \
				 eminusu_y * sigma_p[z, y, x, 1] + \
				 eminusu_z * sigma_p[z, y, x, 2]
			p += 3. * edotu * (e[i, 0] * sigma_p[z, y, x, 0] + \
							   e[i, 1] * sigma_p[z, y, x, 1] + \
							   e[i, 2] * sigma_p[z, y, x, 2])
			p *= 3 * weights[i]

			#Calculate Cf = F1 - Feq
			Cf = F1[z, y, x, i] - weights[i] * Rho[z, y, x] * (1 + 3. * edotu - 1.5 * u2 + 4.5 * edotu**2)

			#Apply first order evolution
			Cf = -itau_f * Cf + p
			zyxi = nextf[z, y, x, i]
			F1[z, y, x, i] += 0.5 * dt * Cf
			F2[zyxi[0], zyxi[1], zyxi[2], zyxi[3]] += dt * Cf

@cuda.jit
def second_order_evol(F1, F2, U, Rho, nextf, sigma_p, fr, Temp, itau_f, dt, e, weights):
	z, y, x = cuda.grid(3)
	if z < F1.shape[0] and y < F1.shape[1] and x < F1.shape[2]:
		#Compute forcing terms AT nextf[z, y, x, i]
		for i in range(F1.shape[3]):
			zz, yy, xx, ii = nextf[z, y, x, i]

			#Calculate p - the external forcing term (e.g. elastic stresses)
			u2 = U[zz, yy, xx, 0]**2 + U[zz, yy, xx, 1]**2 + U[zz, yy, xx, 2]**2
			edotu = e[ii, 0] * U[zz, yy, xx, 0] + e[i, 1] * U[zz, yy, xx, 1] + e[i, 2] * U[zz, yy, xx, 2]
			eminusu_x = e[ii, 0] - U[zz, yy, xx, 0]
			eminusu_y = e[ii, 1] - U[zz, yy, xx, 1]
			eminusu_z = e[ii, 2] - U[zz, yy, xx, 2]
			eminusudotu = eminusu_x * U[zz, yy, xx, 0] + eminusu_y * U[zz, yy, xx, 1] + eminusu_z * U[zz, yy, xx, 2]

			p = -fr * (eminusudotu + 3. * edotu**2) 
			p += eminusu_x * sigma_p[zz, yy, xx, 0] + \
				 eminusu_y * sigma_p[zz, yy, xx, 1] + \
				 eminusu_z * sigma_p[zz, yy, xx, 2]
			p += 3. * edotu * (e[i, 0] * sigma_p[zz, yy, xx, 0] + \
							   e[i, 1] * sigma_p[zz, yy, xx, 1] + \
							   e[i, 2] * sigma_p[zz, yy, xx, 2])
			p *= 3 * weights[i]

			#Calculate F2 - Feq
			Cf = F2[zz, yy, xx, ii] - weights[ii] * Rho[zz, yy, xx] * (1 + 3. * edotu - 1.5 * u2 + 4.5 * edotu**2)
			Cf = -itau_f * Cf + p
			F1[z, y, x, i] += dt * 0.5 * Cf


@cuda.jit
def calculate_U(F, Rho, U, e):
	z, y, x = cuda.grid(3)
	if z < F.shape[0] and y < F.shape[1] and x < F.shape[2]:
		Rho[z, y, x] = 0
		U[z, y, x, 0] = 0
		U[z, y, x, 1] = 0
		U[z, y, x, 2] = 0
		for i in range(F.shape[3]):
			Rho[z, y, x] += F[z, y, x, i]
			U[z, y, x, 0] += F[z, y, x, i] * e[i, 0]
			U[z, y, x, 1] += F[z, y, x, i] * e[i, 1]
			U[z, y, x, 2] += F[z, y, x, i] * e[i, 2]
		for i in range(U.shape[-1]):
			U[z, y, x, i] /= Rho[z, y, x]
	
def evol_F(F1, F2, U, Rho, nextf, sigma_p, fr, Temp, itau_f, e, weights, 
		   blockspergrid, threadsperblock):
	stream[blockspergrid, threadsperblock](F1, F2, nextf)
	first_order_evol[blockspergrid, threadsperblock](F1, F2, U, Rho, nextf, sigma_p, 
		fr, Temp, itau_f, 1., e, weights)
	calculate_U[blockspergrid, threadsperblock](F2, Rho, U, e)
	second_order_evol[blockspergrid, threadsperblock](F1, F2, U, Rho, nextf, sigma_p,
		fr, Temp, itau_f, 1., e, weights)
	stream[blockspergrid, threadsperblock](F1, F2, nextf)
	calculate_U[blockspergrid, threadsperblock](F2, Rho, U, e)

@cuda.jit
def set_F_Feq(F, U, Rho, e, weights):
	z, y, x = cuda.grid(3)
	if z < F.shape[0] and y < F.shape[1] and x < F.shape[2]:
		u2 = U[z, y, x, 0]**2 + U[z, y, x, 1]**2 + U[z, y, x, 2]**2
		for i in range(F.shape[3]):
			ue = e[i, 0] * U[z, y, x, 0] + e[i, 1] * U[z, y, x, 1] + e[i, 2] * U[z, y, x, 2]
			F[z, y, x, i] = weights[i] * Rho[z, y, x] * (1. + 3. * ue - 1.5 * u2 + 4.5 * ue**2)
			#Rui sets E to zero

if __name__=='__main__':
	parser = ArgumentParser('Lattice Boltzmann simulations of liquid crystal hydrodynamics')
	parser.add_argument('-r', '--restart', action='store_true')
	parser.add_argument('-s', '--shape', type=int, nargs=3, default=[200, 200, 3])
	parser.add_argument('--tau_f', type=float, default=1.)
	parser.add_argument('--rho', type=float, default=1.)
	parser.add_argument('--t_max', type=int, default=1000)
	parser.add_argument('--t_write', type=int, default=10)
	parser.add_argument('--A_ldg', type=float, default=0.1)
	parser.add_argument('--U', type=float, default=3.5)
	parser.add_argument('--L1', type=float, default=0.1)
	parser.add_argument('--Gamma_rot', type=float, default=0.1)
	parser.add_argument('--xi', type=float, default=0.8)
	parser.add_argument('--n_evol_Q', type=int, default=2)
	parser.add_argument('--Temp', type=float, default=oneThird)
	parser.add_argument('--fr', type=float, default=0.01)
	parser.add_argument('--zeta', type=float, default=0.005)
	args = parser.parse_args()

	setattr(args,'itau_f', 1. / args.tau_f)
	setattr(args, 'qdt', 1. / args.n_evol_Q)
	setattr(args, 'xi1', 0.5 * (args.xi + 1.))
	setattr(args, 'xi2', 0.5 * (args.xi - 1.))

	#Geometry information
	
	[Nx, Ny, Nz] = args.shape

	#e = [ex, ey, ez]
	e = np.array([[0, 0, 0],
				  [1, 0, 0],	#+x
				  [-1, 0, 0],	#-x
				  [0, 1, 0],	#+y
				  [0, -1, 0],	#-y
				  [0, 0, 1],	#+z
				  [0, 0, -1],	#-z
				  [1, 1, 1],
				  [1, 1, -1],
				  [1, -1, 1],
				  [1, -1, -1],
				  [-1, 1, 1],
				  [-1, 1, -1],
				  [-1, -1, 1],
				  [-1, -1, -1]]).astype(np.int32)

	weights = np.ones(e.shape[0]) / 72.
	weights[1:7] *= 8
	weights[0] *= 16

	oneThirdDelta = np.array([1., 0., 0., 1., 0.], dtype=np.float64) / 3.
	
	#Prep cuda info
	threadsperblock = (1, 16, 16)
	blockspergrid_z = math.ceil(Nz / threadsperblock[0])
	blockspergrid_y = math.ceil(Ny / threadsperblock[1])
	blockspergrid_x = math.ceil(Nx / threadsperblock[2])
	blockspergrid = (blockspergrid_z, blockspergrid_y, blockspergrid_x)

	#Allocate storage
	Q = np.empty((Nz, Ny, Nx, 5), dtype=np.float64)	#[Qxx, Qxy, Qxz, Qyy, Qyz]
	U = np.empty((Nz, Ny, Nx, 3), dtype=np.float64)	#[Ux, Uy, Uz]
	Rho = np.empty((Nz, Ny, Nx), dtype=np.float64)

	F1 = np.ones((Nz, Ny, Nx, e.shape[0]), dtype=np.float64)
	F2 = np.empty_like(F1)
	nextf = np.empty((Nz, Ny, Nx, e.shape[0], 4), dtype=np.int32)	#LB neighborlist

	fd_nbr = np.empty((Nz, Ny, Nx, 6, 3), dtype=np.int32)
	H = np.empty_like(Q)


	if args.restart:
		read_restart('restart.dat', Q, U, Rho)	
	else:
		Q[..., 0] = -1./2.
		Q[..., 1] = 0.
		Q[..., 2] = 0.
		Q[..., 3] = 1./2.
		Q[..., 4] = 0.

		U[..., 0] = 0.
		U[..., 1] = 0.
		U[..., 2] = 0.
		Rho[:] = args.rho

	#Send everything to device
	Qhost = np.empty_like(Q)
	Q = cuda.to_device(Q)
	U = cuda.to_device(U)
	Rho = cuda.to_device(Rho)
	F1 = cuda.to_device(F1)
	F2 = cuda.to_device(F2)
	nextf = cuda.to_device(nextf)
	fd_nbr = cuda.to_device(fd_nbr)
	H = cuda.to_device(H)
	sigma_q = cuda.to_device(np.empty((Nz, Ny, Nx, 3, 3), dtype=np.float64))
	sigma_p = cuda.to_device(np.empty((Nz, Ny, Nx, 3), dtype=np.float64))
	W = cuda.to_device(np.empty((Nz, Ny, Nx, 3, 3), dtype=np.float64))

	oneThirdDelta = cuda.to_device(oneThirdDelta)
	e_d = cuda.to_device(e.astype(np.float64))
	weights = cuda.to_device(weights)
	
	#Initialize F as Feq
	init_stream_functions[blockspergrid, threadsperblock](nextf, e)
	init_neighbor_list[blockspergrid, threadsperblock](fd_nbr)
	set_F_Feq[blockspergrid, threadsperblock](F1, U, Rho, e_d, weights)
	
	t_current = 0
	while t_current < args.t_max:

		t = time()

		for it in range(args.t_write):
			cal_W[blockspergrid, threadsperblock](W, U, nextf)
			for i in range(args.n_evol_Q):
				cal_dQ[blockspergrid, threadsperblock](Q, U, H, fd_nbr, 
					   args.A_ldg, args.U, args.L1, args.Gamma_rot, oneThirdDelta)
				evol_Q[blockspergrid, threadsperblock](Q, H, W, 
					   args.xi, args.xi1, args.xi2, args.qdt)
			
			cal_stress[blockspergrid, threadsperblock](sigma_p, sigma_q, Q, U, H, fd_nbr,
				args.A_ldg, args.U, args.L1, args.xi, args.zeta, oneThirdDelta)
			calc_div_stress[blockspergrid, threadsperblock](sigma_p, sigma_q, nextf)
			evol_F(F1, F2, U, Rho, nextf, sigma_p, 
				args.fr, args.Temp, args.itau_f, e_d, weights,
				blockspergrid, threadsperblock)

			cal_W[blockspergrid, threadsperblock](W, U, nextf)
			for i in range(args.n_evol_Q):
				cal_dQ[blockspergrid, threadsperblock](Q, U, H, fd_nbr, 
					   args.A_ldg, args.U, args.L1, args.Gamma_rot, oneThirdDelta)
				evol_Q[blockspergrid, threadsperblock](Q, H, W, 
					   args.xi, args.xi1, args.xi2, args.qdt)

			cal_stress[blockspergrid, threadsperblock](sigma_p, sigma_q, Q, U, H, fd_nbr,
				args.A_ldg, args.U, args.L1, args.xi, args.zeta, oneThirdDelta)
			calc_div_stress[blockspergrid, threadsperblock](sigma_p, sigma_q, nextf)
			evol_F(F2, F1, U, Rho, nextf, sigma_p, 
				args.fr, args.Temp, args.itau_f, e_d, weights,
				blockspergrid, threadsperblock)

		print(time() - t)
		Q.copy_to_host(Qhost)
		t_current += args.t_write
