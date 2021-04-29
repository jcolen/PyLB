import numpy as np
import matplotlib.pyplot as plt
import itertools
from numba import jit
from time import time
from sys import exit
from argparse import ArgumentParser

@jit(nopython=True)
def build_stream(nextf, e):
	for y in range(nextf.shape[0]):
		for x in range(nextf.shape[1]):
			for i in range(nextf.shape[2]):
				nextf[y, x, i, 0] = (y - e[i, 0]) % nextf.shape[0]
				nextf[y, x, i, 1] = (x - e[i, 1]) % nextf.shape[1]
				nextf[y, x, i, 2] = i

@jit(nopython=True)
def build_neighbor(fd_nbr):
	for y in range(fd_nbr.shape[0]):
		for x in range(fd_nbr.shape[1]):
			fd_nbr[y, x, 0, 0] = (y - 1) % fd_nbr.shape[0]
			fd_nbr[y, x, 0, 1] = x

			fd_nbr[y, x, 1, 0] = (y + 1) % fd_nbr.shape[0]
			fd_nbr[y, x, 1, 1] = x

			fd_nbr[y, x, 2, 0] = y
			fd_nbr[y, x, 2, 1] = (x - 1) % fd_nbr.shape[1]

			fd_nbr[y, x, 3, 0] = y
			fd_nbr[y, x, 3, 1] = (x + 1) % fd_nbr.shape[1]

@jit(nopython=True)
def cal_W(W, U, nextf):
	for y in range(W.shape[0]):
		for x in range(W.shape[1]):
			Uyp = U[nextf[y, x, 1, 0], nextf[y, x, 1, 1]]
			Uym = U[nextf[y, x, 2, 0], nextf[y, x, 2, 1]]
			Uxp = U[nextf[y, x, 3, 0], nextf[y, x, 3, 1]]
			Uxm = U[nextf[y, x, 4, 0], nextf[y, x, 4, 1]]

			W[y, x, 0, 0] = 0.5 * (Uyp[0] - Uym[0])
			W[y, x, 0, 1] = 0.5 * (Uyp[1] - Uym[1])
			W[y, x, 1, 0] = 0.5 * (Uxp[0] - Uxm[0])
			W[y, x, 1, 1] = 0.5 * (Uxp[1] - Uxm[1])

@jit(nopython=True)
def cal_dQ(Q, U, H, fd_nbr, A_ldg, U_lc, L1, Gamma_rot, dij):
	for y in range(Q.shape[0]):
		for x in range(Q.shape[1]):
			trQQ = Q[y, x, 0]**2 + 2. * Q[y, x, 1]**2 + Q[y, x, 2]**2

			ym = fd_nbr[y, x, 0]
			yp = fd_nbr[y, x, 1]
			xm = fd_nbr[y, x, 2]
			xp = fd_nbr[y, x, 3]

			for i in range(Q.shape[2]):
				dyQ = 0.5 * (Q[yp[0], yp[1], i] - Q[ym[0], ym[1], i])
				dxQ = 0.5 * (Q[xp[0], xp[1], i] - Q[xm[0], xm[1], i])

				d2yQ = 0.5 * (Q[yp[0], yp[1], i] - 2. * Q[y, x, i] + Q[ym[0], ym[1], i])
				d2xQ = 0.5 * (Q[xp[0], xp[1], i] - 2. * Q[y, x, i] + Q[xm[0], xm[1], i])

				H[y, x, i] = A_ldg*(1. - 0.5 * U_lc) * Q[y, x, i] \
					- A_ldg*U_lc*(Q[y, x, i]**2 - Q[y, x, 1]**2 - trQQ * (Q[y, x, i] + 0.5 * dij[i])) \
					- L1 * (d2xQ + d2yQ)
				H[y, x, i] = -Gamma_rot * H[y, x, i] - U[y, x, 0] * dyQ - U[y, x, 1] * dxQ

@jit(nopython=True)
def evol_Q(Q, H, W, xi, xi1, xi2, qdt):
	W1 = np.empty((2, 2), dtype=np.float64)
	M1 = np.empty((2, 2), dtype=np.float64)
	S = np.empty((2, 2), dtype=np.float64)
	for y in range(Q.shape[0]):
		for x in range(Q.shape[1]):
			#Compute co-rotation term
			trQW = Q[y, x, 0] * W[y, x, 0, 0] + Q[y, x, 1] * (W[y, x, 1, 0] + W[y, x, 0, 1]) + Q[y, x, 2] * W[y, x, 1, 1]
			
			M1[0, 0] = Q[y, x, 0] + 0.5
			M1[0, 1] = Q[y, x, 1]
			M1[1, 0] = Q[y, x, 1]
			M1[1, 1] = Q[y, x, 2] + 0.5
			W1 = xi1 * W[y, x]

			for i in range(W.shape[-2]):
				for j in range(W.shape[-1]):
					W1[i, j] += xi2 * W[y, x, j, i]

			for i in range(S.shape[0]):
				for j in range(W1.shape[0]):
					S[i, j] = -2. * xi * M1[i, j] * trQW
					for k in range(W1.shape[1]):
						S[i, j] += W1[i, k] * M1[k, j] + M1[i, k] * W1[j, k]

			Q[y, x, 0] += qdt * (H[y, x, 0] + S[0, 0])
			Q[y, x, 1] += qdt * (H[y, x, 1] + S[0, 1])
			Q[y, x, 2] += qdt * (H[y, x, 2] + S[1, 1])

@jit(nopython=True)
def cal_stress(sigma_p, sigma_q, Q, U, H, fd_nbr, nextf, A_ldg, U_lc, L1, xi, zeta):
	M1 = np.empty((2, 2), dtype=np.float64)
	H1 = np.empty_like(M1)
	for y in range(Q.shape[0]):
		for x in range(Q.shape[1]):
			#Compute molecular field
			trQQ = Q[y, x, 0]**2 + 2. * Q[y, x, 1]**2 + Q[y, x, 2]**2

			ym = fd_nbr[y, x, 0]
			yp = fd_nbr[y, x, 1]
			xm = fd_nbr[y, x, 2]
			xp = fd_nbr[y, x, 3]
			
			sigma_p[y, x, 0] = 0
			sigma_p[y, x, 1] = 0

			summ = 0

			for i in range(Q.shape[2]):
				dyQ = 0.5 * (Q[yp[0], yp[1], i] - Q[ym[0], ym[1], i])
				dxQ = 0.5 * (Q[xp[0], xp[1], i] - Q[xm[0], xm[1], i])

				d2yQ = 0.5 * (Q[yp[0], yp[1], i] - 2. * Q[y, x, i] + Q[ym[0], ym[1], i])
				d2xQ = 0.5 * (Q[xp[0], xp[1], i] - 2. * Q[y, x, i] + Q[xm[0], xm[1], i])

				H[y, x, i] = A_ldg*(1. - 0.5 * U_lc) * Q[y, x, i] \
					- A_ldg*U_lc*(Q[y, x, i]**2 - Q[y, x, 1]**2 - trQQ * (Q[y, x, i] + 0.5 * dij[i])) \
					- L1 * (d2xQ + d2yQ)
				
				#Compute trace terms of sigma_p while we have derivatives computed
				summ += Q[y, x, i] * H[y, x, i]
				sigma_p[y, x, 0] -= H[y, x, i] * dyQ
				sigma_p[y, x, 1] -= H[y, x, i] * dxQ

			summ *= 2.
			sigma_p[y, x] *= 2
			
			M1[0, 0] = Q[y, x, 0] + 0.5
			M1[0, 1] = Q[y, x, 1]
			M1[1, 0] = Q[y, x, 1]
			M1[1, 1] = Q[y, x, 2] + 0.5

			H1[0, 0] = H[y, x, 0]
			H1[0, 1] = H[y, x, 1]
			H1[1, 0] = H[y, x, 1]
			H1[1, 1] = H[y, x, 2]

			#Compute stress
			for i in range(sigma_q.shape[-2]):
				for j in range(sigma_q.shape[-1]):
					sigma_q[y, x, i, j] = 2. * xi * summ * M1[i, j]
					for k in range(sigma_q.shape[-1]):
						sigma_q[y, x, i, j] += (1. - xi) * M1[i, k] * H1[k, j]
						sigma_q[y, x, i, j] -= (1. + xi) * H1[i, k] * M1[k, j]
			sigma_q[y, x, 0, 0] += zeta * Q[y, x, 0]
			sigma_q[y, x, 0, 1] += zeta * Q[y, x, 1]
			sigma_q[y, x, 1, 0] += zeta * Q[y, x, 1]
			sigma_q[y, x, 1, 1] += zeta * Q[y, x, 2]
					
	#Compute divergence of stress
	for y in range(Q.shape[0]):
		for x in range(Q.shape[1]):
			for j in range(sigma_q.shape[-1]):
				for i in range(sigma_q.shape[-2]):
					ip = nextf[y, x, 1 + (2 * j)]
					im = nextf[y, x, 2 + (2 * j)]
					sigma_p[y, x, i] += 0.5 * (sigma_q[ip[0], ip[1], i, j] - sigma_q[im[0], im[1], i, j])


#Starting with 1st order LB evolution of F
@jit(nopython=True)
def evol_F(F1, F2, nextf, sigma_p, fr, Temp, itau_f, e, weights):
	for y in range(F1.shape[0]):
		for x in range(F1.shape[1]):
			for i in range(F1.shape[2]):
				yxi = nextf[y, x, i]
				F2[y, x, i] = F1[yxi[0], yxi[1], yxi[2]]

			rho, uy, ux = 0, 0, 0
			for i in range(F2.shape[2]):
				rho += F2[y, x, i]
				uy += e[i, 0] * F2[y, x, i]
				ux += e[i, 1] * F2[y, x, i]
			ux /= rho
			uy /= rho
			u2 = ux * ux + uy * uy
			for i in range(F2.shape[2]):
				ue = e[i, 0] * uy + e[i, 1] * ux
				Feq = 1. + 3. * ue + 4.5 * ue * ue - 1.5 * u2
				Feq *= weights[i] * rho
				F2[y, x, i] += -itau_f * (F2[y, x, i] - Feq)
		
				#Compute forcing term (stresses)
				e_minus_uy = e[i, 0] - uy
				e_minus_ux = e[i, 1] - ux

				eminusudotu = e_minus_uy * uy + e_minus_ux * ux
				#Body force due to walls - (wall_force - u)
				p = -fr * (eminusudotu + 3. * ue**2)

				#Body force due to elastic/active stresses
				p += e_minus_uy * sigma_p[y, x, 0]
				p += e_minus_ux * sigma_p[y, x, 1]
				p += 3. * ue * (e[i, 0] * sigma_p[y, x, 0] + e[i, 1] * sigma_p[y, x, 1])

				F2[y, x, i] += weights[i] * p

@jit(nopython=True)
def set_F_Feq(F, U, Rho, e, weights):
	for y in range(F.shape[0]):
		for x in range(F.shape[1]):
			u2 = U[y, x, 0] * U[y, x, 1]
			for i in range(F.shape[2]):
				ue = e[i, 0] * U[y, x, 0] + e[i, 1] * U[y, x, 1]
				F[y, x, i] = 1. + 3. * ue + 4.5 * ue**2 - 1.5 * u2
				F[y, x, i] *= Rho[y, x] * weights[i]

'''
Read data from restart file
Format is:
Nx Ny Nz wall_x wall_y wall_z
'''
def read_restart(restart_file, Q, U, Rho):
	with open(restart_file, 'r') as fin:
		header = fin.readline().split()
		Nx, Ny, Nz = int(header[0]), int(header[1]), int(header[2])
		
		#Assert we have no walls, since we can't handle that yet
		if len(header) > 3:
			assert int(header[3]) == 0
			assert int(header[4]) == 0
			assert int(header[5]) == 0

		Q[:] = np.loadtxt(itertools.islice(fin, 0, Nx * Ny)).reshape([Ny, Nx, 3])
		#Convert to Q 2d

		Rho[:] = np.loadtxt(itertools.islice(fin, 0, Nx * Ny)).reshape([Ny, Nx])
		U[:] = np.loadtxt(itertools.islice(fin, 0, Nx * Ny)).reshape([Ny, Nx, 2])

def display(ax, Q):
	ax.clear()
	ax.set_xticks([])
	ax.set_yticks([])
	theta = np.arctan2(Q[..., 1], Q[..., 2] - Q[..., 0]) / 2
	theta[theta < 0] += np.pi
	theta[theta > np.pi] -= np.pi
	ax.imshow(np.sin(2 * theta), cmap='BuPu', vmin=-1, vmax=1)
	ax.quiver(np.cos(theta), np.sin(theta))


if __name__=='__main__':
	parser = ArgumentParser('Lattice Boltzmann simulations of liquid crystal hydrodynamics')
	parser.add_argument('-r', '--restart', action='store_true')
	parser.add_argument('-s', '--shape', type=int, nargs=2, default=[100, 100])
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
	parser.add_argument('--Temp', type=float, default=1./3.)
	parser.add_argument('--fr', type=float, default=0.01)
	parser.add_argument('--zeta', type=float, default=0.005)
	args = parser.parse_args()

	setattr(args,'itau_f', 1. / args.tau_f)
	setattr(args, 'qdt', 1. / args.n_evol_Q)
	setattr(args, 'xi1', 0.5 * (args.xi + 1.))
	setattr(args, 'xi2', 0.5 * (args.xi - 1.))

	fig = plt.figure(figsize=(6, 6))
	ax = fig.gca()
	plt.ion()
	plt.show()


	#Geometry information
	
	[Nx, Ny] = args.shape

	e = np.array([[0, 0],
				  [0, 1],
				  [0, -1],
				  [1, 0],
				  [-1, 0],
				  [1, 1],
				  [1, -1],
				  [-1, 1],
				  [-1, -1]]).astype(np.int32)

	weights = np.ones(e.shape[0]) / 36.
	weights[1:5] *= 4
	weights[0] *= 16

	dij = np.array([1, 0, 1], dtype=np.float64)

	#Allocate storage
	
	Q = np.empty((Ny, Nx, 3), dtype=np.float64)	#[Qyy, Qyx, Qxx]
	U = np.empty((Ny, Nx, 2), dtype=np.float64)	#[Uy, Ux]
	Rho = np.empty((Ny, Nx), dtype=np.float64)

	F1 = np.ones((Ny, Nx, e.shape[0]), dtype=np.float64)
	F2 = np.empty_like(F1)
	P = np.empty_like(F1)
	nextf = np.empty((Ny, Nx, e.shape[0], 3), dtype=np.int32)	#LB neighborlist
	build_stream(nextf, e)

	fd_nbr = np.empty((Ny, Nx, 4, 2), dtype=np.int32)
	H = np.empty_like(Q)
	build_neighbor(fd_nbr)	#Neighborlist in +/- x, y directions

	sigma_q = np.empty((Ny, Nx, 2, 2), dtype=np.float64)	#Stress tensor
	sigma_p = np.empty_like(U)								#\partial_j \sigma_{ij}
	W = np.empty((Ny, Nx, 2, 2), dtype=np.float64)			#Velocity gradient

	if args.restart:
		read_restart('restart.dat', Q, U, Rho)	
	else:
		Q[..., 0] = -1./2.
		Q[..., 1] = 0.
		Q[..., 2] = 1./2.

		U[..., 0] = 0.
		U[..., 1] = 0.
		Rho[:] = args.rho
	
	#Initialize F as Feq
	set_F_Feq(F1, U, Rho, e.astype(np.float64), weights)
	
	t_current = 0
	while t_current < args.t_max:

		t = time()

		for it in range(args.t_write):
			cal_W(W, U, nextf)
			for i in range(args.n_evol_Q):
				cal_dQ(Q, U, H, fd_nbr, 
					   args.A_ldg, args.U, args.L1, args.Gamma_rot, dij)
				evol_Q(Q, H, W, 
					   args.xi, args.xi1, args.xi2, args.qdt)
			
			cal_stress(sigma_p, sigma_q, Q, U, H, fd_nbr, nextf,
				args.A_ldg, args.U, args.L1, args.xi, args.zeta)
			evol_F(F1, F2, nextf, sigma_p, 
				args.fr, args.Temp, args.itau_f, e.astype(np.float64), weights)

			cal_W(W, U, nextf)
			for i in range(args.n_evol_Q):
				cal_dQ(Q, U, H, fd_nbr, 
					   args.A_ldg, args.U, args.L1, args.Gamma_rot, dij)
				evol_Q(Q, H, W, 
					   args.xi, args.xi1, args.xi2, args.qdt)

			cal_stress(sigma_p, sigma_q, Q, U, H, fd_nbr, nextf,
				args.A_ldg, args.U, args.L1, args.xi, args.zeta)
			evol_F(F2, F1, nextf, sigma_p, 
				args.fr, args.Temp, args.itau_f, e.astype(np.float64), weights)

		print(time() - t)
		display(ax, Q)
		if input() == 'q':
			exit(0)

		t_current += args.t_write
