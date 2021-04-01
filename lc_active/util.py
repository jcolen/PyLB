import numpy as np
import itertools

def QQ(q):
	Qm = np.array([
		[-q[..., 0]-q[..., 3], q[..., 4], q[..., 2]],
		[q[..., 4], q[..., 3], q[..., 1]],
		[q[..., 2], q[..., 1], q[..., 0]],
	])
	qq =  np.einsum('ij...,jk...', Qm, Qm)
	return np.transpose([
		qq[..., 2, 2],
		qq[..., 2, 1],
		qq[..., 2, 0],
		qq[..., 1, 1],
		qq[..., 1, 0],
	], (1, 2, 3, 0))

def QQQ(q):
	Qm = np.array([
		[-q[..., 0]-q[..., 3], q[..., 4], q[..., 2]],
		[q[..., 4], q[..., 3], q[..., 1]],
		[q[..., 2], q[..., 1], q[..., 0]],
	])
	return np.einsum('ij,jk,ki', Qm, Qm, Qm)

def trQQ(q):
	return 2. * (q[...,0]*q[...,0] + q[...,1]*q[...,1] + q[...,2]*q[...,2] + \
		q[...,3]*q[...,3] + q[...,4]*q[...,4] + q[...,0]*q[...,3])

def getF0(U, S, A_ldg):
	p = np.array([2./3. * S, 0, 0, -2./3.*S, 0])
	trqq = trQQ(p)
	qqq = QQQ(p)
	eld = A_ldg * (0.5 * (1. - U/3.) * trqq - U/3.*qqq + 0.25 * U * trqq * trqq)
	return eld

def n_to_Q(nx, ny, nz, Q, S_lc):
	r2 = nx * nx + ny * ny + nz * nz
	r2i = 1./r2

	Q[..., 0] = nx * nx * r2i - 1./3.	# Qxx
	Q[..., 1] = ny * ny * r2i			# Qxy
	Q[..., 2] = nx * nz * r2i			# Qxz
	Q[..., 3] = ny * ny * r2i - 1./3.	# Qyy
	Q[..., 4] = ny * nz * r2i			# Qyz
	Q *= S_lc

	Q[r2 < 1e-8, :] = 0

'''
Read data from restart file
Format is:
Nx Ny Nz wall_x wall_y wall_z
'''
def read_restart(restart_file, Q, Rho, U):
	with open(restart_file, 'r') as fin:
		header = fin.readline().split()
		Nx, Ny, Nz = int(header[0]), int(header[1]), int(header[2])
		
		#Assert we have no walls, since we can't handle that yet
		if len(header) > 3:
			assert int(header[3]) == 0
			assert int(header[4]) == 0
			assert int(header[5]) == 0

		q = np.loadtxt(itertools.islice(fin, 0, Nx * Ny)).reshape([Ny, Nx, 3])
		rho = np.loadtxt(itertools.islice(fin, 0, Nx * Ny)).reshape([Ny, Nx])
		u = np.loadtxt(itertools.islice(fin, 0, Nx * Ny)).reshape([Ny, Nx, 2])
	
		Q[None, ..., 0] = q[..., 0]
		Q[None, ..., 1] = q[..., 1]
		Q[None, ..., 2] = 0.
		Q[None, ..., 3] = q[..., 2]
		Q[None, ..., 4] = 0.

		Rho[None]		= rho

		U[None, ..., 0] = 0
		U[None, ..., 1] = u[..., 1]
		U[None, ..., 2] = u[..., 0]
		
