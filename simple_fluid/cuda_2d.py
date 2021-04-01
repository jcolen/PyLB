import numpy as np
import math
from numba import jit, cuda
from sys import exit
from argparse import ArgumentParser
from time import time
	
NL = 9
idxs = np.arange(NL)
e = np.array([[0, 0],
			  [0, 1],
			  [0, -1],
			  [1, 0],
			  [-1, 0],
			  [1, 1],
			  [1, -1],
			  [-1, 1],
			  [-1, -1]]).astype(np.float64)

weights = np.ones(e.shape[0]) / 36.
weights[1:5] *= 4
weights[0] *= 16
bounce = np.array([0, 2, 1, 4, 3, 8, 7, 6, 5]).astype(np.int32)

def build_stream(grid, args):
	Ny, Nx = grid.shape[1:]
	prevf = np.empty([Ny, Nx, e.shape[0], 3], dtype=np.int32)

	for i in range(e.shape[0]):
		prevf[:, :, i, :-1] = np.mod(grid.T - e[i], grid.shape[1:]).transpose(1, 0, 2)
		prevf[:, :, i, -1] = i

	return prevf

def iterate_evolution(t_max, F1, F2, pattern, prevf, itau_f, bounce=bounce, e=e, weights=weights):

	#Prep cuda info
	threadsperblock = (32, 32)
	blockspergrid_y = math.ceil(F1.shape[0] / threadsperblock[0])
	blockspergrid_x = math.ceil(F1.shape[1] / threadsperblock[1])
	blockspergrid = (blockspergrid_y, blockspergrid_x)

	#Copy arrays to device
	d_F1 = cuda.to_device(F1)
	d_F2 = cuda.to_device(F2)
	d_pattern = cuda.to_device(pattern)
	d_prevf = cuda.to_device(prevf)
	d_bounce = cuda.to_device(bounce)
	d_e = cuda.to_device(e)
	d_weights = cuda.to_device(weights)

	for t in range(t_max):
		evolve_velocity[blockspergrid, threadsperblock](d_F1, d_F2, d_pattern, d_prevf, itau_f, d_bounce, d_e, d_weights)
		evolve_velocity[blockspergrid, threadsperblock](d_F2, d_F1, d_pattern, d_prevf, itau_f, d_bounce, d_e, d_weights)

	#Copy array back from device
	d_F1.copy_to_host(F1)

	return F1

@cuda.jit
def evolve_velocity(F1, F2, pattern, prevf, itau_f, bounce=bounce, e=e, weights=weights):
	y, x = cuda.grid(2)
	if y < F1.shape[0] and x < F1.shape[1]:
		if pattern[y, x]:
			for i in range(F1.shape[2]):
				yxi = prevf[y, x, bounce[i]]
				F2[y, x, i] = F1[yxi[0], yxi[1], yxi[2]]
		else:
			for i in range(F1.shape[2]):
				yxi = prevf[y, x, i]
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

def display(ax, ux, uy, pattern=None):
	ax.clear()
	ax.set_xticks([])
	ax.set_yticks([])
	vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
	if pattern is not None:
		vorticity[pattern] = np.nan
		plt.cm.bwr.set_bad('black')
	
	img = ax.imshow(vorticity, cmap='bwr')
	
	return img

if __name__=='__main__':
	parser = ArgumentParser('Lattice Boltzmann simulations of liquid crystal hydrodynamics')
	parser.add_argument('-r', '--restart', action='store_true')
	parser.add_argument('-s', '--shape', type=int, nargs=3, default=[400,100,1])
	parser.add_argument('--patch', action='store_true')
	parser.add_argument('--pattern', action='store_true')
	parser.add_argument('--tau_f', type=float, default=0.6)
	parser.add_argument('--rho', type=float, default=1.)
	parser.add_argument('--t_max', type=int, default=1000)
	parser.add_argument('--t_write', type=int, default=10)
	args = parser.parse_args()

	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(12, 4))
	ax = fig.gca()
	plt.ion()
	plt.show()

	[Nx, Ny, Nz] = args.shape
	Y, X = np.mgrid[:Ny, :Nx]
	grid = np.array([Y, X])

	prevf = build_stream(grid, args)
	#prevf = [tuple(prevf[i]) for i in range(prevf.shape[0])]

	F1 = np.ones((Ny, Nx, NL), dtype=np.float64) + 0.01 * np.random.randn(Ny, Nx, NL)
	F1[..., 1] += 2 * (1 + 0.2*np.cos(2 * np.pi*X/Nx*4))
	F2 = np.empty_like(F1)
	rho = np.sum(F1, axis=-1)
	for i in idxs:
		F1[..., i] *= args.rho / rho
	
	if args.pattern:
		pattern = np.load('pattern.npy').astype(bool)[0]
	else:
		pattern = np.zeros([Nz, Ny, Nx], dtype=bool)[0]
		
	t_current = 0

	t = time()

	while t_current < args.t_max:

		t = time()
		F1 = iterate_evolution(args.t_write, F1, F2, pattern, prevf, 1.0 / args.tau_f)

		print(time() - t)
		rho = np.sum(F1, axis=-1)
		u = np.divide(np.einsum('abc,cd', F1, e), rho[..., None])
		img = display(ax, u[..., 1], u[..., 0], pattern=pattern)
		if input() == 'q':
			exit(0)
		t = time()

		t_current += args.t_write
