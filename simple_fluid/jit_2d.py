import numpy as np
from numba import jit
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
	print(grid.shape, grid.T.shape)

	for i in range(e.shape[0]):
		prevf[:, :, i, :-1] = np.mod(grid.T - e[i], grid.shape[1:]).transpose(1, 0, 2)
		prevf[:, :, i, -1] = i

	return prevf

@jit(nopython=True)
def evolve_velocity(F, pattern, prevf, itau_f, bounce=bounce, e=e, weights=weights):
	F2 = np.empty_like(F, dtype=F.dtype)
	Fpattern = np.empty(F.shape[-1], dtype=F.dtype)
	Feq = np.empty(F.shape[-1], dtype=F.dtype)
	rho = 0.
	u = np.zeros(2, dtype=np.float64)
	u2 = 0.
	ue = np.zeros(F.shape[-1])
	for y in range(F.shape[0]):
		for x in range(F.shape[1]):
			p = pattern[y, x]
			
			for i in range(F.shape[2]):
				yxi = prevf[y, x, i]
				F2[y, x, i] = F[yxi[0], yxi[1], yxi[2]]
			
			if pattern[y, x]:
				for i in range(F.shape[2]):
					Fpattern[i] = F2[y, x, bounce[i]]

			u[:] = 0
			ue[:] = 0
			rho = np.sum(F2[y, x])
			for i in range(F2.shape[2]):
				for j in range(e.shape[1]):
					u[j] += e[i, j] * F2[y, x, i]
			u /= rho
			for i in range(F2.shape[2]):
				for j in range(u.shape[0]):
					ue[i] += e[i, j] * u[j]
			u2 = np.sum(u * u)

			for i in range(F2.shape[2]):
				Feq[i] = 1. + 3. * ue[i] + 4.5 * ue[i] * ue[i] - 1.5 * u2
				Feq[i] *= weights[i]
			Feq *= rho

			F2[y, x] += -itau_f * (F2[y, x] - Feq)
			if pattern[y, x]:
				F2[y, x] = Fpattern	

	return F2

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

	F = np.ones((Ny, Nx, NL), dtype=np.float64) + 0.01 * np.random.randn(Ny, Nx, NL)
	F[..., 1] += 2 * (1 + 0.2*np.cos(2 * np.pi*X/Nx*4))
	rho = np.sum(F, axis=-1)
	for i in idxs:
		F[..., i] *= args.rho / rho
	
	if args.pattern:
		pattern = np.load('pattern.npy').astype(bool)[0]
	else:
		pattern = np.zeros([Nz, Ny, Nx], dtype=bool)[0]
		
	t_current = 0

	t = time()

	while t_current < args.t_max:

		F = evolve_velocity(F, pattern, prevf, 1.0 / args.tau_f)

		if t_current % args.t_write == 0:
			print(time() - t)
			rho = np.sum(F, axis=-1)
			u = np.divide(np.einsum('abc,cd', F, e), rho[..., None])
			img = display(ax, u[..., 1], u[..., 0], pattern=pattern)
			if input() == 'q':
				exit(0)
			t = time()

		t_current += 1
