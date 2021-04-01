import numpy as np
from time import time
from sys import exit
from argparse import ArgumentParser
	
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
			  [-1, -1]])

cxs = e[:, 1]
cys = e[:, 0]

weights = np.ones(e.shape[0]) / 36.
weights[1:5] *= 4
weights[0] *= 16
bounce = [0, 2, 1, 4, 3, 8, 7, 6, 5]

def build_stream(grid, args):
	Ny, Nx = grid.shape[1:]
	prevf = np.empty([e.shape[0], 3, Ny, Nx], dtype=int)

	for i in range(e.shape[0]):
		prevf[i, 0] = i
		prevf[i, 1:] = np.mod(grid.T - e[i], grid.shape[1:]).T

	return prevf

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

	'''
	Preallocate arrays
	'''
	print('Allocating memory')
	[Nx, Ny, Nz] = args.shape
	Y, X = np.mgrid[:Ny, :Nx]
	grid = np.array([Y, X])

	prevf = build_stream(grid, args)
	prevf = [tuple(prevf[i]) for i in range(prevf.shape[0])]

	F = np.ones((NL, Ny, Nx)) + 0.01 * np.random.randn(NL, Ny, Nx)
	F[1] += 2 * (1 + 0.2*np.cos(2 * np.pi*X/Nx*4))
	rho = np.sum(F, axis=0)
	for i in idxs:
		F[i] *= args.rho / rho
	
	F2 = np.copy(F)

	if args.pattern:
		pattern = np.load('pattern.npy').astype(bool)[0]
	else:
		pattern = np.zeros([Nz, Ny, Nx], dtype=bool)[0]
		
	t_current = 0

	t = time()

	while t_current < args.t_max:

		for i in idxs:
			F[i] = F[prevf[i]]

		Fpattern = F[:, pattern]
		Fpattern = Fpattern[bounce]

		rho = np.sum(F, axis=0)
		u = np.divide(np.einsum('abc,ad', F, e), rho[..., None])
		u2 = np.einsum('...i,...i', u, u)
		ue = np.einsum('ab,cdb', e, u)


		Feq = np.zeros(F.shape)
		for i in idxs:
			Feq[i] = rho * weights[i] * (1 + 3 * ue[i] + 4.5 * ue[i] * ue[i] - 1.5 * u2)
			
		F += (-1.0/args.tau_f) * (F - Feq)
		F[:, pattern] = Fpattern

		if t_current % args.t_write == 0:
			print(time() - t)
			img = display(ax, u[..., 1], u[..., 0], pattern=pattern)
			if input() == 'q':
				exit(0)
			t = time()

		t_current += 1
