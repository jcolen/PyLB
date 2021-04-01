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

	F1 = np.ones((NL, Ny, Nx)) + 0.01 * np.random.randn(NL, Ny, Nx)
	F1[1] += 2 * (1 + 0.2*np.cos(2 * np.pi*X/Nx*4))
	rho = np.sum(F1, axis=0)
	for i in idxs:
		F1[i] *= args.rho / rho
	
	F2 = np.copy(F1)

	if args.pattern:
		pattern = np.load('pattern.npy').astype(bool)[0]
	else:
		pattern = np.zeros([Nz, Ny, Nx], dtype=bool)[0]
		
	t_current = 0

	def print_stats(a):
		try:
			print(np.max(a), np.min(a), np.average(a))
		except Exception as e:
			print(e)

	while t_current < args.t_max:

		for i in idxs:
			F1[i] = F1[prevf[i]]
			F2[i] = np.roll(F2[i], cxs[i], axis=1)
			F2[i] = np.roll(F2[i], cys[i], axis=0)
			print_stats(F1[i] - F2[i])

		F1pattern = F1[:, pattern]
		F1pattern = F1pattern[bounce]

		F2pattern = F2[:, pattern]
		F2pattern = F2pattern[bounce]

		print_stats(F1pattern - F2pattern)

		rho1 = np.sum(F1, axis=0)
		u1 = np.divide(np.einsum('abc,ad', F1, e), rho1[..., None])
		print(u1.shape)
		u21 = np.einsum('...i,...i', u1, u1)
		ue1 = np.einsum('ab,cdb', e, u1)

		rho2 = np.sum(F2, axis=0)
		ux2 = np.einsum('i...,i', F2, cxs) / rho2
		uy2 = np.einsum('i...,i', F2, cys) / rho2
	
		print_stats(rho1-rho2)
		print_stats(u1[..., 1] - ux2)
		print_stats(u1[..., 0] - uy2)
		print_stats(u21 - (ux2**2 + uy2**2))

		F1eq = np.zeros(F1.shape)
		F2eq = np.zeros(F2.shape)
		for i in idxs:
			F1eq[i] = rho1 * weights[i] * (1 + 3 * ue1[i] + 4.5 * ue1[i] * ue1[i] - 1.5 * u21)
			F2eq[i] = rho2 * weights[i] * (1 + \
				3 * (cxs[i] * ux2 + cys[i] * uy2) + \
				4.5 * (cxs[i] * ux2 + cys[i] * uy2)**2 - \
				1.5 * (ux2**2 + uy2**2))

			print_stats(F1eq[i] - F2eq[i])
			
		F1 += (-1.0/args.tau_f) * (F1 - F1eq)
		F1[:, pattern] = F1pattern
		F2 += (-1.0/args.tau_f) * (F2 - F2eq)
		F2[:, pattern] = F2pattern

		print_stats(F1 - F2)
		
		if t_current % args.t_write == 0:
			#img = display(ax, u1[..., 1], u1[..., 0], pattern=pattern)
			img = display(ax, ux2, uy2, pattern=pattern)
			if input() == 'q':
				exit(0)

		t_current += 1
