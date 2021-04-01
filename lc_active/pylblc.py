import numpy as np
from time import time
from sys import exit
from argparse import ArgumentParser

from evolve_director import cal_dQ, evol_Q, cal_stress
from evolve_velocity import calc_f_equilibrium, cal_W, evol_f
from geometry import build_stream, build_neighbor, init_surf
from read_params import read_params
from util import n_to_Q, read_restart

Temp = 1. / 3.

if __name__=='__main__':
	parser = ArgumentParser('Lattice Boltzmann simulations of liquid crystal hydrodynamics')
	parser.add_argument('-p', '--param_file', type=str, default='param.in')
	parser.add_argument('-r', '--restart_file', type=str, default='restart.dat')
	args = parser.parse_args()

	'''
	Read Parameters
	'''
	print('Reading Parameters')
	params = read_params(args.param_file)
	for key in params:
		print('%10s:\t%10s' % (key, params[key]))

	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(1, 3, figsize=(10, 4))
	plt.ion()
	plt.show()

	'''
	Preallocate arrays
	'''
	print('Allocating memory')
	Nx, Ny, Nz = params['Nx'], params['Ny'], params['Nz']
	Npoints = Nx * Ny * Nz
	Z, Y, X = np.mgrid[:Nz, :Ny, :Nx]
	grid = np.array([Z, Y, X])

	#Q = [Qxx Qxy Qxz Qyy Qyz]
	Q 		= np.empty([Nz, Ny, Nx, 5], dtype=np.float64)	# Nematic order parameter
	Rho 	= np.empty([Nz, Ny, Nx],    dtype=np.float64)	# Fluid Density
	u 		= np.empty([Nz, Ny, Nx, 3], dtype=np.float64)	# Fluid velocity [Uz Uy Ux]

	info 	= np.ones([   Nz, Ny, Nx], dtype=np.int64) * -1

	if params['flow_on']:	# Allocate fields for LB velocity evolution
		f 	= np.empty([15, Nz, Ny, Nx], dtype=np.float64)		# Fluid distribution vector
		f2	= np.empty([15, Nz, Ny, Nx], dtype=np.float64)		# Fluid distribution buffer
		p 	= np.empty([15, Nz, Ny, Nx], dtype=np.float64)		# Forcing term
		nextf0 = np.empty([15, 4, Nz, Ny, Nx], dtype=np.int64)	# Streaming functions [i,z,y,x]
		Cf	= np.empty([15, Nz, Ny, Nx], dtype=np.float64)		# Collision term

	if params['Q_on']:		# Allocate fields for FD director evolution
		neighb0 = np.empty([6, 3, Nz, Ny, Nx], dtype=np.int64)	# Neigbor coordinates [z,y,x] for FD
		H	= np.empty([Nz, Ny, Nx, 5], dtype=np.float64)		# Molecular field

	if params['flow_on'] and params['Q_on']:
		sigma_q = np.empty([Nz, Ny, Nx, 3, 3], dtype=np.float64)	# Stress tensor sigma_{ij}
		sigma_p = np.empty([Nz, Ny, Nx, 3], dtype=np.float64)	# \partial_j sigma_{ij}
		W 		= np.empty([Nz, Ny, Nx, 3, 3], dtype=np.float64)	# Velocity gradient W_{ij}

	'''
	Check auxiliary information
		1. Surface anchoring
		2. Particles in flow field
		3. Patterns (of activity or TBA)
	'''

	#TODO handle surface points
	if params['wall_x'] or params['wall_y'] or params['wall_z'] or params['npar'] > 0:
		#allocate surface points
		pass

	#TODO handle particles
	if params['npar'] > 0:
		#p_allocate()
		pass

	#TODO Check for patterns
	if params['pattern_on']:
		#patt_allocate()
		pass

	'''
	Build evolution functions
		1. Lattice Boltzmann streaming functions 	- velocity field
		2. Finite differences neighbor lists 		- nematic order parameter
	'''

	if params['flow_on']:
		print('Building streaming functions')
		build_stream(nextf0, grid)
		nextf = [tuple(nextf0[i]) for i in range(nextf0.shape[0])]
	if params['Q_on']:
		print('Building FD grid neighbors')
		build_neighbor(neighb0, grid)
		#Neighbor list in +/- x, y, z directions
		neighb = [tuple(neighb0[i]) for i in range(neighb0.shape[0])]
		
	'''
	Initialize auxiliary information
		1. Surface anchoring
		2. Patch (what is this?)
		3. Particles in flow field
		4. Patterns (of activity or TBA)
	'''
	print('Initializing auxiliary fields')
	init_surf()

	if params['patch_on']:
		#add_patch
		pass

	if params['npar'] > 0:
		#p_init
		#p_iden
		#p_set_neighb
		pass
	
	if params['pattern_on']:
		#patt_init
		pass
	
	'''
	Initialize fields
	'''
	if params['newrun_on']:
		print('Initializing nematic fields')
		
		n_top = params['n_top']
		n_bot = params['n_bot']

		sita0 = np.dot(n_top, n_bot)
		sita0 /= np.linalg.norm(n_top) * np.linalg.norm(n_bot)
		if sita0 >  1:	sita0 =  1
		if sita0 < -1:	sita0 = -1
		sita0 = np.arccos(sita0)

		#TODO: Surface points and particles

		if sita0 < 1e-2 and sita0 > -1e-2:
			lambda1 = 0.5 * np.ones([Nz, Ny, Nx]) 
			lambda2 = 0.5 * np.ones([Nz, Ny, Nx])
		else:
			sita = sita0 * (Z + 0.5) / Nz
			lambda1 = np.cos(sita) - np.cos(sita0) * np.cos(sita0 - sita)
			lambda2 = np.cos(sita0 - sita) - np.cos(sita) * np.cos(sita0)

		#Default behavior - uniform director initialization (rand_init = -1)
		nx = lambda1 * n_bot[0] + lambda2 * n_top[0]
		ny = lambda1 * n_bot[1] + lambda2 * n_top[1]
		nz = lambda1 * n_bot[2] + lambda2 * n_top[2]
		
		#rand_init = 0, -101 - Seed with time
		if params['rand_init'] == 0 or params['rand_init'] == -101:
			np.random.seed(time())
		#Otherwise, use specific random seed
		elif params['rand_init'] >= 1 or params['rand_init'] <= 3:
			np.random.seed(params['rand_seed'])

		#rand_init = 0, 1 - Random director field
		if params['rand_init'] == 0 or params['rand_init'] == 1:
			nx = 1. - 2. * np.random.rand(Nz, Ny, Nx)	
			ny = 1. - 2. * np.random.rand(Nz, Ny, Nx)	
			nz = 1. - 2. * np.random.rand(Nz, Ny, Nx)	
		#rand_init = -101 - Small fluctuations from uniform
		elif params['rand_init'] == -101:
			nx += params['q_init'] * (1. - 2. * np.random.rand(Nz, Ny, Nx))
			ny += params['q_init'] * (1. - 2. * np.random.rand(Nz, Ny, Nx))
			nz = np.zeros([Nz, Ny, Nx])
		#rand_init = 2, 3 - Random 2d director field
		elif params['rand_init'] == 2 or params['rand_init'] == 3:
			if params['q_init'] < 0:
				nx = np.zeros([Nz, Ny, Nx])
				ny = 1. - 2. * np.random.rand(Nz, Ny, Nx)
				nz = 1. - 2. * np.random.rand(Nz, Ny, Nx)
			elif params['q_init'] == 0:
				nx = 1. - 2. * np.random.rand(Nz, Ny, Nx)
				ny = np.zeros([Nz, Ny, Nx])
				nz = 1. - 2. * np.random.rand(Nz, Ny, Nx)
			else:
				nx = 1. - 2. * np.random.rand(Nz, Ny, Nx)
				ny = 1. - 2. * np.random.rand(Nz, Ny, Nx)
				nz= np.zeros([Nz, Ny, Nx])

		#Convert director field to nematic order parameter 
		n_to_Q(nx, ny, nz, Q, params['S_lc'])
		q5 = -Q[..., 0] - Q[..., 3]
		if np.any(Q[..., 0] + Q[..., 3] + q5 > 1e-15) or np.any(Q[..., 0] + Q[..., 3] + q5 < -1e-15):
			print('Initial bulk Q not right')
			exit(1)

		#Uniform initial density and velocity field
		if params['flow_on']:
			Rho[:] = params['rho']
			u[:] = 0

	else:
		print('Reading from restart file')
		read_restart(args.restart_file, Q, Rho, u)
		pass
	
	print('Initialized!')

	if params['flow_on']:
		calc_f_equilibrium(f, Rho, u, Temp)
	if params['Q_on']:
		cal_dQ(Q, u, H, neighb, params['A_ldg'], params['U'], 
			params['L1'], params['Gamma_rot'], params['flow_on'])

	#Pre-evolve director before adding activity
	if params['Q_on'] and params['flow_on'] and params['newrun_on']:
		print('Equilibrating initial director field')
		t_current = 0
		qconverge = 0
		cal_W(W, u, nextf)
		while qconverge == 0 and t_current < 500:
			t_current += 1
			for i in range(params['n_evol_Q']):
				cal_dQ(Q, u, H, neighb, params['A_ldg'], params['U'], 
					params['L1'], params['Gamma_rot'], params['flow_on'])
				evol_Q(Q, H, W, 
					params['xi'], params['xi1'], params['xi2'], 
					params['qdt'], params['flow_on'])
			if t_current % 50 == 0:
				print('t=%d' % t_current)

	if params['Q_on'] and params['flow_on']:
		cal_W(W, u, nextf)
		cal_stress(sigma_p, sigma_q, Q, u, H, neighb, nextf, 
			params['A_ldg'], params['U'], params['L1'], params['xi'], params['zetai'])
	#TODO compute energies

	print('Initial director field initialized')

	t_current = 0

	while t_current < params['t_max']:
		ax[0].imshow(Q[0,..., 0])
		ax[1].imshow(u[0,..., 0])
		ax[2].imshow(Rho[0,...])
		if input() == 'q':
			exit(0)
		
		if params['Q_on']:
			if params['flow_on']:
				cal_W(W, u, nextf)
			for i in range(params['n_evol_Q']):
				cal_dQ(Q, u, H, neighb, params['A_ldg'], params['U'], 
					params['L1'], params['Gamma_rot'], params['flow_on'])
				if params['flow_on']:
					evol_Q(Q, H, W, 
						params['xi'], params['xi1'], params['xi2'], 
						params['qdt'], params['flow_on'])
				else:
					evol_Q(Q, H, None, 
						params['xi'], params['xi1'], params['xi2'], 
						params['qdt'], params['flow_on'])
		if params['flow_on']:
			if params['Q_on']:
				cal_stress(sigma_p, sigma_q, Q, u, H, neighb, nextf, 
					params['A_ldg'], params['U'], params['L1'], params['xi'], params['zetai'])
			else:
				sigma_p = None 
			evol_f(f, f2, nextf, p, u, Rho, sigma_p, Cf, 
				params['fr'], Temp, params['itau_f'], params['Q_on'])

		ax[0].imshow(Q[0,..., 0])
		ax[1].imshow(u[0,..., 0])
		ax[2].imshow(Rho[0,...])
		if input() == 'q':
			exit(0)
		
		if params['Q_on']:
			if params['flow_on']:
				cal_W(W, u, nextf)
			for i in range(params['n_evol_Q']):
				cal_dQ(Q, u, H, neighb, params['A_ldg'], params['U'], 
					params['L1'], params['Gamma_rot'], params['flow_on'])
				if params['flow_on']:
					evol_Q(Q, H, W, 
						params['xi'], params['xi1'], params['xi2'], 
						params['qdt'], params['flow_on'])
				else:
					evol_Q(Q, H, None, 
						params['xi'], params['xi1'], params['xi2'], 
						params['qdt'], params['flow_on'])
		if params['flow_on']:
			if params['Q_on']:
				cal_stress(sigma_p, sigma_q, Q, u, H, neighb, nextf, 
					params['A_ldg'], params['U'], params['L1'], params['xi'], params['zetai'])
			else:
				sigma_p = None
			evol_f(f2, f, nextf, p, u, Rho, sigma_p, Cf, 
				params['fr'], Temp, params['itau_f'], params['Q_on'])


		t_current += 1
