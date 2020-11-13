import numpy as np

#D3Q15 Lattice Vectors
e = np.array([
	[ 0,  0,  0],	# 0	
	[-1,  0,  0],	# 1		Z - 1
	[ 1,  0,  0],	# 2		Z + 1
	[ 0, -1,  0],	# 3		Y - 1
	[ 0,  1,  0],	# 4		Y + 1
	[ 0,  0, -1],	# 5		X - 1
	[ 0,  0,  1],	# 6		X + 1
	[ 1,  1,  1],	# 7
	[ 1,  1, -1],	# 8
	[ 1, -1,  1],	# 9
	[ 1, -1, -1],	# 10
	[-1,  1,  1],	# 11
	[-1,  1, -1],	# 12
	[-1, -1,  1],	# 13
	[-1, -1, -1],	# 14
])

# Compute the neighbor functions for the streaming step
def build_stream(nextf, grid):
	Nz, Nx, Ny = grid.shape[1:]
	gridT = grid.T
	for i in range(e.shape[0]):
		tmp = (gridT + e[i]).T

		#No walls - periodic BCs
		tmp[0, tmp[0, ...] < 0] = Nz - 1
		tmp[0, tmp[0, ...] >= Nz] = 0
		tmp[1, tmp[1, ...] < 0] = Ny - 1
		tmp[1, tmp[1, ...] >= Ny] = 0
		tmp[2, tmp[2, ...] < 0] = Nx -1
		tmp[2, tmp[2, ...] >= Nx] = 0
		
		nextf[i, 0, ...] = i
		nextf[i, 1:, ...] = tmp
	
#Compute neighbor list for FD calculations
#TODO compute next-neighbor lists in each direction
def build_neighbor(neighb, grid):
	Nz, Nx, Ny = grid.shape[1:]
	
	#Z - 1
	neighb[0, :] = grid
	neighb[0, 0] -= 1
	neighb[0, 0, 0] += Nz

	#Z + 1
	neighb[1, :] = grid
	neighb[1, 0] += 1
	neighb[1, 0, -1] -= Nz

	#Y - 1
	neighb[2, :] = grid
	neighb[2, 1] -= 1
	neighb[2, 1, :, 0] += Ny

	#Y + 1
	neighb[3, :] = grid
	neighb[3, 1] -= 1
	neighb[3, 1, :, -1] -= Ny

	#X - 1
	neighb[4, :] = grid
	neighb[4, 2] -= 1
	neighb[4, 2, ..., 0] += Nx

	#X + 1
	neighb[5, :] = grid
	neighb[5, 2] -= 1
	neighb[5, 2, ..., -1] -= Nx
	
	#TODO handle walls

def init_surf():
	pass
