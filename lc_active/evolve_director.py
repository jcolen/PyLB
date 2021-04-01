import numpy as np

from util import trQQ, QQ

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
Compute dQ for finite differences calculations
This step computes the molecular field H for the director field
'''
def cal_dQ(Q, u, H, neighb, A_ldg, U_lc, L1, Gamma_rot, flow_on):
	dzQ = 0.5 * (Q[neighb[1]] - Q[neighb[0]])
	d2zQ = Q[neighb[1]] - 2 * Q + Q[neighb[0]]	
	
	dyQ = 0.5 * (Q[neighb[3]] - Q[neighb[2]])
	d2yQ = Q[neighb[3]] - 2 * Q + Q[neighb[2]]	
	
	dxQ = 0.5 * (Q[neighb[5]] - Q[neighb[4]])
	d2xQ = Q[neighb[5]] - 2 * Q + Q[neighb[4]]	

	trqq = trQQ(Q)
	qq = QQ(Q)

	#Compute molecular field H from free energy minimization
	H[:] = 0
	H +=  A_ldg * (1. - U_lc * one3rd) * Q \
		- A_ldg * U_lc * qq \
		+ A_ldg * U_lc * trqq[..., None] * Q \
		- L1 * (d2xQ + d2yQ + d2zQ)
	
	#Additional terms for diagonal elements
	H[..., 0] += A_ldg * U_lc * one3rd * trqq
	H[..., 3] += A_ldg * U_lc * one3rd * trqq
	H *= -1
	
	#TODO Include diagonal effects if not in single elastic constant approximation
	#TODO Figure out that additional term in the L3 coefficient
	#Compute dQ matrix dQ_{ij} = d_i Q_{ab} d_j Q_{ab}
	#dQ = np.empty(Q.shape[:-1] + (3, 3), dtype=Q.dtype)
	#dQ[..., 0, 0] = 2 * np.sum(dxQ * dxQ, axis=-1)
	#dQ[..., 0, 1] = 2 * np.sum(dxQ * dyQ, axis=-1)
	#dQ[..., 0, 2] = 2 * np.sum(dxQ * dzQ, axis=-1)
	#dQ[..., 1, 1] = 2 * np.sum(dyQ * dyQ, axis=-1)
	#dQ[..., 1, 2] = 2 * np.sum(dyQ * dzQ, axis=-1)
	#dQ[..., 2, 2] = 2 * np.sum(dzQ * dzQ, axis=-1)

	#Combine H and convective of Q
	H *= Gamma_rot
	if flow_on:
		#H -= u[..., 0, None] * dzQ + u[..., 1, None] * dyQ + 0. * u[..., 2, None] * dxQ
		H -= + u[..., 1, None] * dyQ + 0. * u[..., 2, None] * dxQ

def evol_Q(Q, H, W, xi, xi1, xi2, qdt, flow_on):
	#Compute co-rotation term
	if flow_on:
		#Here A = (W + W^T) / 2, \Omega = (W - W^T) / 2
		XAO = np.empty(Q.shape[:-1] + (3, 3), dtype=Q.dtype)	#xi A + \Omega
		M = np.empty(Q.shape[:-1] + (3, 3), dtype=Q.dtype)		#Q_{ij} + \delta_{ij} / 3
		WT = np.transpose(W, (0, 1, 2, 4, 3))
		
		XAO = xi2 * W + xi1 * WT

		M[..., 2, 2] = Q[..., 0] + one3rd
		M[..., 2, 1] = Q[..., 1]
		M[..., 1, 2] = M[..., 2, 1]
		M[..., 2, 0] = Q[..., 2]
		M[..., 0, 2] = M[..., 2, 0]
		M[..., 1, 1] = Q[..., 3] + one3rd
		M[..., 1, 0] = Q[..., 4]
		M[..., 0, 1] = M[..., 1, 0]
		M[..., 0, 0] = 1. - M[..., 2, 2] - M[..., 1, 1]

		trQA = Q[..., 0] * (W[..., 2, 2] - W[..., 0, 0]) \
			 + Q[..., 3] * (W[..., 1, 1] - W[..., 0, 0]) \
			 + Q[..., 1] * (W[..., 2, 1] + W[..., 1, 2]) \
			 + Q[..., 2] * (W[..., 2, 0] + W[..., 0, 2]) \
			 + Q[..., 4] * (W[..., 1, 0] + W[..., 0, 1])

		S = np.einsum('...ik,...kj', XAO, M) + np.einsum('...ik,...jk', M, XAO) \
			- 2. * xi * M * trQA[..., None, None]
		S[..., 0, :] = 0
	else:
		S = np.zeros(Q.shape[:-1] + (3, 3), dtype=Q.dtype)

	Q[..., 0] += qdt * (H[..., 0] + S[..., 2, 2]) 
	Q[..., 1] += qdt * (H[..., 1] + S[..., 2, 1]) 
	Q[..., 2] = 0 
	Q[..., 3] += qdt * (H[..., 2] + S[..., 1, 1]) 
	Q[..., 4] = 0

'''
Compute the active and elastic stress terms in the momentum equation
In reality, this actually computes the $derivative$ of the stress \partial_j \sigma_{ij}

Be careful - recall that the order of directions is Z Y X not X Y Z

NOTE: Combining cal_stress and cal_sigma_p into a single function
'''
def cal_stress(sigma_p, sigma_q, Q, u, H, neighb, nextf, A_ldg, U_lc, L1, xi, zetai):
	dzQ = 0.5 * (Q[neighb[1]] - Q[neighb[0]])
	d2zQ = Q[neighb[1]] - 2 * Q + Q[neighb[0]]	
	
	dyQ = 0.5 * (Q[neighb[3]] - Q[neighb[2]])
	d2yQ = Q[neighb[3]] - 2 * Q + Q[neighb[2]]	
	
	dxQ = 0.5 * (Q[neighb[5]] - Q[neighb[4]])
	d2xQ = Q[neighb[5]] - 2 * Q + Q[neighb[4]]	

	trqq = trQQ(Q)
	qq = QQ(Q)

	#Compute molecular field H from free energy minimization
	H[:] = A_ldg * (1. - U_lc * one3rd) * Q \
		 - A_ldg * U_lc * qq \
		 + A_ldg * U_lc * trqq[..., None] * Q \
		 - L1 * (d2xQ + d2yQ + d2zQ)
	
	#Additional terms for diagonal elements
	H[..., 0] += A_ldg * U_lc * one3rd * trqq
	H[..., 3] += A_ldg * U_lc * one3rd * trqq
	H *= -1

	M = np.empty(Q.shape[:-1] + (3, 3), dtype=Q.dtype)
	h = np.empty_like(M)

	# M_{ij} = Q_{ij} + 1/3 \delta_{ij}
	M[..., 0, 0] = -Q[..., 0] - Q[..., 3] + one3rd	#ZZ
	M[..., 0, 1] = Q[..., 4]						#ZY
	M[..., 0, 2] = Q[..., 2]						#ZX
	M[..., 1, 0] = Q[..., 4]						#YZ
	M[..., 1, 1] = Q[..., 3] + one3rd				#YY
	M[..., 1, 2] = Q[..., 1]						#YX
	M[..., 2, 0] = Q[..., 2]						#XZ
	M[..., 2, 1] = Q[..., 1]						#XY
	M[..., 2, 2] = Q[..., 0] + one3rd				#XX
	
	h[..., 2, 2] = H[..., 0]
	h[..., 2, 1] = H[..., 1]
	h[..., 2, 0] = H[..., 2]
	h[..., 1, 2] = H[..., 1]
	h[..., 1, 1] = H[..., 3]
	h[..., 1, 0] = H[..., 4]
	h[..., 0, 2] = H[..., 2]
	h[..., 0, 1] = H[..., 4]
	h[..., 0, 0] = -H[..., 0] - H[..., 3]


	summ = 2. * np.einsum('...i,...i', Q, H) + Q[..., 0] * H[..., 3] + Q[..., 3] * H[..., 0]

	sigma_q[:] = 0
	sigma_q += 2. * xi * summ[..., None, None] * M
	sigma_q += (1. - xi) * np.einsum('...im,...mj', M, h)
	sigma_q -= (1. + xi) * np.einsum('...im,...mj', h, M)

	#TODO include activity patterning

	sigma_q -= zetai * M

	# Compute sigma_p_{i} = \partial_{j} sigma_q_{ij}

	sigma_p[:] = 0

	# Trace terms can be computed directly
	sigma_p[..., 0] += -2. * np.einsum('...i,...i', H, dzQ) - H[..., 0] * dzQ[..., 3] - H[..., 3] * dzQ[..., 0]
	sigma_p[..., 1] += -2. * np.einsum('...i,...i', H, dyQ) - H[..., 0] * dyQ[..., 3] - H[..., 3] * dyQ[..., 0]
	sigma_p[..., 2] += -2. * np.einsum('...i,...i', H, dxQ) - H[..., 0] * dxQ[..., 3] - H[..., 3] * dxQ[..., 0]

	idz_m = nextf[1][1:]
	idz_p = nextf[2][1:]
	idy_m = nextf[3][1:]
	idy_p = nextf[4][1:]
	idx_m = nextf[5][1:]
	idx_p = nextf[6][1:]

	sigma_p += 0.5 * (sigma_q[idz_p][..., 0] - sigma_q[idz_m][..., 0])
	sigma_p += 0.5 * (sigma_q[idy_p][..., 1] - sigma_q[idy_m][..., 1])
	sigma_p += 0.5 * (sigma_q[idx_p][..., 2] - sigma_q[idx_m][..., 2])
