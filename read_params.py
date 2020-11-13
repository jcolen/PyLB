from math import sqrt
from util import getF0

param_keys = [
	'newrun_on',
	'Nx', 'Ny', 'Nz',
	'npar',
	'patch_on',
	'pattern_on',
	'wall_x', 'wall_y', 'wall_z',
	'flow_on',
	'Q_on',
	'rand_init', 'rand_seed', 'q_init',
	't_max', 't_print', 't0_write', 't_write',
	'n_evol_Q',
	'type_xlo', 'W_xlo', 'n_xlo',
	'type_xhi', 'W_xhi', 'n_xhi',
	'type_ylo', 'W_ylo', 'n_ylo',
	'type_yhi', 'W_yhi', 'n_yhi',
	'type_bot', 'W_bot', 'n_bot',
	'type_top', 'W_top', 'n_top',
	'tau_f',
	'ux_lo', 'uy_lo', 'uz_lo',
	'ux_hi', 'uy_hi', 'uz_hi',
	'xforce', 'yforce', 'zforce',
	'L1', 'L2', 'L3', 'L4',
	'K1', 'K2', 'K3', 'K24',
	'rho',
	'A_ldg',
	'q_ch',
	'U',
	'xi',
	'fr',
	'zetai', 'zetao',
	'Gamma_rot',
	'debug_on',
	'Q_tol', 'u_tol'
]

def read_params(filename):
	params_dict = {}
	with open(filename, 'r') as infile:
		for line in infile:
			if line[0] == '#':
				continue
			toks = line.split()

			if len([i for i in ['n_xlo', 'n_xhi', 'n_ylo', 'n_yhi', 'n_bot', 'n_top'] \
				if i in toks]) > 0:
				keys = toks[:3]
				vals = toks[3:-3] + [toks[-3:]]
			else:
				keys = toks[:len(toks) // 2]
				vals = toks[len(toks) // 2:]
			for key, val in zip(keys, vals):
				if not key in param_keys:
					print('No parameter named %s' % key)
					continue
				if isinstance(val, list):
					params_dict[key] = [int(v) for v in val]
				else:
					try:
						params_dict[key] = int(val)
					except:
						params_dict[key] = float(val)

	params_dict['itau_f'] = 1. / params_dict['tau_f']
	params_dict['qdt'] = 1. / params_dict['n_evol_Q']
	params_dict['S_lc'] = 0.25 + 0.75 * sqrt(1 - 8./3. / params_dict['U']) 
	params_dict['xi1'] = 0.5 * (params_dict['xi'] + 1.)
	params_dict['xi2'] = 0.5 * (params_dict['xi'] - 1.)

	slc = params_dict['S_lc']
	if 'K1' in params_dict:
		params_dict['L1'] = (params_dict['K2'] + (params_dict['K3'] - params_dict['K1']) * 1./3.) \
			* 0.5 / (slc**2)
		params_dict['L2'] = (params_dict['K1'] - params_dict['K24']) / (slc**2)
		params_dict['L3'] = (params_dict['K3'] - params_dict['K1']) * 0.5 / (slc**3)
		params_dict['L4'] = (params_dict['K24'] - params_dict['K2']) / (slc**2)
	elif 'L1' in params_dict:
		params_dict['K1'] = (2. * params_dict['L1'] + params_dict['L2'] \
			- 2./3. * slc * params_dict['L3'] + params_dict['L4']) * slc * slc
		params_dict['K2'] = (params_dict['L1'] - 1./3. * slc * params_dict['L3']) * slc * slc * 2
		params_dict['K3'] = (2. * params_dict['L1'] + params_dict['L2'] \
			+ 4./3. * slc * params_dict['L3'] + params_dict['L4']) * slc * slc
		params_dict['K4'] = params_dict['L4'] * slc * slc
		params_dict['K24'] = params_dict['K2'] + params_dict['K4']
	
	params_dict['twqL_ch'] = 2 * params_dict['q_ch'] * params_dict['L1']
	params_dict['Fld0'] =  getF0(params_dict['U'], params_dict['S_lc'], params_dict['A_ldg'])
	if params_dict['flow_on'] != 0:
		params_dict['uconverge'] = 0
	if params_dict['Q_on'] != 0:
		params_dict['qconverge'] = 0


	flag = False
	for key in param_keys:
		if not key in params_dict:
			print('Could not find entry for %s' % key)
			flag = True	
	if flag:
		return None

	return params_dict
