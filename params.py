import numpy as np

class Params:
    def __init__(self):
        self.default = {  # Set default parameters
            'temp_l' : 26,
            'temp_h' : 37,

            'od_setpoint' : 0.5,

            'ts' : 1, # seconds
            'dt' : 1, # seconds

            'avg_temp' : True,
            'lag' : 3, # hours
            'lag_ind' : 0, # Lag indeces
            'min_fl' : [0.037, 0.064, 0.057, 0.058, 0.036, 0.068, 0.064, 0.061],
            'max_fl' : [0.280, 0.408, 0.355, 0.375, 0.323, 0.391, 0.297, 0.310],
            'fl_ofs' : [0.06681171028640953, 0.10124284006186575, 0.012420329020951698, 0.09948044745557398, 0.08938092574497708, 0.11260643351414071, 0.05869055434532918, 0.048870744085821975],
            'od_ofs' : 0.2,


            ### Growth rate parameters
            # Linear line fitting to 062_5 data (without 36°C)
            'beta_e' : 0.08388,
            'alpha_e' : -1.934,

            # Cubic line fitting to 062_5 data
            'del_p' : -0.001184,
            'gam_p' :  0.09397,
            'beta_p' : -2.413,
            'alpha_p' : 20.74,

            # Cubic line fitting to 062_5 data
            'gr_fp' : [-0.00997995390810818, 0.5878457848575587, -8.328492791512256],


            ### Parameters for the Kalman filter
            'od_init' : 0.25, # initial belief optical density
            'e_rel_init' : 0.5, # %, initial relative belief of e. coli abundance
            'fp_init' : 0.056, # initial belief of fluorescence

            'sigma_e_init' : 0.03,
            'sigma_p_init' : 0.03,
            'sigma_fp_init' : 0.07,

            # Process noise standard deviation
            'sigma_e_dil' : 1e-2,
            'sigma_p_dil' : 1e-2,
            'sigma_fp_dil' : 1e-2,
            'q_dil' : np.zeros((3,3)),
            'sigma_e' : 1e-2,
            'sigma_p' : 1e-2,
            'sigma_fp' : 1e-2,
            'sigma_fl_ofs' : 1e-1,
            'q' : np.zeros((3,3)),

            # Measurement noise standard deviation
            'sigma_od' : 5e-2,
            'sigma_fl' : 1e-1,
            'r' : np.zeros((1,1))
        }