import numpy as np

class Params:
    def __init__(self):
        self.default = {  # Set default parameters
            'temp_pre_e' : 37,
            'temp_pre_p' : 26,

            'od_setpoint' : 0.5,

            'ts' : 1, # seconds

            'avg_temp' : True,
            'lag' : 3, # hours
            'lag_ind' : 0, # Lag indeces
            'min_fl' : [0.037, 0.064, 0.057, 0.058, 0.036, 0.068, 0.064, 0.061],
            'max_fl' : [0.280, 0.408, 0.355, 0.375, 0.323, 0.391, 0.297, 0.310],
            # 'fl_ofs' : [0.06681171028640953, 0.10124284006186575, 0.012420329020951698, 0.09948044745557398, 0.08938092574497708, 0.11260643351414071, 0.05869055434532918, 0.048870744085821975],
            'od_fac' : 250,
            'e_fac' :  {'Allan': 0.8406, 'Arthur': 0.9445, 'Morlock': 1.0252, 'Swift': 1.1362, 'Stacey': 0.7536, 'Faith': 1.1945, 'Alabama': 0.9118, 'George': 1.0, 'Tammy': 0.7893, 'Taylor': 0.9146, 'Garth': 0.98, 'Willi': 1.0267, 'Reba': 1.1088, 'Shania': 0.9263, 'Johnny': 1.1502, 'Dolly': 1.0304, 'Carrie': 1.1627},
            'e_ofs' :  {'Allan': 214.4, 'Arthur': 213.8, 'Morlock': 291.9, 'Swift': 339.3, 'Stacey': 221.4, 'Faith': 241.2, 'Alabama': 173.5, 'George': 194.4, 'Tammy': 175.7, 'Taylor': 151.1, 'Garth': 186.2, 'Willi': 181.9, 'Reba': 208.9, 'Shania': 196.1, 'Johnny': 210.9, 'Dolly': 222.4, 'Carrie': 219.6},


            ### Growth rate parameters
            # Linear line fitting to 075_1 data
            'gr_e' : [0.08023, -1.81796],

            # Quadratic line fitting to 075_1 data
            'gr_p' : [-0.02087, 1.30128, -19.31919],

            # Quadratic line fitting to 062_4 data
            'gr_fp' : [-0.00997995390810818, 0.5878457848575587, -8.328492791512256],


            ### Parameters for the Kalman filter
            'od_init' : 0.25, # initial belief optical density
            'e_rel_init' : 0.5, # %, initial relative belief of e. coli abundance
            # 'fp_init' : 0.056, # initial belief of fluorescence
            'fp_init' : 620, # initial belief of fluorescence

            'sigma_e_init' : 0.03,
            'sigma_p_init' : 0.03,
            'sigma_fp_init' : 20, # 0.07

            # Process noise standard deviation
            'sigma_e_dil' : 1e-2,
            'sigma_p_dil' : 1e-2,
            'sigma_fp_dil' : 1e2,
            'q_dil' : np.zeros((3,3)),
            'sigma_e' : 1e-4,
            'sigma_p' : 1e-4,
            'sigma_fp' : 5e5,
            'q' : np.zeros((3,3)),

            # Measurement noise standard deviation
            'sigma_od' : 5e-2,
            'sigma_fl' : 1e1,
            'r' : np.zeros((2,2))
        }