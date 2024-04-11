import numpy as np

class Params:
    def __init__(self):
        self.default = {
            'temp_pre_e' : 24,
            'temp_pre_p' : 24,
            'crit_temp' : 33.2, # around this temperature both bacteria should be growing at the same rate

            'od_setpoint' : 0.5,
            'dil_rate' : 0.048, # 1/cycle

            'ts' : 1, # seconds

            'avg_temp' : True,
            'lp_ht' : [2/60,1], # IIR Filter halftime in h for temperature delay of [e,p] production
            'min_fl' : [0.037, 0.064, 0.057, 0.058, 0.036, 0.068, 0.064, 0.061],
            'max_fl' : [0.280, 0.408, 0.355, 0.375, 0.323, 0.391, 0.297, 0.310],
            # 'fl_ofs' : [0.06681171028640953, 0.10124284006186575, 0.012420329020951698, 0.09948044745557398, 0.08938092574497708, 0.11260643351414071, 0.05869055434532918, 0.048870744085821975],
            'od_fac' : 150,
            'e_fac' :  {'Allan': 0.8406, 'Arthur': 0.9445, 'Morlock': 1.0252, 'Swift': 1.1362, 'Stacey': 0.7536, 'Faith': 1.1945, 'Alabama': 0.9118, 'George': 1.0, 'Tammy': 0.7893, 'Taylor': 0.9146, 'Garth': 0.98, 'Willi': 1.0267, 'Reba': 1.1088, 'Shania': 0.9263, 'Johnny': 1.1502, 'Dolly': 1.0304, 'Carrie': 1.1627},
            'e_ofs' :  {'Allan': 214.4, 'Arthur': 213.8, 'Morlock': 291.9, 'Swift': 339.3, 'Stacey': 221.4, 'Faith': 241.2, 'Alabama': 173.5, 'George': 194.4, 'Tammy': 175.7, 'Taylor': 151.1, 'Garth': 186.2, 'Willi': 181.9, 'Reba': 208.9, 'Shania': 196.1, 'Johnny': 210.9, 'Dolly': 222.4, 'Carrie': 219.6},


            ### Growth rate parameters
            # Linear line fitting to 075_1 data
            'gr_e' : [0.07727, -1.67954 + 0.01],

            # Quadratic line fitting to 075_1 data
            'gr_p' : [-0.02208, 1.3796, -20.56996],

            # Quadratic line fitting to 075-1 data
            # 'gr_fp' : [-487.10807, 30798.23482, -480631.03115],
            'gr_fp' : [44.49572512791427, 31.820479677943325, -917.5864128224449, -886.6854718162117, 6318.711894680335],
            'min_gr_fp' : 20,


            ### Parameters for the Kalman filter
            'od_init' : 0.25, # initial belief optical density
            'e_rel_init' : 0.7, # initial relative belief of e. coli abundance
            # 'fp_init' : 0.056, # initial belief of fluorescence
            'fp_init' : 620, # initial belief of fluorescence

            'sigma_e_init' : 0.1,
            'sigma_p_init' : 0.1,
            'sigma_fp_init' : 200,

            # Process noise standard deviation
            'sigma_e_dil' : 1e-2,
            'sigma_p_dil' : 1e-2,
            'sigma_fp_dil' : 1e2,

            'sigma_e_dil_dit' : 5e-3,
            'sigma_p_dil_dit' : 5e-3,
            'sigma_fp_dil_dit' : 1e2,

            'sigma_e' : 1e-4,
            'sigma_p' : 1e-4,
            'sigma_fp' : 8e5,

            # Measurement update selection
            'od_update' : True,
            'fl_update' : True,
            'od_gr_update' : True,
            'fl_gr_update' : True,

            # Measurement noise standard deviation
            # 'sigma_od' : 1e-2,
            'sigma_od_mod' : 5e-3,
            'sigma_od' : 3e-2,
            'sigma_fl' :2e1,
            'fl_temp2_prox_sigma_max': 1e2,
            'sigma_od_gr' : 3e-2,
            'sigma_fl_gr' : 2e-2,
            'gr_res_sigma' : 1, #this lstsq residual corresponds to an putida estimation error usually smaller than of 0.05 (approximated from the simulation)
            'od_gr_prox_sigma_max' : 0.4, #this is the maximum value the uncertainty gets increased through proximity of both growth rates
            'od_gr_prox_sigma_decay' : 0.05, #at 0.05 growth rate difference the growth rate uncertainty is 1/e of the maximum growth rate uncertainty
            'fl_gr_temp_prox_sigma_max' : 0.4, #this is the maximum value the uncertainty gets increased through proximity to 35.5 degrees
            'fl_gr_temp_sigma_decay' : 0.7, #at 1 degree from the max temperature the temp uncertainty is approx 1/10th of the maximum temp uncertainty
            'fl_gr_temp2_prox_sigma_max' : 0.7, #this is the maximum value the uncertainty gets increased through proximity to 29 degrees
            'fl_gr_temp2_sigma_decay' : 1.0, #at 0.5 degree from the max temperature the temp uncertainty is approx 1/e of the maximum temp uncertainty


            ### Parameters for the Controller
            # PID gains
            'kp' : 15,
            'ki' : 0.01,
            'kd' : 0,
        }