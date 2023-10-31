class Param:
    def __init__(self):
        self.Ts = 1 # seconds
        self.Sim_h = 10 # hours

        self.T_l = 26
        self.T_h = 37
        self.Dil_th = 0.51
        self.Dil_amount = 0.02

        self.T_pre_e = 37
        self.T_pre_p = 26

        self.Avg_temp = False
        self.Lag = 0 # hours
        self.Lag_ind = int(self.Lag*3600/self.Ts) # Lag in indeces

        # TODO: Adapt min/max fluorescense values of the respective reactors
        self.min_fl = [0.061, 0.059]
        self.max_fl = [0.285, 0.290]

        # Linear line fitting to 062_5 data (without 36°C)
        self.Del_e = 0
        self.Gam_e =  0 # 0.008511
        self.Beta_e = 0.08388 # -0.45
        self.Alpha_e = -1.934 # 6.334

        # Cubic line fitting to 062_5 data
        self.Del_p = -0.001184
        self.Gam_p =  0.09397
        self.Beta_p = -2.413
        self.Alpha_p = 20.74

        # Quadratic model for fluorescent protein dynamcs
        self.Gam_f =  0
        self.Beta_f = -0.1
        self.Alpha_f = 3.6
        self.m = 1
        self.c = 0

        # TODO: Add fluorescense growth parameters

        self.Init_e = 0.25
        self.Init_p = 0.25
        self.Init_f = 0