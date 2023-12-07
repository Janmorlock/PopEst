import numpy as np
import pandas as pd

class ModParam:
    def __init__(self):
        self.Ts = 1 # seconds

        self.temp_l = 26
        self.temp_h = 37

        self.T_pre_e = 37
        self.T_pre_p = 26

        self.dil_th = 0.54
        self.dil_amount = 0.11
        self.Dil_sp = 0.5
        self.Avg_temp = True
        self.Lag = 3 # hours
        # TODO: Adapt min/max fluorescense values of the respective reactors
        # self.min_fl = [0.0384, 0.141, 0.112, 0.119, 0.104, 0.13, 0.093, 0.108]
        self.min_fl = np.array([0.06670286988458429, 0.09954091472779605, 0.01339878054716663, 0.09732706551691282, 0.08975873832801415, 0.1128096218942311, 0.061012361789911165, 0.04765733175961176]).T # [0.037, 0.064, 0.057, 0.058, 0.036, 0.068, 0.064, 0.061]
        self.max_fl = []
        self.minv = 0.059
        self.maxv = 0.290

        self.od_ofs = np.array([0.2])
        self.n_samples = 10000
        self.mcmc = True

        # Linear line fitting to 062_5 data (without 36°C)
        self.Beta_e = np.array([0.08388]) # -0.45
        self.Alpha_e =np.array([-1.934]) # 6.334

        # Cubic line fitting to 062_5 data
        self.Del_p = np.array([-0.001184])
        self.Gam_p =  np.array([0.09397])
        self.Beta_p = np.array([-2.413])
        self.Alpha_p = np.array([20.74])

        # Pw linear model for fluorescent protein dynamcs
        self.Del_fp = np.array([0.0])
        self.Gam_fp = np.array([-0.010030269158100553])
        self.Beta_fp = np.array([0.5909218418486065])
        self.Alpha_fp = np.array([-8.374536265928038])
        self.gr_fp = np.zeros(1,dtype='float')
                          

        self.Lag_ind = int(self.Lag*3600/self.Ts) # Lag in indeces