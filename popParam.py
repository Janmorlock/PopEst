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
        self.dil_sp = 0.5
        self.Avg_temp = True
        self.Lag = 3 # hours
        # TODO: Adapt min/max fluorescense values of the respective reactors
        self.min_fl = [0.037, 0.064, 0.057, 0.058, 0.036, 0.068, 0.064, 0.061]
        # self.min_fl = np.array([0.06668681763440748, 0.10129968019248045, 0.014508302984426407, 0.09772901252859235, 0.08988247242606058, 0.11187820931518658, 0.059842450984781456, 0.050616139008507456]).T # [0.037, 0.064, 0.057, 0.058, 0.036, 0.068, 0.064, 0.061]
        self.max_fl = []
        self.minv = 0.059
        self.maxv = 0.290

        self.od_ofs = np.array([0.2])
        self.e1_ofs = [120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0]
        self.od_fac = 200
        self.n_samples = 10000
        self.fit_e1 = True
        self.mcmc = False

        # Linear line fitting to 062_5 data (without 36°C)
        self.Beta_e = np.array([0.08388]) # -0.45
        self.Alpha_e =np.array([-1.934]) # 6.334

        # Cubic line fitting to 062_5 data
        self.Del_p = np.array([-0.001184])
        self.Gam_p =  np.array([0.09397])
        self.Beta_p = np.array([-2.413])
        self.Alpha_p = np.array([20.74])

        # Quadratic line fitting to 062_4 data
        # self.Gam_fp = np.array([-0.009919789860461197])
        # self.Beta_fp = np.array([0.5842676792191441])
        # self.Alpha_fp = np.array([-8.275973495562294])
        self.Gam_fp = np.array([-138.38010305512265])
        self.Beta_fp = np.array([8199.738520671823])
        self.Alpha_fp = np.array([-116924.32600350716])
        self.gr_fp = np.zeros(1,dtype='float')
                          

        self.Lag_ind = int(self.Lag*3600/self.Ts) # Lag in indeces