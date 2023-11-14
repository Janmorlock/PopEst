import numpy as np
import pandas as pd

class ModParam:
    def __init__(self):
        self.Ts = 1 # seconds

        self.T_l = 26
        self.T_h = 37

        self.T_pre_e = 37
        self.T_pre_p = 26

        self.Dil_th = 0.51
        self.Dil_amount = 0.02
        self.Avg_temp = True
        self.Lag = 3 # hours
        # TODO: Adapt min/max fluorescense values of the respective reactors
        self.min_fl = [0.0384, 0.141, 0.112, 0.119, 0.104, 0.13, 0.093, 0.108]
        self.max_fl = []
        self.minv = 0.059
        self.maxv = 0.290

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

        # Pw linear model for fluorescent protein dynamcs
        self.T_sl = 33
        self.c_sl = 0.16
        self.T_sh = 35
        self.c_sh = 0.005

        self.Lag_ind = int(self.Lag*3600/self.Ts) # Lag in indeces

def getCbDataInfo(dataName):
    match dataName:
        case '064-1':
            path = '../../Ting/Experiments/064-1'
            # Indeces of files of interest
            file_ind = [3, 5]
            # Sampcycle indices as in Matlab
            sampcycle = np.array([[1004, 1140, 1262, 1376, 1505, 1637, 1760, 2487],
                                  [2, 138, 260, 374, 502, 636, 759, 1486]])
            sampcycle -= 1 # adapts indeces to python
            titles = ['C8M2','C8M3']
        case '064-2':
            path = '../../Ting/Experiments/064-2'
            # Indeces of files of interest
            file_ind = [3,4,5,6,0,1,7,2]
            # Sampcycle indices as in Matlab
            sampcycle = [np.array([946, 1032, 1145, 1259, 1368, 1485, 2303, 2426, 2543, 2668]),
                        np.array([946, 1032, 1145, 1258, 1367, 1484, 2302, 2425, 2542, 2667]),
                        np.array([946, 1032, 1145, 1258, 1367, 1483, 2301, 2424, 2540, 2665]),
                        np.array([946, 1032, 1145, 1258, 1367, 1483, 2301, 2424, 2541, 2665]),
                        np.array([946, 1032, 1145, 1258, 1367, 1483, 2301, 2425, 2542, 2666]),
                        np.array([946, 1032, 1145, 1258, 1366, 1483, 2301, 2425, 2542, 2667]),
                        np.array([946, 1032, 1145, 1258, 1366, 1483, 2301, 2425, 2542, 2667])-942,
                        np.array([946, 1032, 1145, 1258, 1366, 1483, 2301, 2425, 2542])]
            sampcycle = [sampcycle[i][0:7]-1 for i in range(len(file_ind))] # adapts indeces to python
            titles = ['C8M0','C8M1','C8M2','C8M3','C9M0','C9M1','C9M2','C9M3']
        case '062-4':
            path = '../../Ting/Experiments/062-4'
            # Indeces of files of interest
            file_ind = [2,4,6,7,0,1,3,5]
            n_reactors = len(file_ind)
            # Sampcycle indices as in Matlab
            sampcycle = np.array([[0,1400] for i in range(n_reactors)])
            titles = ['26','27','29','30','31','33','35','37']
        case _:
            raise Exception('Data information of {} not given.'.format(dataName))
    
    return path, file_ind, sampcycle, titles

def getFcData(dataName):
    match dataName:
        case '064-1':
            FC_file = pd.read_excel('../../Ting/Experiments/064-1/231016 Facility Analysis Manual Count.xlsx',header=[1])
            FC_data = FC_file['% Parent.1'] + FC_file['% Parent.2']
            cb_fc_ec = np.array([FC_data[4::4].to_numpy(),
                                FC_data[5::4].to_numpy()])
        case '064-2':
            FC_file = pd.read_excel('../../Ting/Experiments/064-2/231027 Facility Analysis Manual Count.xlsx',header=[1])
            FC_data = FC_file['% Parent.1'] + FC_file['% Parent.2']
            cb_fc_ec = [FC_data[4::8].to_numpy(),
                        FC_data[5::8].to_numpy(),
                        FC_data[6::8].to_numpy(),
                        FC_data[7::8].to_numpy(),
                        FC_data[8::8].to_numpy(),
                        FC_data[9::8].to_numpy(),
                        FC_data[10::8].to_numpy(),
                        FC_data[11::8].to_numpy()]
        case _:
            raise Exception('Data information of {} not given.'.format(dataName))
    return cb_fc_ec