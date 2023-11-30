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
        case '064-2-test':
            path = '../../Ting/Experiments/064-2'
            # Indeces of files of interest
            file_ind = [3,6]
            # Sampcycle indices as in Matlab
            sampcycle = [np.array([946, 1032, 1145, 1259, 1368, 1485, 2303, 2426, 2543, 2668]),
                        np.array([946, 1032, 1145, 1258, 1367, 1483, 2301, 2424, 2540, 2665])]
            sampcycle = [sampcycle[i][0:7]-1 for i in range(len(file_ind))] # adapts indeces to python
            titles = ['C8M0','C8M2']
        case '062-4':
            path = '../../Ting/Experiments/062-4'
            # Indeces of files of interest
            file_ind = [2,4,6,7,0,1,3,5]
            n_reactors = len(file_ind)
            # Sampcycle indices as in Matlab
            sampcycle = np.array([[0,1400] for i in range(n_reactors)])
            titles = ['26','27','29','30','31','33','35','37']
        case '062-5':
            path = '../../Ting/Experiments/062-5'
            # Indeces of files of interest
            file_ind = [4,5,6,7]
            n_reactors = len(file_ind)
            # Sampcycle indices as in Matlab
            sampcycle = np.array([[0,1400] for i in range(n_reactors)])
            titles = ['34','35','36','37']
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
        case '064-2-test':
            FC_file = pd.read_excel('../../Ting/Experiments/064-2/231027 Facility Analysis Manual Count.xlsx',header=[1])
            FC_data = FC_file['% Parent.1'] + FC_file['% Parent.2']
            cb_fc_ec = [FC_data[4::8].to_numpy(),
                        FC_data[6::8].to_numpy()]
        case _:
            raise Exception('Data information of {} not given.'.format(dataName))
    return cb_fc_ec