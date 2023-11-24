import numpy as np
import pandas as pd

class Params:
    def __init__(self):
        self.default = {  # Set default parameters
            'temp_l' : 26,
            'temp_h' : 37,

            'dil_dithered' : False,
            # Dither values are calculated by:
            # For a setpoint of 0.5, the highest point should be centre + zig (0.54) and the lowest point should be centre - zig*1.5 (0.44)
            'zig' : 0.04, # (arbitrary value we use in the CB code)
            'od_setpoint' : 0.5,
            'dil_th' : 0,
            'dil_amount' : 0,

            'ts' : 1, # seconds
            'dt' : 1, # seconds

            'od_init' : 0.25, # initial belief optical density
            'e_rel_init' : 0.5, # %, initial relative belief of e. coli abundance
            'fl_init' : 0.2, # initial belief of fluorescence

            'sigma_e_init' : 0.01,
            'sigma_p_init' : 0.01,
            'sigma_fp_init' : 0.07,

            'avg_temp' : True,
            'lag' : 3, # hours
            'lag_ind' : 0, # Lag indeces
            # 'min_fl' : [0.06681911858766822, 0.10092813517299981, 0.036706517103062705, 0.09824800951992305, 0.08935696216429849, 0.11279752402152422, 0.06290159139625404, 0.050540617661143504],
            'min_fl' : [0.037, 0.064, 0.057, 0.058, 0.036, 0.068, 0.064, 0.061],
            'max_fl' : [0.280, 0.408, 0.355, 0.375, 0.323, 0.391, 0.297, 0.310],

            # Linear line fitting to 062_5 data (without 36°C)
            'beta_e' : 0.08388, # -0.45
            'alpha_e' : -1.934, # 6.334

            # Cubic line fitting to 062_5 data
            'del_p' : -0.001184,
            'gam_p' :  0.09397,
            'beta_p' : -2.413,
            'alpha_p' : 20.74,

            # Pw linear model for fluorescent protein dynamcs
            'temp_sl' : 33,
            'c_sl' : 0.16,
            'temp_sh' : 35,
            'c_sh' : 0.005,
            'gr_fp' : [0.000193358172191664, -0.02344361901656315, 1.0521220115703511, -20.72204666758777, 151.45884447928148],


            # Process noise standard deviation
            'sigma_e_dil' : 5e-3,
            'sigma_p_dil' : 5e-3,
            'sigma_fp_dil' : 5e-3,
            'sigma_e' : 1e-4,
            'sigma_p' : 1e-4,
            'sigma_fp' : 1e-1,
            'q' : np.zeros((3,3)),

            # Measurement noise standard deviation
            'sigma_od' : 5e-1,
            'sigma_fl' : 1e-1,
            'r' : np.zeros((2,2))
        }


class CbDataParam:
    def __init__(self, dataName):
        self.dataName = dataName
        self.path, self.file_ind, self.sampcycle, self.titles = self.getCbDataParam(dataName)
        self.n_reactors = len(self.file_ind)
        self.cb_fc_ec = getFcData(dataName)

    def getCbDataParam(self, dataName):
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
                file_ind = [3,5]
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
            cb_fc_ec = [cb_fc_ec[i][0:7] for i in range(len(cb_fc_ec))]
        case _:
            raise Exception('Data information of {} not given.'.format(dataName))
    return cb_fc_ec