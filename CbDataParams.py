import numpy as np
import pandas as pd

class CbDataParam:
    def __init__(self, dataName):
        self.dataName = dataName
        self.path, self.file_ind, self.sampcycle, self.titles, self.reactors = self.getCbDataParam(dataName)
        self.n_reactors = len(self.file_ind)
        self.cb_fc_ec = getFcData(dataName)

    def getCbDataParam(self, dataName):
        reactors = []
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
                sampcycle = [sampcycle[i] for i in range(len(file_ind))] # adapts indeces to python
                titles = ['C8M0','C8M1','C8M2','C8M3','C9M0','C9M1','C9M2','C9M3']
            case '064-2-test':
                path = '../../Ting/Experiments/064-2'
                # Indeces of files of interest
                file_ind = [3,5]
                # Sampcycle indices as in Matlab
                sampcycle = [np.array([946, 1032, 1145, 1259, 1368, 1485, 2303, 2426, 2543, 2668]),
                            np.array([946, 1032, 1145, 1258, 1367, 1483, 2301, 2424, 2540, 2665])]
                sampcycle = [sampcycle[i] for i in range(len(file_ind))] # adapts indeces to python
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
            case '051-1':
                path = '../../Ting/Experiments/051-1'
                # Indeces of files of interest
                file_ind = [2,3,4,5,0,1]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = np.array([[0,1000] for i in range(n_reactors)])
                titles = ['C7M0','C7M1','C7M2','C7M3','C9M2','C9M3']
            case '067-2':
                path = '../../Ting/Experiments/067-2/CB data'
                # Indeces of files of interest
                file_ind = [0,1,2,3,4,5,6,7]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = np.array([[0,1800] for i in range(n_reactors)])
                titles = ['C8M0','C8M1','C8M2','C8M3','C9M0','C9M1','C9M2','C9M3']
            case '067-3':
                path = '../../Ting/Experiments/067-3'
                # Indeces of files of interest
                file_ind = [0,1,2,3]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = np.array([[0,1800] for i in range(n_reactors)])
                titles = ['C9M0','C9M1','C9M2','C9M3']
            case '067-4':
                path = '../../Ting/Experiments/067-4'
                # Indeces of files of interest
                file_ind = [0,1,2,3]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = np.array([[0,1800] for i in range(n_reactors)])
                titles = ['C8M0','C8M1','C6M0','C6M1']
            case '069-2':
                path = '../../Ting/Experiments/069-2'
                # Indeces of files of interest
                file_ind = [0,1,2,3,8,9,10,11,4,5,6,7]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = np.array([[0,1800] for i in range(n_reactors)])
                titles = ['C7M0','C7M1','C7M2','C7M3','C8M0','C8M1','C8M2','C8M3','C9M0','C9M1','C9M2','C9M3']
                reactors = ['Faith', 'Alabama', 'George', 'Tammy', 'Taylor', 'Garth', 'Willi', 'Reba', 'Shania', 'Johnny', 'Dolly', 'Carrie']
            case '069-3':
                path = '../../Ting/Experiments/069-3'
                # Indeces of files of interest
                file_ind = [0,1,2,3]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = np.array([[0,1800] for i in range(n_reactors)])
                titles = ['C6M0','C6M1','C6M2','C6M3']
                reactors = ['Allan', 'Arthur', 'Morlock', 'Swift']
            case '069-4':
                path = '../../Ting/Experiments/069-4'
                # Indeces of files of interest
                file_ind = [0,1,2,3,4]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = np.array([[0,1800] for i in range(n_reactors)])
                titles = ['C6M0','C6M1','C6M2','C6M3','C6M4']
                reactors = ['Allan', 'Arthur', 'Morlock', 'Swift', 'Stacey']
            case '069-5':
                path = '../../Ting/Experiments/temp 069-5'
                # Indeces of files of interest
                file_ind = [0,1,3]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = np.array([[0,1800] for i in range(n_reactors)])
                titles = ['C6M0','C6M1','C6M3']
                reactors = ['Allan', 'Arthur', 'Swift']
            case '073-1':
                path = '../../Ting/Experiments/073-1'
                # Indeces of files of interest
                file_ind = [0,1,2,3,4]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = np.array([[0,1800] for i in range(n_reactors)])
                titles = ['C6M0','C6M1','C6M2','C6M3','C6M4']
                reactors = ['Allan', 'Arthur', 'Morlock', 'Swift', 'Stacey']
            case '074-1':
                path = '../../Ting/Experiments/074-1'
                # Indeces of files of interest
                file_ind = [0,1,2,3]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = np.array([[0,1800] for i in range(n_reactors)])
                titles = ['C6M0','C6M1','C6M2','C6M3']
            case '075-1':
                path = '../../Ting/Experiments/075-1'
                # Indeces of files of interest
                file_ind = [4,5,6,7,8,10,12,14,9,11,13,15,0,1,2,3]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = np.array([[0,1800] for i in range(n_reactors)])
                sampcycle[12:15] = np.array([[0,1000] for i in range(3)])
                sampcycle[1] = np.array([0,800])
                titles = ['C7M0','C7M1','C7M2','C7M3','C8M0','C8M1','C8M2','C8M3','C9M0','C9M1','C9M2','C9M3','C6M0','C6M1','C6M2','C6M3']
                reactors = ['Faith', 'Alabama', 'George', 'Tammy', 'Taylor', 'Garth', 'Willi', 'Reba', 'Shania', 'Johnny', 'Dolly', 'Carrie', 'Allan', 'Arthur', 'Morlock', 'Swift']
            case '076-1':
                path = '../../Ting/Experiments/076-1'
                # Indeces of files of interest
                file_ind = [8,9,10,11,12]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = np.array([[0,1800] for i in range(n_reactors)])
                titles = ['C7M0','C7M1','C7M2','C7M3','C8M0','C8M1','C8M2','C8M3','C9M0','C9M1','C9M2','C9M3','C6M0','C6M1','C6M2','C6M3']
            case '081-1':
                path = '../../Ting/Experiments/081-1'
                # Indeces of files of interest
                file_ind = [4,5,6,7]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = [np.array([1, 118, 244, 352, 469, 579, 693, 1433, 1537]),
                            np.array([1, 118, 244, 352, 469, 579, 693, 1433, 1537]),
                            np.array([1, 118, 244, 351, 468, 578, 692, 1432, 1536]),
                            np.array([1, 118, 244, 351, 468, 578, 692, 1432, 1536])]
                sampcycle = [sampcycle[i] - 1 for i in range(len(file_ind))] # adapts indeces to python
                titles = ['C7M0','C7M1','C7M2','C7M3']
                reactors = ['Faith', 'Alabama', 'George', 'Tammy']
            case '081-2':
                path = '../../Ting/Experiments/081-2'
                # Indeces of files of interest
                file_ind = [[7,8,12],[4,9],[5,10],[6,11]]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = [np.array([1, 116, 216, 326, 462, 462, 576, 576, 700, 700, 1476, 1476]),
                            np.array([1, 116, 216, 352, 462, 462, 576, 576, 700, 700, 1476, 1476]),
                            np.array([1, 116, 216, 351, 462, 462, 576, 576, 700, 700, 1476, 1476]),
                            np.array([1, 116, 216, 351, 462, 462, 576, 576, 700, 700, 1476, 1476])]
                sampcycle = [sampcycle[i] - 1 for i in range(len(file_ind))] # adapts indeces to python
                titles = ['C7M0','C7M1','C7M2','C7M3']
                reactors = ['Faith', 'Allan', 'George', 'Tammy']
            case '081-3':
                path = '../../Ting/Experiments/081-3'
                # Indeces of files of interest
                file_ind = [[6,12],[7,13],[8,14],[9,15],[10,16],[11,17]]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = [np.array([1, 1, 76, 151, 197, 252, 302, 358, 412, 467, 555, 598, 648, 694, 743, 791, 849, 883, 1147, 1198, 1248, 1303]),
                            np.array([1, 1, 76, 151, 197, 252, 302, 358, 412, 467, 555, 598, 648, 694, 743, 791, 849, 883, 1147, 1198, 1248, 1303]),
                            np.array([1, 1, 76, 151, 197, 252, 302, 358, 412, 467, 555, 598, 648, 694, 743, 791, 849, 883, 1147, 1198, 1248, 1303]),
                            np.array([1, 1, 76, 151, 197, 252, 302, 358, 412, 467, 555, 598, 648, 694, 743, 791, 849, 883, 1147, 1198, 1248, 1303]),
                            np.array([1, 1, 76, 151, 197, 252, 302, 358, 412, 467, 555, 598, 648, 694, 743, 791, 849, 883, 1147, 1198, 1248, 1303]),
                            np.array([1, 1, 76, 151, 197, 252, 302, 358, 412, 467, 555, 598, 648, 694, 743, 791, 849, 883, 1147, 1198, 1248, 1303])]
                sampcycle = [sampcycle[i] - 1 for i in range(len(file_ind))] # adapts indeces to python
                titles = ['C7M0','C7M1','C7M2','C7M3','C7M4','C7M5']
                reactors = ['Faith', 'Allan', 'George', 'Tammy', 'Taylor', 'Garth']
            case '081-4':
                path = '../../Ting/Experiments/081-4'
                # Indeces of files of interest
                file_ind = [4,5,6,7]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = [np.array([1, 112, 230, 344, 459, 576, 689, 802, 916, 1028, 1149, 1266, 1396, 1545]),
                            np.array([1, 112, 230, 344, 459, 576, 689, 802, 916, 1028, 1149, 1266, 1396, 1545]),
                            np.array([1, 112, 230, 344, 459, 576, 689, 802, 916, 1028, 1149, 1266, 1396, 1545]),
                            np.array([1, 112, 230, 344, 459, 576, 689, 802, 916, 1028, 1149, 1266, 1396, 1545])]
                sampcycle = [sampcycle[i] - 1 for i in range(len(file_ind))] # adapts indeces to python
                titles = ['C7M0','C7M1','C7M2','C7M3']
                reactors = ['Faith', 'Allan', 'George', 'Tammy']
            case '083-2':
                path = '../../Ting/Experiments/083-2'
                # Indeces of files of interest
                file_ind = [4,5,6,7]
                n_reactors = len(file_ind)
                # Sampcycle indices as in Matlab
                sampcycle = [np.array([1, 1369, 3018, 4407, 5820, 7305]),
                            np.array([1, 1369, 3018, 4407, 5820, 7305]),
                            np.array([1, 1369, 3018, 4407, 5820, 7305]),
                            np.array([1, 1369, 3018, 4407, 5820, 7305])]
                sampcycle = [sampcycle[i] - 1 for i in range(len(file_ind))] # adapts indeces to python
                titles = ['C6M0','C6M1','C6M2','C6M3']
                reactors = ['Faith', 'Arthur', 'George', 'Tammy']
            case _:
                raise Exception('Data information of {} not given.'.format(dataName))
        
        file_ind = [[file_ind[j]] if type(file_ind[j]) == int else file_ind[j] for j in range(n_reactors)]
        
        return path, file_ind, sampcycle, titles, reactors

def getFcData(dataName):
    match dataName:
        case '064-1':
            FC_file = pd.read_excel('../../Ting/Experiments/064-1/231016 Facility Analysis Manual Count.xlsx',header=[1])
            FC_data = FC_file['% Parent.1'] + FC_file['% Parent.2']
            cb_fc_ec = np.array([FC_data[4::4].to_numpy(),
                                FC_data[5::4].to_numpy()])
        case '064-2':
            FC_file1 = pd.read_excel('../../Ting/Experiments/064-2/231027 Facility Analysis Manual Count.xlsx',header=[1])
            FC_file2 = pd.read_excel('../../Ting/Experiments/064-2/231103 Facility Analysis Manual Count.xlsx',header=[1])
            FC_data1 = FC_file1['% Parent.1'] + FC_file1['% Parent.2']
            FC_data2 = FC_file2['% Parent.1'] + FC_file2['% Parent.2']
            FC_data = np.append(FC_data1.to_numpy(),FC_data2.to_numpy())
            cb_fc_ec = [FC_data[4::8],
                        FC_data[5::8],
                        FC_data[6::8],
                        FC_data[7::8],
                        FC_data[8::8],
                        FC_data[9::8],
                        FC_data[10::8],
                        FC_data[11:-1:8]]
        case '064-2-test':
            FC_file1 = pd.read_excel('../../Ting/Experiments/064-2/231027 Facility Analysis Manual Count.xlsx',header=[1])
            FC_file2 = pd.read_excel('../../Ting/Experiments/064-2/231103 Facility Analysis Manual Count.xlsx',header=[1])
            FC_data1 = FC_file1['% Parent.1'] + FC_file1['% Parent.2']
            FC_data2 = FC_file2['% Parent.1'] + FC_file2['% Parent.2']
            FC_data = np.append(FC_data1.to_numpy(),FC_data2.to_numpy())
            cb_fc_ec = [FC_data[4::8],
                        FC_data[6::8]]
        case '081-1':
            FC_file = pd.read_excel('../../Ting/Experiments/081-1/240221 Facility Analysis Manual Count.xlsx',header=[1])
            FC_data = FC_file['% Parent.1'] + FC_file['% Parent.2']
            cb_fc_ec = np.array([FC_data[4:-4:4].to_numpy(),
                                FC_data[5:-4:4].to_numpy(),
                                FC_data[6:-4:4].to_numpy(),
                                FC_data[7:-4:4].to_numpy()])
        case '081-2':
            FC_file = pd.read_excel('../../Ting/Experiments/081-2/240227 Facility Analysis Manual Count.xlsx',header=[1])
            FC_data = FC_file['% Parent.1'] + FC_file['% Parent.2']
            cb_fc_ec = np.array([FC_data[4::4].to_numpy(),
                                FC_data[5::4].to_numpy(),
                                FC_data[6::4].to_numpy(),
                                FC_data[7::4].to_numpy()])
        case '081-3':
            FC_file = pd.read_excel('../../Ting/Experiments/081-3/240304 Facility Analysis Manual Count.xlsx',header=[1])
            FC_data = FC_file['% Parent.1'] + FC_file['% Parent.2']
            cb_fc_ec = np.array([np.array(list(FC_data[6:-30:6])+list(FC_data[-24::6])),
                                np.array(list(FC_data[7:-30:6])+list(FC_data[-23::6])),
                                np.array(list(FC_data[8:-30:6])+list(FC_data[-22::6])),
                                np.array(list(FC_data[9:-30:6])+list(FC_data[-21::6])),
                                np.array(list(FC_data[10:-30:6])+list(FC_data[-20::6])),
                                np.array(list(FC_data[11:-30:6])+list(FC_data[-19::6]))])
        case '081-4':
            FC_file = pd.read_excel('../../Ting/Experiments/081-4/240311 Facility Analysis Manual Count.xlsx',header=[1])
            FC_data = FC_file['% Parent.1'] + FC_file['% Parent.2']
            cb_fc_ec = np.array([FC_data[4::4].to_numpy(),
                                FC_data[5::4].to_numpy(),
                                FC_data[6::4].to_numpy(),
                                FC_data[7::4].to_numpy()])
        case '083-2':
            FC_file = pd.read_excel('../../Ting/Experiments/083-2/240319 Facility Analysis Manual Count.xlsx',header=[1])
            FC_data = FC_file['% Parent.1'] + FC_file['% Parent.2']
            cb_fc_ec = np.array([FC_data[8::8].to_numpy(),
                                FC_data[9::8].to_numpy(),
                                FC_data[10::8].to_numpy(),
                                FC_data[11::8].to_numpy()])
        case _:
            cb_fc_ec = np.array([[]])
            print('cb_fc_ec information of {} not given.'.format(dataName))
    return cb_fc_ec
