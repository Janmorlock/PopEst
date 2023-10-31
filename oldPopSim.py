from popParam import Param
from bactCult import BactCult, FlProtein
import growthRates as gR

import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import math


def simulateCultures(param, cb_fc_ec, cb_hrs, cb_od, cb_tem):
    """
    Simulates one E coli. and P. Putida coculture using Euler forward discretization.

    Returns a time array with the corresponding values of the temperature and simulated population sizes.
    """

    Init_e = cb_fc_ec[0]*cb_od[0]/100
    Init_p = (100-cb_fc_ec[0])*cb_od[0]/100

    sim_h = cb_hrs[-1]-cb_hrs[0]
    data_l = int(sim_h*3600/param.Ts)

    sim_hrs = np.empty(data_l)
    e_coli = BactCult("e", Init_e)
    p_puti = BactCult("p", Init_p)

    temp_arr_e = np.full(data_l+param.Lag_ind,param.T_pre_e,'float')
    temp_arr_p = np.full(data_l+param.Lag_ind,param.T_pre_p,'float')
    temp_arr_e_in = np.empty(data_l)
    temp_arr_p_in = np.empty(data_l)
    count = 0
    e_coli_arr = np.empty(data_l)
    p_puti_arr = np.empty(data_l)

    # Update
    for i in range(data_l):
        # log
        sim_hrs[i] = i*param.Ts/3600
        e_coli_arr[i] = e_coli.pop
        p_puti_arr[i] = p_puti.pop

        pop_e = e_coli.pop
        pop_p = p_puti.pop

        # Dilute if combined OD above threshold
        if (pop_e + pop_p > param.Dil_th):
            e_coli.dilute(pop_p)
            p_puti.dilute(pop_e)
            
        # Let them grow
        i_lag = i+param.Lag_ind
        if cb_hrs[count] <= sim_hrs[i]:
            count += 1
        temp_arr_e[i_lag] = cb_tem[count-1]
        temp_arr_p[i_lag] = cb_tem[count-1]
        if (param.Avg_temp and param.Lag):
            temp_arr_e_in[i] = np.average(temp_arr_e[i+1:i+param.Lag_ind+1])
            temp_arr_p_in[i] = np.average(temp_arr_p[i+1:i+param.Lag_ind+1])
        else:
            temp_arr_e_in[i] = temp_arr_e[i]
            temp_arr_p_in[i] = temp_arr_p[i]
        e_coli.grow(temp_arr_e_in[i])
        p_puti.grow(temp_arr_p_in[i])
    
    return temp_arr_e_in, sim_hrs, e_coli_arr, p_puti_arr


def interpolateCbToSim(cb_hrs, cb_data, sim_hrs):
    count = 0
    sim_data = np.empty_like(sim_hrs)
    for i in range(len(sim_hrs)):
        if cb_hrs[count] <= sim_hrs[i]:
            count += 1
        sim_data[i] = cb_data[count-1]
    return sim_data


def simulateFlProtein(flp_init, p_puti, temp, dil):
    fl_arr = np.empty_like(p_puti)
    fl_p = FlProtein(flp_init)
    for i in range(len(p_puti)):
        fl_arr[i] = fl_p.count
        fl_p.produce(p_puti[i],temp[i])
        if dil[i]:
            fl_p.dilute()
    return fl_arr


if __name__ == "__main__":

    # LOAD DATA
    # TODO: Add path to chiBio data directory
    path = '../../Ting/Experiments/064-1'
    # TODO: Add indeces of files of interest
    file_ind = [3, 5]
    # TODO: Add sampcycle indices as in Matlab
    sampcycle = np.array([[1004, 1140, 1262, 1376, 1505, 1637, 1760, 2487],
                          [2, 138, 260, 374, 502, 636, 759, 1486]])
    sampcycle -= 1 # adapts indeces to python
    # TODO: Add excel with flow cytometry data
    FC_file = pd.read_excel('../../Ting/Experiments/064-1/231016 Facility Analysis Manual Count.xlsx',header=[1])
    FC_data = FC_file['% Parent.1'] + FC_file['% Parent.2']
    # TODO: Add indeces
    cb_fc_ec = np.array([FC_data[4::4].to_numpy(),
                         FC_data[5::4].to_numpy()])
    # TODO: Adapt min/max fluorescense values of the respective reactors
    min_fl = [0.061, 0.059]
    max_fl = [0.285, 0.290]

    cb_files = sorted(glob.glob(path + "/*.csv"))
    cb_dfs = []
    for i in file_ind:
        df = pd.read_csv(cb_files[i], index_col=None, header=0)
        cb_dfs.append(df)
    n_reactors = len(file_ind)

    cb_hrs, cb_od, cb_tem, cb_fl = [], [], [], []
    for i in range(n_reactors):
        cb_hrs.append(cb_dfs[i]["exp_time"][sampcycle[i][0]:sampcycle[i][-1]+1].to_numpy()/3600)
        cb_od.append(cb_dfs[i]["od_measured"][sampcycle[i][0]:sampcycle[i][-1]+1].to_numpy())
        cb_tem.append(cb_dfs[i]["media_temp"][sampcycle[i][0]:sampcycle[i][-1]+1].to_numpy())
        cb_fl.append(cb_dfs[i]["FP1_emit1"][sampcycle[i][0]:sampcycle[i][-1]+1].to_numpy())
    cb_hrs = [cb_hrs[i]-cb_hrs[i][0] for i in range(n_reactors)]

    param = Param()

    temp_arr_e_in_all, time_all, e_coli_all, p_puti_all = [], [], [], []
    fl_p_all = []
    # SIMULATION
    for j in range(n_reactors):
        # TODO: include reactor fluorescense offsets
        temp_arr_e_in, sim_hrs, e_coli_arr, p_puti_arr = simulateCultures(param, cb_fc_ec[j], cb_hrs[j], cb_od[j], cb_tem[j])
        temp_arr_e_in_all.append(temp_arr_e_in)
        time_all.append(sim_hrs)
        e_coli_all.append(e_coli_arr)
        p_puti_all.append(p_puti_arr)



    gR.plotGrowthRates()
    critTemp = gR.getCritTemp()
    assert(26 < critTemp[0] and critTemp[0] < 37)

    # ANALYSIS
    n_rows = math.ceil(n_reactors/2)
    fig, ax = plt.subplots(n_rows,2,sharey='all')
    fig.set_figheight(n_rows*7)
    fig.set_figwidth(20)
    for j in range(n_reactors):
        od = e_coli_all[j] + p_puti_all[j]
        e_coli_percent = e_coli_all[j]/od*100
        p_puti_percent = p_puti_all[j]/od*100

        axr = ax[j].twinx()
        ax[j].set_zorder(2)
        axr.set_zorder(1)
        ax[j].patch.set_visible(False)

        axr.plot(cb_hrs[j],cb_tem[j],'--r',lw=0.5,alpha=0.4)
        axr.plot(time_all[j],temp_arr_e_in_all[j],'r',lw=1)
        axr.hlines(critTemp[0],time_all[j][0]-1,time_all[j][-1]+1,'r',lw=0.5)
        ax[j].plot(time_all[j],e_coli_percent, 'b', label = 'e coli. sim')
        ax[j].plot(time_all[j],p_puti_percent, 'g', label = 'p. putida sim')
        ax[j].plot(cb_hrs[j][sampcycle[j]-sampcycle[j][0]],cb_fc_ec[j], 'b--x', label = 'e coli. fc')
        ax[j].plot(cb_hrs[j][sampcycle[j]-sampcycle[j][0]],100-cb_fc_ec[j], 'g--x', label = 'p. putida fc')
        ax[j].plot(cb_hrs[j],(cb_fl[j]-min_fl[j])/(max_fl[j]-min_fl[j])*100,'.k',markersize = 0.8, label = '% max fluorescense')
        # ax[j].plot(time_all[j],od*100,'-k',lw = 0.5, label = 'od sim')
        # ax[j].plot(cb_hrs[j],cb_od[j]*100,'--m',lw = 0.5, label = 'od')

        ax[j].legend(loc="lower left")
        if (j%2 == 0):
            ax[j].set_ylabel("Relative composition [%]")
        else:
            axr.set_ylabel('Temperature [°C]', color='r')
            ax[j].tick_params(axis='y', labelleft=True)
        axr.set_yticks(np.append(axr.get_yticks(), critTemp[0]))
        axr.tick_params(axis='y', color='r', labelcolor='r')
        axr.text(1, 1, 'e coli. prefered',
                horizontalalignment='right',
                verticalalignment='top',
                transform=axr.transAxes,
                color='w',
                bbox={'facecolor': 'red', 'alpha': 1, 'pad': 0, 'edgecolor': 'r'})
        axr.text(1, 0, 'p. putida prefered',
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=axr.transAxes,
                color='w',
                bbox={'facecolor': 'red', 'alpha': 1, 'pad': 0, 'edgecolor': 'r'})
        ax[j].set_xlabel("Time [h]")
        ax[j].set_xlim([time_all[j][0]-0.5,time_all[j][-1]+0.5])
        ax[j].set_ylim([-5,105])
    # TODO: Set titles
    ax[0].set_title("C8M2")
    ax[1].set_title("C8M3")
    fig.suptitle("064-1")
    fig.tight_layout()
    fig.savefig("Images/064-1_lag_3h_avg_fl_tem.png")
    # fig.savefig("Images/064-1_fl.png")
    # plt.show()