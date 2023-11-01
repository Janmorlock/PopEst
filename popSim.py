from popParam import ModParam, getCbDataInfo, getFcData
from bactCult import BactCult, FlProtein
import growthRates as gR

import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as ss

def interpolateCbToSim(cb_hrs, cb_data, sim_hrs, method='hold'):
    count = 0
    sim_data = np.empty_like(sim_hrs)
    if method == 'zero':
        sim_data = np.zeros(len(sim_hrs),dtype=int)
    for i in range(len(sim_hrs)):
        if cb_hrs[count] <= sim_hrs[i]:
            count += 1
            sim_data[i] = cb_data[count-1]
        else:
            if method == 'hold':
                sim_data[i] = cb_data[count-1]
            elif method == 'zero':
                sim_data[i] = 0
    return sim_data

def simulateCultures(e_init, p_init, sim_tem_e, sim_tem_p):
    """
    Simulates one E coli. and P. Putida coculture using Euler forward discretization.

    Returns a time array with the corresponding values of the temperature and simulated population sizes.
    """
    e_coli = BactCult("e", e_init)
    p_puti = BactCult("p", p_init)

    data_l = len(sim_tem_e)

    sim_pop_e = np.empty(data_l)
    sim_pop_p = np.empty(data_l)
    sim_dil = np.empty(data_l)

    # Update
    for i in range(data_l):
        # log
        sim_pop_e[i] = e_coli.pop
        sim_pop_p[i] = p_puti.pop

        pop_e = e_coli.pop
        pop_p = p_puti.pop
        dil = 0

        # Dilute if combined OD above threshold
        if (pop_e + pop_p > param.Dil_th):
            e_coli.dilute(pop_p)
            p_puti.dilute(pop_e)
            dil = 1

        sim_dil[i] = dil
            
        # Let them grow
        e_coli.grow(sim_tem_e[i])
        p_puti.grow(sim_tem_p[i])
    
    return sim_pop_e, sim_pop_p, sim_dil


def simulateFlProtein(flp_init, p_puti, temp, dil, dil_am ,dil_th):
    fl_arr = np.empty_like(p_puti)
    fl_p = FlProtein(flp_init)
    for i in range(len(p_puti)):
        fl_arr[i] = fl_p.count
        fl_p.produce(p_puti[i],temp[i])
        if dil[i]:
            fl_p.dilute(dil_am, dil_th)
    return fl_arr


def loadData(path, file_ind, scope, n_reactors):
    cb_files = sorted(glob.glob(path + "/*.csv"))
    cb_dfs = []
    for i in file_ind:
        df = pd.read_csv(cb_files[i], index_col=None, header=0)
        cb_dfs.append(df)

    cb_hrs, cb_od, cb_tem, cb_fl, cb_p1 = [], [], [], [], []
    for i in range(n_reactors):
        cb_hrs.append(cb_dfs[i]["exp_time"][scope[i][0]:scope[i][-1]+1].to_numpy()/3600)
        cb_od_temp = cb_dfs[i]["od_measured"][scope[i][0]:scope[i][-1]+1].to_numpy()
        cb_od_temp[cb_od_temp < 0.005] = 0.005
        cb_od.append(cb_od_temp)
        cb_tem.append(cb_dfs[i]["media_temp"][scope[i][0]:scope[i][-1]+1].to_numpy())
        cb_fl.append(cb_dfs[i]["FP1_emit1"][scope[i][0]:scope[i][-1]+1].to_numpy())
        cb_p1.append(cb_dfs[i]["pump_1_rate"][scope[i][0]:scope[i][-1]+1].to_numpy())
    cb_hrs = [cb_hrs[i]-cb_hrs[i][0] for i in range(n_reactors)]
    return cb_hrs, cb_od, cb_tem, cb_fl, cb_p1


if __name__ == "__main__":

    # SPECIFY DATA
    dataName = '064-2'
    path, file_ind, sampcycle = getCbDataInfo(dataName)
    cb_fc_ec = getFcData(dataName)
    # min_fl = [0.061, 0.059]

    n_reactors = len(file_ind)
    cb_hrs, cb_od, cb_tem, cb_fl, cb_p1 = loadData(path, file_ind, sampcycle, n_reactors)

    param = ModParam()

    sim_tem_e, sim_tem_p, sim_hrs, e_coli_all, p_puti_all = [], [], [], [], []
    fl_p_all = []
    # SIMULATION
    for j in range(n_reactors):
        # Collect inputs
        sim_hrs.append(np.arange(0,cb_hrs[j][-1]-cb_hrs[j][0],param.Ts/3600))
        sim_tem = interpolateCbToSim(cb_hrs[j], cb_tem[j], sim_hrs[j])
        e_init = cb_fc_ec[j][0]*cb_od[j][0]/100
        p_init = (100-cb_fc_ec[j][0])*cb_od[j][0]/100
        if (param.Lag):
            tem_lag_e = np.concatenate((np.full(param.Lag_ind,param.T_pre_e),sim_tem))
            tem_lag_p = np.concatenate((np.full(param.Lag_ind,param.T_pre_p),sim_tem))
            if param.Avg_temp:
                sim_tem_e.append(ss.convolve(tem_lag_e,np.full(param.Lag_ind+1,1/(param.Lag_ind+1)),mode='valid'))
                sim_tem_p.append(ss.convolve(tem_lag_p,np.full(param.Lag_ind+1,1/(param.Lag_ind+1)),mode='valid'))
            else:
                sim_tem_e.append(tem_lag_e[:len(sim_hrs[j])])
                sim_tem_p.append(tem_lag_p[:len(sim_hrs[j])])
        else:
            sim_tem_e.append(sim_tem)
            sim_tem_p.append(sim_tem)
        # Simulate cultures
        sim_pop_e, sim_pop_p, sim_dil = simulateCultures(e_init, p_init, sim_tem_e[j], sim_tem_p[j])
        e_coli_all.append(sim_pop_e)
        p_puti_all.append(sim_pop_p)
        fl_init = (cb_fl[j][0] - param.min_fl[file_ind[j]])*cb_od[j][0]
        sim_od = interpolateCbToSim(cb_hrs[j], cb_od[j], sim_hrs[j])
        fl_p_all.append(simulateFlProtein(fl_init, sim_pop_p, sim_tem, sim_dil, param.Dil_amount, param.Dil_th)/sim_od)


    # gR.plotGrowthRates()
    critTemp = gR.getCritTemp()
    assert(26 < critTemp[0] and critTemp[0] < 37)

    # ANALYSIS
    n_rows = math.ceil(n_reactors/2)
    fig, ax = plt.subplots(n_rows,2,sharey='all')
    fig.set_figheight(n_rows*7)
    fig.set_figwidth(20)
    for j in range(n_reactors):
        r = j//2
        c = j%2
        od = e_coli_all[j] + p_puti_all[j]
        e_coli_percent = e_coli_all[j]/od*100
        p_puti_percent = p_puti_all[j]/od*100

        axr = ax[r][c].twinx()
        ax[r][c].set_zorder(2)
        axr.set_zorder(1)
        ax[r][c].patch.set_visible(False)

        axr.plot(cb_hrs[j],cb_tem[j],'--r',lw=0.5,alpha=0.4)
        axr.plot(sim_hrs[j],sim_tem_e[j],'r',lw=1)
        axr.hlines(critTemp[0],sim_hrs[j][0]-1,sim_hrs[j][-1]+1,'r',lw=0.5)
        ax[r][c].plot(sim_hrs[j],e_coli_percent, 'b', label = 'e coli. sim')
        ax[r][c].plot(sim_hrs[j],p_puti_percent, 'g', label = 'p. putida sim')
        ax[r][c].plot(cb_hrs[j][sampcycle[j]-sampcycle[j][0]],cb_fc_ec[j], 'b--x', label = 'e coli. fc')
        ax[r][c].plot(cb_hrs[j][sampcycle[j]-sampcycle[j][0]],100-cb_fc_ec[j], 'g--x', label = 'p. putida fc')
        ax[r][c].plot(cb_hrs[j],cb_fl[j]*100,'.k',markersize = 0.8, label = '$100*fl$')
        ax[r][c].plot(sim_hrs[j],(fl_p_all[j]+param.min_fl[file_ind[j]])*100,'m',lw = 0.5, label = '$100*fl_{sim}$')
        # ax[r][c].plot(time_all[j],od*100,'-k',lw = 0.5, label = 'od sim')
        # ax[r][c].plot(cb_hrs[j],cb_od[j]*100,'--m',lw = 0.5, label = 'od')

        ax[r][c].legend(loc="upper left")
        if (j%2 == 0):
            ax[r][c].set_ylabel("Relative composition [%]")
        else:
            axr.set_ylabel('Temperature [°C]', color='r')
            ax[r][c].tick_params(axis='y', labelleft=True)
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
        ax[r][c].set_xlabel("Time [h]")
        ax[r][c].set_xlim([sim_hrs[j][0]-0.5,sim_hrs[j][-1]+0.5])
        ax[r][c].set_ylim([-5,105])
    # TODO: Set titles
    ax[0][0].set_title("C8M0")
    ax[0][1].set_title("C8M1")
    fig.suptitle("064-2")
    fig.tight_layout()
    fig.savefig("Images/064-2_lag_3_h_avg_fl.png")
    # fig.savefig("Images/064-1_fl.png")
    # plt.show()