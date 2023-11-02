import numpy as np
import math
import matplotlib.pyplot as plt

from popParam import ModParam, getCbDataInfo
from popSim import loadData, simulateFlProtein, interpolateCbToSim
from growthRates import plotGrowthRates

if __name__ == "__main__":    
    # SPECIFY DATA
    dataName = '062-4'
    path, file_ind, sampcycle, titles = getCbDataInfo(dataName)
    dil_th = 0.54
    dil_am = 0.1
    # tem = [26, 27, 29, 30, 31, 33, 35, 37]

    n_reactors = len(file_ind)
    cb_hrs, cb_od, cb_tem, cb_fl, cb_p1 = loadData(path, file_ind, sampcycle, n_reactors)

    param = ModParam()

    # SIMULATE
    sim_hrs, sim_fl, sim_dil = [], [], []
    for j in range(n_reactors):
        sim_hrs.append(np.arange(0,cb_hrs[j][-1]-cb_hrs[j][0],param.Ts/3600))
        fl_init = (cb_fl[j][0] - param.min_fl[file_ind[j]])*cb_od[j][0]
        cb_dil = (np.diff(cb_p1[j], prepend=[0,0,0]) > 0.015)
        # sim_tem = np.full(len(sim_hrs[j]),tem[j])
        sim_tem = interpolateCbToSim(cb_hrs[j], cb_tem[j], sim_hrs[j])
        sim_od = interpolateCbToSim(cb_hrs[j], cb_od[j], sim_hrs[j])
        sim_dil.append(interpolateCbToSim(cb_hrs[j], cb_dil, sim_hrs[j],'zero'))
        sim_fl.append(simulateFlProtein(fl_init, sim_od, sim_tem, sim_dil[j], dil_am, dil_th)/sim_od)


    # ANALYSIS
    # plotGrowthRates()

    n_rows = math.ceil(n_reactors/2)
    fig, ax = plt.subplots(n_rows,2,sharey='all')
    fig.set_figheight(n_rows*7)
    fig.set_figwidth(20)
    for j in range(n_reactors):
        r = j//2
        c = j%2
        axr = ax[r][c].twinx()
        ax[r][c].set_zorder(2)
        axr.set_zorder(1)
        ax[r][c].patch.set_visible(False)

        axr.plot(cb_hrs[j],cb_tem[j],'r',lw=1)
        ax[r][c].plot(cb_hrs[j],cb_fl[j],'.k',markersize = 0.8, label = '$(fl-fl_{min})$')
        ax[r][c].plot(sim_hrs[j],sim_fl[j]+param.min_fl[file_ind[j]],'m',lw = 0.5, label = '$(fl_{sim}-fl_{min})$')
        ax[r][c].plot(cb_hrs[j],cb_od[j],'g',lw = 0.5, label = 'p. putida od')
        ax[r][c].vlines(sim_hrs[j][sim_dil[j]==1],-2,2,'g',lw = 0.5, alpha=0.5, label = 'p. putida dil')
        # ax[j].plot(cb_hrs[j],cb_od[j]*100,'--m',lw = 0.5, label = 'od')

        ax[r][c].legend(loc="upper left")
        if (c == 0):
            ax[r][c].set_ylabel("Relative composition [%]")
        else:
            axr.set_ylabel('Temperature [°C]', color='r')
            ax[r][c].tick_params(axis='y', labelleft=True)
        if r == n_rows-1:
            ax[r][c].set_xlabel("Time [h]")
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
        ax[r][c].set_xlim([sim_hrs[j][0]-0.5,sim_hrs[j][-1]+0.5])
        ax[r][c].set_ylim([-0.05,1])
        axr.set_ylim([25,38])
        ax[0][0].set_title(titles[j])
    # TODO: Set titles
    fig.suptitle(dataName)
    fig.tight_layout()
    fig.savefig("Images/062-4_fl_sim_prox.png")