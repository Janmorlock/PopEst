import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

from popParam import ModParam, getCbDataInfo
from popSim import loadData, simulateFlProtein, interpolateCbToSim

if __name__ == "__main__":    
    # SPECIFY DATA
    dataName = '062-4'
    n_samples = 10000
    path, file_ind, sampcycle, titles = getCbDataInfo(dataName)
    dil_th = 0.54
    dil_am = 0.12
    tem = [26, 27, 29, 30, 31, 33, 35, 37]

    n_reactors = len(file_ind)
    cb_hrs, cb_od, cb_tem, cb_fl, cb_p1, cb_sp = loadData(path, file_ind, sampcycle, n_reactors)

    param = ModParam()

    param.gr_fp = np.random.uniform(0, 0.4, (n_samples,n_reactors)).T
    param.min_fl = np.random.uniform(0, [cb_fl[r][0] for r in range(n_reactors)], (n_samples,n_reactors)).T

    # SIMULATE
    s_min = np.zeros(n_reactors, dtype=int)
    sim_hrs, sim_fl, cb_dil = [], [], []
    for j in range(n_reactors):
        print("Simulating temperature: ", titles[j])
        sim_hrs.append(np.arange(0,cb_hrs[j][-1]-cb_hrs[j][0],param.Ts/3600))
        dil = np.diff(cb_p1[j], prepend=[0,0]) > 0.015
        cb_dil.append(dil[:-1])
        # sim_tem = np.full(len(sim_hrs[j]),tem[j])
        # sim_tem = interpolateCbToSim(cb_hrs[j], cb_tem[j], sim_hrs[j])
        sim_od = interpolateCbToSim(cb_hrs[j], cb_od[j], sim_hrs[j])
        # param.min_fl[j][cb_fl[j][0] - param.min_fl[j] < 0] = cb_fl[j][0]
        fl_init = (cb_fl[j][0] - param.min_fl[j])*(cb_od[j][0]+0.2)
        sim_fp = simulateFlProtein(fl_init, cb_hrs[j], sim_hrs[j], sim_od, cb_dil[j], dil_am, dil_th, j, param).T
        sim_fl.append(((sim_fp/(cb_od[j]+0.2)).T+param.min_fl[j]).T)
        rmse = np.sqrt(np.mean((sim_fl[j] - cb_fl[j])**2,axis=1))
        s_min[j] = np.argmin(rmse)
    param.gr_fp = np.array([param.gr_fp[j][s_min[j]] for j in range(n_reactors)])
    print("Growth rates:")
    print(*param.gr_fp, sep = ", ")
    print("Min Fl:")
    print(*[param.min_fl[j][s_min[j]] for j in range(n_reactors)], sep = ", ")

    gr_model3 = np.poly1d(np.polyfit(tem, param.gr_fp, 3))
    gr_model4 = np.poly1d(np.polyfit(tem, param.gr_fp, 4))
    print("Growth rate model:")
    print(*gr_model3.coefficients, sep = ", ")
    print(*gr_model4.coefficients, sep = ", ")
    x_mod = np.linspace(param.temp_l, param.temp_h, 100)

    # ANALYSIS
    fig, ax = plt.subplots()
    ax.plot(tem, param.gr_fp, 'o', label = 'data')
    ax.plot(x_mod, gr_model3(x_mod), '-', label = '3rd order poly')
    ax.plot(x_mod, gr_model4(x_mod), '-', label = '4th order poly')
    plt.legend(loc='best')
    fig.savefig("Images/062-4/gr_model.png")

    n_rows = math.ceil(n_reactors/2)
    fig, ax = plt.subplots(n_rows,2,sharey='row')
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
        ax[r][c].plot(cb_hrs[j],sim_fl[j][0],'k',lw = 0.5, label = '$(fl_{sim}-fl_{min})$ train', alpha = 0.1)
        for s in range(1,min(n_samples,100)):
            ax[r][c].plot(cb_hrs[j],sim_fl[j][s],'k',lw = 0.5, alpha = 0.1)
        ax[r][c].plot(cb_hrs[j],sim_fl[j][s_min[j]],'m', lw = 0.5, label = '$(fl_{sim}-fl_{min})$ opt')
        ax[r][c].plot(cb_hrs[j],cb_fl[j],'.g',markersize = 0.8, label = '$(fl-fl_{min})$ meas')
        # ax[r][c].plot(cb_hrs[j],cb_od[j],'g',lw = 0.5, label = 'p. putida od')
        # ax[r][c].vlines(cb_hrs[j][cb_dil[j]==1],-2,2,'g',lw = 0.5, alpha=0.5, label = 'p. putida dil')
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
        # ax[r][c].set_ylim([-0.05,1])
        axr.set_ylim([25,38])
        ax[r][c].set_title(titles[j])
    # TODO: Set titles
    fig.suptitle(dataName)
    fig.tight_layout()
    fig.savefig("Images/062-4/fl_sim_prox.png")