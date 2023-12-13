import numpy as np
import math
import matplotlib.pyplot as plt
import os

from popParam import ModParam
from popSim import simulateFlProtein, interpolateCbToSim
from run import CbData
from paramsData import CbDataParam


if __name__ == "__main__":    
    # SPECIFY DATA
    dataName = '062-4'
    cbParam = CbDataParam(dataName)
    tem = [26, 27, 29, 30, 31, 33, 35, 37]
    results_dir = "Images/{}".format(dataName)

    cbData = CbData(cbParam.path, cbParam.file_ind, cbParam.sampcycle, cbParam.n_reactors)

    param = ModParam()
    if param.mcmc:
        param.gr_fp = np.random.uniform(0, 0.35, (param.n_samples,cbParam.n_reactors)).T
        param.min_fl = np.random.uniform(0, [cbData.fl[r][0] for r in range(cbParam.n_reactors)], (param.n_samples,cbParam.n_reactors)).T
        # param.od_ofs = np.random.uniform(0.05, 0.3, (n_samples))

    # SIMULATE
    sim_hrs, cb_dil, fl_init, sim_puti = [], [], [], []
    sim_fl_train, sim_fl_test = [], []
    s_min = np.zeros(cbParam.n_reactors, dtype=int)
    for j in range(cbParam.n_reactors):
        sim_hrs.append(np.arange(0,cbData.time_h[j][-1]-cbData.time_h[j][0],param.Ts/3600))
        dil = np.diff(cbData.p1[j], prepend=[0,0]) > 0.015
        cb_dil.append(dil[:-1])
        sim_puti.append(interpolateCbToSim(cbData.time_h[j], cbData.od[j], sim_hrs[j]))
        fl_init.append((cbData.fl[j][0] - param.min_fl[j])*(cbData.od[j][0]+param.od_ofs))

    if param.mcmc:
        for j in range(cbParam.n_reactors):
            print("Training at temperature: ", cbParam.titles[j])
            sim_fp = simulateFlProtein(fl_init[j], cbData.time_h[j], sim_hrs[j], cbData.temp[j], sim_puti[j], cb_dil[j], j, param).T
            sim_fl_train.append(((sim_fp/(cbData.od[j]+np.full((len(cbData.od[j]),param.n_samples),param.od_ofs).T)).T+param.min_fl[j]).T)
            rmse = np.sqrt(np.mean((sim_fl_train[j] - cbData.fl[j])**2,axis=1))
            s_min[j] = np.argmin(rmse)

        param.gr_fp = np.array([param.gr_fp[j][s_min[j]] for j in range(cbParam.n_reactors)])
        param.min_fl = np.array([param.min_fl[j][s_min[j]] for j in range(cbParam.n_reactors)])
        print("Growth rates:")
        print(*param.gr_fp, sep = ", ")
        print("Min Fl:")
        print(*param.min_fl, sep = ", ")
        # print("od_ofs:")
        # print(*[param.od_ofs[s_min[j]] for j in range(n_reactors)], sep = ", ")

        gr_model2 = np.poly1d(np.polyfit(tem[:-1], param.gr_fp[:-1], 2))
        gr_model2_all = np.poly1d(np.polyfit(tem, param.gr_fp, 2))
        print("Growth rate model:")
        print(*gr_model2.coefficients, sep = ", ")
        x_mod = np.linspace(param.temp_l, param.temp_h, 110)

        y_all = gr_model2_all(x_mod)
        y_all[y_all < 0] = 0
        y = gr_model2(x_mod)
        y[y < 0] = 0
        fig, ax = plt.subplots()
        fig.set_figheight(7)
        fig.set_figwidth(10)
        ax.plot(tem, param.gr_fp, 'om', label = 'data')
        ax.plot(x_mod, y, '-b', label = '2nd order poly')
        ax.plot(x_mod, y_all, '-k', label = '2nd order poly all')
        ax.set_xlabel('Temperature [°C]')
        ax.set_ylabel('Growth rate [1/h]')
        ax.legend(loc='best')
        ax.set_title("Fluorescent protein growth rate model")
        fig.savefig(results_dir + "/gr_model.png")

        param.Alpha_fp = np.array([gr_model2.coefficients[2]])
        param.Beta_fp = np.array([gr_model2.coefficients[1]])
        param.Gam_fp = np.array([gr_model2.coefficients[0]])

    mcmc = param.mcmc
    param.mcmc = False
    print("Running with estimated parameters")
    for j in range(cbParam.n_reactors):
        fl_init_j = (cbData.fl[j][0] - param.min_fl[j])*(cbData.od[j][0]+param.od_ofs)
        sim_fp = simulateFlProtein(fl_init_j, cbData.time_h[j], sim_hrs[j], cbData.temp[j], sim_puti[j], cb_dil[j], j, param).T
        sim_fl_test.append(((sim_fp/(cbData.od[j]+np.full((len(cbData.od[j]),param.n_samples),param.od_ofs).T)).T+param.min_fl[j]).T)
    param.mcmc = mcmc
    
    # ANALYSIS
    n_rows = math.ceil(cbParam.n_reactors/2)
    fig, ax = plt.subplots(n_rows,2,sharey='row')
    fig.set_figheight(n_rows*7)
    fig.set_figwidth(20)
    for j in range(cbParam.n_reactors):
        r = j//2
        c = j%2
        axr = ax[r][c].twinx()
        ax[r][c].set_zorder(2)
        axr.set_zorder(1)
        ax[r][c].patch.set_visible(False)

        axr.plot(cbData.time_h[j],cbData.temp[j],'r',lw=0.5)
        # ax[r][c].plot(cbData.time_h[j],cbData.od[j],'k',lw = 0.5, label = 'p. putida od')
        ax[r][c].plot(cbData.time_h[j],cbData.fl[j],'.g',markersize = 0.8, label = '$fl_{meas}$')
        if param.mcmc:
            ax[r][c].plot(cbData.time_h[j],sim_fl_train[j][0],'k',lw = 0.5, label = '$fl_{sim, train}$', alpha = 0.1)
            for s in range(1,min(param.n_samples,100)):
                ax[r][c].plot(cbData.time_h[j],sim_fl_train[j][s],'k',lw = 0.5, alpha = 0.1)
            ax[r][c].plot(cbData.time_h[j],sim_fl_train[j][s_min[j]],'m', lw = 0.7, label = '$fl_{sim, opt}$')
        ax[r][c].plot(cbData.time_h[j],sim_fl_test[j][0],'b',lw = 1, label = '$fl_{sim, model}$')

        # ax[r][c].vlines(cbData.time_h[j][cb_dil[j]==1],-2,2,'g',lw = 0.5, alpha=0.5, label = 'p. putida dil')
        # ax[j].plot(cbData.time_h[j],cbData.od[j]*100,'--m',lw = 0.5, label = 'od')

        ax[r][c].legend(loc="upper left")
        if (c == 0):
            ax[r][c].set_ylabel("Fluorescense [a.u.]")
        else:
            axr.set_ylabel('Temperature [°C]', color='r')
            ax[r][c].tick_params(axis='y', labelleft=True)
        if r == n_rows-1:
            ax[r][c].set_xlabel("Time [h]")
        axr.tick_params(axis='y', color='r', labelcolor='r')
        ax[r][c].set_xlim([sim_hrs[j][0]-0.5,sim_hrs[j][-1]+0.5])
        ax[r][c].set_ylim([0,0.5])
        axr.set_ylim([25,38])
        ax[r][c].set_title(cbParam.titles[j])
    # TODO: Set titles
    fig.suptitle(dataName)
    fig.tight_layout()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    fig.savefig(results_dir + "/fl_sim_{}.png".format("train" if param.mcmc else "test"))