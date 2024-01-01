import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import os

from popParam import ModParam
from popSim import simulateFlProtein, interpolateCbToSim
from run import CbData
from paramsData import CbDataParam


if __name__ == "__main__":    
    # SPECIFY DATA
    dataName = '067-3'
    cbParam = CbDataParam(dataName)
    tem = [26, 27, 29, 30, 31, 33, 35, 37]
    results_dir = "Images/{}".format(dataName)

    cbData = CbData(cbParam.path, cbParam.file_ind, cbParam.sampcycle, cbParam.n_reactors)

    param = ModParam()
    if param.mcmc:
        if param.fit_e1:
            param.gr_fp = np.random.uniform(np.array([2000, 2000, 2000, 2000, 2000, 2000, 0, 0]), np.array([6000, 6000, 6000, 6000, 6000, 6000, 200, 200]), (param.n_samples,cbParam.n_reactors)).T
        else:
            param.gr_fp = np.random.uniform(0, np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.1, 0.1]), (param.n_samples,cbParam.n_reactors)).T
            param.min_fl = np.random.uniform(0, [cbData.fl[r][0] for r in range(cbParam.n_reactors)], (param.n_samples,cbParam.n_reactors)).T

    # SIMULATE
    sim_hrs, cb_dil, fp_init, sim_puti = [], [], [], []
    sim_fl_train, sim_fl_test = [], []
    s_min = np.zeros(cbParam.n_reactors, dtype=int)
    for j in range(cbParam.n_reactors):
        sim_hrs.append(np.arange(0,cbData.time_h[j][-1]-cbData.time_h[j][0],param.Ts/3600))
        dil = np.diff(cbData.p1[j], prepend=[0,0]) > 0.015
        cb_dil.append(dil[:-1])
        sim_puti.append(interpolateCbToSim(cbData.time_h[j], cbData.od[j], sim_hrs[j]))
        if param.fit_e1:
            fp_init.append(cbData.fl[j][0]*cbData.b1[j][0] - param.e1_ofs[j] - param.od_fac*cbData.od[j][0])
        else:
            fp_init.append((cbData.fl[j][0] - param.min_fl[j])*(cbData.od[j][0]+param.od_ofs))

    if param.mcmc:
        for j in range(cbParam.n_reactors):
            print("Training at temperature: ", cbParam.titles[j])
            sim_fp = simulateFlProtein(fp_init[j], cbData.time_h[j], sim_hrs[j], cbData.temp[j], sim_puti[j], cb_dil[j], j, param).T
            if param.fit_e1:
                sim_fl_train.append(sim_fp + param.e1_ofs[j] + param.od_fac*cbData.od[j])
                rmse = np.sqrt(np.mean((sim_fl_train[j] - cbData.fl[j]*cbData.b1[j])**2,axis=1))
            else:
                sim_fl_train.append(((sim_fp/(cbData.od[j]+np.full((len(cbData.od[j]),param.n_samples),param.od_ofs).T)).T+param.min_fl[j]).T)
                rmse = np.sqrt(np.mean((sim_fl_train[j] - cbData.fl[j])**2,axis=1))
            s_min[j] = np.argmin(rmse)

        # Print obtrained parameters
        if not param.fit_e1:
            param.min_fl = np.array([param.min_fl[j][s_min[j]] for j in range(cbParam.n_reactors)])
            print("Min Fl:")
            print(*param.min_fl, sep = ", ")

        param.gr_fp = np.array([param.gr_fp[j][s_min[j]] for j in range(cbParam.n_reactors)])
        print("Growth rates:")
        print(*param.gr_fp, sep = ", ")

        # Optain, print and plot growth rate model
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
        fig.savefig(results_dir + "/gr_model{}.png".format("_e1" if param.fit_e1 else ""))

        param.Alpha_fp = np.array([gr_model2.coefficients[2]])
        param.Beta_fp = np.array([gr_model2.coefficients[1]])
        param.Gam_fp = np.array([gr_model2.coefficients[0]])

    mcmc = param.mcmc
    param.mcmc = False
    print("Running with estimated parameters")
    for j in range(cbParam.n_reactors):
        if param.fit_e1:
            fp_init_j = cbData.fl[j][0]*cbData.b1[j][0] - param.e1_ofs[j]
            sim_fp = simulateFlProtein(fp_init_j, cbData.time_h[j], sim_hrs[j], cbData.temp[j], sim_puti[j], cb_dil[j], j, param).T
            sim_fl_test.append(sim_fp + param.e1_ofs[j] + param.od_fac*cbData.od[j])
        else:
            fp_init_j = (cbData.fl[j][0] - param.min_fl[j])*(cbData.od[j][0]+param.od_ofs)
            sim_fp = simulateFlProtein(fp_init_j, cbData.time_h[j], sim_hrs[j], cbData.temp[j], sim_puti[j], cb_dil[j], j, param).T
            sim_fl_test.append(((sim_fp/(cbData.od[j]+np.full((len(cbData.od[j]),param.n_samples),param.od_ofs).T)).T+param.min_fl[j]).T)
    param.mcmc = mcmc
    
    # ANALYSIS
    n_rows = math.ceil(cbParam.n_reactors/2)
    n_culumns = 2 if cbParam.n_reactors > 1 else 1
    matplotlib.style.use('default')
    fig, ax = plt.subplots(n_rows,2)
    fig.set_figheight(n_rows*7)
    fig.set_figwidth(n_culumns*10)
    if n_culumns == 1:
        ax = [ax]
    if n_rows == 1:
        ax = [ax]
    for j in range(cbParam.n_reactors):
        r = j//2
        c = j%2
        axr = ax[r][c].twinx()
        ax[r][c].set_zorder(2)
        axr.set_zorder(1)
        ax[r][c].patch.set_visible(False)

        axr.plot(cbData.time_h[j],cbData.temp[j],'r',lw=0.5, alpha = 0.4)
        ax[r][c].plot(cbData.time_h[j],cbData.od[j],'k',lw = 0.5, alpha = 0.4, label = 'p. putida od')
        if param.fit_e1:
            ax[r][c].plot(cbData.time_h[j],cbData.fl[j]*cbData.b1[j],'.g',markersize = 0.8, label = '$e1_{meas}$')
        else:
            ax[r][c].plot(cbData.time_h[j],cbData.fl[j],'.g',markersize = 0.8, label = '$fl_{meas}$')
        if param.mcmc:
            ax[r][c].plot(cbData.time_h[j],sim_fl_train[j][0],'k',lw = 0.5, label = '$fl_{sim, train}$', alpha = 0.1)
            for s in range(1,min(param.n_samples,20)):
                ax[r][c].plot(cbData.time_h[j],sim_fl_train[j][s],'k',lw = 0.5, alpha = 0.1)
            ax[r][c].plot(cbData.time_h[j],sim_fl_train[j][s_min[j]],'m', lw = 0.7, label = '$fl_{sim, opt}$')
        ax[r][c].plot(cbData.time_h[j],sim_fl_test[j][0],'b',lw = 1, label = '$fl_{sim, model}$')

        # ax[r][c].vlines(cbData.time_h[j][cb_dil[j]==1],-2,2,'g',lw = 0.5, alpha=0.5, label = 'p. putida dil')
        # ax[j].plot(cbData.time_h[j],cbData.od[j]*100,'--m',lw = 0.5, label = 'od')

        ax[r][c].legend(loc="upper left")
        if (c == 0):
            if param.fit_e1:
                axr.set_ylabel("Intensity")
            else:
                ax[r][c].set_ylabel("Fluorescense [a.u.]")
        else:
            axr.set_ylabel('Temperature [°C]', color='r')
            ax[r][c].tick_params(axis='y', labelleft=True)
        if r == n_rows-1:
            ax[r][c].set_xlabel("Time [h]")
        axr.tick_params(axis='y', color='r', labelcolor='r')
        ax[r][c].set_xlim([sim_hrs[j][0]-0.5,sim_hrs[j][-1]+0.5])
        # ax[r][c].set_ylim([0,0.5])
        if param.mcmc:
            ax[r][c].set_ylim([0,1.1*max(max(sim_fl_test[j][0]),max(sim_fl_train[j][s_min[j]]))])
        else:
            ax[r][c].set_ylim([0,1.1*max(sim_fl_test[j][0])])
        axr.set_ylim([25,38])
        ax[r][c].set_title(cbParam.titles[j])
    # TODO: Set titles
    fig.suptitle(dataName)
    fig.tight_layout()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    fig.savefig(results_dir + "/fl_sim{}_{}.png".format("_e1" if param.fit_e1 else "","train" if param.mcmc else "test"))