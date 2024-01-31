import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import os

from config.params import Params
from CbData import CbData
from paramsData import CbDataParam

def simulateFlProtein(cb_hrs, fp_init, p_puti, dil, temp, parameters, temp_arr = np.empty, gr = np.empty, train = False, e = np.empty(1), err = []):
    data_l = len(p_puti)
    n_s = gr.shape[1] if train else 1
    x = np.zeros((data_l,n_s))
    x_curr = fp_init
    k = 0
    dil_th = 0.54
    dil_am = 0.11
    temp_ind = 0

    for k in range(len(p_puti)):
        if train:
            if temp[k] == 0:
                temp[k] = temp[k-1]
            temp_ind = np.where(temp[k] == temp_arr)[0][0] 
        if dil[k]:
            x_curr -= dil_am/dil_th*x_curr
        # Euler forward (not significantly less accurate than Runge - Kutta 4th order)
        dt = cb_hrs[k] - cb_hrs[max(0,k-1)]
        if train:
            x_curr = x_curr + dt*gr[temp_ind]*p_puti[k]
        else:
            gr = 0 # TODO: Add growth rate model
            x_curr = x_curr + dt*parameters['gr_fp'][1]*temp[k]*p_puti[k]
        x[k] = x_curr
        if train:
            err[temp_ind].append(x_curr-e[k])
            
    return x

if __name__ == "__main__":    
    # SPECIFY DATA
    dataName = '075-1'
    cbParam = CbDataParam(dataName)
    cbParam.n_reactors = 8
    cbParam.file_ind = [4,5,6,7,8,10,12,14]
    # cbParam.sampcycle = np.array([[0,600] for i in range(cbParam.n_reactors)])

    results_dir = "Images/{}".format(dataName)
    parameters = Params().default
    train = True
    n_samples = 50000
    
    # Get occuring temperatures
    cbData = CbData(cbParam.path, cbParam.file_ind, cbParam.sampcycle, cbParam.n_reactors)
    temp = []
    for j in range(cbParam.n_reactors):
        temp += list(set(cbData.temp_sp[j]))
    temp_arr = sorted(list(set(temp)))
    if temp_arr[0] == 0:
        temp_arr.remove(0)
    temp_arr = np.array(temp_arr)


    low = np.array([2000 if temp_arr[t] < 35 else 0 for t in range(len(temp_arr))])
    high = np.array([6000 if temp_arr[t] < 35 else 500 for t in range(len(temp_arr))])
    gr_fp = np.random.uniform(low, high, (n_samples,len(temp_arr))).T

    e_ofs = [parameters['e_ofs'][cbParam.reactors[j]] for j in range(cbParam.n_reactors)]
    e_fac = [parameters['e_fac'][cbParam.reactors[j]] for j in range(cbParam.n_reactors)]
    e = [(cbData.fl[j]*cbData.b1[j] - e_ofs[j])/e_fac[j] + e_ofs[j] for j in range(cbParam.n_reactors)]

            
    # SIMULATE
    cb_dil, fp_init = [], []
    sim_fl_train, sim_fl_test = [], []
    s_min = np.zeros(len(temp_arr), dtype=int)
    for j in range(cbParam.n_reactors):
        dil = np.diff(cbData.p1[j], prepend=[0,0]) > 0.015
        cb_dil.append(dil[:-1])
        fp_init.append(e[j][0] - e_ofs[j])

    if train:
        err = [[] for t in range(len(temp_arr))]
        for j in range(cbParam.n_reactors):
            print("Training at reactor: ", cbParam.titles[j])
            sim_fp = simulateFlProtein(cbData.time_h[j],fp_init[j], cbData.od[j], cb_dil[j], cbData.temp_sp[j], parameters, temp_arr, gr_fp, True, e[j] - e_ofs[j], err).T
            sim_fl_train.append(sim_fp + e_ofs[j])
        
        for t in range(len(temp_arr)):
            rmse = np.sqrt(np.mean(np.array(err[t])**2,axis=0))
            s_min[t] = np.argmin(rmse)
        # Print obtrained parameters
        gr_fp_opt = np.array([gr_fp[t][s_min[t]] for t in range(len(temp_arr))])
        print("Growth rates:")
        print(*gr_fp_opt, sep = ", ")

        # Obtain, print and plot growth rate model
        # gr_model1 = np.poly1d(np.polyfit(tem[:-3], param.gr_fp[:-3], 1))
        # gr_model2 = np.poly1d(np.polyfit(tem[4:-1], param.gr_fp[4:-1], 2))
        gr_model2_all = np.poly1d(np.polyfit(temp_arr, gr_fp_opt, 2))
        print("Growth rate model:")
        print(*gr_model2_all.coefficients, sep = ", ")
        x_mod = np.linspace(29, 36, 110)

        # y1 = gr_model1(x_mod)
        # y2 = gr_model2(x_mod)
        y_all = gr_model2_all(x_mod)
        # y_all[y_all < 20] = 20
        # y2[y2 < 20] = 20
        fig, ax = plt.subplots()
        fig.set_figheight(7)
        fig.set_figwidth(10)
        ax.plot(temp_arr, gr_fp_opt, 'om', label = 'data')
        # ax.plot(x_mod, y1, '-b')
        # ax.plot(x_mod, y2, '-b', label = '1st and 2nd order pw poly')
        ax.plot(x_mod, y_all, '-k', label = '2nd order poly all')
        ax.set_xlabel('Temperature [°C]')
        ax.set_ylabel('Growth rate [1/h]')
        ax.legend(loc='best')
        ax.set_title("Fluorescent protein growth rate model")
        fig.savefig(results_dir + "/gr_model{}_e.png", transparent=True)

        parameters['gr_fp'] = gr_model2_all.coefficients

    print("Running with estimated parameters")
    for j in range(cbParam.n_reactors):
        sim_fp = simulateFlProtein(cbData.time_h[j], fp_init[j], cbData.od[j], cb_dil[j], cbData.temp[j], parameters).T
        sim_fl_test.append(sim_fp + e_ofs[j])
    
    # ANALYSIS
    n_rows = math.ceil(cbParam.n_reactors/2)
    n_culumns = 2 if cbParam.n_reactors > 1 else 1
    matplotlib.style.use('default')
    fig, ax = plt.subplots(n_rows,2,sharex='all')
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
        ax[r][c].plot(cbData.time_h[j],e[j],'.g',markersize = 1, label = '$e_{meas}$')
        if train:
            ax[r][c].plot(cbData.time_h[j],sim_fl_train[j][0],'k',lw = 0.5, label = '$fl_{sim, train}$', alpha = 0.1)
            for s in range(1,min(n_samples,20)):
                ax[r][c].plot(cbData.time_h[j],sim_fl_train[j][s],'k',lw = 0.5, alpha = 0.1)
            ax[r][c].plot(cbData.time_h[j],sim_fl_train[j][s_min[j]],'m', lw = 1, label = '$fl_{sim, opt}$')
        # ax[r][c].plot(cbData.time_h[j],sim_fl_test[j][0],'b',lw = 1, label = '$fl_{sim, model}$')

        # ax[r][c].vlines(cbData.time_h[j][cb_dil[j]==1],-2,2,'g',lw = 0.5, alpha=0.5, label = 'p. putida dil')
        # ax[j].plot(cbData.time_h[j],cbData.od[j]*100,'--m',lw = 0.5, label = 'od')

        ax[r][c].legend(loc="upper left")
        if (c == 0):
            axr.set_ylabel("Em 440 norm. Intensity")
        else:
            axr.set_ylabel('Temperature [°C]', color='r')
            ax[r][c].tick_params(axis='y', labelleft=True)
        if r == n_rows-1:
            ax[r][c].set_xlabel("Time [h]")
        axr.tick_params(axis='y', color='r', labelcolor='r')
        ax[r][c].set_xlim([cbData.time_h[j][0]-0.5,cbData.time_h[j][-1]+0.5])
        # ax[r][c].set_ylim([0,0.5])
        if train:
            pass
            # ax[r][c].set_ylim([0,1.1*max(max(sim_fl_test[j][0]),max(sim_fl_train[j][s_min[j]]))])
        else:
            ax[r][c].set_ylim([0,1.1*max(sim_fl_test[j][0])])
        axr.set_ylim([28,37])
        ax[r][c].set_title(cbParam.titles[j])
    # TODO: Set titles
    fig.suptitle(dataName)
    fig.tight_layout()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    fig.savefig(results_dir + "/fl_sim_e_{}.png".format("train" if train else "test"),transparent=True)