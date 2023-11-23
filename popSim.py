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

def dilute(x_curr, parameters):
        """
        Take self.dil_am out of the reactor and update self.pop correspondingly.
        """
        # self.pop -= self.param.Dil_amount*self.pop/(self.pop + pop_other)
        dil = x_curr[:,0] + x_curr[:,1] > parameters.Dil_sp
        x_curr[,:] = (2*parameters.Dil_sp/(self.pop + pop_other) - 1)*self.pop

def simulateCultures(e_init, p_init, cb_hrs, sim_hrs, sim_tem_e, sim_tem_p, parameters):
    """
    Simulates one E coli. and P. Putida coculture using Euler forward discretization.

    Returns a time array with the corresponding values of the temperature and simulated population sizes.
    """
    e_coli = BactCult("e", e_init)
    p_puti = BactCult("p", p_init)

    data_l = len(cb_hrs)

    x = np.zeros((data_l,n_samples,2))
    x_curr = np.full((n_samples,2),[e_init, p_init])

    cb_pop_e = np.empty(data_l)
    cb_pop_p = np.empty(data_l)
    cb_dil = np.empty(data_l)
    k = 0
    # Update
    for i in range(len(sim_hrs)):
        pop_e = e_coli.pop
        pop_p = p_puti.pop
        dil = 0

        if sim_hrs[i] >= cb_hrs[k]:
            # log
            cb_pop_e[k] = e_coli.pop
            cb_pop_p[k] = p_puti.pop
            x[k,:,:] = x_curr
            x_curr = dilute(x_curr, parameters)
            k += 1
            # Dilute if combined OD above threshold
            if (pop_e + pop_p > parameters.Dil_sp):
                e_coli.dilute(pop_p)
                p_puti.dilute(pop_e)
            if (pop_e + pop_p > parameters.Dil_sp):
                e_coli.dilute(pop_p)
                p_puti.dilute(pop_e)
                dil = 1
            cb_dil[k] = dil
            
        # Let them grow
        e_coli.grow(sim_tem_e[i],parameters)
        p_puti.grow(sim_tem_p[i],parameters)

    cb_pop_e[k] = e_coli.pop
    cb_pop_p[k] = p_puti.pop
    return cb_pop_e, cb_pop_p, cb_dil


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

    cb_hrs, cb_od, cb_tem, cb_fl, cb_p1, cb_sp = [], [], [], [], [], []
    for i in range(n_reactors):
        cb_hrs.append(cb_dfs[i]["exp_time"][scope[i][0]:scope[i][-1]+1].to_numpy()/3600)
        cb_od_temp = cb_dfs[i]["od_measured"][scope[i][0]:scope[i][-1]+1].to_numpy()
        cb_od_temp[cb_od_temp < 0.005] = 0.005
        cb_od.append(cb_od_temp)
        cb_tem.append(cb_dfs[i]["media_temp"][scope[i][0]:scope[i][-1]+1].to_numpy())
        cb_fl.append(cb_dfs[i]["FP1_emit1"][scope[i][0]:scope[i][-1]+1].to_numpy())
        cb_p1.append(cb_dfs[i]["pump_1_rate"][scope[i][0]:scope[i][-1]+1].to_numpy())
        cb_sp.append(cb_dfs[i]["custom_prog_status"][scope[i][0]:scope[i][-1]+1].to_numpy())
    cb_hrs = [cb_hrs[i]-cb_hrs[i][0] for i in range(n_reactors)]
    return cb_hrs, cb_od, cb_tem, cb_fl, cb_p1, cb_sp


if __name__ == "__main__":

    # SPECIFY DATA
    dataName = '064-2'
    n_samples = 5000
    train_ind = [0,2]
    test_ind = [1,3]
    all_ind = train_ind+test_ind
    n_reactors = len(all_ind)
    path, file_ind, sampcycle, titles = getCbDataInfo(dataName)
    cb_fc_ec = getFcData(dataName)

    cb_hrs, cb_od, cb_tem, cb_fl, cb_p1, cb_sp = loadData(path, file_ind, sampcycle, n_reactors)

    param = ModParam()

    s_alpha_e = np.random.normal(param.Alpha_e, 0.05*abs(param.Alpha_e), n_samples)
    s_beta_e = np.random.normal(param.Beta_e, 0.01*abs(param.Beta_e), n_samples)
    s_alpha_p = np.random.normal(param.Alpha_p, 0.05*abs(param.Alpha_p), n_samples)
    s_beta_p = np.random.normal(param.Beta_p, 0.01*abs(param.Beta_p), n_samples)
    s_gamma_p = np.random.normal(param.Gam_p, 0.01*abs(param.Gam_p), n_samples)
    s_delta_p = np.random.normal(param.Del_p, 0.01*abs(param.Del_p), n_samples)


    sim_tem_e, sim_tem_p, sim_hrs = [], [], []
    p_rel_prior= []
    # SIMULATION
    for i in range(n_reactors):
        # Collect inputs
        sim_hrs.append(np.arange(0,cb_hrs[i][-1],param.Ts/3600))
        sim_tem = interpolateCbToSim(cb_hrs[i], cb_tem[i], sim_hrs[i])
        if (param.Lag):
            tem_lag_e = np.concatenate((np.full(param.Lag_ind,param.T_pre_e),sim_tem))
            tem_lag_p = np.concatenate((np.full(param.Lag_ind,param.T_pre_p),sim_tem))
            if param.Avg_temp:
                sim_tem_e.append(ss.convolve(tem_lag_e,np.full(param.Lag_ind+1,1/(param.Lag_ind+1)),mode='valid'))
                sim_tem_p.append(ss.convolve(tem_lag_p,np.full(param.Lag_ind+1,1/(param.Lag_ind+1)),mode='valid'))
            else:
                sim_tem_e.append(tem_lag_e[:len(sim_hrs[i])])
                sim_tem_p.append(tem_lag_p[:len(sim_hrs[i])])
        else:
            sim_tem_e.append(sim_tem)
            sim_tem_p.append(sim_tem)

        e_init = cb_fc_ec[i][0]*cb_od[i][0]/100
        p_init = (100-cb_fc_ec[i][0])*cb_od[i][0]/100
        # Simulate cultures
        cb_pop_e, cb_pop_p, cb_dil = simulateCultures(e_init, p_init, cb_hrs[i], sim_hrs[i], sim_tem_e[i], sim_tem_p[i], param)
        p_rel_prior.append(np.array(cb_pop_p)/(np.array(cb_pop_p)+np.array(cb_pop_e))*100)

    # Run multiple simulations with different growth rates
    p_rel_train, fl_p_train = [], []
    rmse = np.zeros((1,n_samples))
    for i in train_ind:
        print("Training with reactor {}".format(titles[i]))
        e_coli_samples, p_puti_samples = [], []
        e_init = cb_fc_ec[i][0]*cb_od[i][0]/100
        p_init = (100-cb_fc_ec[i][0])*cb_od[i][0]/100
        for s in range(n_samples):
            param.Alpha_e = s_alpha_e[s]
            param.Beta_e = s_beta_e[s]
            param.Alpha_p = s_alpha_p[s]
            param.Beta_p = s_beta_p[s]
            param.Gam_p = s_gamma_p[s]
            param.Del_p = s_delta_p[s]
            # Simulate cultures
            cb_pop_e, cb_pop_p, cb_dil = simulateCultures(e_init, p_init, cb_hrs[i], sim_hrs[i], sim_tem_e[i], sim_tem_p[i], param)
            e_coli_samples.append(cb_pop_e)
            p_puti_samples.append(cb_pop_p)
        # Calculate RMSE
        p_rel_train.append(np.array(p_puti_samples)/(np.array(p_puti_samples)+np.array(e_coli_samples))*100)
        rmse += np.sqrt(np.mean((p_rel_train[-1][:,sampcycle[i]-sampcycle[i][0]] - (100-cb_fc_ec[i]))**2,axis=1))
        # fl_init = (cb_fl[i][0] - param.min_fl[file_ind[i]])*cb_od[i][0]
        # sim_od = interpolateCbToSim(cb_hrs[i], cb_od[i], sim_hrs[i])
        # fl_p_train.append(simulateFlProtein(fl_init, cb_pop_p, sim_tem, cb_dil, param.Dil_amount, param.Dil_th)/sim_od)
    s_min = np.argmin(rmse)

    print("Best fit:\n{0:0.4f}, {1:0.4f}\n{2:0.4f}, {3:0.4f}, {4:0.4f}, {5:0.4f}".format(s_alpha_e[s_min], s_beta_e[s_min], s_alpha_p[s_min], s_beta_p[s_min],s_gamma_p[s_min], s_delta_p[s_min]))

    # Run multiple simulations with different growth rates
    p_rel_test= []
    print("Testing...")
    for i in test_ind:
        e_init = cb_fc_ec[i][0]*cb_od[i][0]/100
        p_init = (100-cb_fc_ec[i][0])*cb_od[i][0]/100
        param.Alpha_e = s_alpha_e[s_min]
        param.Beta_e = s_beta_e[s_min]
        param.Alpha_p = s_alpha_p[s_min]
        param.Beta_p = s_beta_p[s_min]
        param.Gam_p = s_gamma_p[s_min]
        param.Del_p = s_delta_p[s_min]
        # Simulate cultures
        cb_pop_e, cb_pop_p, cb_dil = simulateCultures(e_init, p_init, cb_hrs[i], sim_hrs[i], sim_tem_e[i], sim_tem_p[i], param)
        p_rel_test.append(np.array(cb_pop_p)/(np.array(cb_pop_p)+np.array(cb_pop_e))*100)


    # gR.plotGrowthRates()
    # critTemp = gR.getCritTemp()
    # assert(26 < critTemp[0] and critTemp[0] < 37)
    # ANALYSIS
    # n_reactors = 1
    n_rows = math.ceil(n_reactors/2)
    n_culumns = 2 if n_reactors > 1 else 1
    fig, ax = plt.subplots(n_rows,n_culumns,sharey='all')
    fig.set_figheight(n_rows*7)
    fig.set_figwidth(n_culumns*10)
    if n_culumns == 1:
        ax = [ax]
    if n_rows == 1:
        ax = [ax]
    axr = [[]]*n_reactors
    for i in range(n_reactors):
        r = i//2
        c = i%2
        axr[i] = ax[r][c].twinx()
        ax[r][c].set_zorder(2)
        axr[i].set_zorder(1)
        ax[r][c].patch.set_visible(False)

        if (i%2 == 0):
            ax[r][c].set_ylabel("Relative composition [%]")
        else:
            axr[i].set_ylabel('Temperature [°C]', color='r')
            ax[r][c].tick_params(axis='y', labelleft=True)
        # axr[i].set_yticks(np.append(axr[i].get_yticks(), critTemp[0]))
        axr[i].tick_params(axis='y', color='r', labelcolor='r')
        axr[i].text(1, 1, 'e coli. prefered',
                horizontalalignment='right',
                verticalalignment='top',
                transform=axr[i].transAxes,
                color='w',
                bbox={'facecolor': 'red', 'alpha': 1, 'pad': 0, 'edgecolor': 'r'})
        axr[i].text(1, 0, 'p. putida prefered',
                horizontalalignment='right',
                verticalalignment='bottom',
                transform=axr[i].transAxes,
                color='w',
                bbox={'facecolor': 'red', 'alpha': 1, 'pad': 0, 'edgecolor': 'r'})
        ax[r][c].set_xlabel("Time [h]")
        ax[r][c].set_xlim([sim_hrs[i][0]-0.5,sim_hrs[i][-1]+0.5])
        ax[r][c].set_ylim([-5,105])
        ax[r][c].set_title(titles[i])

    for j in range(len(train_ind)):
        i = train_ind[j]
        r = i//2
        c = i%2

        axr[i].plot(cb_hrs[i],cb_tem[i],'r',lw=0.5)
        # axr.plot(sim_hrs[i],sim_tem_e[i],'r',lw=1)
        # axr.hlines(critTemp[0],sim_hrs[i][0]-1,sim_hrs[i][-1]+1,'r',lw=0.5)
        ax[r][c].plot(cb_hrs[i],p_rel_train[j][0], 'g', label = 'p. putida train', alpha=0.1)
        for s in range(1,n_samples):
            ax[r][c].plot(cb_hrs[i],p_rel_train[j][s], 'g', alpha=0.1)
        ax[r][c].plot(cb_hrs[i],p_rel_train[j][s_min], 'k', label = 'p. putida opt')
        ax[r][c].plot(cb_hrs[i][sampcycle[i]-sampcycle[i][0]],100-cb_fc_ec[i], 'gx', label = 'p. putida fc',lw=0.4)
        ax[r][c].plot(cb_hrs[i],p_rel_prior[i], '--k', label = 'p. putida prior')
        # ax[r][c].plot(cb_hrs[i],(cb_fl[i]-param.minv)/(param.maxv-param.minv)*100,'.k',markersize = 0.8, label = 'fluorescense measured')
        # ax[r][c].plot(sim_hrs[i],(fl_p_train[i]+param.min_fl[file_ind[i]])*100,'m',lw = 0.5, label = '$100*fl_{sim}$')
        # ax[r][c].plot(time_all[i],od*100,'-k',lw = 0.5, label = 'od sim')
        # ax[r][c].plot(cb_hrs[i],cb_od[i]*100,'--m',lw = 0.5, label = 'od')
        # ax[r][c].plot(cb_hrs[i],(cb_sp[i]-param.minv)/(param.maxv-param.minv)*100,'g--',markersize = 0.8, label = 'fluorescense setpoint')
        ax[r][c].legend(loc="upper left")
    
    for j in range(len(test_ind)):
        i = test_ind[j]
        r = i//2
        c = i%2

        axr[i].plot(cb_hrs[i],cb_tem[i],'r',lw=0.5)
        ax[r][c].plot(cb_hrs[i],p_rel_test[j], 'k', label = 'p. putida opt')
        ax[r][c].plot(cb_hrs[i][sampcycle[i]-sampcycle[i][0]],100-cb_fc_ec[i], 'gx', label = 'p. putida fc',lw=0.4)
        ax[r][c].plot(cb_hrs[i],p_rel_prior[i], '--k', label = 'p. putida prior')
        ax[r][c].legend(loc="upper left")

    # TODO: Set titles
    fig.suptitle(dataName)
    fig.tight_layout()
    fig.savefig("Images/{}/{}r_{}h{}lag_est.png".format(dataName,n_reactors,param.Lag,'avg' if param.Avg_temp else ''))