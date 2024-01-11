import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import scipy.signal as ss
from scipy.optimize import fsolve

from popParam import ModParam
from CbData import CbData
from paramsData import CbDataParam

param = ModParam()

def f(xy):
    x, y = xy
    gr = getGrowthRate(np.full(2,x),param)
    z = np.array([y - gr[0,:][0],
                  y - gr[1,:][0]])
    return z

def getCritTemp():
    temp = fsolve(f, [33.0, 0.8])
    return temp

def plotGrowthRates(ax, parameters, label):
    temp = np.full((2,11),np.arange(parameters.temp_l,parameters.temp_h,1))
    gr = getGrowthRate(temp, parameters)
    # critT = getCritTemp()[0]
    ax.plot(temp[0],gr[0,:],"-x{}".format("k" if (label == "prior") else "b"),label="E coli. {}".format(label))
    ax.plot(temp[0],gr[1,:],"-o{}".format("k" if (label == "prior") else "g"),label='P. Putida {}'.format(label))
    # plt.plot(temp[0],gr[2],'-xk',label='Fl. Protein')
    # ax.vlines(critT,0,1,colors='r',label='Critical Temperature')
    return

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
    od = np.array([x_curr[:,0] + x_curr[:,1], x_curr[:,0] + x_curr[:,1]]).T
    dil = od[:,0] > parameters.dil_sp
    # dil = 1 if od > parameters.dil_sp else 0
    x_curr[dil,:] = (2*parameters.dil_sp/od[dil] - 1)*x_curr[dil,:]
    return x_curr

def getGrowthRate(temp, parameters):
    gr = np.array([parameters.Beta_e*temp[0] + parameters.Alpha_e,
                      parameters.Del_p*temp[1]**3 + parameters.Gam_p*temp[1]**2 + parameters.Beta_p*temp[1] + parameters.Alpha_p])
    gr[gr < 0] = 0
    return gr

def getGrowthRateFl(temp, parameters):
    gr = np.array([parameters.Gam_fp*temp**2 + parameters.Beta_fp*temp + parameters.Alpha_fp])
    gr[gr < 20] = 20
    return gr
            

def simulateCultures(e_init, p_init, cb_hrs, sim_hrs, sim_tem_e, sim_tem_p, parameters):
    """
    Simulates one E coli. and P. Putida coculture using Euler forward discretization.

    Returns a time array with the corresponding values of the temperature and simulated population sizes.
    """
    data_l = len(cb_hrs)
    n_s = len(parameters.Alpha_e)

    x = np.zeros((data_l,n_s,2))
    x_curr = np.full((n_s,2),[e_init, p_init])

    k = 0
    # Update
    for i in range(len(sim_hrs)):

        if sim_hrs[i] >= cb_hrs[k]:
            # log
            x[k,:,:] = x_curr
            # Dilute if combined OD above threshold
            x_curr = dilute(x_curr, parameters)
            k += 1
        
        # Euler forward (less accurate)
        # x_curr *= 1 + parameters.Ts/3600*getGrowthRate([sim_tem_e[i],sim_tem_p[i]], parameters).T
            
        # Runge - Kutta 4th order
        k1 = getGrowthRate([sim_tem_e[i],sim_tem_p[i]], parameters).T*x_curr
        k2 = getGrowthRate([sim_tem_e[i],sim_tem_p[i]], parameters).T*(x_curr+parameters.Ts/3600/2*k1)
        k3 = getGrowthRate([sim_tem_e[i],sim_tem_p[i]], parameters).T*(x_curr+parameters.Ts/3600/2*k2)
        k4 = getGrowthRate([sim_tem_e[i],sim_tem_p[i]], parameters).T*(x_curr+parameters.Ts/3600*k3)
        x_curr = x_curr + parameters.Ts/3600/6*(k1+2*k2+2*k3+k4)
    
    x[k,:,:] = x_curr
    return x


def simulateFlProtein(fp_init, cb_hrs, sim_hrs, cb_temp, p_puti, dil, r_ind, parameters):
    data_l = len(cb_hrs)
    n_s = parameters.n_samples if parameters.mcmc else 1
    x = np.zeros((data_l,n_s))
    x_curr = fp_init
    k = 0

    for i in range(len(sim_hrs)):
        if sim_hrs[i] >= cb_hrs[k]:
            x[k] = x_curr
            if dil[k]:
                x_curr -= parameters.dil_amount/parameters.dil_th*x_curr
            k += 1
        if parameters.mcmc:
            gr = parameters.gr_fp[r_ind]
        else:
            gr = getGrowthRateFl(cb_temp[k-1], parameters)
        
        # Euler forward (not significantly less accurate than Runge - Kutta 4th order)
        x_curr = x_curr + parameters.Ts/3600*gr*p_puti[i]
        
        # Runge - Kutta 4th order
        # k1 = gr*p_puti[i]
        # k2 = gr*(p_puti[i]+parameters.Ts/3600/2*k1)
        # k3 = gr*(p_puti[i]+parameters.Ts/3600/2*k2)
        # k4 = gr*(p_puti[i]+parameters.Ts/3600*k3)
        # x_curr = x_curr + parameters.Ts/3600/6*(k1+2*k2+2*k3+k4)
    
    x[k] = x_curr
    return x


if __name__ == "__main__":

    # SPECIFY DATA
    dataName = '064-2-test'
    cbParam = CbDataParam(dataName)

    if param.mcmc:
        train_ind = [0,2,4,6]
        test_ind = [1,3,5,7]
        all_ind = train_ind+test_ind
        cbParam.n_reactors = len(all_ind)

    cbData = CbData(cbParam.path, cbParam.file_ind, cbParam.sampcycle, cbParam.n_reactors)

    fig, ax = plt.subplots()
    plotGrowthRates(ax, param, "prior")

    sim_tem_e, sim_tem_p, sim_hrs = [], [], []
    p_rel_prior= []
    # SIMULATION
    for i in range(cbParam.n_reactors):
        # Collect inputs
        sim_hrs.append(np.arange(0,cbData.time_h[i][-1],param.Ts/3600))
        sim_tem = interpolateCbToSim(cbData.time_h[i], cbData.temp[i], sim_hrs[i])
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

        e_init = cbParam.cb_fc_ec[i][0]*cbData.od[i][0]/100
        p_init = (100-cbParam.cb_fc_ec[i][0])*cbData.od[i][0]/100
        # Simulate cultures
        x = simulateCultures(e_init, p_init, cbData.time_h[i], sim_hrs[i], sim_tem_e[i], sim_tem_p[i], param)
        p_rel_prior.append(np.array(x[:,:,1])/(np.array(x[:,:,1])+np.array(x[:,:,0]))*100)


    if param.mcmc:
        # Run multiple simulations with different growth rates
        p_rel_train, fl_p_train = [], []
        rmse = np.zeros((1,param.n_samples))
        param.Alpha_e = np.random.normal(param.Alpha_e[0], 0.03*abs(param.Alpha_e[0]), param.n_samples)
        param.Beta_e = np.random.normal(param.Beta_e[0], 0.05*abs(param.Beta_e[0]), param.n_samples)
        param.Alpha_p = np.random.normal(param.Alpha_p[0], 0.03*abs(param.Alpha_p[0]), param.n_samples)
        param.Beta_p = np.random.normal(param.Beta_p[0], 0.05*abs(param.Beta_p[0]), param.n_samples)
        param.Gam_p = np.random.normal(param.Gam_p[0], 0.007*abs(param.Gam_p[0]), param.n_samples)
        param.Del_p = np.random.normal(param.Del_p[0], 0.002*abs(param.Del_p[0]), param.n_samples)
        for i in train_ind:
            print("Training with reactor {}".format(cbParam.titles[i]))
            e_coli_samples, p_puti_samples = [], []
            e_init = cbParam.cb_fc_ec[i][0]*cbData.od[i][0]/100
            p_init = (100-cbParam.cb_fc_ec[i][0])*cbData.od[i][0]/100
            # Simulate cultures
            x = simulateCultures(e_init, p_init, cbData.time_h[i], sim_hrs[i], sim_tem_e[i], sim_tem_p[i], param)
            # Calculate RMSE
            p_rel_train.append(np.array(x[:,:,1])/(np.array(x[:,:,1])+np.array(x[:,:,0]))*100)
            rmse += np.sqrt(np.mean((p_rel_train[-1][cbParam.sampcycle[i]-cbParam.sampcycle[i][0],:].T - (100-cbParam.cb_fc_ec[i]))**2,axis=1))
        s_min = np.argmin(rmse)

        print("Best fit: \nself.Beta_e = np.array([{0:0.5f}]) \nself.Alpha_e =np.array([{1:0.5f}]) \nself.Del_p = np.array([{2:0.5f}]) \nself.Gam_p =  np.array([{3:0.5f}]) \nself.Beta_p = np.array([{4:0.5f}]) \nself.Alpha_p = np.array([{5:0.5f}])".format(param.Beta_e[s_min], param.Alpha_e[s_min], param.Del_p[s_min], param.Gam_p[s_min],param.Beta_p[s_min], param.Alpha_p[s_min]))

        # Run with optimal growth rates
        param.Alpha_e = np.array([param.Alpha_e[s_min]])
        param.Beta_e = np.array([param.Beta_e[s_min]])
        param.Alpha_p = np.array([param.Alpha_p[s_min]])
        param.Beta_p = np.array([param.Beta_p[s_min]])
        param.Gam_p = np.array([param.Gam_p[s_min]])
        param.Del_p = np.array([param.Del_p[s_min]])
        p_rel_test= []
        print("Testing...")
        for i in test_ind:
            e_init = cbParam.cb_fc_ec[i][0]*cbData.od[i][0]/100
            p_init = (100-cbParam.cb_fc_ec[i][0])*cbData.od[i][0]/100
            # Simulate cultures
            x = simulateCultures(e_init, p_init, cbData.time_h[i], sim_hrs[i], sim_tem_e[i], sim_tem_p[i], param)
            p_rel_test.append(np.array(x[:,:,1])/(np.array(x[:,:,1])+np.array(x[:,:,0]))*100)

        # critTemp = getCritTemp()
        # assert(26 < critTemp[0] and critTemp[0] < 37)
        plotGrowthRates(ax, param, "opt")

    # ANALYSIS
    ax.set_xlim(param.temp_l-1,param.temp_h+1)
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Growth Rate [$h^{-1}$]")
    ax.legend()
    fig.savefig("Images/{}/growthRates{}.png".format(dataName,"_mcmc" if param.mcmc else ""))

    # cbParam.n_reactors = 1
    n_rows = math.ceil(cbParam.n_reactors/2)
    n_culumns = 2 if cbParam.n_reactors > 1 else 1
    matplotlib.style.use('default')
    fig, ax = plt.subplots(n_rows,n_culumns,sharey='all')
    fig.set_figheight(n_rows*7)
    fig.set_figwidth(n_culumns*10)
    if n_culumns == 1:
        ax = [ax]
    if n_rows == 1:
        ax = [ax]
    axr = [[]]*cbParam.n_reactors
    for i in range(cbParam.n_reactors):
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
        ax[r][c].set_title(cbParam.titles[i])

    if param.mcmc:
        for j in range(len(train_ind)):
            i = train_ind[j]
            r = i//2
            c = i%2
            axr[i].plot(cbData.time_h[i],cbData.temp[i],'r',lw=0.5)
            ax[r][c].plot(cbData.time_h[i],p_rel_train[j][:,0], 'k', label = 'p. putida train', alpha=0.1, lw=0.5)
            for s in range(1,min(param.n_samples,200)):
                ax[r][c].plot(cbData.time_h[i],p_rel_train[j][:,s], 'k', alpha=0.1, lw=0.5)
            ax[r][c].plot(cbData.time_h[i],p_rel_train[j][:,s_min], 'g', label = 'p. putida opt')
            ax[r][c].plot(cbData.time_h[i][cbParam.sampcycle[i]-cbParam.sampcycle[i][0]],100-cbParam.cb_fc_ec[i], 'gx', label = 'p. putida fc',lw=0.4)
            ax[r][c].plot(cbData.time_h[i],p_rel_prior[i][:,0], '--g', label = 'p. putida prior')
            ax[r][c].legend(loc="upper left")
        
        for j in range(len(test_ind)):
            i = test_ind[j]
            r = i//2
            c = i%2
            axr[i].plot(cbData.time_h[i],cbData.temp[i],'r',lw=0.5)
            ax[r][c].plot(cbData.time_h[i],p_rel_test[j][:,0], 'g', label = 'p. putida opt')
            ax[r][c].plot(cbData.time_h[i][cbParam.sampcycle[i]-cbParam.sampcycle[i][0]],100-cbParam.cb_fc_ec[i], 'gx', label = 'p. putida fc',lw=0.4)
            ax[r][c].plot(cbData.time_h[i],p_rel_prior[i][:,0], '--g', label = 'p. putida prior')
            ax[r][c].legend(loc="upper left")
    
    else:
        for i in range(cbParam.n_reactors):
            r = i//2
            c = i%2
            axr[i].plot(cbData.time_h[i],cbData.temp[i],'--r',lw=0.5)
            axr[i].plot(sim_hrs[i],sim_tem_p[i],'r',lw=0.5)
            ax[r][c].plot(cbData.time_h[i],p_rel_prior[i][:,0], 'g', label = 'p. putida')
            ax[r][c].plot(cbData.time_h[i][cbParam.sampcycle[i]-cbParam.sampcycle[i][0]],100-cbParam.cb_fc_ec[i], 'gx', label = 'p. putida fc',lw=0.4)
            ax[r][c].legend(loc="upper left")

    # TODO: Set titles
    fig.suptitle(dataName)
    fig.tight_layout()
    fig.savefig("Images/{}/{}r_{}h{}lag_est.png".format(dataName,cbParam.n_reactors,param.Lag,'avg' if param.Avg_temp else ''))