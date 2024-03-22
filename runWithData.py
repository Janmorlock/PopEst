import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib
import os

from CbDataParams import CbDataParam
from CbData import CbData
from lib.Estimator import EKF
from config.params import Params
from lib.Model import CustModel


class CritTemp:
    def __init__(self, gr_e = np.zeros(1), gr_p = np.zeros(1)):
        self.model = CustModel()
        if gr_e[0]:
            self.model.parameters['gr_e'] = gr_e
            self.model.parameters['gr_p'] = gr_p

    def f(self, xy):
        x, y = xy
        gr = self.model.getSteadyStateGrowthRates(x)
        z = np.array([y - gr[0],
                    y - gr[1]])
        return z

    def getCritTemp(self):
        temp = fsolve(self.f, [33.0, 0.8])
        return temp[0]

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


if __name__ == "__main__":

    # SPECIFY DATA
    data_name = '081-1'
    test_est = False
    paper = True
    control = False
    single_est = {'od_update':      [True, True, True],
                    'fl_update':    [True, False, False],
                    'od_gr_update': [False, True, False],
                    'fl_gr_update': [False, False, True]}
    cbParam = CbDataParam(data_name)
    if paper:
        cbParam.n_reactors = 1
        cbParam.file_ind = [[5]]
        cbParam.cb_fc_ec = [cbParam.cb_fc_ec[1]]
    cbData = CbData(cbParam)

    len_test = len(single_est['od_update']) if test_est else 1
    parameters = Params().default

    cT = CritTemp()
    critTemp = cT.getCritTemp()
    assert(29 < critTemp and critTemp < 36)
    print("Critical temperature: {}".format(critTemp))

    estimates_pred = [np.empty((len_test,len(cbData.time[j])),dtype=dict) for j in range(cbParam.n_reactors)]
    estimates = [np.empty((len_test,len(cbData.time[j])),dtype=dict) for j in range(cbParam.n_reactors)]
    variances = [np.empty((len_test,len(cbData.time[j]),3,3)) for j in range(cbParam.n_reactors)]
    pp_rel_od_arr = [np.empty((len_test,len(cbData.time[j]))) for j in range(cbParam.n_reactors)]
    pp_rel_fl_arr = [np.empty((len_test,len(cbData.time[j]))) for j in range(cbParam.n_reactors)]
    pp_rel_od_res_arr = [np.empty((len_test,len(cbData.time[j]))) for j in range(cbParam.n_reactors)]
    pp_rel_fl_res_arr = [np.empty((len_test,len(cbData.time[j]))) for j in range(cbParam.n_reactors)]
    for j in range(cbParam.n_reactors):
        for t in range(len_test):
            # Construct State estimator
            pred = EKF(dev_ind = j, update = False)
            ekf = EKF(dev_ind = j)
            pred.model.dithered = bool(cbData.dil)
            ekf.model.dithered = bool(cbData.dil)
            if test_est:
                ekf.model.parameters['od_update'] = single_est['od_update'][t]
                ekf.model.parameters['fl_update'] = single_est['fl_update'][t]
                ekf.model.parameters['od_gr_update'] = single_est['od_gr_update'][t]
                ekf.model.parameters['fl_gr_update'] = single_est['fl_gr_update'][t]
            ekf.set_r_coeff(cbParam.reactors[j])
            for k in range(len(cbData.time[j])):
                # Run the filter
                dil = cbData.dil[j][k] if cbData.dil else 0
                u_pred = [cbData.temp[j][k], dil]
                pred.estimate(cbData.time[j][k], u_pred)
                estimates_pred[j][t,k] = pred.est.copy()
                u = np.array([cbData.temp[j][k], dil])
                y = np.array([cbData.od[j][k], cbData.fl[j][k]*cbData.b1[j][k]])
                ekf.estimate(cbData.time[j][k], u, y)
                estimates[j][t,k] = ekf.est.copy()
                variances[j][t,k] = ekf.var.copy()
                pp_rel_od_arr[j][t,k] = ekf.p_est_od/cbData.od[j][k]
                pp_rel_fl_arr[j][t,k] = ekf.p_est_fl/cbData.od[j][k]
                pp_rel_od_res_arr[j][t,k] = ekf.p_est_od_res
                pp_rel_fl_res_arr[j][t,k] = ekf.p_est_fl_res

                if k == len(cbData.time[j])-1:
                    print("Final variance [{}][{}]: {}".format(j, t, variances[j][t,k]))

    e = [(cbData.fl[j]*cbData.b1[j] - parameters['e_ofs'][cbParam.reactors[j]])/parameters['e_fac'][cbParam.reactors[j]] + parameters['e_ofs'][cbParam.reactors[j]] for j in range(cbParam.n_reactors)]

    # ANALYSIS
    # cbParam.n_reactors = 1
    n_plots = cbParam.n_reactors * len_test
    n_rows = cbParam.n_reactors
    if paper:
        n_rows += 2
    n_culumns = len_test
    matplotlib.style.use('default')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 11})
    fig, ax = plt.subplots(n_rows,n_culumns,sharex='all')
    if paper:
        fig, ax = plt.subplots(n_rows,n_culumns,sharex='all',height_ratios=[1,1,4])
    if paper:
        fig.set_figheight((n_rows-2)*6)
        fig.set_figwidth(n_culumns*7)
    else:
        fig.set_figheight(n_rows*7)
        fig.set_figwidth(n_culumns*10)
    if n_culumns == 1:
        if n_rows == 1:
            ax = [ax]
        else:
            ax = [[ax[i]] for i in range(n_rows)]
    if n_rows == 1:
        ax = [ax]
    for j in range(cbParam.n_reactors):
        for t in range(len_test):
            r = j
            if paper:
                r += 2
            c = t
            # j = 1
            e_coli_pred = np.array([estimates_pred[j][t,k]['e'] for k in range(len(estimates_pred[j][t]))])
            p_puti_pred = np.array([estimates_pred[j][t,k]['p'] for k in range(len(estimates_pred[j][t]))])
            fp_pred = np.array([estimates_pred[j][t,k]['fp'] for k in range(len(estimates_pred[j][t]))])
            od_pred = e_coli_pred + p_puti_pred
            p_puti_pred_percent = p_puti_pred/od_pred

            e_coli = np.array([estimates[j][t,k]['e'] for k in range(len(estimates[j][t]))])
            p_puti = np.array([estimates[j][t,k]['p'] for k in range(len(estimates[j][t]))])
            var = np.array([variances[j][t,k][:2,:2] for k in range(len(variances[j][t]))])
            fp = np.array([estimates[j][t,k]['fp'] for k in range(len(estimates[j][t]))])
            od = e_coli + p_puti
            p_puti_percent = p_puti/od
            sq = e_coli**2/p_puti**2*var[:,1,1] + var[:,0,0] - 2*e_coli/p_puti*np.sign(var[:,0,1])*np.sqrt(np.abs(var[:,0,1]))
            p_puti_per_sigma = np.abs(p_puti_percent/(e_coli+p_puti))*np.sqrt(np.abs(sq))

            max_e = max(e[j])

            if not paper:
                axr = ax[r][c].twinx()
                ax[r][c].set_zorder(2)
                axr.set_zorder(1)
                ax[r][c].patch.set_visible(False)
            else:
                axr = ax[0][c]
            axr.hlines(critTemp,cbData.time_h[j][0]-1,cbData.time_h[j][-1]+1,'r',linestyles = '--',lw=0.5)
            if not paper:
                axr.plot(cbData.time_h[j],cbData.temp_sp[j],'--r',lw=0.6,alpha=1, label = '$temp_{sp}$')
            else:
                axr.plot(cbData.time_h[j],cbData.temp[j],'r',lw=0.6, label = '$temp_{meas}$')


            ax[r][c].fill_between(cbData.time_h[j], (p_puti_percent-p_puti_per_sigma)*100, (p_puti_percent+p_puti_per_sigma)*100, color='g',alpha=0.1)
            od_j = cbData.od[j]
            if paper:
                axod = ax[1][c]
                axod.set_ylabel("OD")
                ax[r][c].plot(cbData.time_h[j],p_puti_pred_percent*100, 'g', lw = 0.5, label = 'Est. through Model, $\hat{p}_{mod}$')
                axod.plot(cbData.time_h[j],od_j,'k',lw = 0.6, label = '$od_{meas}$')
            else:
                axod = ax[r][c]
                ax[r][c].plot(cbData.time_h[j],e[j]/max_e*100,'.', color = '#0000ff',markersize = 0.5,label = '$fl_{meas}$')
                od_j *= 100
                # ax[r][c].plot(cbData.time_h[j],(fp + parameters['od_fac']*od + parameters['e_ofs'][cbParam.reactors[j]])/max_e,color = '#0000ff',lw = 0.5, label = '$fl_{est}$')
            # ax[r][c].plot(cbData.time_h[j][pp_rel_od_arr[j][t] > 0],pp_rel_od_arr[j][t,pp_rel_od_arr[j][t] > 0],'+g',markersize = 6, alpha = 0.7, label = '$pp_{od}$')
            ax[r][c].plot(cbData.time_h[j][pp_rel_od_arr[j][t] > 0][1:-1],moving_average(pp_rel_od_arr[j][t,pp_rel_od_arr[j][t] > 0]*100,3),'--',color = 'k',lw = 0.6, alpha = 1, label = 'Est. through OD, $\hat{p}_{od}$')
            # ax[r][c].plot(cbData.time_h[j][pp_rel_fl_arr[j][t] > 0],pp_rel_fl_arr[j][t,pp_rel_fl_arr[j][t] > 0],'+',color = '#0000FF',markersize = 6, alpha = 0.7, label = '$pp_{fl}$')
            ax[r][c].plot(cbData.time_h[j][pp_rel_fl_arr[j][t] > 0][1:-1],moving_average(pp_rel_fl_arr[j][t,pp_rel_fl_arr[j][t] > 0]*100,3),'--',color = '#0000FF',lw = 0.6, alpha = 1, label = 'Est. through Fluorescence, $\hat{p}_{fl}$')
            # ax[r][c].plot(cbData.time_h[j][pp_rel_od_res_arr[j][t] > 0],pp_rel_od_res_arr[j][t,pp_rel_od_res_arr[j][t] > 0]/max(pp_rel_od_res_arr[j][t]), '+g', markersize = 7, label = '$pp_{res,od}$')
            # ax[r][c].plot(cbData.time_h[j][pp_rel_fl_res_arr[j][t] > 0],pp_rel_fl_res_arr[j][t,pp_rel_fl_res_arr[j][t] > 0]/max(pp_rel_fl_res_arr[j][t]), '+b', markersize = 7, label = '$pp_{res,fl}$')
            
            # time_fl = cbData.time_h[j][pp_rel_fl_arr[j][t] > 0]
            # temp_fl = np.array(cbData.temp[j][pp_rel_fl_arr[j][t] > 0])
            # pp_rel_fl_res = pp_rel_fl_res_arr[j][t,pp_rel_fl_arr[j][t] > 0]
            # pp_rel_fl = pp_rel_fl_arr[j][t,pp_rel_fl_arr[j][t] > 0]
            # fl_unc = (parameters['sigma_fl_gr']
            #             + pp_rel_fl_res*parameters['gr_res_sigma']*p_puti[pp_rel_fl_arr[j][t] > 0]
            #             + np.exp(-abs(temp_fl - 35.5)/parameters['fl_gr_temp_sigma_decay'])*parameters['fl_gr_temp_prox_sigma_max']
            #             + np.exp(-abs(temp_fl - 29)/parameters['fl_gr_temp2_sigma_decay'])*parameters['fl_gr_temp2_prox_sigma_max']
            #             )
            # max_fl_unc = max(fl_unc)
            # sz = 20*(1 - fl_unc/max_fl_unc)
            # ax[r][c].scatter(time_fl,pp_rel_fl*100,marker = 'x', color = '#0000FF',s = sz, alpha = 1)
            # for k in range(len(fl_unc)):
            #     al = 1 - fl_unc[k]/max_fl_unc
            #     ax[r][c].scatter(time_fl[k],pp_rel_fl[k]*100,color = '#0000FF',s = 6, alpha = al)

            if control:
                ax[r][c].plot(cbData.time_h[j][cbData.p_targ[j] > 0],cbData.p_targ[j][cbData.p_targ[j] > 0]*100, '--g', lw = 1, label = '$p_{targ}$')
                ax[r][c].plot(cbData.time_h[j],cbData.p_est[j]*100, 'g', linestyle = 'dashdot', lw = 1, label = '$\hat{p}_{live}$')
            ax[r][c].plot(cbData.time_h[j],p_puti_percent*100, 'g', lw = 2, label = 'Est. through EKF, $\hat{p}$')
            fc_std = 5.3
            ax[r][c].errorbar(cbData.time_h[j][cbParam.sampcycle[j]-cbParam.sampcycle[j][0]],100-cbParam.cb_fc_ec[j], yerr = fc_std*2, fmt = 'x', markersize = 10, color = 'g', capsize = 3, label = 'Flow Cytometry Data, $p_{fc}$ with $2 \sigma_{fc}$')

            ax[r][c].legend(loc="best")
            if not paper:
                axr.legend(loc="lower right")
            if c == 0:
                ax[r][c].set_ylabel("Relative P. putida Abundance [%]")
            if c == n_culumns-1:
                axr.set_ylabel('Temperature [°C]', color='r')
                yticks = np.array([29, parameters['crit_temp'], 36])
                axr.set_yticks(yticks, labels=yticks)
                axr.tick_params(axis='y', color='r', labelcolor='r')
            else:
                axr.tick_params(axis='y', color='r', labelright=False)
            axr.text(1, 1, 'E. coli Dominates',
                    horizontalalignment='right',
                    verticalalignment='top',
                    transform=axr.transAxes,
                    color='r',
                    bbox={'facecolor': 'red', 'alpha': 0, 'pad': 0, 'edgecolor': 'r'})
            axr.text(1, 0, 'P. putida Dominates',
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    transform=axr.transAxes,
                    color='r',
                    bbox={'facecolor': 'red', 'alpha': 0, 'pad': 0, 'edgecolor': 'r'})
            if r == n_rows-1:
                ax[r][c].set_xlabel("Time [h]")
            ax[r][c].set_xlim([cbData.time_h[j][0]-0.5,cbData.time_h[j][-1]+0.5])
            ax[r][c].set_ylim([0,100])
            axr.set_ylim([28,37])
            if not paper:
                ax[r][c].set_title(cbParam.titles[j])

    if not paper:
        fig.suptitle(data_name)
    fig.tight_layout()
    
    # hm, hm_ax = plt.subplots(n_plots, 1, figsize=(14, 10), sharey=True)

    results_dir = "Images/{}".format(data_name)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if paper:
        fig.savefig('/Users/janmorlock/Documents/Ausbildung/Master/MasterProject/FiguresCDC/3_estResults.pdf', transparent=True)
    else:
        fig.savefig(results_dir+"/ekf_{}r{}.png".format(cbParam.n_reactors, '_test_est' if test_est else ''),transparent=True)