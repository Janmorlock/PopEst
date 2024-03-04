import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib
import os

from paramsData import CbDataParam
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
        gr = self.model.getGrowthRates(np.full(3,x))
        z = np.array([y - gr[0],
                    y - gr[1]])
        return z

    def getCritTemp(self):
        temp = fsolve(self.f, [33.0, 0.8])
        return temp[0]


if __name__ == "__main__":

    # SPECIFY DATA
    data_name = '081-3'
    test_est = False
    single_est = {'od_update':      [True, True, True],
                    'fl_update':    [True, False, False],
                    'od_gr_update': [False, True, False],
                    'fl_gr_update': [False, False, True]}
    cbParam = CbDataParam(data_name)
    # cbParam.n_reactors = 1
    cbData = CbData(cbParam)

    len_test = len(single_est['od_update']) if test_est else 1
    parameters = Params().default

    cT = CritTemp()
    critTemp = cT.getCritTemp()
    assert(29 < critTemp and critTemp < 36)

    # dim = reactor, time, state
    estimates_pred = [np.empty((len_test,len(cbData.time[j])),dtype=dict) for j in range(cbParam.n_reactors)]
    estimates = [np.empty((len_test,len(cbData.time[j])),dtype=dict) for j in range(cbParam.n_reactors)]
    variances = [np.empty((len_test,len(cbData.time[j]),3,3)) for j in range(cbParam.n_reactors)]
    pp_rel_od_arr = [np.empty((len_test,len(cbData.time[j]))) for j in range(cbParam.n_reactors)]
    pp_rel_fl_arr = [np.empty((len_test,len(cbData.time[j]))) for j in range(cbParam.n_reactors)]
    for j in range(cbParam.n_reactors):
        for t in range(len_test):
            # Construct State estimator
            est_pred = EKF(dev_ind = j, update = False)
            est = EKF(dev_ind = j)
            est.model.dithered = bool(cbData.dil)
            if test_est:
                est.model.parameters['od_update'] = single_est['od_update'][t]
                est.model.parameters['fl_update'] = single_est['fl_update'][t]
                est.model.parameters['od_gr_update'] = single_est['od_gr_update'][t]
                est.model.parameters['fl_gr_update'] = single_est['fl_gr_update'][t]
            est.set_r_coeff(cbParam.reactors[j])
            for k in range(len(cbData.time[j])):
                # Run the filter
                dil = cbData.dil[j][k] if cbData.dil else 0
                u_pred = [cbData.temp[j][k], dil]
                est_pred.estimate(cbData.time[j][k], u_pred)
                estimates_pred[j][t,k] = est_pred.est.copy()
                u = np.array([cbData.temp[j][k], dil])
                y = np.array([cbData.od[j][k], cbData.fl[j][k]*cbData.b1[j][k]])
                est.estimate(cbData.time[j][k], u, y)
                estimates[j][t,k] = est.est.copy()
                variances[j][t,k] = est.var.copy()
                pp_rel_od_arr[j][t,k] = est.p_est_od/cbData.od[j][k]
                pp_rel_fl_arr[j][t,k] = est.p_est_fl/cbData.od[j][k]

                if k == len(cbData.time[j])-1:
                    print("Final variance [{}][{}]: {}".format(j, t, variances[j][t,k]))

    e = [(cbData.fl[j]*cbData.b1[j] - parameters['e_ofs'][cbParam.reactors[j]])/parameters['e_fac'][cbParam.reactors[j]] + parameters['e_ofs'][cbParam.reactors[j]] for j in range(cbParam.n_reactors)]

    # ANALYSIS
    # cbParam.n_reactors = 1
    err = np.empty((cbParam.n_reactors,len_test,len(cbParam.cb_fc_ec[j])))
    n_plots = cbParam.n_reactors * len_test
    n_rows = cbParam.n_reactors
    n_culumns = len_test
    matplotlib.style.use('default')
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots(n_rows,n_culumns,sharey='all')
    # Set font to Times New Roman
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

            axr = ax[r][c].twinx()
            ax[r][c].set_zorder(2)
            axr.set_zorder(1)
            ax[r][c].patch.set_visible(False)

            axr.hlines(critTemp,cbData.time_h[j][0]-1,cbData.time_h[j][-1]+1,'r',lw=0.5,alpha=0.5)
            axr.plot(cbData.time_h[j],cbData.temp_sp[j],'--r',lw=0.5,alpha=0.5, label = '$temp_{sp}$')
            axr.plot(cbData.time_h[j],cbData.temp[j],'r',lw=0.5,alpha=0.5, label = '$temp_{meas}$')

            ax[r][c].plot(cbData.time_h[j],e[j]/max_e,'.', color = '#0000ff',markersize = 0.5,label = '$fl_{meas}$')
            # ax[r][c].plot(cbData.time_h[j],(fp_pred + parameters['od_fac']*od_pred + parameters['e_ofs'][cbParam.reactors[j]])/max_e,'--',color='#0000ff',lw = 0.5, label = '$fl_{pred}$')
            ax[r][c].plot(cbData.time_h[j],(fp + parameters['od_fac']*od + parameters['e_ofs'][cbParam.reactors[j]])/max_e,color = '#0000ff',lw = 0.5, label = '$fl_{est}$')
            ax[r][c].plot(cbData.time_h[j],cbData.od[j],'.k',markersize = 0.5, alpha = 0.5, label = '$od_{meas}$')
            ax[r][c].plot(cbData.time_h[j],od,'k',lw = 0.5, alpha = 0.5, label = '$od_{est}$')
            ax[r][c].plot(cbData.time_h[j][pp_rel_od_arr[j][t] > 0],pp_rel_od_arr[j][t,pp_rel_od_arr[j][t] > 0],'+g',markersize = 6, alpha = 0.7, label = '$pp_{od}$')
            ax[r][c].plot(cbData.time_h[j][pp_rel_fl_arr[j][t] > 0],pp_rel_fl_arr[j][t,pp_rel_fl_arr[j][t] > 0],'+',color = '#0000FF',markersize = 6, alpha = 0.7, label = '$pp_{fl}$')

            # ax[r][c].fill_between(cbData.time_h[j], p_puti_percent-p_puti_per_sigma, p_puti_percent+p_puti_per_sigma, color='g',alpha=0.2)
            # ax[r][c].plot(cbData.time_h[j],(cbData.fl[j]-parameters['min_fl'][j])/(parameters['max_e'][j]-parameters['min_fl'][j]),'.g',markersize = 0.8, alpha = 0.5, label = '$puti_{est,old}$')
            ax[r][c].plot(cbData.time_h[j],p_puti_pred_percent, 'g', lw = 0.5, label = '$pp_{pred}$')
            ax[r][c].plot(cbData.time_h[j][cbData.p_targ[j] > 0],cbData.p_targ[j][cbData.p_targ[j] > 0], '--g', lw = 1, label = '$pp_{targ}$')
            ax[r][c].plot(cbData.time_h[j],cbData.p_est[j], 'g', linestyle = 'dashdot', lw = 1.4, label = '$pp_{est,live}$')
            ax[r][c].plot(cbData.time_h[j],p_puti_percent, 'g', lw = 1.4, label = '$pp_{est}$')
            ax[r][c].plot(cbData.time_h[j][cbParam.sampcycle[j]-cbParam.sampcycle[j][0]],1-cbParam.cb_fc_ec[j]/100, 'gx', markersize = 10, label = '$pp_{fc}$')

            ax[r][c].legend(loc="upper left")
            axr.legend(loc="lower right")
            if c == 0:
                ax[r][c].set_ylabel("Normalized abunance")
            if c == n_culumns-1:
                axr.set_ylabel('Temperature [°C]', color='r')
                axr.set_yticks(np.append(axr.get_yticks(), critTemp))
                axr.tick_params(axis='y', color='r', labelcolor='r')
                # ax[r][c].tick_params(axis='y', labelleft=True)
            else:
                axr.tick_params(axis='y', color='r', labelright=False)
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
            if r == n_rows-1:
                ax[r][c].set_xlabel("Time [h]")
            ax[r][c].set_xlim([cbData.time_h[j][0]-0.5,cbData.time_h[j][-1]+0.5])
            ax[r][c].set_ylim([0,1.05])
            axr.set_ylim([28,37])
            ax[r][c].set_title(cbParam.titles[j])

            err[j,t] = abs(1-cbParam.cb_fc_ec[j]/100 - p_puti_percent[cbParam.sampcycle[j]-cbParam.sampcycle[j][0]])
    fig.suptitle(data_name)
    fig.tight_layout()
    
    hm, hm_ax = plt.subplots(n_plots, 1, figsize=(14, 10), sharey=True)

    results_dir = "Images/{}".format(data_name)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    fig.savefig(results_dir+"/ekf_{}r{}_opt.png".format(cbParam.n_reactors, '_test_est' if test_est else ''),transparent=True)