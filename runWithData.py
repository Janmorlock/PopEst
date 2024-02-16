import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib
import math
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
    data_name = '064-2-test'
    cbParam = CbDataParam(data_name)
    cbData = CbData(cbParam.path, cbParam.file_ind, cbParam.sampcycle, cbParam.n_reactors)

    parameters = Params().default

    cT = CritTemp()
    critTemp = cT.getCritTemp()
    assert(26 < critTemp and critTemp < 37)

    # dim = reactor, time, state
    estimates_pred = [[] for j in range(cbParam.n_reactors)]
    estimates = [[] for j in range(cbParam.n_reactors)]
    variances = [[] for j in range(cbParam.n_reactors)]
    for j in range(cbParam.n_reactors):
        # Construct State estimator
        est_pred = EKF(dev_ind = j, update = False)
        est = EKF(dev_ind = j)
        est.set_r_coeff('Faith')
        for k in range(len(cbData.time[j])):
            # Run the filter
            u_pred = cbData.temp[j][k]
            est_pred.estimate(cbData.time[j][k], u_pred)
            estimates_pred[j].append(est_pred.est)
            dil = cbData.dil[j][k] if cbData.dil else 0
            u = np.array([cbData.temp[j][k], dil])
            y = np.array([cbData.od[j][k], cbData.fl[j][k]*cbData.b1[j][k]])
            est.estimate(cbData.time[j][k], u, y)
            estimates[j].append(est.est)
            variances[j].append(est.var)

            if k == len(cbData.time[j])-1:
                print("Final variance ({}): {}".format(j, variances[j][k]))


    # ANALYSIS
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
    for j in range(cbParam.n_reactors):
        r = j//2
        c = j%2
        # j = 1
        e_coli_pred = np.array([estimates_pred[j][k]['e'] for k in range(len(estimates_pred[j]))])
        p_puti_pred = np.array([estimates_pred[j][k]['p'] for k in range(len(estimates_pred[j]))])
        fp_pred = np.array([estimates_pred[j][k]['fp'] for k in range(len(estimates_pred[j]))])
        od_pred = e_coli_pred + p_puti_pred
        p_puti_pred_percent = p_puti_pred/od_pred*100

        e_coli = np.array([estimates[j][k]['e'] for k in range(len(estimates[j]))])
        p_puti = np.array([estimates[j][k]['p'] for k in range(len(estimates[j]))])
        var = np.array([variances[j][k][:2,:2] for k in range(len(variances[j]))])
        fp = np.array([estimates[j][k]['fp'] for k in range(len(estimates[j]))])
        od = e_coli + p_puti
        p_puti_percent = p_puti/od*100
        sq = e_coli**2/p_puti**2*var[:,1,1] + var[:,0,0] - 2*e_coli/p_puti*np.sign(var[:,0,1])*np.sqrt(np.abs(var[:,0,1]))
        p_puti_per_sigma = np.abs(p_puti_percent/(e_coli+p_puti))*np.sqrt(np.abs(sq))

        max_fl = max(cbData.fl[j]*cbData.b1[j])/100

        axr = ax[r][c].twinx()
        ax[r][c].set_zorder(2)
        axr.set_zorder(1)
        ax[r][c].patch.set_visible(False)

        axr.plot(cbData.time_h[j],cbData.temp[j],'--r',lw=0.5,alpha=0.5)
        axr.hlines(critTemp,cbData.time_h[j][0]-1,cbData.time_h[j][-1]+1,'r',lw=0.5,alpha=0.5)

        ax[r][c].plot(cbData.time_h[j][cbParam.sampcycle[j]-cbParam.sampcycle[j][0]],100-cbParam.cb_fc_ec[j], 'gx', markersize = 10, label = '$puti_{fc}$')
        ax[r][c].plot(cbData.time_h[j],(cbData.fl[j]-parameters['min_fl'][j])/(parameters['max_fl'][j]-parameters['min_fl'][j])*100,'.g',markersize = 0.8, alpha = 0.5, label = '$puti_{est,old}$')
        ax[r][c].plot(cbData.time_h[j],p_puti_pred_percent, '--g', lw = 0.8, label = '$puti_{pred}$')
        ax[r][c].plot(cbData.time_h[j],p_puti_percent, 'g', lw = 1.2, label = '$puti_{est}$')

        # ax[r][c].plot(cbData.time_h[j],cbData.fl[j]*100,'.m',markersize = 0.8,label = '$fl_{meas}*100$')
        # ax[r][c].plot(cbData.time_h[j],(fp_pred/(od_pred+parameters['od_ofs']) + parameters['fl_ofs'][j])*100,'--',color='#0000ff',lw = 0.8, label = '$fl_{pred}*100$')
        # ax[r][c].plot(cbData.time_h[j],(fp/(od+parameters['od_ofs']) + parameters['fl_ofs'][j])*100,color = '#0000ff',lw = 1.2, label = '$fl_{est}*100$')

        ax[r][c].plot(cbData.time_h[j],cbData.fl[j]*cbData.b1[j]/max_fl,'.m',markersize = 0.8,label = '$fl_{meas}$')
        ax[r][c].plot(cbData.time_h[j],(fp_pred + parameters['od_fac']*od_pred + parameters['e1_ofs'])/max_fl,'--',color='#0000ff',lw = 0.8, label = '$fl_{pred}$')
        ax[r][c].plot(cbData.time_h[j],(fp + parameters['od_fac']*od + parameters['e1_ofs'])/max_fl,color = '#0000ff',lw = 1.2, label = '$fl_{est}$')

        ax[r][c].plot(cbData.time_h[j],cbData.od[j]*100,'.k',markersize = 0.8, alpha = 0.5, label = '$od_{meas}*100$')
        ax[r][c].plot(cbData.time_h[j],od*100,'k',lw = 0.5, alpha = 1, label = '$od_{est}*100$')
        # ax[r][c].fill_between(cbData.time_h[j], p_puti_percent-p_puti_per_sigma, p_puti_percent+p_puti_per_sigma, color='g',alpha=0.2)

        ax[r][c].legend(loc="upper left")
        if (j%2 == 0):
            ax[r][c].set_ylabel("Relative composition [%]")
        else:
            axr.set_ylabel('Temperature [°C]', color='r')
            ax[r][c].tick_params(axis='y', labelleft=True)
        axr.set_yticks(np.append(axr.get_yticks(), critTemp))
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
        ax[r][c].set_xlim([cbData.time_h[j][0]-0.5,cbData.time_h[j][-1]+0.5])
        ax[r][c].set_ylim([-5,105])
        ax[r][c].set_title(cbParam.titles[j])
    fig.suptitle(data_name)
    fig.tight_layout()
    results_dir = "Images/{}".format(data_name)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    fig.savefig(results_dir+"/ekf_{}r_e.png".format(cbParam.n_reactors),transparent=True)