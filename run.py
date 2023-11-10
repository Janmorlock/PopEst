import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import glob
import pandas as pd
import math

from estimator import EKF
from model import getGrowthRates
from params import ExpParam, EstParam, ModelParam, CbDataParam, getFcData

def f(xy):
    x, y = xy
    gr = getGrowthRates(np.full(3,x))
    z = np.array([y - gr[0],
                  y - gr[1]])
    return z

def getCritTemp():
    temp = fsolve(f, [33.0, 0.8])
    return temp

def plotGrowthRates(expParam):
    temp = np.full((3,3),np.arange(expParam.T_l,expParam.T_h,1))
    gr = getGrowthRates(temp)
    critT = getCritTemp()[0]
    plt.plot(temp,gr[0],'-xb',label='E coli.')
    plt.plot(temp,gr[1],'-xg',label='P. Putida')
    plt.plot(temp,gr[2],'-xk',label='Fl. Protein')
    plt.vlines(critT,0,1,colors='r',label='Critical Temperature')
    plt.xlim(expParam.T_l-1,expParam.T_h+1)
    plt.xlabel("Temperature [°C]")
    plt.ylabel("Growth Rate [$h^{-1}$]")
    plt.legend()
    plt.savefig("Images/growthRates_<critT>.png")
    return

class CbData:
    def __init__(self, path, file_ind, scope, n_reactors):
        cb_files = sorted(glob.glob(path + "/*.csv"))
        cb_dfs = []
        for i in file_ind:
            df = pd.read_csv(cb_files[i], index_col=None, header=0)
            cb_dfs.append(df)

        self.time, self.time_h, self.od, self.temp, self.fl, self.p1 = [], [], [], [], [], []
        for j in range(n_reactors):
            time = cb_dfs[j]["exp_time"][scope[j][0]:scope[j][-1]+1].to_numpy()
            self.time.append(time[j]-time[j][0])
            self.time_h.append(self.time[j]/3600)
            od = cb_dfs[j]["od_measured"][scope[j][0]:scope[j][-1]+1].to_numpy()
            od[od < 0.005] = 0.005
            self.od.append(od)
            self.temp.append(cb_dfs[j]["media_temp"][scope[j][0]:scope[j][-1]+1].to_numpy())
            self.fl.append(cb_dfs[j]["FP1_emit1"][scope[j][0]:scope[j][-1]+1].to_numpy())
            self.p1.append(cb_dfs[j]["pump_1_rate"][scope[j][0]:scope[j][-1]+1].to_numpy())
        # self.time = [self.time[j]-self.time[j][0] for j in range(n_reactors)]

if __name__ == "__main__":

    # SPECIFY DATA
    data_name = '064-2'
    expParam = ExpParam()
    modelParam = ModelParam()
    cbParam = CbDataParam(data_name)
    cbData = CbData(cbParam.path, cbParam.file_ind, cbParam.cb_fc_ec, cbParam.n_reactors)
    # Initialize the estimator
    ekf = EKF(cbParam.n_reactors, expParam.Dil_dithered)

    # reactor, time, state
    estimates = [[] for j in range(cbParam.n_reactors)]
    variances = [[] for j in range(cbParam.n_reactors)]

    for j in range(cbParam.n_reactors):
        for k in range(1, len(cbData.time[j])):
            # Run the EKF
            pred, pred_variance = ekf.prediction(j, cbData.time[j][k], cbData.temp[j][k])

            estimates[j][k] = pred
            variances[j][k] = pred_variance

    plotGrowthRates(expParam)
    critTemp = getCritTemp()[0]
    assert(26 < critTemp and critTemp < 37)

    # ANALYSIS
    # cbParam.n_reactors = 1
    n_rows = math.ceil(cbParam.n_reactors/2)
    n_culumns = 2 if cbParam.n_reactors > 1 else 1
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
        od = estimates[j][:][0] + estimates[j][:][0]
        e_coli_percent = estimates[j][:][0]/od*100
        p_puti_percent = estimates[j][:][1]/od*100

        axr = ax[r][c].twinx()
        ax[r][c].set_zorder(2)
        axr.set_zorder(1)
        ax[r][c].patch.set_visible(False)

        axr.plot(cbData.time_h[j],cbData.temp[j],'--r',lw=0.5,alpha=0.4)
        # axr.plot(cbData.time_h[j],sim_tem_e[j],'r',lw=1)
        axr.hlines(critTemp,cbData.time_h[j][0]-1,cbData.time_h[j][-1]+1,'r',lw=0.5)
        ax[r][c].plot(cbData.time_h[j],e_coli_percent, 'b', label = 'e coli. sim')
        ax[r][c].plot(cbData.time_h[j],p_puti_percent, 'g', label = 'p. putida sim')
        ax[r][c].plot(cbData.time_h[j][cbParam.sampcycle[j]-cbParam.sampcycle[j][0]],cbParam.cb_fc_ec[j], 'b--x', label = 'e coli. fc')
        ax[r][c].plot(cbData.time_h[j][cbParam.sampcycle[j]-cbParam.sampcycle[j][0]],100-cbParam.cb_fc_ec[j], 'g--x', label = 'p. putida fc')
        ax[r][c].plot(cbData.time_h[j],cbData.fl[j]*100,'.k',markersize = 0.8, label = '$100*fl$')
        ax[r][c].plot(cbData.time_h[j],(estimates[j][:][2][j]/od + modelParam.min_fl[cbParam.file_ind[j]])*100,'m',lw = 0.5, label = '$100*fl_{sim}$')
        # ax[r][c].plot(time_all[j],od*100,'-k',lw = 0.5, label = 'od sim')
        # ax[r][c].plot(cbData.time_h[j],cb_od[j]*100,'--m',lw = 0.5, label = 'od')

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
    # TODO: Set titles
    fig.suptitle(data_name)
    fig.tight_layout()
    fig.savefig("Images/{}/{}r_{}h{}lag_fl_new.svg".format(data_name,cbParam.n_reactors,modelParam.Lag,'avg' if modelParam.Avg_temp else ''))