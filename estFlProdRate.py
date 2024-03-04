from cProfile import label
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import os

from config.params import Params
from CbData import CbData
from paramsData import CbDataParam

def simulateFlProtein(cb_hrs, fp_init, p_puti, dil, temp, parameters, temp_lst = np.empty, gr = np.empty, train = False, fp = np.empty(1), err = []):
    data_l = len(p_puti)
    n_s = gr.shape[1] if train else 1
    x = np.zeros((data_l,n_s))
    x_curr = fp_init
    k = 0
    dil_th = 0.55
    dil_am = 0.12
    temp_ind = 0
    dilute = False

    for k in range(len(p_puti)):
        if train:
            if temp[k] == 0:
                temp[k] = temp[k-1]
            temp_ind = np.where(temp[k] == temp_lst)[0][0] 
        if dilute:
            x_curr -= dil_am/dil_th*x_curr
        # Euler forward (not significantly less accurate than Runge - Kutta 4th order)
        dt = cb_hrs[k] - cb_hrs[max(0,k-1)]
        if train:
            x_curr = x_curr + dt*gr[temp_ind]*p_puti[k]
        else:
            gr = parameters['gr_fp'][0]*temp[k]**2 + parameters['gr_fp'][1]*temp[k] + parameters['gr_fp'][2]
            gr = max(gr,100)
            x_curr = x_curr + dt*gr*p_puti[k]
        x[k] = x_curr
        if train:
            err[temp_ind].append(x_curr-fp[k])
        
        if dil[k] and not dil[max(0,k-1)]:
            dilute = True
        else:
            dilute = False
            
    return x

### Explicit production rate estimation
def get_prs(cbData: CbData, cbParam: CbDataParam, temp_lst: list, parameters: dict, e: list):
    
    temp_sp = []
    gr = []
    for j in range(cbParam.n_reactors):
        time_h_list = []
        od_list = []
        dfl_list = []
        dp_mat = []
        dfl_mat = []
        temp_sp_list = []
        diluted = False
        gr_constant = True
        time_beg = 0
        for i in range(len(cbData.time[j])):
            if cbData.temp_sp[j][i] != cbData.temp_sp[j][max(0,i-1)]:
                gr_constant = False
                time_beg = cbData.time_h[j][i]
            if cbData.time_h[j][i] - time_beg > 2.5 and not gr_constant:
                gr_constant = True
            if cbData.dil[j][i] and not cbData.dil[j][max(0,i-1)]:
                dfl_list.append(e[j][i] - cbData.od[j][i]*parameters['od_fac'])
                od_list.append(cbData.od[j][i])
                time_h_list.append(cbData.time_h[j][i])
                if diluted and gr_constant: # don't get gradient before first dilution, after temperature change
                    p_ln_fit = np.polyfit(time_h_list-time_h_list[0], np.log(od_list), 1, w = np.sqrt(od_list)) # fit exponential, adjusted weight to account for transformation
                    mu = p_ln_fit[0]
                    p0 = np.exp(p_ln_fit[1])
                    dp_list = p0/mu*(np.exp(mu*np.array(time_h_list-time_h_list[0]))-1)# integrate exp
                    dfl_mat.append(dfl_list)
                    dp_mat.append(np.array(dp_list))
                    temp_sp_list.append(cbData.temp_sp[j][i])
                diluted = True
                dfl_list = []
                time_h_list = []
                od_list = []
            if gr_constant and not cbData.dil[j][i]:
                dfl_list.append(e[j][i] - cbData.od[j][i]*parameters['od_fac'])
                od_list.append(cbData.od[j][i])
                time_h_list.append(cbData.time_h[j][i])
        # Fit lines and get growth rates
        for r in range(len(temp_sp_list)):
            if len(dfl_mat[r]) > 5:
                gr.append(np.polyfit(dp_mat[r],dfl_mat[r],1)[0])
                temp_sp.append(temp_sp_list[r])
    # Sort gr according to temperature
    prs = [[] for t in range(len(temp_lst))]
    for r in range(len(temp_sp)):
        prs[temp_lst.index(temp_sp[r])].append(gr[r])
    
    return prs

if __name__ == "__main__":
    # SPECIFY DATA
    dataName = '075-1'
    cbParam = CbDataParam(dataName)
    cbParam.n_reactors = 8
    cbParam.file_ind = cbParam.file_ind[0:cbParam.n_reactors]
    cbParam.sampcycle[0][0] = 700
    cbParam.sampcycle[1] = [0,100]
    train = False
    n_samples = 10000

    cbData = CbData(cbParam)
    # Get occuring temperatures
    temp = []
    for j in range(cbParam.n_reactors):
        temp += list(set(cbData.temp_sp[j]))
    temp_lst = sorted(list(set(temp)))
    if temp_lst[0] == 0:
        temp_lst.remove(0)

    results_dir = "Images/{}".format(dataName)
    parameters = Params().default
    e_ofs = [parameters['e_ofs'][cbParam.reactors[j]] for j in range(cbParam.n_reactors)]
    e_fac = [parameters['e_fac'][cbParam.reactors[j]] for j in range(cbParam.n_reactors)]
    e = [(cbData.fl[j]*cbData.b1[j] - e_ofs[j])/e_fac[j] + e_ofs[j] for j in range(cbParam.n_reactors)]

    matplotlib.style.use('default')
    gr_fig, gr_ax = plt.subplots()
    gr_fig.set_figheight(7)
    gr_fig.set_figwidth(10)
            
    ### Brute-force search of production rates
    sim_e_train, sim_fl_opt, pr_fp_opt = [], [], []
    s_min = np.zeros(len(temp_lst), dtype=int)
    fp_init = [e[j][0] - parameters['od_fac']*cbData.od[j][0] - e_ofs[j] for j in range(cbParam.n_reactors)]
    if train:
        low = np.array([3000 if temp_lst[t] < 35 else 0 for t in range(len(temp_lst))])
        high = np.array([8000 if temp_lst[t] < 35 else 1000 for t in range(len(temp_lst))])
        gr_fp = np.random.uniform(low, high, (n_samples,len(temp_lst))).T
        err = [[] for t in range(len(temp_lst))]
        for j in range(cbParam.n_reactors):
            print("Training at reactor: ", cbParam.titles[j])
            sim_fp = simulateFlProtein(cbData.time_h[j],fp_init[j], cbData.od[j], cbData.dil[j], cbData.temp_sp[j], parameters, temp_lst, gr_fp, True, e[j] - e_ofs[j] - parameters['od_fac']*cbData.od[j], err).T
            sim_e_train.append(sim_fp + parameters['od_fac']*cbData.od[j] + e_ofs[j])
        
        for t in range(len(temp_lst)):
            rmse = np.sqrt(np.mean(np.array(err[t])**2,axis=0))
            s_min[t] = np.argmin(rmse)
        sim_fl_opt = [[sim_e_train[j][s_min[temp_lst.index(cbData.temp_sp[j][i])]][i] for i in range(len(cbData.temp_sp[j]))] for j in range(cbParam.n_reactors)]
        # Print obtrained parameters
        pr_fp_opt = np.array([gr_fp[t][s_min[t]] for t in range(len(temp_lst))])
        print("Brute-Force Production Rates:")
        print(*pr_fp_opt, sep = ", ")

        gr_ax.plot(temp_lst, pr_fp_opt, 'x', markersize = 7, color = '#0000ff', label = 'Brute-Force')


    ### Explicit production rate estimation
    prs = get_prs(cbData, cbParam, temp_lst, parameters, e)

    boxdata = prs
    median = [np.median(prs[t]) for t in range(len(prs))]
    mean = [np.mean(prs[t]) for t in range(len(prs))]
    std = [np.std(prs[t]) for t in range(len(prs))]
    print("Explicit Production Rates:")
    print(*mean, sep = ", ")
    bp = gr_ax.boxplot(boxdata, positions=temp_lst, widths=0.4, showfliers=False, showmeans=True, patch_artist=True,
                        meanprops=dict(markerfacecolor = '#0000ff', markeredgecolor = '#0000ff'),
                        boxprops=dict(color = '#0000ff'),
                        medianprops=dict(color = '#0000ff'),
                        flierprops=dict(markeredgecolor = '#0000ff'),
                        capprops=dict(color = '#0000ff'),
                        whiskerprops=dict(color = '#0000ff'))
    # fill with colors
    for patch in bp['boxes']:
        patch.set_facecolor((0,0,0,0))

    ### Plot both production rate estimates
    gr_model2 = np.poly1d(np.polyfit(np.array(temp_lst[:-1])-32.5, mean[:-1], 2, w = 1/np.array(std[:-1])))
    gr_model4 = np.poly1d(np.polyfit(np.array(temp_lst)-32.5, mean, 4, w = 1/np.array(std)))
    print("Production rate model:")
    model_coeff = np.round(gr_model2.coefficients, 5)
    print(*model_coeff, sep = ", ")
    print(*gr_model4.coefficients, sep = ", ")
    print(median[-1])
    x_mod = np.linspace(29, 36, 110)
    y2 = np.maximum(gr_model2(x_mod-32.5),median[-1])
    y4 = np.maximum(gr_model4(x_mod-32.5),median[-1])
    gr_ax.plot(x_mod, y2, '-', color = '#0000FF', label = '2nd order poly fit')
    gr_ax.plot(x_mod, y4, '-.', color = '#0000FF', label = '4th order poly fit')

    gr_ax.set_xlabel('Temperature [°C]')
    gr_ax.set_ylabel('Production Rate [1/h]')
    h,l = gr_ax.get_legend_handles_labels()
    h = [bp["boxes"][0], *h]
    l = ["Explicit", *l]
    gr_ax.legend(h,l, loc='best')
    gr_ax.set_title("Fluorescent Protein Production Rate Model")
    gr_fig.savefig(results_dir + "/pr_model_e.png", transparent=True)

    parameters['gr_fp'] = model_coeff
    sim_e_test = []
    print("Running with estimated parameters")
    for j in range(cbParam.n_reactors):
        sim_fp = simulateFlProtein(cbData.time_h[j], fp_init[j], cbData.od[j], cbData.dil[j], cbData.temp[j], parameters).T
        sim_e_test.append(sim_fp + e_ofs[j] + parameters['od_fac']*cbData.od[j])
    

    # ANALYSIS
    n_rows = math.ceil(cbParam.n_reactors/2)
    n_culumns = 2 if cbParam.n_reactors > 1 else 1
    gr_fig, ax = plt.subplots(n_rows,2,sharex='all',sharey='all')
    gr_fig.set_figheight(n_rows*7)
    gr_fig.set_figwidth(n_culumns*10)
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
        # ax[r][c].plot(cbData.time_h[j],cbData.od[j],'k',lw = 0.5, alpha = 0.4, label = 'p. putida od')
        if train:
            ax[r][c].plot(cbData.time_h[j],sim_e_train[j][0],'k',lw = 0.5, label = '$e_{sim, train}$', alpha = 0.1)
            for s in range(1,min(n_samples,10)):
                ax[r][c].plot(cbData.time_h[j],sim_e_train[j][s],'k',lw = 0.5, alpha = 0.1)
            ax[r][c].plot(cbData.time_h[j],sim_fl_opt[j],'m', lw = 0.5, alpha = 1, label = '$e_{sim, opt}$')
        ax[r][c].plot(cbData.time_h[j],sim_e_test[j][0],'b',lw = 1, label = '$e_{sim, model}$')
        ax[r][c].plot(cbData.time_h[j],e[j],'.g',markersize = 1, label = '$e_{meas}$')

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
        axr.set_ylim([28,37])
        ax[r][c].set_title(cbParam.titles[j])
    # TODO: Set titles
    gr_fig.suptitle(dataName)
    gr_fig.tight_layout()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    gr_fig.savefig(results_dir + "/fl_sim_e_{}.png".format("train" if train else "test"),transparent=True)