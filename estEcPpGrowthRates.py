import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
import os

from CbData import CbData
from CbDataParams import CbDataParam
from runWithData import CritTemp
from config.params import Params
from estFlProdRate import get_prs

if __name__ == "__main__":

    dataName = '075-1'
    bac = [0,8]
    batch_size = 8
    cbParam = CbDataParam(dataName)
    results_dir = "Images/{}".format(dataName)
    paper = True
    symbol = False

    cbData = CbData(cbParam)
    temps = []
    mean = [[] for b in range(len(bac))]
    median = [[] for b in range(len(bac))]
    std = [[] for b in range(len(bac))]

    boxdata = [[] for b in range(len(bac))]

    delay = [3, 0]

    matplotlib.style.use('default')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 11})
    n_rows = math.ceil(cbParam.n_reactors/2)
    n_culumns = 2 if cbParam.n_reactors > 1 else 1
    fig, ax = plt.subplots(n_rows,2,sharex='all',sharey='all')
    fig.set_figheight(n_rows*7)
    fig.set_figwidth(n_culumns*10)
    if n_culumns == 1:
        if n_rows == 1:
            ax = [ax]
        else:
            ax = [[ax[i]] for i in range(n_rows)]
    if n_rows == 1:
        ax = [ax]
    
    # for each reactor
    for b in range(len(bac)):
        temp_sp = []
        gr = []
        for j in range(bac[b],batch_size+bac[b]):
            gr_fit = []
            od_mat = []
            temp_sp_mat = []
            time_h_mat = []
            time_h = []
            # Resize data
            diluted = False
            od_list = []
            time_h_list = []
            temp_sp_beg = cbData.temp_sp[j][0]
            gr_constant = False
            time_beg = 0
            for i in range(len(cbData.time[j])):
                if cbData.temp_sp[j][i] != cbData.temp_sp[j][max(0,i-1)]:
                    gr_constant = False
                    time_beg = cbData.time_h[j][i]
                if cbData.dil[j][i] and not cbData.dil[j][max(0,i-1)]:
                    od_list.append(cbData.od[j][i])
                    time_h_list.append(cbData.time_h[j][i])
                    if diluted and gr_constant: # don't get gradient before first dilution, after temperature change
                        od_mat.append(od_list)
                        time_h_mat.append(time_h_list)
                        temp_sp_mat.append(temp_sp_beg)
                    diluted = True
                    od_list = []
                    time_h_list = []
                    temp_sp_beg = cbData.temp_sp[j][i]
                    if cbData.time_h[j][i] - time_beg > delay[b]:
                        gr_constant = True
                if gr_constant and not cbData.dil[j][i]:
                    od_list.append(cbData.od[j][i])
                    time_h_list.append(cbData.time_h[j][i])
            temps += list(set(temp_sp_mat))
            # Fit lines and get growth rates
            for r in range(len(temp_sp_mat)):
                if len(od_mat[r]) > 5:
                    fit = np.poly1d(np.polyfit(time_h_mat[r], np.log(od_mat[r]), 1, w = np.sqrt(od_mat[r])))
                    gr_fit.append(fit)
                    gr.append(fit.coefficients[0])
                    temp_sp.append(temp_sp_mat[r])
                    time_h.append(time_h_mat[r])
            
            # Plot gradients
            r = j//2
            c = j%2
            axr = ax[r][c].twinx()
            ax[r][c].set_zorder(2)
            axr.set_zorder(1)
            ax[r][c].patch.set_visible(False)

            axr.plot(cbData.time_h[j],cbData.temp[j],'r',lw=0.5, alpha = 0.5)
            ax[r][c].plot(cbData.time_h[j],np.log(cbData.od[j]),'k',lw = 0.5, alpha = 1, label = 'log(od)')
            ax[r][c].plot(time_h[0],gr_fit[0](time_h[0]),'b',lw = 0.5, alpha = 1, label = 'gradients')
            for row in range(1, len(gr_fit)):
                ax[r][c].plot(time_h[row],gr_fit[row](time_h[row]),'b',lw = 0.5, alpha = 1)

            ax[r][c].legend(loc="upper left")
            if (c == 0):
                ax[r][c].set_ylabel("Optical Density")
            else:
                axr.set_ylabel('Temperature [°C]', color='r')
                ax[r][c].tick_params(axis='y', labelleft=True)
            if r == n_rows-1:
                ax[r][c].set_xlabel("Time [h]")
            axr.tick_params(axis='y', color='r', labelcolor='r')
            ax[r][c].set_ylim([-1,-0.5])
            # ax[r][c].set_xlim([10,20])
            axr.set_ylim([28,37])
            ax[r][c].set_title(cbParam.titles[j])

        # Sort gr according to temperature
        temps = list(set(temps))
        temps.sort()
        grs = [[] for t in range(len(temps))]
        for r in range(len(temp_sp)):
            grs[temps.index(temp_sp[r])].append(gr[r])

        boxdata[b] = grs
        mean[b] = [np.mean(grs[t]) for t in range(len(grs))]
        median[b] = [np.median(grs[t]) for t in range(len(grs))]
        std[b] = [np.std(grs[t]) for t in range(len(grs))]
    # Fit growth rates
    p_model = np.poly1d(np.polyfit(temps, mean[0], 2, w=1/np.array(std[0])))
    e_model = np.poly1d(np.polyfit(temps, mean[1], 1, w=1/np.array(std[1])))
    p_coefficients = np.round(p_model.coefficients, 5)
    e_coefficients = np.round(e_model.coefficients, 5)
    print("P. putida growth rate fit:")
    print(*p_coefficients, sep = ", ")
    print("E. coli growth rate fit:")
    print(*e_coefficients, sep = ", ")
    cT = CritTemp(e_coefficients, p_coefficients)
    critical_temp = round(cT.getCritTemp(),2)
    print("Critical temperature: ", critical_temp)

    parameters = Params().default
    cbParam.n_reactors = 8
    cbParam.file_ind = cbParam.file_ind[0:cbParam.n_reactors]
    cbParam.sampcycle[0][0] = 700
    cbParam.sampcycle[1] = [0,100]
    cbData = CbData(cbParam)
    e_ofs = [parameters['e_ofs'][cbParam.reactors[j]] for j in range(cbParam.n_reactors)]
    e_fac = [parameters['e_fac'][cbParam.reactors[j]] for j in range(cbParam.n_reactors)]
    e = [(cbData.fl[j]*cbData.b1[j] - e_ofs[j])/e_fac[j] + e_ofs[j] for j in range(cbParam.n_reactors)]
    prs = get_prs(cbData, cbParam, temps, parameters, e)
    mean_fl = [np.mean(prs[t]) for t in range(len(prs))]
    std_fl = [np.std(prs[t]) for t in range(len(prs))]
    media_fl = [np.median(prs[t]) for t in range(len(prs))]
    pr_model4 = np.poly1d(np.polyfit(np.array(temps)-32.5, mean_fl, 4, w = 1/np.array(std_fl)))

    fig_gr, (ax, pr_ax) = plt.subplots(2,1,height_ratios=[3, 2],sharex='all')
    if symbol:
        fig_gr, ax = plt.subplots(1,1,sharex='all')
        fig_gr.set_figheight(1.5)
        fig_gr.set_figwidth(1.5)
    else:
        plt.subplots_adjust(hspace=0.09) # 0.09
        fig_gr.set_figheight(4) #6
        fig_gr.set_figwidth(4) #7
        # Plot gradients
        bp_p = ax.boxplot(boxdata[0], positions=temps, widths=0.4, showfliers=False, showmeans=True, patch_artist=True,
                    meanprops=dict(markerfacecolor = 'g', markeredgecolor = 'g'),
                    boxprops=dict(color="g", alpha=0.5),
                    medianprops=dict(color="g", alpha=0.5),
                    flierprops=dict(markeredgecolor="g", alpha=0.5),
                    capprops=dict(color="g", alpha=0.5),
                    whiskerprops=dict(color="g", alpha=0.5))
        bp_e = ax.boxplot(boxdata[1], positions=temps, widths=0.4, showfliers=False, showmeans=True, patch_artist=True,
                    meanprops=dict(markerfacecolor = 'm', markeredgecolor = 'm'),
                    boxprops=dict(color="m", alpha=0.5),
                    medianprops=dict(color="m", alpha=0.5),
                    flierprops=dict(markeredgecolor="m", alpha=0.5),
                    capprops=dict(color="m", alpha=0.5),
                    whiskerprops=dict(color="m", alpha=0.5))
        bp_fl = pr_ax.boxplot(prs, positions=temps, widths=0.4, showfliers=False, showmeans=True, patch_artist=True,
                            meanprops=dict(markerfacecolor = '#0000ff', markeredgecolor = '#0000ff'),
                            boxprops=dict(color = '#0000ff'),
                            medianprops=dict(color = '#0000ff'),
                            flierprops=dict(markeredgecolor = '#0000ff'),
                            capprops=dict(color = '#0000ff'),
                            whiskerprops=dict(color = '#0000ff'))
        # fill with colors
        for bp in [bp_p, bp_e, bp_fl]:
            for patch in bp['boxes']:
                patch.set_facecolor((1,1,1,0))

    # Plot fit
    x = np.linspace(temps[0],temps[-1],100)
    ax.plot(x, p_model(x), 'g', linewidth=2, label = 'Polynomial Fit')
    ax.plot(x, e_model(x), 'm', linewidth=2, label = 'Polynomial Fit')
    if symbol:
        ax.set_xlabel('Temperature')
        ax.set_ylabel('Growth Rate')
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        pr_ax.plot(x, np.maximum(pr_model4(x-32.5),media_fl[-1]), '#0000ff', linewidth=2, label = 'Polynomial Fit')
        xticks = list(set(np.int16(pr_ax.get_xticks())))# + [critical_temp]
        xticks_lb = ['29', '30', '31', '32', '33', '34', '35', '36']
        xticks.sort()
        pr_ax.set_xticks(xticks, labels=xticks_lb)
        y_ticks = [0,4e3,8e3]
        y_ticks_lb = ['0', '4e3', '8e3']
        pr_ax.set_yticks(y_ticks, labels=y_ticks_lb)
        ax.set_ylabel(r'Growth Rate $[\frac{1}{h}]$')
        pr_ax.set_xlabel(r'Temperature $[^{\circ}C]$')
        pr_ax.set_ylabel(r'Production Rate $[\frac{1}{h}]$')
        h,l = ax.get_legend_handles_labels()
        h = [bp_p["boxes"][0], bp_e["boxes"][0], *h]
        l = [r'P. putida, $\mu_{P,meas}$', r'E. coli, $\mu_{E,meas}$', *l]
        ax.legend(h,l, loc='best')
        h,l = pr_ax.get_legend_handles_labels()
        h = [bp_fl["boxes"][0], *h]
        l = [r'Pyoverdine, $\mu_{F,meas}$', *l]
        pr_ax.legend(h,l, loc='best')
        pr_ax.set_ylim([0,8000])

    ax.set_ylim([0,1.3])
    fig_gr.align_ylabels()
    fig_gr.tight_layout()
    # Save figures
    fig.suptitle(dataName)
    # fig.tight_layout()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if paper:
        # fig_gr.savefig('/Users/janmorlock/Documents/Ausbildung/Master/MasterProject/FiguresCDC/2_productionRates.pdf', transparent=True)
        if symbol:
            fig_gr.savefig('/Users/janmorlock/Documents/Ausbildung/Master/MasterProject/FiguresPresentation/SymbolProductionRates.pdf', transparent=True)
        fig_gr.savefig('/Users/janmorlock/Documents/Ausbildung/Master/MasterProject/FiguresPresentation/ProductionRates.pdf', transparent=True)
    else:
        fig.savefig(results_dir + "/odGradient.pdf", transparent=True)