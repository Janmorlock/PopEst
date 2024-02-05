
from turtle import back
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
import os

from CbData import CbData
from paramsData import CbDataParam

if __name__ == "__main__":

    dataName = '075-1'
    bac = [0,8]
    batch_size = 8
    cbParam = CbDataParam(dataName)
    results_dir = "Images/{}".format(dataName)

    cbData = CbData(cbParam.path, cbParam.file_ind, cbParam.sampcycle, cbParam.n_reactors)
    temps = []
    mean = [[] for b in range(len(bac))]
    median = [[] for b in range(len(bac))]
    std = [[] for b in range(len(bac))]

    boxdata = [[] for b in range(len(bac))]

    matplotlib.style.use('default')
    n_rows = math.ceil(cbParam.n_reactors/2)
    n_culumns = 2 if cbParam.n_reactors > 1 else 1
    fig, ax = plt.subplots(n_rows,2,sharex='all',sharey='all')
    fig.set_figheight(n_rows*7)
    fig.set_figwidth(n_culumns*10)
    if n_culumns == 1:
        ax = [ax]
    if n_rows == 1:
        ax = [ax]
    
    # for each reactor
    for b in range(len(bac)):
        counter = 0
        temp_sp = [[] for j in range(batch_size)]
        gr = [[] for j in range(batch_size)]
        for j in range(bac[b],batch_size+bac[b]):
            gr_fit = []
            od_mat = []
            temp_sp_mat = []
            time_h_mat = []
            time_h = []
            # Get dilution times
            dil = np.diff(cbData.p1[j], prepend=[0,0]) > 0.015
            # Resize data
            r = 0
            od_list = []
            time_h_list = []
            temp_sp_beg = cbData.temp_sp[j][0]
            gr_constant = False
            time_beg = 0
            for i in range(len(cbData.time[j])):
                if cbData.temp_sp[j][i] != cbData.temp_sp[j][max(0,i-1)]:
                    gr_constant = False
                    time_beg = cbData.time_h[j][i]
                if cbData.time_h[j][i] - time_beg > 3 and not gr_constant:
                    gr_constant = True
                if dil[i]:
                    if r > 0 and gr_constant: # don't get gradient before first dilution, after temperature change
                        od_mat.append(od_list)
                        time_h_mat.append(time_h_list)
                        temp_sp_mat.append(temp_sp_beg)
                    r += 1
                    od_list = []
                    time_h_list = []
                    temp_sp_beg = cbData.temp_sp[j][i]
                if gr_constant:
                    od_list.append(cbData.od[j][i])
                    time_h_list.append(cbData.time_h[j][i])
            temps += list(set(temp_sp_mat))
            # Fit lines and get growth rates
            for r in range(len(temp_sp_mat)):
                time_h_mat[r] = time_h_mat[r][3:]
                od_mat[r] = od_mat[r][3:]
                if len(od_mat[r]) > 5:
                    fit = np.poly1d(np.polyfit(time_h_mat[r], np.log(od_mat[r]), 1, w = np.sqrt(od_mat[r])))
                    gr_fit.append(fit)
                    gr[counter].append(fit.coefficients[0])
                    temp_sp[counter].append(temp_sp_mat[r])
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

            counter += 1

        # Sort gr according to temperature
        temps = list(set(temps))
        temps.sort()
        grs = [[] for t in range(len(temps))]
        for count in range(batch_size):
            for r in range(len(temp_sp[count])):
                grs[temps.index(temp_sp[count][r])].append(gr[count][r])

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
    fig_gr, ax = plt.subplots()
    fig_gr.set_figheight(6)
    fig_gr.set_figwidth(8)
    # Plot gradients
    ax.boxplot(boxdata[0], positions=temps, widths=0.4, showfliers=False,
               boxprops=dict(color="g", alpha=0.5),
               medianprops=dict(color="g", alpha=0.5),
               flierprops=dict(markeredgecolor="g", alpha=0.5),
               capprops=dict(color="g", alpha=0.5),
               whiskerprops=dict(color="g", alpha=0.5))
    ax.boxplot(boxdata[1], positions=temps, widths=0.4, showfliers=False,
               boxprops=dict(color="m", alpha=0.5),
               medianprops=dict(color="m", alpha=0.5),
               flierprops=dict(markeredgecolor="m", alpha=0.5),
               capprops=dict(color="m", alpha=0.5),
               whiskerprops=dict(color="m", alpha=0.5))
    # Plot fit
    x = np.linspace(temps[0],temps[-1],100)
    ax.plot(x, p_model(x), 'g', linewidth=2, label = 'P. putida (ivw fit)')
    ax.plot(x, e_model(x), 'm', linewidth=2, label = 'E. coli (ivw fit)')
    ax.set_xlabel(r'Temperature $[^{\circ}C]$')
    ax.set_ylabel(r'Growth rate $[\frac{1}{h}]$')
    ax.legend(loc='best')
    ax.set_title("Bacteria Growth Rates")
    ax.set_ylim([0,1.5])

    # Save figures
    fig.suptitle(dataName)
    fig.tight_layout()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    fig.savefig(results_dir + "/odGradient.png", transparent=True)
    fig_gr.savefig(results_dir + "/growthRates.png", transparent=True)
