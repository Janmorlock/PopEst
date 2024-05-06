from re import T
from lib.Model import CustModel
from lib.Estimator import EKF

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    # Simulation parameters
    pre_time_m = 60
    timeEnd_m = 60 * 17
    alpha_temp = 0.8
    e_init = 0.5

    od_th = 0.53
    od_l_th = 0.46
    dithered = True
    noise = False
    control = False
    paper = True
    plos = False
    if plos:
        pp_color = '#F6B343'
    else:
        pp_color = 'g'
    test_var = '' # 'pid' or 'single_est' or 'combi_est'
    pid = [{'Kp': 15, 'Ki': 0, 'Kd': 0},
           {'Kp': 15, 'Ki': 0.01, 'Kd': 0},
           {'Kp': 15, 'Ki': 0.05, 'Kd': 5}]
    single_est = {'od_update':      [False, True, False, False, False],
                    'fl_update':    [False, False, True, False, False],
                    'od_gr_update': [False, False, False, True, False],
                    'fl_gr_update': [False, False, False, False, True]}
    combi_est = {'od_update':       [True, True, True, True, True, True],
                    'fl_update':    [False, False, True, True, True, True],
                    'od_gr_update': [True, True, False, True, False, True],
                    'fl_gr_update': [False, True, False, False, True, True]}
    match test_var:
        case 'pid':
            n_culumns = len(pid)
            titles = ['Kp = {}, Ki = {}, Kd = {}'.format(pid[i]['Kp'], pid[i]['Ki'], pid[i]['Kd']) for i in range(n_culumns)]
        case 'single_est':
            n_culumns = len(single_est['od_update'])
            titles = ['No measurement update', 'OD update', 'FL update', 'OD curvature update', 'FL curvature update']
        case 'combi_est':
            n_culumns = len(combi_est['od_update'])
            titles = ['OD + OD Curvature', 'OD + OD curvature + FL curvature update', 'OD + FL update', 'OD + FL + OD curvature update', 'OD + FL + FL curvature update', 'OD + FL + OD curvature + FL curvature update']
        case _:
            n_culumns = 1
            if control:
                titles = ['Kp = 15, Ki = 0.01, Kd = 0']
            else:
                titles = ['']
    if control:
        timeEnd_m += pre_time_m

    # Initialize plot
    matplotlib.style.use('default')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 8})
    plt.rcParams["mathtext.fontset"] = 'stix'
    n_rows = 1
    fig, ax = plt.subplots(n_rows,n_culumns)
    if paper:
        n_rows = 3
        fig, ax = plt.subplots(n_rows,n_culumns, height_ratios=[1, 3, 2])
    if n_culumns == 1:
        if n_rows == 1:
            ax = [ax]
        else:
            ax = [[ax[i]] for i in range(n_rows)]
    if n_rows == 1:
        ax = [ax]
    if paper:
        fig.set_figheight(5)
        fig.set_figwidth(7.5)
    else:
        fig.set_figheight(n_rows*7)
        fig.set_figwidth(n_culumns*10)

    # Run simulation in different configurations
    for j in range(n_culumns):
        model_sim = CustModel()
        model_pred = CustModel()
        model_est = CustModel()
        model_sim.parameters['e_rel_init'] = e_init
        model_sim.parameters['temp_pre_e'] =  24
        model_sim.parameters['temp_pre_p'] =  24
        x0 = {'e': model_sim.parameters['od_init']*model_sim.parameters['e_rel_init'],
            'p': model_sim.parameters['od_init']*(1-model_sim.parameters['e_rel_init']),
            'fp': model_sim.parameters['fp_init']}
        p0 = np.diag([model_sim.parameters['sigma_e_init']**2, model_sim.parameters['sigma_p_init']**2, model_sim.parameters['sigma_fp_init']**2])
        x = x0
        p = p0
        model_sim.dithered = dithered
        model_pred.dithered = dithered
        model_est.dithered = dithered
        pred = EKF(model_pred, 0, False)
        ekf = EKF(model_est, 0, True)
        ekf.u_prev = np.array([36, False])
        pred.set_r_coeff('Faith')
        ekf.set_r_coeff('Faith')
        err_cum = 0
        err_prev = 0
        dil = False
        temp = 24
        temp_target = 0
        time_m = 0
        time_s = 0
        time_s_prev = 0
        if test_var == 'pid':
            model_est.parameters['kp'] = pid[j]['Kp']
            model_est.parameters['ki'] = pid[j]['Ki']
            model_est.parameters['kd'] = pid[j]['Kd']
        if test_var == 'single_est':
            model_est.parameters['od_update'] = single_est['od_update'][j]
            model_est.parameters['fl_update'] = single_est['fl_update'][j]
            model_est.parameters['od_gr_update'] = single_est['od_gr_update'][j]
            model_est.parameters['fl_gr_update'] = single_est['fl_gr_update'][j]
        if test_var == 'combi_est':
            model_est.parameters['od_update'] = combi_est['od_update'][j]
            model_est.parameters['fl_update'] = combi_est['fl_update'][j]
            model_est.parameters['od_gr_update'] = combi_est['od_gr_update'][j]
            model_est.parameters['fl_gr_update'] = combi_est['fl_gr_update'][j]
        # Log
        len = int(timeEnd_m)
        time_h_arr = np.empty(len)
        fl_arr = np.empty(len)
        fl_est_arr = np.empty(len)
        temp_arr = np.empty(len)
        temp_target_arr = np.empty(len)
        od_arr = np.empty(len)
        od_est_arr = np.empty(len)
        pp_rel_arr = np.empty(len)
        pp_rel_est_arr = np.empty(len)
        pp_rel_pred_arr = np.empty(len)
        pp_rel_target_arr = np.full(len, -1, dtype=float)
        pp_rel_od_arr = np.empty(len)
        pp_rel_fl_arr = np.empty(len)

        pp_rel_od_res_arr = np.empty(len)
        pp_rel_fl_res_arr = np.empty(len)
        while time_m < timeEnd_m:
            ### Simulate
            time_h = time_m/60
            u = np.array([temp, dil])
            dt = time_s - time_s_prev
            
            # Add deviations, noise to the system model dynamics
            # if noise:
                # model_sim.parameters['gr_e'][-1] = model_est.parameters['gr_e'][-1] + 0.05*model_est.getSteadyStateGrowthRates(temp)[0]
                # model_sim.parameters['gr_p'][-1] = model_est.parameters['gr_p'][-1] - 0.1*model_est.getGrowthRates(np.full(3,temp))[1]
                # model_sim.parameters['gr_fp'][-1] = model_est.parameters['gr_fp'][-1] + (0.15*np.sin(time_h/2*2*np.pi) + 0.1)*model_est.getSteadyStateGrowthRates(temp)[2]
            
            # Get simulated states and measurements
            x, p = model_sim.predict(x, p, u, dt)
            od = x['e'] + x['p']
            fl = (x['fp'] + model_sim.parameters['od_fac']*od) * model_sim.parameters['e_fac']['Faith'] + model_sim.parameters['e_ofs']['Faith']
            
            # Add measurement noise
            od_meas = od
            if noise:
                od_meas = od + np.random.normal(0, model_sim.parameters['sigma_od_mod'])
                fl = fl + np.random.normal(0, 30)

            y = np.array([od_meas, fl])
            pp_rel = x['p']/(x['e'] + x['p'])
            if dithered:
                if x['e'] + x['p'] > od_th:
                    dil = True
                if x['e'] + x['p'] < od_l_th:
                    dil = False
            else:
                dil = 0
            if dt > 0:
                temp = alpha_temp * temp + (1 - alpha_temp) * temp_target
            # Log
            time_h_arr[time_m] = time_h
            od_arr[time_m] = od_meas
            fl_arr[time_m] = fl
            temp_arr[time_m] = temp
            pp_rel_arr[time_m] = pp_rel

            ### State Estimation
            u_ekf = np.array([temp, dil])
            pred.estimate(time_s, u_ekf)
            ekf.estimate(time_s, u_ekf, y)
            x_pred = pred.est
            p_pred = pred.var
            x_est = ekf.est
            p_est = ekf.var
            od_est = x_est['e'] + x_est['p']
            fl_est = (x_est['fp'] + model_est.parameters['od_fac']*od_est) * model_est.parameters['e_fac']['Faith'] + model_est.parameters['e_ofs']['Faith']
            pp_rel_pred = x_pred['p']/(x_pred['e'] + x_pred['p'])
            pp_rel_est = x_est['p']/od_est
            # Log
            od_est_arr[time_m] = od_est
            fl_est_arr[time_m] = fl_est
            pp_rel_od_arr[time_m] = ekf.p_est_od/y[0]
            pp_rel_fl_arr[time_m] = ekf.p_est_fl/y[0]
            pp_rel_pred_arr[time_m] = pp_rel_pred
            pp_rel_est_arr[time_m] = pp_rel_est
            pp_rel_od_res_arr[time_m] = ekf.p_est_od_res
            pp_rel_fl_res_arr[time_m] = ekf.p_est_fl_res

            if control:
                ### Temperature Control
                if time_m < pre_time_m:
                    temp_target = model_est.parameters['crit_temp']
                else:
                    # get desired composition ratio
                    if (time_m < 6 * 60 + pre_time_m):
                        pp_rel_target = 0.8
                    elif (time_m < 12 * 60 + pre_time_m):
                        pp_rel_target = 0.2
                    else:
                        pp_rel_target = 0.8
                    # pp_rel_target = -0.4*math.sin(2*math.pi*(time_m-pre_time_m)/(20*60)) + 0.5
                    # log
                    pp_rel_target_arr[time_m] = pp_rel_target
                    
                    # calculate required temperature
                    pp_rel_err = pp_rel_target - pp_rel_est
                    err_cum += pp_rel_err * model_est.parameters['ki']
                    err_dif = (pp_rel_err - err_prev) * model_est.parameters['kd']
                    th_target = model_est.parameters['crit_temp'] - (pp_rel_err * model_est.parameters['kp'] + err_cum + err_dif)
                    # Constrain the target temperature to desired range
                    temp_target = th_target
                    temp_target = min(temp_target, 36)
                    temp_target = max(temp_target, 29)
                    err_prev = pp_rel_err
                    # Anti Reset Windup
                    if model_est.parameters['ki'] > 0:
                        if abs(th_target - temp_target) > 0:
                            err_cum -= pp_rel_err * model_est.parameters['ki']
            else:
                if time_h < 3:
                    temp_target = 29
                elif time_h < 6:
                    temp_target = 36
                elif time_h < 9:
                    temp_target = 31
                elif time_h < 12:
                    temp_target = 34
                else:
                    temp_target = model_est.parameters['crit_temp']
                
            #log
            temp_target_arr[time_m] = temp_target
                    
            # Get data for next iteration
            time_s_prev = time_s
            time_m += 1
            time_s = time_m*60
    
        ### Plot Results
        r = 1; c = j
        max_fl = max(fl_arr)

        axr = ax[0][c]
        axr.hlines(model_sim.parameters['crit_temp'],time_h_arr[0]-0.5,time_h_arr[-1]+0.5,'r','--',lw=1, label = '$T_{crit}$')
        axr.plot(time_h_arr,temp_arr,'r',lw=1,alpha=1, label = '$T_{sim}$')

        if not paper:
            ax[r][c].plot(time_h_arr,od_arr*100, '.k', markersize = 0.5, label = '$od_{meas}$')
            ax[r][c].plot(time_h_arr,fl_arr/max_fl*100, '.', color = '#0000FF', markersize =  0.5, label = '$em_{meas}$')
        ax[r][c].plot(time_h_arr,pp_rel_arr*100, color = pp_color, lw = 1.5, label = 'Ground Truth, $p$')
        ax[r][c].plot(time_h_arr[pp_rel_od_arr > 0],pp_rel_od_arr[pp_rel_od_arr > 0]*100, 'xk', markersize = 4, label = 'OD Curvature Estimate, $\hat{p}_{od}$')
        ax[r][c].plot(time_h_arr[pp_rel_fl_arr > 0],pp_rel_fl_arr[pp_rel_fl_arr > 0]*100, 'x', color = '#0000FF', markersize = 4, label = 'fl Curvature Estimate, $\hat{p}_{fl}$')
        if not paper:
            ax[r][c].plot(time_h_arr,od_est_arr*100, 'k', lw = 0.5, label = '$\hat{od}$', alpha = 0.5)
            ax[r][c].plot(time_h_arr,fl_est_arr/max_fl*100, color = '#0000FF', lw = 0.5, label = '$\hat{em}$', alpha = 0.5)
        if control:
            ax[r][c].plot(time_h_arr[pp_rel_target_arr >= 0],pp_rel_target_arr[pp_rel_target_arr >= 0]*100, '--', color = pp_color, lw = 1.2, label = 'Reference $p_{target}$')
        # if noise:
        #     ax[r][c].plot(time_h_arr,pp_rel_pred_arr*100, color = pp_color, lw = 0.5, label = '$p_{pred}$')
        if not paper:
            ax[r][c].plot(time_h_arr,pp_rel_est_arr*100, color = pp_color, lw = 1.2, label = '$\hat{p}$')
            ax[r][c].plot(time_h_arr[pp_rel_od_res_arr > 0],pp_rel_od_res_arr[pp_rel_od_res_arr > 0]*100, '.k', markersize = 4, label = '$r_{od}$')
            ax[r][c].plot(time_h_arr[pp_rel_fl_res_arr > 0],pp_rel_fl_res_arr[pp_rel_fl_res_arr > 0]*100, '.' , color = '#0000FF', markersize = 4, label = '$r_{fl}$')

        if c == 0:
            ax[r][c].set_ylabel("Relative $P. putida$ Abundance [%]")
        if c == n_culumns-1:
            axr.set_ylabel('Temp. [°C]', color='r')
            yticks = np.array([29, model_sim.parameters['crit_temp'], 36])
            axr.set_yticks(yticks, labels=yticks)
            axr.tick_params(axis='y', color='r', labelcolor='r')
        else:
            axr.tick_params(axis='y', color='r', labelright=False)
        ax[r][c].set_xlabel("Time [h]")
        ax[r][c].set_xlim([-0.5,timeEnd_m/60+0.5])
        ax[r][c].set_ylim([0,100])
        axr.set_xlim([-0.5,timeEnd_m/60+0.5])
        axr.set_ylim([28,37])
        axr.tick_params(axis='x', labelbottom=False)
        # axr.set_xticks(ax[n_rows-2][c].get_xticks())
        axr.legend(loc="lower right")
        if not paper:
            axr.legend(loc="lower right")
            ax[r][c].set_title(titles[j])

        if paper:
            ax[2][0].scatter(np.log10(pp_rel_fl_res_arr[pp_rel_fl_arr > 0])*20,abs(pp_rel_fl_arr[pp_rel_fl_arr > 0]-pp_rel_arr[pp_rel_fl_arr > 0])/pp_rel_arr[pp_rel_fl_arr > 0]*100, color = '#0000FF', label = 'fl Curvature Estimate, $\hat{p}_{fl}$')
            ax[2][0].scatter(np.log10(pp_rel_od_res_arr[pp_rel_od_arr > 0][time_h_arr[pp_rel_od_arr > 0] < 13])*20,abs(pp_rel_od_arr[pp_rel_od_arr > 0][time_h_arr[pp_rel_od_arr > 0] < 13]-pp_rel_arr[pp_rel_od_arr > 0][time_h_arr[pp_rel_od_arr > 0] < 13])/pp_rel_arr[pp_rel_od_arr > 0][time_h_arr[pp_rel_od_arr > 0] < 13]*100, color = 'k', label = 'OD Curvature Estimate, $\hat{p}_{od}$, $t<13$')
            ax[2][0].scatter(np.log10(pp_rel_od_res_arr[pp_rel_od_arr > 0][time_h_arr[pp_rel_od_arr > 0] > 13])*20,abs(pp_rel_od_arr[pp_rel_od_arr > 0][time_h_arr[pp_rel_od_arr > 0] > 13]-pp_rel_arr[pp_rel_od_arr > 0][time_h_arr[pp_rel_od_arr > 0] > 13])/pp_rel_arr[pp_rel_od_arr > 0][time_h_arr[pp_rel_od_arr > 0] > 13]*100, color = '#909090', label = 'OD Curvature Estimate, $\hat{p}_{od}$, $t>13$')
            ax[2][0].set_xlabel('Normalized Least Squares Residual [dB]')
            ax[2][0].set_ylabel('Relative Estimation Error [%]')
            ax[1][0].text(0.01, 0.98, 'A',
                    horizontalalignment='left',
                    verticalalignment='top',
                    fontsize=20,
                    transform=ax[1][0].transAxes,
                    color='k',
                    bbox={'facecolor': 'red', 'alpha': 0, 'pad': 0, 'edgecolor': 'k'})
            ax[2][0].text(0.01, 0.98, 'B',
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax[2][0].transAxes,
                    fontsize=20,
                    color='k',
                    bbox={'facecolor': 'red', 'alpha': 0, 'pad': 0, 'edgecolor': 'k'})
            ax[2][0].legend(loc = 'upper left', bbox_to_anchor=(0.09,1))
            ax[2][0].set_ylim([-5, 100])
        ax[r][c].legend()
        
    # Save figures
    dataName = "sim"
    results_dir = "Images/{}".format(dataName)
    if not paper:
        if noise:
            fig.suptitle('Simulation with noise and imperfect knowledge of the model', fontsize=14)
        else:
            fig.suptitle('Simulation without noise and perfect knowledge of the model', fontsize=14)
    fig.tight_layout()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    if paper:
        if plos:
            fig.savefig('/Users/janmorlock/Library/CloudStorage/GoogleDrive-janmorlock2001@gmail.com/.shortcut-targets-by-id/1YB52wg5TU4QbAKwShK0fWVAjnrvxlZvh/Pp Ec/Figures/Fig 4 - model and state estimation/SiMuLaTiOn.svg', transparent=True)
        else:
            # fig.savefig('/Users/janmorlock/Documents/Ausbildung/Master/MasterProject/FiguresCDC/2_estQuality.pdf', transparent=True)
            fig.savefig('/Users/janmorlock/Documents/Ausbildung/Master/MasterProject/FiguresReport/Z_estCorr.pdf', transparent=True)
    else:
        fig.savefig(results_dir + "/sim{}{}{}.png".format('_control' if control else '', '_' + test_var if test_var else '', '_noise' if noise else ''), transparent=True)
