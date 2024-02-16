from cProfile import label
import time
from turtle import pu
from lib.Model import CustModel
from lib.Estimator import EKF

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":

    model_sim = CustModel()
    model_pred = CustModel()
    model_est = CustModel()

    pre_time_m = 45
    timeEnd_m = 60 * 24 + pre_time_m
    alpha_temp = 0.8
    od_th = 0.55
    od_l_th = 0.44
    dithered = True
    CritTemp = 33.5 # around this temperature both bacteria should be growing at the same rate
    x0 = {'e': model_sim.parameters['od_init']*model_sim.parameters['e_rel_init'],
        'p': model_sim.parameters['od_init']*(1-model_sim.parameters['e_rel_init']),
        'fp': model_sim.parameters['fp_init']}
    p0 = np.diag([model_sim.parameters['sigma_e_init']**2, model_sim.parameters['sigma_p_init']**2, model_sim.parameters['sigma_fp_init']**2])
    pid = {'Kp': 15, 'Ki': 0, 'Kd': 0}

    x = x0
    p = p0
    model_sim.dithered = dithered
    model_pred.dithered = dithered
    model_est.dithered = dithered
    pred = EKF(model_pred, 0, False)
    ekf = EKF(model_est, 0, True)
    pred.set_r_coeff('Faith')
    ekf.set_r_coeff('Faith')
    err_cum = 0
    err_prev = 0
    dil = False
    temp = 23
    temp_target = 0
    time_m = 0
    time_s = 0
    time_s_prev = 0
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
    while time_m < timeEnd_m:
        ### Simulate
        time_h = time_m/60
        u = np.array([temp, dil])
        dt = time_s - time_s_prev
        # TODO: Add deviations, noise to the system model dynamics
        # model_sim.parameters['gr_e'][-1] = model_est.parameters['gr_e'][-1] + 0.05*model_est.getGrowthRates(np.full(3,temp))[0]
        # model_sim.parameters['gr_p'][-1] = model_est.parameters['gr_p'][-1] - 0.1*model_est.getGrowthRates(np.full(3,temp))[1]
        # model_sim.parameters['gr_fp'][-1] = model_est.parameters['gr_fp'][-1] + 0.2*np.sin(time_h/2*2*np.pi)*model_est.getGrowthRates(np.full(3,temp))[2]
        # Get simulated states and measurements
        x, p = model_sim.predict(x, p, u, dt)
        od = x['e'] + x['p']
        fl = (x['fp'] + model_sim.parameters['od_fac']*od) * model_sim.parameters['e_fac']['Faith'] + model_sim.parameters['e_ofs']['Faith']
        # TODO: Add measurement noise
        od_meas = od
        # od_meas = od + np.random.normal(0, model_sim.parameters['sigma_e'])
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
        pp_rel_est = x_est['p']/(x_est['e'] + x_est['p'])
        # Log
        od_est_arr[time_m] = od_est
        fl_est_arr[time_m] = fl_est
        pp_rel_od_arr[time_m] = ekf.p_est_od/y[0]
        pp_rel_fl_arr[time_m] = ekf.p_est_fl/y[0]
        pp_rel_pred_arr[time_m] = pp_rel_pred
        pp_rel_est_arr[time_m] = pp_rel_est

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
            # log
            pp_rel_target_arr[time_m] = pp_rel_target
            
            # calculate required temperature
            pp_rel_err = pp_rel_target - pp_rel_est
            err_cum += pp_rel_err * pid['Ki']
            err_dif = (pp_rel_err - err_prev) * pid['Kd']
            th_target = model_est.parameters['crit_temp'] - pp_rel_err * pid['Kp'] - err_cum - err_dif
            # Constrain the target temperature to desired range
            temp_target = th_target
            temp_target = min(temp_target, 36)
            temp_target = max(temp_target, 29)
            err_prev = pp_rel_err
            # Anti Reset Windup
            if pid['Ki'] > 0:
                err_cum -= (th_target - temp_target)

        #log
        temp_target_arr[time_m] = temp_target
                
        # Get data for next iteration
        time_s_prev = time_s
        time_m += 1
        time_s = time_m*60
    
    ### Plot Results
    n_rows = 1
    n_culumns = 1
    matplotlib.style.use('default')
    fig, ax = plt.subplots(n_rows,n_culumns,sharey='all')
    if n_culumns == 1:
        ax = [ax]
    if n_rows == 1:
        ax = [ax]
    fig.set_figheight(n_rows*7)
    fig.set_figwidth(n_culumns*10)

    r = 0; c = 0
    max_fl = max(fl_arr)
    max_fl_est = max(fl_est_arr)

    axr = ax[r][c].twinx()
    ax[r][c].set_zorder(2)
    axr.set_zorder(1)
    ax[r][c].patch.set_visible(False)
    axr.plot(time_h_arr, temp_target_arr, '--r', lw = 0.5, alpha = 0.5, label = '$temp_{sp}$')
    axr.plot(time_h_arr,temp_arr,'r',lw=0.5,alpha=1, label = '$temp_{meas}$')

    ax[r][c].plot(time_h_arr,od_arr, '.k', markersize = 0.5, label = '$OD_{meas}$')
    ax[r][c].plot(time_h_arr,fl_arr/max_fl, '.', color = '#0000FF', markersize =  0.5, label = '$fl_{meas}$')
    ax[r][c].plot(time_h_arr[pp_rel_od_arr > 0],pp_rel_od_arr[pp_rel_od_arr > 0], 'xg', markersize = 10, label = '$pp_{est,od}$')
    ax[r][c].plot(time_h_arr[pp_rel_fl_arr > 0],pp_rel_fl_arr[pp_rel_fl_arr > 0], 'xb', markersize = 7, label = '$pp_{est,fl}$')
    ax[r][c].plot(time_h_arr,od_est_arr, 'k', lw = 0.5, label = '$OD_{est}$')
    ax[r][c].plot(time_h_arr,fl_est_arr/max_fl_est, color = '#0000FF', lw = 0.5, label = '$fl_{est}$')
    ax[r][c].plot(time_h_arr[pp_rel_target_arr >= 0],pp_rel_target_arr[pp_rel_target_arr >= 0], '--g', lw = 1.2, label = '$pp_{target}$')
    ax[r][c].plot(time_h_arr,pp_rel_arr, '.g', markersize = 0.8, label = '$pp_{truth}$')
    ax[r][c].plot(time_h_arr,pp_rel_pred_arr, 'g', lw = 0.5, label = '$pp_{pred}$')
    ax[r][c].plot(time_h_arr,pp_rel_est_arr, 'g', lw = 1.2, label = '$pp_{est}$')


    ax[r][c].legend(loc="upper left")
    if c == 0:
        ax[r][c].set_ylabel("Normalized abundance")
    if c == n_culumns-1:
        axr.set_ylabel('Temperature [°C]', color='r')
        # ax[r][c].tick_params(axis='y', labelleft=True)
    if r == n_rows-1:
        ax[r][c].set_xlabel("Time [h]")
    axr.tick_params(axis='y', color='r', labelcolor='r')
    ax[r][c].set_ylim([0,1])
    # ax[r][c].set_xlim([10,20])
    axr.set_ylim([28,37])
    axr.legend(loc="upper right")

    # Save figures
    dataName = "sim"
    results_dir = "Images/{}".format(dataName)
    # fig.suptitle('Simulation with noise and imperfect knowledge of the model', fontsize=14)
    fig.suptitle('Simulation without noise and perfect knowledge of the model', fontsize=14)
    fig.tight_layout()
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    fig.savefig(results_dir + "/sim_00.png", transparent=True)