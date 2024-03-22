import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

from CbData import CbData
from CbDataParams import CbDataParam
from lib.Estimator import EKF 
from config.params import Params

cdict = {
    'red': (
        (0.0,  1.0, 1.0),
        (0.5,  1.0, 1.0),
        (1.0,  0.0, 0.0),
    ),
    'green': (
        (0.0,  0.0, 0.0),
        (0.5,  1.0, 1.0),
        (1.0,  1.0, 1.0),
    ),
    'blue': (
        (0.0,  0.0, 0.0),
        (0.5,  1.0, 1.0),
        (1.0,  0.0, 0.0),
    )
}
red_green = LinearSegmentedColormap('RedGreen', cdict)

def highlight_cell(x,y, ax=None, **kwargs):
    rect = plt.Rectangle((x-.5, y-.5), 1,1, fill=True, linewidth = 0, hatch = '/', color = 'k', **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect

if __name__ == "__main__":
    # SPECIFY DATA
    data_names = ['081-1', '081-2', '081-3', '081-4']
    # data_names = ['081-2', '081-3', '081-4']
    # data_names = ['081-1']

    len_d = len(data_names)
    cbParamAll = [CbDataParam(data_names[d]) for d in range(len_d)]
    n_exp = 0
    max_s = 0
    for d in range(len_d):
        # cbParamAll[d].n_reactors = 2
        n_exp += cbParamAll[d].n_reactors
        max_s = max(max_s, cbParamAll[d].sampcycle[0].shape[0])
    cbDataAll = [CbData(cbParamAll[d]) for d in range(len_d)]
    parameters = Params().default
    p_rel = [[np.empty(len(cbDataAll[d].time_h[j])) for j in range(cbParamAll[d].n_reactors)] for d in range(len_d)]
    p_err = [np.empty(cbParamAll[d].n_reactors) for d in range(len_d)]
    p_err_pred = [np.empty(cbParamAll[d].n_reactors) for d in range(len_d)]
    p_c_err = [np.empty(cbParamAll[d].n_reactors) for d in range(len_d)]
    sample_comp = np.full((n_exp, max_s), 0.01)
    e = 0
    for d in range(len_d):
        for j in range(cbParamAll[d].n_reactors):
            # Construct State estimator
            pred = EKF(dev_ind = j, update = False)
            ekf = EKF(dev_ind = j)
            pred.model.dithered = bool(cbDataAll[d].dil)
            ekf.model.dithered = bool(cbDataAll[d].dil)
            pred.set_r_coeff(cbParamAll[d].reactors[j])
            ekf.set_r_coeff(cbParamAll[d].reactors[j])
            p_rel_pred = np.empty(len(cbDataAll[d].time[j]))
            for k in range(len(cbDataAll[d].time[j])):
                # Run the filter
                dil = cbDataAll[d].dil[j][k] if cbDataAll[d].dil else 0
                u = np.array([cbDataAll[d].temp[j][k], dil])
                y = np.array([cbDataAll[d].od[j][k], cbDataAll[d].fl[j][k]*cbDataAll[d].b1[j][k]])
                pred.estimate(cbDataAll[d].time[j][k], u)
                ekf.estimate(cbDataAll[d].time[j][k], u, y)
                predictions = pred.est.copy()
                estimates = ekf.est.copy()
                p_rel_pred[k] = predictions['p']/(predictions['e']+predictions['p'])
                p_rel[d][j][k] = estimates['p']/(estimates['e']+estimates['p'])
            for s in range(len(cbParamAll[d].cb_fc_ec[j])):
                err_pred = abs(1-cbParamAll[d].cb_fc_ec[j,s]/100 - p_rel_pred[cbParamAll[d].sampcycle[j][s]-cbParamAll[d].sampcycle[j][0]])
                err_ekf = abs(1-cbParamAll[d].cb_fc_ec[j,s]/100 - p_rel[d][j][cbParamAll[d].sampcycle[j][s]-cbParamAll[d].sampcycle[j][0]])
                sample_comp[e,s] = (err_pred - err_ekf)*100
            e += 1
            p_err_pred[d][j] = np.abs(1-cbParamAll[d].cb_fc_ec[j]/100 - p_rel_pred[cbParamAll[d].sampcycle[j]-cbParamAll[d].sampcycle[j][0]]).mean()
            p_err[d][j] = np.abs(1-cbParamAll[d].cb_fc_ec[j]/100 - p_rel[d][j][cbParamAll[d].sampcycle[j]-cbParamAll[d].sampcycle[j][0]]).mean()
            p_c_err[d][j] = np.abs(1-cbParamAll[d].cb_fc_ec[j]/100 - cbDataAll[d].p_targ[j][cbParamAll[d].sampcycle[j]-cbParamAll[d].sampcycle[j][0]]).mean()
            print("Mean estimation error [{}][{}]: {:.3f}    Mean control error: {:.3f}".format(d, j, p_err[d][j], p_c_err[d][j]))
    flat_list = [item for sublist in p_err for item in sublist]
    p_err_f = np.array(flat_list)
    flat_list_c = [item for sublist in p_c_err for item in sublist]
    p_c_err_f = np.array(flat_list_c)
    print(np.quantile(p_err_f, [0.25, 0.75]))
    print(np.quantile(p_c_err_f, [0.25, 0.75]))


    matplotlib.style.use('default')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # Set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 11})

    fig = plt.figure(figsize=(7, 4))
    extr = np.max(np.abs(sample_comp))
    im = plt.imshow(sample_comp, cmap=red_green, aspect='auto', vmin=-extr, vmax=extr)
    ax = plt.gca()
    for e in range(n_exp):
        for s in range(max_s):
            if sample_comp[e,s] == 0.01:
                highlight_cell(s,e,ax)
    ax.set_xticks(np.arange(0,max_s), labels=np.arange(1,max_s+1))
    ax.set_yticks(np.arange(0,n_exp+0), labels=np.arange(1,n_exp+1))
    ax.set_xlabel('Flow Cytometer Sample')
    ax.set_ylabel('Experiment')

    cbar = fig.colorbar(im)
    # cbar.set_label(r'$| \hat{p}_{pred}-p_{fc} | - | \hat{p}-p_{fc} |$')
    cbar.set_label('Change of EKF Estimation Error\nthrough Measurement Update [%]')
    fig.tight_layout()

    fig.savefig('/Users/janmorlock/Documents/Ausbildung/Master/MasterProject/FiguresCDC/3_estComp.pdf', transparent=True)