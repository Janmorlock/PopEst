import glob
import pandas as pd

from CbDataParams import CbDataParam

class CbData:
    def __init__(self, cbParam: CbDataParam):
        scope = cbParam.sampcycle
        cb_files = sorted(glob.glob(cbParam.path + "/*.csv"))
        cb_dfs = [[] for j in range(cbParam.n_reactors)]
        for j in range(len(cbParam.file_ind)):
            for i in range(len(cbParam.file_ind[j])):
                df = pd.read_csv(cb_files[cbParam.file_ind[j][i]], index_col=None, header=0)
                if i > 0:
                    df["exp_time"] = df["exp_time"] + cb_dfs[j]["exp_time"].iloc[-1]
                    df = pd.concat([cb_dfs[j],df], ignore_index=True)
                cb_dfs[j] = df

        self.time, self.time_h, self.od, self.temp, self.fl, self.b1, self.temp_sp, self.dil, self.p_targ, self.p_est = ([[] for j in range(cbParam.n_reactors)] for i in range(10))
        for j in range(cbParam.n_reactors):
            time = cb_dfs[j]["exp_time"][scope[j][0]:scope[j][-1]+1].to_numpy()
            self.time[j] = (time-time[0])
            self.time_h[j] = self.time[j]/3600
            od = cb_dfs[j]["od_measured"][scope[j][0]:scope[j][-1]+1].to_numpy()
            od[od < 0.005] = 0.005
            self.od[j] = od
            self.temp[j] = cb_dfs[j]["media_temp"][scope[j][0]:scope[j][-1]+1].to_numpy()
            self.fl[j] = cb_dfs[j]["FP1_emit1"][scope[j][0]:scope[j][-1]+1].to_numpy()
            self.b1[j] = cb_dfs[j]["FP1_base"][scope[j][0]:scope[j][-1]+1].to_numpy()
            self.temp_sp[j] = cb_dfs[j]["thermostat_setpoint"][scope[j][0]:scope[j][-1]+1].to_numpy()
            try:
                self.p_targ[j] = cb_dfs[j]["comm_tar"][scope[j][0]:scope[j][-1]+1].to_numpy()
                self.p_est[j] = cb_dfs[j]["comm_est"][scope[j][0]:scope[j][-1]+1].to_numpy()
            except:
                if j == 0:
                    print("No comm_tar and/or comm_est in file")
            try:
                zzt = cb_dfs[j]["zigzag_target"][scope[j][0]:scope[j][-1]+1].to_numpy()
                self.dil[j] = zzt < 1
            except:
                if j == 0:
                    print("No zigzag_target in file")