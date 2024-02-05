import glob
import pandas as pd

class CbData:
    def __init__(self, path, file_ind, scope, n_reactors):
        cb_files = sorted(glob.glob(path + "/*.csv"))
        cb_dfs = []
        for i in file_ind:
            df = pd.read_csv(cb_files[i], index_col=None, header=0)
            cb_dfs.append(df)

        self.time, self.time_h, self.od, self.temp, self.fl, self.p1 = [], [], [], [], [], []
        self.hr, self.b1, self.temp_sp, self.temp_ext, self.temp_int, self.dil = [], [], [], [], [], []
        for j in range(n_reactors):
                time = cb_dfs[j]["exp_time"][scope[j][0]:scope[j][-1]+1].to_numpy()
                self.time.append(time-time[0])
                self.time_h.append(self.time[j]/3600)
                od = cb_dfs[j]["od_measured"][scope[j][0]:scope[j][-1]+1].to_numpy()
                od[od < 0.005] = 0.005
                self.od.append(od)
                self.temp.append(cb_dfs[j]["media_temp"][scope[j][0]:scope[j][-1]+1].to_numpy())
                self.fl.append(cb_dfs[j]["FP1_emit1"][scope[j][0]:scope[j][-1]+1].to_numpy())
                self.p1.append(cb_dfs[j]["pump_1_rate"][scope[j][0]:scope[j][-1]+1].to_numpy())
                self.hr.append(cb_dfs[j]["heating_rate"][scope[j][0]:scope[j][-1]+1].to_numpy())
                self.b1.append(cb_dfs[j]["FP1_base"][scope[j][0]:scope[j][-1]+1].to_numpy())
                self.temp_sp.append(cb_dfs[j]["thermostat_setpoint"][scope[j][0]:scope[j][-1]+1].to_numpy())
                self.temp_ext.append(cb_dfs[j]["external_air_temp"][scope[j][0]:scope[j][-1]+1].to_numpy())
                self.temp_int.append(cb_dfs[j]["internal_air_temp"][scope[j][0]:scope[j][-1]+1].to_numpy())
                try:
                    zzt = cb_dfs[j]["zigzag_target"][scope[j][0]:scope[j][-1]+1].to_numpy()
                    self.dil.append(zzt < 1)
                except:
                    if j == 0:
                        print("No zigzag_target in file")