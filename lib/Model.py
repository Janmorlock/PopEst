import numpy as np
from typing import Tuple

from ..config.params import Params

# initialize
class CustModel:
    def __init__(self, n_reactors: int):
        self.parameters = Params().default

        self.parameters['lag_ind'] = int(self.parameters['lag']*3600/self.parameters['ts']) # Lag indeces
        self.parameters['r'] = np.diag([self.parameters['sigma_od']**2])
        self.parameters['q'] = np.diag([self.parameters['sigma_e']**2, self.parameters['sigma_p']**2, self.parameters['sigma_fp']**2])
        self.parameters['q_dil'] = np.diag([self.parameters['sigma_e_dil']**2, self.parameters['sigma_p_dil']**2, self.parameters['sigma_fp_dil']**2])

        temps = np.full((self.parameters['lag_ind']+1,2),[self.parameters['temp_h'],self.parameters['temp_l']],dtype='float').T
        self.temps = np.full((n_reactors,temps.shape[0],temps.shape[1]),temps)
        self.ts = self.parameters['ts']
        self.ts_h = self.ts/3600

        self.A_dil = np.eye(3)
        self.L_dil = np.eye(3)

        self.A = np.eye(3)
        self.L = np.eye(3)

        self.H = np.zeros((1,3))
        self.M = np.eye(1)


    def dilute(self, x_prev: np.ndarray, p_prev: np.ndarray):
        x_dil = x_prev.copy()
        p_dil = p_prev.copy()
        # Dilution, checked at same rate as real system TODO: add dithered dilution
        if x_prev[0] + x_prev[1] > self.parameters['od_setpoint']:
            if x_prev[0] + x_prev[1] < 1.1*self.parameters['od_setpoint']:
                x_dil *= (2*self.parameters['od_setpoint']/(x_prev[0] + x_prev[1]) - 1)
            else: # Prevent too extensive bounce back from high od values
                print('diluting differently')
                x_dil *= self.parameters['od_setpoint']/(x_prev[0] + x_prev[1])
            # Jacobian of dilution
            den = (x_prev[0] + x_prev[1])**2
            self.A_dil = np.array([[x_prev[1]/den-1, -x_prev[0]/den, 0],
                                    [-x_prev[1]/den, x_prev[0]/den-1, 0],
                                    [-x_prev[2]/den, -x_prev[2]/den, 1/(x_prev[0]+x_prev[1]) - 1]])
            p_dil = self.A_dil @ p_prev @ self.A_dil.T + self.L_dil @ self.parameters['q_dil'] @ self.L_dil.T
        return x_dil, p_dil


    def predict(self, r_ind, x_prev_dic: dict, p_prev: np.ndarray, u: float, dt: float) -> Tuple[dict, np.ndarray]:
        """
        Dilutes if needed and predicts states their variance after dt seconds.
        """ 
        x_prev = np.fromiter(x_prev_dic.values(),dtype=float)
        x_pred, p_pred = self.dilute(x_prev, p_prev)

        # Approximate abundance and their varaince after dt seconds of growth
        temp = np.full(3,u,dtype='float')
        for i in range(round(dt/self.ts)):
            # Modifiy temperature
            self.temps[r_ind] = np.append(self.temps[r_ind,:,1:],np.full((2,1),u),axis=1)
            if self.parameters['avg_temp']:
                temp[0:2] = np.average(self.temps[r_ind], axis=1)
            else:
                temp[0:2] = self.temps[:,0]
            # Jacobian of growth
            self.A = np.array([[1 + self.ts_h*self.getGrowthRates(temp)[0], 0, 0],
                      [0, 1 + self.ts_h*self.getGrowthRates(temp)[1], 0],
                      [0, self.ts_h*self.getGrowthRates(temp)[2], 1]])
            self.L = np.diag([self.ts_h*x_pred[0], self.ts_h*x_pred[1], self.ts_h*x_pred[1]])

            # Abundance after Ts seconds
            # Euler forward (not significantly less accurate than Runge - Kutta 4th order)
            x_pred = x_pred + self.ts_h*self.getGrowthRates(temp)*np.array([x_pred[0], x_pred[1], x_pred[1]])

            # Runge - Kutta 4th order
            # k1 = self.getGrowthRates(temp)*np.array([x_pred[0], x_pred[1], x_pred[1]])
            # k2 = self.getGrowthRates(temp)*np.array([x_pred[0]+self.ts_h/2*k1[0], x_pred[1]+self.ts_h/2*k1[1], x_pred[1]+self.ts_h/2*k1[1]])
            # k3 = self.getGrowthRates(temp)*np.array([x_pred[0]+self.ts_h/2*k2[0], x_pred[1]+self.ts_h/2*k2[1], x_pred[1]+self.ts_h/2*k2[1]])
            # k4 = self.getGrowthRates(temp)*np.array([x_pred[0]+self.ts_h*k3[0], x_pred[1]+self.ts_h*k3[1], x_pred[1]+self.ts_h*k3[1]])
            # x_pred = x_pred + self.ts_h/6*(k1+2*k2+2*k3+k4)

            p_pred = self.A @ p_pred @ self.A.T + self.L @ self.parameters['q'] @ self.L.T

        return dict(zip(x_prev_dic, x_pred)), p_pred
    
    def update(self, x_pred_dic: dict, p_pred: np.ndarray, y: np.ndarray) -> Tuple[dict, np.ndarray]:
        x_pred = np.fromiter(x_pred_dic.values(),dtype=float)
        od = x_pred[0] + x_pred[1]
        # self.H = np.array([[-x_pred[2]/(od+self.parameters['od_ofs'])**2, -x_pred[2]/(od+self.parameters['od_ofs'])**2, 1/(od+self.parameters['od_ofs'])],
        #                    [1, 1, 0]])
        self.H = np.array([[1, 1, 0]])
        # K = np.linalg.solve(self.H @ p_pred.T @ self.H.T + self.M @ self.parameters['r'].T @ self.M.T, self.H @ p_pred.T).T
        K = p_pred @ self.H.T @ np.linalg.inv(self.H @ p_pred @ self.H.T + self.M @ self.parameters['r'] @ self.M.T)
        # y_est = np.array([x_pred[2]/(od+self.parameters['od_ofs']) + self.parameters['fl_ofs'][r_ind], od])
        y_est = np.array([od])
        xm = x_pred + K @ (y - y_est)
        xm[np.isnan(xm)] = x_pred[np.isnan(xm)]
        Pm = (np.eye(3) - K @ self.H) @ p_pred

        if xm[0] < 0:
            xm[1] += xm[0]
            xm[0] = 1e-4
        if xm[1] < 0:
            xm[0] += xm[1]
            xm[1] = 1e-4
        if xm[2] < 0:
            xm[2] = 1e-4

        return dict(zip(x_pred_dic, xm)), Pm
    
    def getGrowthRates(self, temp: np.ndarray) -> np.ndarray:
        """
        Given the temperatures, return the corresponding growth rates
        """
        gr_e = self.parameters['beta_e']*temp[0] + self.parameters['alpha_e']
        gr_p = self.parameters['del_p']*temp[1]**3 + self.parameters['gam_p']*temp[1]**2 + self.parameters['beta_p']*temp[1] + self.parameters['alpha_p']
        gr_f = self.parameters['gr_fp'][0]*temp[2]**2 + self.parameters['gr_fp'][1]*temp[2] + self.parameters['gr_fp'][2]
        gr_f = max(0,gr_f)

        return np.array([gr_e, gr_p, gr_f])