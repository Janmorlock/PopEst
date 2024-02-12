import numpy as np
from typing import Tuple

from config.params import Params

# initialize
class CustModel:
    def __init__(self):
        self.parameters = Params().default
        self.dithered = False

        self.temps = np.array([self.parameters['temp_pre_e'],self.parameters['temp_pre_p'],self.parameters['temp_pre_p']],dtype='float')
        self.lp_fac = np.exp(-self.parameters['ts']/np.array(self.parameters['lp_ht']))
        self.ts = self.parameters['ts']
        self.ts_h = self.ts/3600

        self.Q = np.diag([self.parameters['sigma_e']**2, self.parameters['sigma_p']**2, self.parameters['sigma_fp']**2])
        self.Q_dil = np.diag([self.parameters['sigma_e_dil']**2, self.parameters['sigma_p_dil']**2, self.parameters['sigma_fp_dil']**2])
        self.Q_dil_dit = np.diag([self.parameters['sigma_e_dil_dit']**2, self.parameters['sigma_p_dil_dit']**2, self.parameters['sigma_fp_dil_dit']**2])
        self.L = np.eye(3)
        self.L_dil = np.eye(3)

        self.R = np.diag([self.parameters['sigma_od']**2, self.parameters['sigma_fl']**2, self.parameters['sigma_od_gr']**2, self.parameters['sigma_fl_gr']**2])
        self.M = np.eye(4)

    def dilute(self, x_prev: np.ndarray, p_prev: np.ndarray):
        '''
        Continuous Dilution

        Similar to the chiBio, this model checks every iteration if dilution is required.
        '''
        x_dil = x_prev.copy()
        p_dil = p_prev.copy()

        # Dilution
        if x_prev[0] + x_prev[1] > self.parameters['od_setpoint']:
            if x_prev[0] + x_prev[1] < 1.1*self.parameters['od_setpoint']:
                x_dil *= (2*self.parameters['od_setpoint']/(x_prev[0] + x_prev[1]) - 1)
            else: # Prevent too extensive bounce back from high od values
                print('diluting differently')
                x_dil *= self.parameters['od_setpoint']/(x_prev[0] + x_prev[1])
            den = (x_prev[0] + x_prev[1])**2
            # Covariance estimation after dilution
            A_dil = np.array([[x_prev[1]/den-1, -x_prev[0]/den, 0],
                                    [-x_prev[1]/den, x_prev[0]/den-1, 0],
                                    [-x_prev[2]/den, -x_prev[2]/den, 1/(x_prev[0]+x_prev[1]) - 1]])
            p_dil = A_dil @ p_prev @ A_dil.T + self.L_dil @ self.Q_dil @ self.L_dil.T
        return x_dil, p_dil

    def dilute_dithered(self, x_prev: np.ndarray, p_prev: np.ndarray):
        '''
        Dithered Dilution

        This model tries to dilute the same amount as the chiBio during dithered dilution.
        '''
        dil_r = self.parameters['dil_rate']
        od = x_prev[0] + x_prev[1]
        # Dilution
        x_dil = x_prev * (1 - dil_r/od)
        # Covariance estimation after dilution
        A_dil = np.array([[1 - dil_r*x_prev[1]/od**2, dil_r*x_prev[0]/od**2, 0],
                            [dil_r*x_prev[1]/od**2, 1 - dil_r*x_prev[0]/od**2, 0],
                            [dil_r*x_prev[2]/od**2, dil_r*x_prev[2]/od**2, 1 - dil_r/od]])
        p_dil = A_dil @ p_prev @ A_dil.T + self.L_dil @ self.Q_dil_dit @ self.L_dil.T
        return x_dil, p_dil
    
    def predict(self, x_prev_dic: dict, p_prev: np.ndarray, u: np.ndarray, dt: float) -> Tuple[dict, np.ndarray]:
        """
        Prediction Step

        Dilutes if needed and predicts states and their variance after dt seconds.
        To calculate the variance, the Jacobian of the system model is used.
        """ 
        x_prev = np.fromiter(x_prev_dic.values(),dtype=float)
        x_pred = x_prev.copy()
        p_pred = p_prev.copy()

        # Dilute if needed
        if self.dithered:
            if u[1]:
                x_pred, p_pred = self.dilute_dithered(x_prev, p_prev)
        else:
            x_pred, p_pred = self.dilute(x_prev, p_prev)

        # Approximate abundance and their varaince after dt seconds of growth
        temp = np.full(3,u[0],dtype='float')
        for i in range(round(dt/self.ts)):
            # Get delayed temperature values
            self.temps = self.lp_fac*self.temps + (1-self.lp_fac)*temp
            # Get growth rates
            gr = self.getGrowthRates(self.temps)
            # Jacobian of growth model
            A = np.array([[1 + self.ts_h*gr[0], 0, 0],
                      [0, 1 + self.ts_h*gr[1], 0],
                      [0, self.ts_h*gr[2], 1]])
            self.L = np.diag([self.ts_h*x_pred[0], self.ts_h*x_pred[1], self.ts_h*x_pred[1]])

            # Abundance after Ts seconds
            # Euler forward (not significantly less accurate than Runge - Kutta 4th order)
            x_pred = x_pred + self.ts_h*gr*np.array([x_pred[0], x_pred[1], x_pred[1]])

            p_pred = A @ p_pred @ A.T + self.L @ self.Q @ self.L.T

        return dict(zip(x_prev_dic, x_pred)), p_pred
    
    def update(self, x_pred_dic: dict, p_pred: np.ndarray, y: np.ndarray, p_est_od: float, p_est_fl: float) -> Tuple[dict, np.ndarray]:
        '''
        Update Step
        
        Updates the states and their variance after a measurement y.
        '''
        x_pred = np.fromiter(x_pred_dic.values(),dtype=float)
        od = x_pred[0] + x_pred[1]

        H = np.array([[self.parameters['od_fac'], self.parameters['od_fac'], 1]]) # [-x_pred[2]/(od+self.parameters['od_ofs'])**2, -x_pred[2]/(od+self.parameters['od_ofs'])**2, 1/(od+self.parameters['od_ofs'])]
        y_est = x_pred[2] + self.parameters['od_fac']*od # x_pred[2]/(od+self.parameters['od_ofs']) + self.parameters['fl_ofs'][r_ind]
        m = 1
        if y[0]:
            H = np.append([1, 1, 0], H)
            y_est = np.array([od, y_est])
            m += 1
        if p_est_od:
            H = np.append(H, [0, 1, 0])
            y_est = np.append(y_est, x_pred[1])
            m += 1
        if p_est_fl:
            H = np.append(H, [0, 1, 0])
            y_est = np.append(y_est, x_pred[1])
            m += 1
        
        # K = np.linalg.solve(H @ p_pred.T @ H.T + self.M @ self.parameters['r'].T @ self.M.T, H @ p_pred.T).T
        K = p_pred @ H.T @ np.linalg.inv(H @ p_pred @ H.T + self.M[:m,:m] @ self.R[:m,:m] @ self.M[:m,:m].T)
        xm = x_pred + K @ (y - y_est)
        xm[np.isnan(xm)] = x_pred[np.isnan(xm)]
        Pm = (np.eye(3) - K @ H) @ p_pred

        # Constrain states to be positive
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
        gr_e = self.parameters['gr_e'][0]*temp[0] + self.parameters['gr_e'][1]
        gr_p = self.parameters['gr_p'][0]*temp[1]**2 + self.parameters['gr_p'][1]*temp[1] + self.parameters['gr_p'][2]
        gr_f = self.parameters['gr_fp'][0]*temp[2]**2 + self.parameters['gr_fp'][1]*temp[2] + self.parameters['gr_fp'][2]
        gr_f = max(100,gr_f)

        return np.array([gr_e, gr_p, gr_f])