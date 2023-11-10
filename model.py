import numpy as np
from typing import Tuple

from params import ModelParam, ExpParam
from params import ExpParam

# initialize
class Model:
    def __init__(self, dithered: bool, ts: int):
        self.modelParam = ModelParam()
        self.expParam = ExpParam()
        self.dithered = dithered
        self.ts = ts
        self.lag_ind = int(self.modelParam.Lag*3600/self.ts) # Lag indeces
        self.temps = np.array([np.full(self.lag_ind+1,self.expParam.T_pre_e),np.full(self.lag_ind+1,self.expParam.T_pre_p)])

        self.A_dil = np.zeros((3,3))
        self.L_dil = np.eye(3)
        self.Q_dil = np.diag([self.modelParam.sigma_e_dil**2, self.modelParam.sigma_p_dil**2, self.modelParam.sigma_fp_dil**2])

        self.A = np.zeros((3,3))
        self.L = np.eye(3)
        self.Q = np.diag([self.modelParam.sigma_e**2, self.modelParam.sigma_p**2, self.modelParam.sigma_fp**2])

    def predict(self, x_prev: np.ndarray, p_prev: np.ndarray, u: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dilutes if needed and predicts states their variance after dt seconds.
        """
        x_dil = x_prev
        p_dil = p_prev
        # Dilution, checked at same rate as real system
        if self.dithered:
            if x_prev[0] + x_prev[1] > self.expParam.Dil_th:
                x_dil = (1 - self.expParam.Dil_amount/self.expParam.Dil_th)*x_prev
                # TODO: Jacobian of dilution
        else:
            if x_prev[0] + x_prev[1] > self.expParam.Od_setpoint:
                x_dil = (2*self.expParam.Od_setpoint/(x_prev[0] + x_prev[1]) - 1)*x_prev
                # Jacobian of dilution
                den = (x_prev[0] + x_prev[1])**2
                self.A_dil = np.array([[x_prev[1]/den-1, -x_prev[0]/den, 0],
                                       [-x_prev[1]/den, x_prev[0]/den-1, 0],
                                       [-x_prev[2]/den, -x_prev[2]/den, x_prev[2]/den-1]])
                p_dil = self.A_dil @ p_prev @ self.A_dil.T + self.L_dil @ self.Q_dil @ self.L_dil.T
            
        p_pred = p_dil
        x = x_dil
        # Approximate abundance and their varaince after dt seconds of growth
        temp = np.full(3,u)
        for i in range(round(dt/self.ts)):
            # Modifiy temperature
            self.temps = np.append(self.temps[:,1:],np.full(2,u),axis=1)
            if self.modelParam.Avg_temp:
                temp[0:2] = np.average(self.temps, axis=1)
            else:
                temp[0:2] = self.temps[:,0]
            # Jacobian of growth
            self.A = np.array([[1 + self.ts/3600*getGrowthRates(temp)[0], 0, 0],
                      [0, 1 + self.ts/3600*getGrowthRates(temp)[1], 0],
                      [0, self.ts/3600*getGrowthRates(temp)[1], 1]])
            self.L = np.diag([self.ts/3600*x[0], self.ts/3600*x[1], self.ts/3600*x[1]])
            # Abundance after Ts seconds
            x = x_dil + self.ts/3600*getGrowthRates(temp)*np.array([x_dil[0], x_dil[1], x_dil[1]])

            p_pred = self.A @ p_pred @ self.A.T + self.L @ self.Q @ self.L.T

        return x, p_pred

def getGrowthRates(temp: np.ndarray) -> np.ndarray:
    """
    Given the temperatures, return the corresponding growth rates
    """
    modelParam = ModelParam()
    gr_e = modelParam.Del_e*temp[0]**3 + modelParam.Gam_e*temp[0]**2 + modelParam.Beta_e*temp[0] + modelParam.Alpha_e
    gr_p = modelParam.Del_p*temp[1]**3 + modelParam.Gam_p*temp[1]**2 + modelParam.Beta_p*temp[1] + modelParam.Alpha_p

    beta_f = (modelParam.c_sl - modelParam.c_sh)/(modelParam.T_sl - modelParam.T_sh)
    alpha_f = modelParam.c_sl - beta_f*modelParam.T_sl
    gr_f = beta_f*temp[3] + alpha_f
    if gr_f.size > 1:
        gr_f[gr_f<modelParam.c_sh] = modelParam.c_sh
        gr_f[gr_f>modelParam.c_sl] = modelParam.c_sl
    else:
        gr_f = max(modelParam.c_sh, gr_f)
        gr_f = min(modelParam.c_sl, gr_f)

    return np.array([gr_e, gr_p, gr_f])