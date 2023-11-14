import numpy as np
from typing import Tuple

from params import EstParam, ModelParam
from model import Model

class EKF:
    """
    Extended Kalman Filter class used to estimate the states (abundance of p. putida, e.coli and fluorescent protein) of the system.

    Attributes
    ----------
    estParam : EstConst
        Estimation constants
    """

    def __init__(self, n_reactors: int, dithered: bool): # TODO: add od setpoint, dilution threshold and dilution amount as arguments
        """
        Initialize the estimator. Sets the mean and covariance of the initial
        estimate.

        Parameters
        ----------
        estParam : EstConst TODO: update
            Estimation constants
        """
        self.n_reactors = n_reactors
        self.dithered = dithered
        self.estParam = EstParam()
        self.modelParam = ModelParam()
        self.model = Model(self.dithered, self.estParam.Ts)
        self.time_prev = np.zeros(self.n_reactors)
        self.temp_prev = np.zeros(self.n_reactors)

        e0 = self.estParam.e_rel_init * self.estParam.od_init
        p0 = (1 - self.estParam.e_rel_init) * self.estParam.od_init
        fp0 = (np.full((1,self.n_reactors), self.estParam.fl_init) - self.modelParam.min_fl)*self.estParam.od_init
        est = np.full((self.n_reactors, 2),[e0, p0])
        self.est = np.concatenate((est, fp0.T), axis=1)

        self.var = np.full((self.n_reactors, self.estParam.num_states, self.estParam.num_states),np.diag([0.1, 0.1, 0.1]))
        
    def reset(self):
        self.model = Model(self.dithered, self.estParam.Ts)
    
    def estimate(self, r_ind: int, time: float, u: float, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform prediction step of the states of the system using the extended Kalman filter.

        Parameters
        ----------
        xm : np.ndarray, dim: (num_states,) TODO: update
            The mean of the previous state estimate. The order of states is
            given by x = [e, p, fp].
        Pm : np.ndarray, dim: (num_states, num_states)
            The covariance of the previous state estimate. The order of states
            is given by x = [e, p, fp].
        u : np.ndarray, dim: (num_inputs,)
            The input to the system. The order of inputs is given by u = temp_prev.

        Returns
        -------
        xp : np.ndarray, dim: (num_states,)
            The mean of the current state estimate. The order of states is
            given by x = [e, p, fp].
        Pp : np.ndarray, dim: (num_states, num_states)
            The covariance of the current state estimate. The order of states
            is given by x = [e, p, fp].
        """
        # Prediction
        if self.time_prev[r_ind] == 0: # First time step
            self.time_prev[r_ind] = time
            self.temp_prev[r_ind] = u
        else: # Prediction step
            dt = time - self.time_prev[r_ind]
            self.est[r_ind], self.var[r_ind] = self.model.predict(self.est[r_ind], self.var[r_ind], self.temp_prev[r_ind], dt)

            self.time_prev[r_ind] = time
            self.temp_prev[r_ind] = u

        # MeasurementUpdate
        self.est[r_ind], self.var[r_ind] = self.model.update(r_ind, self.est[r_ind], self.var[r_ind], y)

        return self.est[r_ind], self.var[r_ind]