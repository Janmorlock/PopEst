import numpy as np

class EKF:
    """
    Extended Kalman Filter class used to estimate the states (abundance of p. putida, e.coli and fluorescent protein) of the system.

    Attributes
    ----------
    estParam : EstConst
        Estimation constants
    """
    def __init__(self, model, r_ind, x0, p0: np.ndarray):
        """
        Initialize the estimator. Sets the mean and covariance of the initial
        estimate.

        Parameters
        ----------
        estParam : EstConst TODO: update
            Estimation constants
        """
        self.model = model
        self.time_prev = 0
        self.temp_prev = 0

        self.est = x0
        self.var = p0
        self.r_ind = r_ind

    def estimate(self, time: float, u: float, y: np.ndarray):
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
        if self.time_prev == 0: # First time step
            self.time_prev = time
            self.temp_prev = u
        else: # Prediction step
            dt = time - self.time_prev
            self.est, self.var = self.model.predict(self.r_ind, self.est, self.var, self.temp_prev, dt)

            self.time_prev = time
            self.temp_prev = u

        # MeasurementUpdate
        # self.est, self.var = self.model.update(self.r_ind, self.est, self.var, y)