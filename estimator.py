import numpy as np

class EKF:
    """
    Extended Kalman Filter class used to estimate the states (abundance of p. putida, e.coli and fluorescent protein) of the system.

    Attributes
    ----------
    model : CustModel or CustProgModel
        Model that is used to predict and update the state estimate.
    est : np.ndarray, dim: (num_states,)
        The mean of the previous state estimate. The order of states is
        given by x = [e, p, fp, fl_ofs].
    var : np.ndarray, dim: (num_states, num_states)
        The covariance of the previous state estimate. The order of states
        is given by x = [e, p, fp, fl_ofs].
    r_ind : int
        The index of the reactor.
    update : bool
        Whether to perform the measurement update step.
    time_prev : float
        The time of the previous measurement.
    temp_prev : float
        The temperature of the previous measurement.
    """
    def __init__(self, model, r_ind: int, x0, p0: np.ndarray, update: bool = True):
        """
        Initialize the estimator. Sets the mean and covariance of the initial
        estimate.

        Parameters
        ----------
        model : CustModel or CustProgModel
            Model that is used to predict and update the state estimate.
        r_ind : int
            The index of the reactor.
        x0 : dict[str, float], dim: (num_states,)
            The mean of the initial state estimate. The order of states is given by x = [e, p, fp, fl_ofs].
        p0 : np.ndarray, dim: (num_states, num_states)
            The covariance of the initial state estimate. The order of states is given by x = [e, p, fp, fl_ofs].
        update : bool, optional
            Whether to perform the measurement update step. Default is True.
        """
        self.model = model

        self.est = x0.copy()
        self.var = p0.copy()

        self.r_ind = r_ind
        self.update = update
        self.time_prev = 0.0
        self.temp_prev = 0.0

    def estimate(self, time: float, u: float, y: np.ndarray = np.zeros(0)):
        """
        Update system state and variance estimate by performing prediction and measurement update step with the Extended Kalman Filter.

        Parameters
        ----------
        time : float
            The time in s at which the measurement is obtained.
        u : float
            The input u = temp_prev to the system.
        y : np.ndarray, dim: (num_outputs,), optional
            The measurement of the system. The order of outputs is given by y = [fl, od].
            Will be ignored when update set to False.

        Returns
        -------
        None
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
        if self.update:
            self.est, self.var = self.model.update(self.r_ind, self.est, self.var, y)