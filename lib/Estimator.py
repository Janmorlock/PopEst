import numpy as np
import math

from lib.Model import CustModel

class EKF:
    """
    Extended Kalman Filter class used to estimate the states (abundance of p. putida, e.coli and fluorescent protein) of the system.

    Attributes
    ----------
    model : CustModel
        Model that is used to predict and update the state estimate.
    est : np.ndarray, dim: (num_states,)
        The mean of the previous state estimate. The order of states is
        given by x = [e, p, fp, fl_ofs].
    var : np.ndarray, dim: (num_states, num_states)
        The covariance of the previous state estimate. The order of states
        is given by x = [e, p, fp, fl_ofs].
    dev_ind : int
        The index of the reactor.
    update : bool
        Whether to perform the measurement update step.
    time_prev : float
        The time of the previous measurement.
    temp_prev : float
        The temperature of the previous measurement.
    """
    def __init__(self, model: CustModel = CustModel(), dev_ind: int = 0, update: bool = True):
        """
        Initialize the estimator. Sets the mean and covariance of the initial
        estimate.

        Parameters
        ----------
        model : CustModel
            Model that is used to predict and update the state estimate.
        dev_ind : int
            The index of the reactor.
        update : bool, optional
            Whether to perform the measurement update step. Default is True.
        """
        self.model = model
        self.dev_ind = dev_ind
        self.update = update

        self.cc_ind = 0
        self.time_prev = 0.0
        self.e_ofs = 0
        self.e_fac = 1

        self.time_lst = []
        self.temp_lst = []
        self.od_lst = []
        self.fl_lst = []
        self.dil_prev = False

        self.est = {  # Initial state
            'e': model.parameters['od_init']*model.parameters['e_rel_init'],
            'p': model.parameters['od_init']*(1-model.parameters['e_rel_init']),
            'fp': model.parameters['fp_init']
        }
        self.var = np.diag([model.parameters['sigma_e_init']**2, model.parameters['sigma_p_init']**2, model.parameters['sigma_fp_init']**2])

    def initialise(self, m0_key):
        e_ofs_lst = list(self.model.parameters['e_ofs'].values())
        e_fac_lst = list(self.model.parameters['e_fac'].values())
        self.e_ofs = e_ofs_lst[self.model.parameters['e_ofs'].index(m0_key) + self.dev_ind]
        self.e_fac = e_fac_lst[self.model.parameters['e_fac'].index(m0_key) + self.dev_ind]

    def estimate(self, time: float, u: np.ndarray, y: np.ndarray = np.zeros(0)):
        """
        Update system state and variance estimate by performing prediction and measurement update step with the Extended Kalman Filter.

        Parameters
        ----------
        time : float
            The time in s at which the measurement is obtained.
        u : np.ndarray, dim: (num_inputs,)
            The input u = [temp_prev, dilute] to the system.
        y : np.ndarray, dim: (num_outputs,), optional
            The measurement of the system. The order of outputs is given by y = [fl, od].
            Will be ignored when update set to False.

        Returns
        -------
        None
        """

        # Prediction
        if self.time_prev != 0: # Skip on first time step
            dt = time - self.time_prev
            self.est, self.var = self.model.predict(self.est, self.var, u, dt)

        self.time_prev = time

        # Measurement Update
        if self.update:
            # Normalize fluorescent measurements
            y[1] = (y[1] - self.e_ofs)/self.e_fac
            # Store time, od and fl for later measurement update
            self.time_lst.append(time)
            self.temp_lst.append(u[0])
            self.od_lst.append(y[0])
            self.fl_lst.append(y[1])
            p_est_od = 0
            p_est_fl = 0
            if u[1]:
                if not self.dil_prev: # TODO: Discard if time too long
                    temp_avg = np.mean(self.temp_lst)
                    gr = self.model.getGrowthRates(np.full(3,temp_avg))
                    self.temp_lst -= self.temp_lst[-1]
                    p_est_od = np.linalg.lstsq(np.exp(gr[1]*self.time_lst) - np.exp(gr[0]*self.time_lst), self.od_lst - self.od_lst[-1]*np.exp(gr[0]*self.time_lst))
                    p_est_fl = np.linalg.lstsq(gr[2]/gr[1]*(np.exp(gr[1]*self.time_lst) - 1), self.fl_lst - self.fl_lst[-1])
                self.time_lst, self.temp_lst, self.od_lst, self.fl_lst = [], [], [], []
            if abs(self.est['e'] + self.est['p'] - y[0]) < 0.1:
                y[0] = 0
            else:
                print('no measurement update [{}] [{}:{}]'.format(self.dev_ind, math.floor(time/3600), math.floor((time/3600-math.floor(time/3600))*60)))
            self.est, self.var = self.model.update(self.est, self.var, y, p_est_od, p_est_fl)
            self.dil_prev = u[1]