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
        self.time_prev = -1
        self.e_ofs = 0
        self.e_fac = 1

        self.time_lst = []
        self.temp_lst = []
        self.od_lst = []
        self.fp_lst = []
        self.u_prev = np.array([23, False])

        self.p_est_fl = 0
        self.p_est_od = 0

        self.p_est_od_res = 0
        self.p_est_fl_res = 0

        self.est = {  # Initial state
            'e': model.parameters['od_init']*model.parameters['e_rel_init'],
            'p': model.parameters['od_init']*(1-model.parameters['e_rel_init']),
            'fp': model.parameters['fp_init']
        }
        self.var = np.diag([model.parameters['sigma_e_init']**2, model.parameters['sigma_p_init']**2, model.parameters['sigma_fp_init']**2])

    def set_r_coeff(self, m0_key):
        e_ofs_lst = list(self.model.parameters['e_ofs'].values())
        e_fac_lst = list(self.model.parameters['e_fac'].values())
        self.e_ofs = e_ofs_lst[list(self.model.parameters['e_ofs'].keys()).index(m0_key) + self.dev_ind]
        self.e_fac = e_fac_lst[list(self.model.parameters['e_fac'].keys()).index(m0_key) + self.dev_ind]

    def estimate(self, time: float, u: np.ndarray, y: np.ndarray = np.zeros(0)):
        """
        Update system state and variance estimate by performing prediction and measurement update step with the Extended Kalman Filter.

        Parameters
        ----------
        time : float
            The time in s at which the measurement is obtained.
        u : np.ndarray, dim: (num_inputs,)
            The next input u = [temp, dilute] to the system.
        y : np.ndarray, dim: (num_outputs,), optional
            The measurement of the system. The order of outputs is given by y = [fl, od].
            Will be ignored when update set to False.

        Returns
        -------
        None
        """

        # Prediction
        if self.time_prev >= 0: # Skip on first time step
            dt = time - self.time_prev
            self.est, self.var = self.model.predict(self.est, self.var, self.u_prev, dt)

        # Measurement Update
        if self.update:
            # Normalize fluorescent measurements
            y[1] = (y[1] - self.e_ofs)/self.e_fac
            # Store time, od and fl for later measurement update
            self.time_lst.append(time/3600)
            self.temp_lst.append(self.model.temps)
            self.od_lst.append(y[0])
            self.fp_lst.append(y[1] - self.model.parameters['od_fac']*(self.est['e'] + self.est['p']))
            self.p_est_od = 0
            self.p_est_fl = 0
            self.p_est_od_res = 0
            self.p_est_fl_res = 0
            temp_avg = 0
            if u[1]:
                if not self.u_prev[1] and len(self.temp_lst) > 4: # TODO: Discard if time too long
                    # calculate self.p_est_od and self.p_est_fl just before dilution
                    temp_avg = np.mean(self.temp_lst[:-1], axis = 0) # exclude current temp
                    gr = self.model.getGrowthRates(temp_avg)
                    self.time_lst = np.array(self.time_lst) - self.time_lst[-1]

                    A_od = np.vstack(np.exp(gr[1]*self.time_lst) - np.exp(gr[0]*self.time_lst))
                    b_od = self.od_lst - self.od_lst[-1]*np.exp(gr[0]*self.time_lst)
                    [self.p_est_od], [self.p_est_od_res] = np.linalg.lstsq(A_od, b_od, rcond = None)[0:2]
                    self.p_est_od_res = np.sqrt(self.p_est_od_res/len(self.od_lst))
                    self.p_est_od = max(0, self.p_est_od)
                    self.p_est_od = min(y[0], self.p_est_od)

                    A_fl = np.vstack(gr[2]/gr[1]*(np.exp(gr[1]*self.time_lst) - 1))
                    b_fl = self.fp_lst - self.fp_lst[-1]
                    [self.p_est_fl], [self.p_est_fl_res] = np.linalg.lstsq(A_fl, b_fl, rcond = None)[0:2]
                    self.p_est_fl_res = np.sqrt(self.p_est_fl_res/len(self.fp_lst))
                    self.p_est_fl = max(0, self.p_est_fl)
                    self.p_est_fl = min(y[0], self.p_est_fl)
                self.time_lst, self.temp_lst, self.od_lst, self.fp_lst = [], [], [], []
            if abs(self.est['e'] + self.est['p'] - y[0]) > 0.1:
                print('WARNING: no od measurement update [{}] [{}:{}]'.format(self.dev_ind, math.floor(time/3600), math.floor((time/3600-math.floor(time/3600))*60)))
            self.est, self.var = self.model.update(self.est, self.var, y, self.p_est_od, self.p_est_fl, self.p_est_od_res, self.p_est_fl_res, temp_avg)

        self.time_prev = time
        self.u_prev = u