# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

# Note: To preserve vectorization use numpy math functions (e.g., maximum, minimum, sign, sqrt, etc.) instead of non-vectorized functions (max, min, etc.)

from numpy import inf
import numpy as np
from progpy import PrognosticsModel
from params import Params


# Each function defines one or more derived parameters as a function of the other parameters.
def dil_callback(params):
    # Return format: dict of key: new value pair for at least one derived parameter
    return {
        'dil_th' : params['od_setpoint'] + params['zig'],
        'dil_amount' : params['zig']*(1.5 + 1),
    }

def lag_callback(params):
    return {
        'lag_ind' : int(params['lag']*3600/params['ts']) # Lag indeces
    }

def r_callback(params):
    return {
        'r' : np.diag([params['sigma_fl']**2, params['sigma_od']**2])
    }

def q_callback(params):
    return {
        'q' : np.diag([params['sigma_e']**2, params['sigma_p']**2, params['sigma_fp']**2])
    }


class CustProgModel(PrognosticsModel):
    """
    Template for Prognostics Model
    """

    # V Uncomment Below if the class is vectorized (i.e., if it can accept input to all functions as arrays) V
    is_vectorized = True

    # REPLACE THE FOLLOWING LIST WITH EVENTS BEING PREDICTED
    # events = [
    #     'Example Event'
    # ]
    
    inputs = [
        'temp'
    ]

    states = [
        'e',
        'p',
        'fp'
    ]

    outputs = [
        'fl',
        'od'
    ]

    # REPLACE THE FOLLOWING LIST WITH PERFORMANCE METRICS
    # i.e., NON-MEASURED VALUES THAT ARE A FUNCTION OF STATE
    # e.g., maximum torque of a motor
    # performance_metric_keys = [
    #     'metric 1',
    # ]

    # REPLACE THE FOLLOWING LIST WITH CONFIGURED PARAMETERS
    # Note- everything required to configure the model
    # should be in parameters- this is to enable the serialization features
    default_parameters = Params().default

    # Instance specific variables
    temps = np.array([np.full(default_parameters['lag_ind']+1,default_parameters['temp_h']),
                      np.full(default_parameters['lag_ind']+1,default_parameters['temp_l'])])
    ind = 0

    state_limits = {
        # 'state': (lower_limit, upper_limit)
        # only specify for states with limits
        'e': (0, inf),
        'p': (0, inf),
        'fp': (0, inf)
    }

    # Identify callbacks used by this model
    # See examples.derived_params
    # Format: "trigger": [callbacks]
    # Where trigger is the parameter that the derived parameters are derived from.
    # And callbacks are one or more callback functions that define parameters that are
    # derived from that paramete
    param_callbacks = {
        'zig' : [dil_callback],
        'od_setpoint' : [dil_callback],
        'lag' : [lag_callback],
        'ts' : [lag_callback],
        'sigma_od' : [r_callback],
        'sigma_fl' : [r_callback],
        'sigma_e' : [q_callback],
        'sigma_p' : [q_callback],
        'sigma_fp' : [q_callback]
    }

    # UNCOMMENT THIS FUNCTION IF YOU NEED CONSTRUCTION LOGIC (E.G., INPUT VALIDATION)
    # def __init__(self, **kwargs):
    #     """
    #     Constructor for model

    #     Note
    #     ----
    #     To use the JSON serialization capabilities in to_json and from_json, model.parameters must include everything necessary for initialize, including any keyword arguments.
    #     """
    #     # ADD OPTIONS CHECKS HERE

    #     # e.g., Checking for required parameters
    #     # if not 'required_param' in kwargs:
    #     #   throw Exception;

    #     super().__init__(**kwargs) # Run Parent constructor

    # Model state initialization - there are two ways to provide the logic to initialize model state.
    # 1. Provide the initial state in parameters['x0'], or
    # 2. Provide an Initialization function
    #
    # If following method 2, uncomment the initialize function, below.
    # Sometimes initial input (u) and initial output (z) are needed to initialize the model
    # In that case remove the '= None' for the appropriate argument
    # Note: If they are needed, that requirement propagated through to the simulate_to* functions
    # UNCOMMENT THIS FUNCTION FOR COMPLEX INITIALIZATION
    def initialize(self, j, n_reactors):
        """
        Calculate initial state given inputs and outputs
    
        Parameters
        ----------
        u : InputContainer
            Inputs, with keys defined by model.inputs.
            e.g., u = {'i':3.2} given inputs = ['i']
        z : OutputContainer
            Outputs, with keys defined by model.outputs.
            e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']
    
        Returns
        -------
        x : StateContainer
            First state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        """
    
        # REPLACE BELOW WITH LOGIC TO CALCULATE INITIAL STATE
        # NOTE: KEYS FOR x0 MATCH 'states' LIST ABOVE
    
        self.ind = j
        self.temps = np.array([np.full(self.parameters['lag_ind']+1,self.parameters['temp_h']),
                               np.full(self.parameters['lag_ind']+1,self.parameters['temp_l'])])
        x0 = {  # Initial state
            'e': self.parameters['od_init']*self.parameters['e_rel_init'],
            'p': self.parameters['od_init']*(1-self.parameters['e_rel_init']),
            'fp': (self.parameters['fl_init']-self.parameters['min_fl'][self.ind])*self.parameters['od_init'],
        }

        return self.StateContainer(x0)

    def getGrowthRates(self, temp: np.ndarray) -> np.ndarray:
        """
        Given the temperatures, return the corresponding growth rates
        """
        gr_e = self.parameters['beta_e']*temp[0] + self.parameters['alpha_e']
        gr_p = self.parameters['del_p']*temp[1]**3 + self.parameters['gam_p']*temp[1]**2 + self.parameters['beta_p']*temp[1] + self.parameters['alpha_p']

        beta_f = (self.parameters['c_sl'] - self.parameters['c_sh'])/(self.parameters['temp_sl'] - self.parameters['temp_sh'])
        alpha_f = self.parameters['c_sl'] - beta_f*self.parameters['temp_sl']
        gr_f = beta_f*temp[2] + alpha_f
        if gr_f.size > 1:
            gr_f[gr_f<self.parameters['c_sh']] = self.parameters['c_sh']
            gr_f[gr_f>self.parameters['c_sl']] = self.parameters['c_sl']
        else:
            gr_f = max(self.parameters['c_sh'], gr_f)
            gr_f = min(self.parameters['c_sl'], gr_f)

        return np.array([gr_e, gr_p, gr_f])
    
    # UNCOMMENT THIS FUNCTION FOR DISCRETE MODELS
    def next_state(self, x, u, dt):
        """
        State transition equation: Calculate next state
    
        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        u : InputContainer
            Inputs, with keys defined by model.inputs.
            e.g., u = {'i':3.2} given inputs = ['i']
        dt : number
            Timestep size in seconds (≥ 0)
            e.g., dt = 0.1
    
        Returns
        -------
        x : StateContainer
            Next state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        """
        x_dil = x.values()
        # Dilution, checked at same rate as real system
        if x['e'] + x['p'] > self.parameters['od_setpoint']:
            x_dil = (2*self.parameters['od_setpoint']/(x['e'] + x['p']) - 1)*x.values()

        # Approximate abundance and their varaince after dt seconds of growth
        x_next = x_dil
        temp = np.full(3,u['temp'])
        # for i in range(round(dt/self.parameters['ts'])):
        # Modifiy temperature
        self.temps = np.append(self.temps[:,1:],np.full((2,1),u['temp']),axis=1)
        if self.parameters['avg_temp']:
            temp[0:2] = np.average(self.temps, axis=1)
        else:
            temp[0:2] = self.temps[:,0]
        # Abundance after Ts seconds
        x_next = x_next + self.parameters['ts']/3600*self.getGrowthRates(temp)*np.array([x_next[0], x_next[1], x_next[1]])

        return self.StateContainer(dict(zip(x, x_next)))

    def output(self, x):
        """
        Calculate output, z (i.e., measurable values) given the state x

        Parameters
        ----------
        x : StateContainer
            state, with keys defined by model.states
            e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
        Returns
        -------
        z : OutputContainer
            Outputs, with keys defined by model.outputs.
            e.g., z = {'t':12.4, 'v':3.3} given inputs = ['t', 'v']
        """

        # REPLACE BELOW WITH LOGIC TO CALCULATE OUTPUTS
        # NOTE: KEYS FOR z MATCH 'outputs' LIST ABOVE
        z = self.OutputContainer({
            'fl': x['fp']/(x['e'] + x['p']) + self.parameters['min_fl'][self.ind],
            'od': x['e'] + x['p']
        })

        return z

    # def event_state(self, x):
    #     """
    #     Calculate event states (i.e., measures of progress towards event (0-1, where 0 means event has occurred))

    #     Parameters
    #     ----------
    #     x : StateContainer
    #         state, with keys defined by model.states
    #         e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
    #     Returns
    #     -------
    #     event_state : dict
    #         Event States, with keys defined by prognostics_model.events.
    #         e.g., event_state = {'EOL':0.32} given events = ['EOL']
    #     """

    #     # REPLACE BELOW WITH LOGIC TO CALCULATE EVENT STATES
    #     # NOTE: KEYS FOR event_x MATCH 'events' LIST ABOVE
    #     event_x = {
    #         'Example Event': 0.95
    #     }

    #     return event_x
        
    # Note: Thresholds met equation below is not strictly necessary.
    # By default, threshold_met will check if event_state is ≤ 0 for each event
    # def threshold_met(self, x):
    #     """
    #     For each event threshold, calculate if it has been met

    #     Parameters
    #     ----------
    #     x : StateContainer
    #         state, with keys defined by model.states
    #         e.g., x = {'abc': 332.1, 'def': 221.003} given states = ['abc', 'def']
        
    #     Returns
    #     -------
    #     thresholds_met : dict
    #         If each threshold has been met (bool), with keys defined by prognostics_model.events
    #         e.g., thresholds_met = {'EOL': False} given events = ['EOL']
    #     """

    #     # REPLACE BELOW WITH LOGIC TO CALCULATE IF THRESHOLDS ARE MET
    #     # NOTE: KEYS FOR t_met MATCH 'events' LIST ABOVE
    #     t_met = {
    #         'Example Event': False
    #     }

    #     return t_met

    # def performance_metrics(self, x) -> dict:
    #     """
    #     Calculate performance metrics where

    #     Parameters
    #     ----------
    #     x : StateContainer
    #         state, with keys defined by model.states \n
    #         e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
        
    #     Returns
    #     -------
    #     pm : dict
    #         Performance Metrics, with keys defined by model.performance_metric_keys. \n
    #         e.g., pm = {'tMax':33, 'iMax':19} given performance_metric_keys = ['tMax', 'iMax']

    #     Example
    #     -------
    #     | m = PrognosticsModel() # Replace with specific model being simulated
    #     | u = m.InputContainer({'u1': 3.2})
    #     | z = m.OutputContainer({'z1': 2.2})
    #     | x = m.initialize(u, z) # Initialize first state
    #     | pm = m.performance_metrics(x) # Returns {'tMax':33, 'iMax':19}
    #     """

    #     # REPLACE BELOW WITH LOGIC TO CALCULATE PERFORMANCE METRICS
    #     # NOTE: KEYS FOR p_metrics MATCH 'performance_metric_keys' LIST ABOVE
    #     p_metrics = {
    #         'metric1': 23
    #     }
    #     return p_metrics

    # V UNCOMMENT THE BELOW FUNCTION FOR DIRECT FUNCTIONS V
    # V i.e., a function that directly estimate ToE from  V
    # V x and future loading function                     V
    # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    # def time_of_event(self, x, future_loading_eqn = lambda t,x=None: {}, **kwargs) -> dict:
    #     """
    #     Calculate the time at which each :term:`event` occurs (i.e., the event :term:`threshold` is met). time_of_event must be implemented by any direct model. For a state transition model, this returns the time at which threshold_met returns true for each event. A model that implements this is called a "direct model".

    #     Args:
    #         x (StateContainer):
    #             state, with keys defined by model.states \n
    #             e.g., x = m.StateContainer({'abc': 332.1, 'def': 221.003}) given states = ['abc', 'def']
    #         future_loading_eqn (abc.Callable, optional)
    #             Function of (t) -> z used to predict future loading (output) at a given time (t). Defaults to no outputs

    #     Returns:
    #         time_of_event (dict)
    #             time of each event, with keys defined by model.events \n
    #             e.g., time_of_event = {'impact': 8.2, 'falling': 4.077} given events = ['impact', 'falling']

    #     Note:
    #         Also supports arguments from :py:meth:`simulate_to_threshold`

    #     See Also:
    #         threshold_met
    #     """
    #     # REPLACE BELOW WITH LOGIC TO CALCULATE IF TIME THAT EVENT OCCURS
    #     # NOTE: KEYS FOR t_event MATCH 'events' LIST ABOVE
    #     t_event = {
    #         'Example Event': 2330
    #     }

    #     return t_event
