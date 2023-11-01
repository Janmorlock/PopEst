import numpy as np

from popParam import ModParam
from growthRates import getGrowthRate

class BactCult:
    """
    Class containing all relevant attributes of the bacterial culture.
    """

    def __init__(self, bacType, pop_init):
        self.bacType = bacType
        self.pop = pop_init
        self.param = ModParam()

    def grow(self, T):
        """
        Given the temperature, go one time step further and update the self.pop attribute.
        """
        self.pop *= 1 + self.param.Ts/3600*getGrowthRate(self.bacType, T)
        
    def dilute(self, pop_other):
        """
        Take self.dil_am out of the reactor and update self.pop correspondingly.
        """
        self.pop -= self.param.Dil_amount*self.pop/(self.pop + pop_other)

class FlProtein:
    """
    Class containing all relevant attributes of the flourescense.
    """

    def __init__(self, count_init):
        self.type = 'f'
        self.count = count_init
        self.param = ModParam()

    def produce(self, pop_p, T):
        """
        Given the temperature, go one time step further and update the self.count attribute.
        """
        self.count = self.count + self.param.Ts/3600*getGrowthRate(self.type, T)*pop_p
        
    def dilute(self, dil_am, dil_th):
        """
        Take self.dil_am out of the reactor and update self.count correspondingly.
        """
        self.count -= dil_am/dil_th*self.count