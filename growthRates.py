from popParam import ModParam

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

param = ModParam()

def getGrowthRate(bact, temp, parameters):
    match bact:
        case "e":
            gr = parameters.Del_e*temp**3 + parameters.Gam_e*temp**2 + parameters.Beta_e*temp + parameters.Alpha_e
        case "p":
            gr = parameters.Del_p*temp**3 + parameters.Gam_p*temp**2 + parameters.Beta_p*temp + parameters.Alpha_p
        case "f":
            # beta_f = (parameters.c_sl - parameters.c_sh)/(parameters.T_sl - parameters.T_sh)
            # alpha_f = parameters.c_sl - beta_f*parameters.T_sl
            # gr = beta_f*temp + alpha_f
            # if gr.size > 1:
            #     gr[gr<parameters.c_sh] = parameters.c_sh
            #     gr[gr>parameters.c_sl] = parameters.c_sl
            # else:
            #     gr = max(parameters.c_sh, gr)
            #     gr = min(parameters.c_sl, gr)
            gr = 0
            if temp == 26:
                gr = 0.17
            elif temp == 27:
                gr = 0.25
            elif temp == 29:
                gr = 0.22
            elif temp == 30:
                gr = 0.22
            elif temp == 31:
                gr = 0.24
            elif temp == 33:
                gr = 0.22
            elif temp == 35:
                gr = 0.004
            elif temp == 37:
                gr = 0.004
        case _:
            raise Exception('bact should be e or p. The value of bact was: {}'.format(bact))
    return gr

def plotGrowthRates():
    temp = np.arange(param.T_l,param.T_h,1)
    grP = getGrowthRate("p", temp)
    grE = getGrowthRate("e", temp)
    grF = getGrowthRate("f", temp)
    plt.plot(temp,grE,'-xb',label='E coli.')
    plt.plot(temp,grP,'-xg',label='P. Putida')
    plt.plot(temp,grF,'-xk',label='Fl. Protein')
    plt.xlabel("Temperature [°C]")
    plt.ylabel("Growth Rate [$h^{-1}$]")
    plt.legend()
    plt.savefig("Images/growthRates.png")
    return

def f(xy):
    x, y = xy
    z = np.array([y - getGrowthRate("e", x),
                  y - getGrowthRate("p", x)])
    return z

def getCritTemp():
    temp = fsolve(f, [33.0, 0.8])
    return temp