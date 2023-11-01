from popParam import ModParam

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

param = ModParam()

def getGrowthRate(bact, temp):
    match bact:
        case "e":
            gr = param.Del_e*temp**3 + param.Gam_e*temp**2 + param.Beta_e*temp + param.Alpha_e
        case "p":
            gr = param.Del_p*temp**3 + param.Gam_p*temp**2 + param.Beta_p*temp + param.Alpha_p
        case "f":
            beta_f = (param.c_sl - param.c_sh)/(param.T_sl - param.T_sh)
            alpha_f = param.c_sl - beta_f*param.T_sl
            gr = beta_f*temp + alpha_f
            if gr.size > 1:
                gr[gr<param.c_sh] = param.c_sh
                gr[gr>param.c_sl] = param.c_sl
            else:
                gr = max(param.c_sh, gr)
                gr = min(param.c_sl, gr)
            # if temp == 26:
            #     gr = 0.15
            # elif temp == 27:
            #     gr = 0.17
            # elif temp == 29:
            #     gr = 0.15
            # elif temp == 30:
            #     gr = 0.17
            # elif temp == 31:
            #     gr = 0.18
            # elif temp == 33:
            #     gr = 0.15
            # elif temp == 35:
            #     gr = 0.002
            # elif temp == 37:
            #     gr = 0.002
        case _:
            raise Exception('bact should be e or p. The value of bact was: {}'.format(bact))
    return gr

def plotGrowthRates():
    temp = np.arange(26,38,1)
    grP = getGrowthRate("p", temp)
    grE = getGrowthRate("e", temp)
    grF = getGrowthRate("f", temp)
    plt.plot(temp,grE,'-xb',label='E coli.')
    plt.plot(temp,grP,'-xg',label='P. Putida')
    plt.plot(temp,grF,'-xk',label='Fl. Protein')
    plt.xlabel("Temperature [°C]")
    plt.ylabel("Growth Rate [h^{-h}]")
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