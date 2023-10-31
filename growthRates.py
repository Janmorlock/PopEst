from popParam import Param

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

param = Param()

def getGrowthRate(bact, temp):
    match bact:
        case "e":
            gr = param.Del_e*temp**3 + param.Gam_e*temp**2 + param.Beta_e*temp + param.Alpha_e
        case "p":
            gr = param.Del_p*temp**3 + param.Gam_p*temp**2 + param.Beta_p*temp + param.Alpha_p
        case "f":
            gr = param.Gam_f*temp**2 + param.Beta_f*temp + param.Alpha_f
            if gr.size > 1:
                gr[gr<0.2] = 0.2
            else:
                gr = max(0.2, gr)
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