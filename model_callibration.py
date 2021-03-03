# imports
import numpy as np
import math as m
import os
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

def lpm_model(tO,O,tP,P,tS,S,tT,T,tW,W,tq,q):
    ''' Calibrates data to form pressure and temperature best-fit models using curve_fit and plots the models.
    
        Parameters:
        -----------
        tO : array-like
            Time array of oil production rate data (days).
        O : array-like
            Data for oil production rate at tO from the study (kg/day).
        tP : array-like
            Time array of pressure data (days).
        P : array-like
            Recorded pressure values during the study (kPA).
        tS : array-like
            Time array of steam injection rate data (days).
        S : array-like
            Data for steam injection rate at tS from the study (kg/day).
        tT : array-like
            Time array of the temperature data (days).
        T : array-like
            Recorded temperature values during the study (degrees C).
        tW : array-like
            Time array of the water production rate data (days).
        W : array-like
            Data of water production rate at tW from the study (kg/day).
        tq : array-like
            Time array of the combined mass rate (days).
        q : array-like
            Combined mass rate from the data ; W + O - S (kg/day).

        Returns:
        --------
        None    

    '''
    # define derivative function
    def lpm_pressure(pi, t, a, b):
        ''' Solve pressure LPM.

        Parameters:
        -----------
        pi : float
            pressure difference
        t :  array-like
            time vector
        a : float
            lumped parameter
        b : float
            lumped parameter

        Returns:
        --------
        Change in pressure over time.

        '''
        qi = np.interp(t, tq, q)
        return -a*qi -b*pi

    t_step_P = np.linspace(0,215,2151)

    # implement improved Euler step to solve ODE
    def solve_lpm_pressure(t, a, b, P0):
        ''' Solve pressure ODE using Improved Euler Method.

        Parameters:
        -----------
        t : array-like
            time vector
        a : float
            lumped parameter
        b : float
            lumped parameter
        P0 : float
            calibrated initial pressure parameter

        Returns:
        --------
        Interpolated pressure array.

        '''
        Pm = [P[0],]
        for t0,t1 in zip(t_step_P[:-1],t_step_P[1:]):
            dPdt1 = lpm_pressure(Pm[-1]-P0, t0, a, b)
            Pp = Pm[-1] + dPdt1*(t1-t0)
            dPdt2 = lpm_pressure(Pp-P0, t1, a, b)
            Pm.append(Pm[-1] + 0.5*(t1-t0)*(dPdt2+dPdt1))
        return np.interp(t, t_step_P, Pm)

    # use curve_fit to find best model
    pars,cov = curve_fit(solve_lpm_pressure, tP, P, [1,1,100])

    # print(pars)
    # print(cov)

    Pm = solve_lpm_pressure(t_step_P, *pars)
    f,ax = plt.subplots(nrows=1,ncols=1)
    ax.plot(tP, P, 'ro', label='observations')
    ax.plot(t_step_P, Pm, 'k-', label='model')
    ax.set_xlabel("time [days]",size=14)
    ax.set_ylabel("pressure [kPa]",size=14)
    ax.legend(prop={'size':14})
    ax.set_title('aP={:2.1e},   bP={:2.1e},    P0={:2.1e}'.format(*pars),size=14)
    
    plt.savefig("figures/lpm_pressure.png")
    # plt.show()

    # pressure model misfit
    Pm_obs = np.interp(tP, t_step_P, Pm) # get pressure model values at tP (time at observations of pressure)
    Stheta_P = Pm_obs-P
    f,ax = plt.subplots(nrows=1,ncols=1)
    ax.plot(tP, Stheta_P, 'rx')
    ax.set_xlabel("time [days]",size=14)
    ax.set_ylabel("pressure misfit [kPa]",size=14)
    plt.hlines(0, 0, 215, colors='grey', linestyles='dashed')
    
    plt.savefig("figures/model_misfit_pressure.png")
    # plt.show()

    ap = pars[0]
    bp = pars[1]
    P0 = pars[2]
    


    def lpm_temp(Ti, t, b, M0, T0):
        ''' Solve temperature LPM.

        Parameters:
        -----------
        Ti : float
            temperature difference
        t : array-like
            time vector
        b : float
            lumped parameter
        M0 : float
            initial mass
        T0 : float
            calibrated initial temperature outside the system

        Returns:
        --------
        dTdt: float
            Change in temperature over time.

        '''
        Tc = Ti + T0
        Sc = np.interp(t, tq, S)
        Pc = np.interp(t, t_step_P, Pm)
        pi = Pc - P0
        if pi > 0:
            Tf = 0
        else:
            Tf = Ti*-1
        dTdt = Sc*(260-Tc)/M0 - bp*pi*Tf/(ap*M0) - b*Ti
        return dTdt

    t_step_T = np.linspace(0,221,2211)

    # implement improved Euler step to solve ODE
    def solve_lpm_temp(t, b, M0, T0):
        ''' Solve temperature ODE using Improved Euler Method.

        Parameters:
        -----------
        t : array-like
            time vector
        b : float
            lumped parameter
        M0 : float
            initial mass
        T0 : float
            calibrated initial temperature outside the system

        Returns:
        --------
        Interpolated temperature array.

        '''
        Tm = [T[0],]
        for t0,t1 in zip(t_step_T[:-1],t_step_T[1:]):
            dTdt1 = lpm_temp(Tm[-1]-T0, t0, b, M0, T0)
            Tp = Tm[-1] + dTdt1*(t1-t0)
            dTdt2 = lpm_temp(Tp-T0, t1, b, M0, T0)
            Tm.append(Tm[-1] + 0.5*(t1-t0)*(dTdt2+dTdt1))
        return np.interp(t, t_step_T, Tm)

    # use curve_fit to find best model
    pars,cov = curve_fit(solve_lpm_temp, tT, T, [0.001,100000,100])

    # print(pars)
    # print(cov)

    Tm = solve_lpm_temp(t_step_T, *pars)
    f,ax = plt.subplots(nrows=1,ncols=1)
    ax.plot(tT, T, 'ro', label='observations')
    ax.plot(t_step_T, Tm, 'k-', label='model')
    ax.set_ylabel("temperature [°C]",size=14); ax.set_xlabel("time [days]",size=14)
    ax.legend(prop={'size':14})
    ax.set_title('aP={:2.1e},   bP={:2.1e}   P0={:2.1e}\nbT={:2.1e},   M0={:2.1e},   T0={:2.1e}'.format(ap,bp,P0,*pars),size=14)
    
    plt.savefig("figures/lpm_temp.png")
    # plt.show()

    # temperature model misfit
    Tm_obs = np.interp(tT, t_step_T, Tm) # get temperature model values at tT (time at observations of temperature)
    Stheta_T = Tm_obs-T
    f,ax = plt.subplots(nrows=1,ncols=1)
    ax.plot(tT, Stheta_T, 'rx')
    ax.set_xlabel("time [days]",size=14)
    ax.set_ylabel("temperature misfit [°C]",size=14)
    plt.hlines(0, 0, 221, colors='grey', linestyles='dashed')
    
    plt.savefig("figures/model_misfit_temperature.png")
    # plt.show()