# imports
import numpy as np
import math as m
import os
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

def lpm_model_prediction(tO,O,tP,P,tS,S,tT,T,tW,W,tq,q):
    ''' Plots extrapolation of the model into the future for 2 cycles of injection/extraction.
    
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
    # for future q - an estimate of the production
    average_production = (sum(O) + sum(W)) / (np.count_nonzero(O) + np.count_nonzero(W))

    num_cycles = 2 # future cycles of injection/extraction

    # define derivative function
    def lpm_pressure(pi, t, a, b, q): 
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
        q : array-like
            source/sink rates

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
        for t0,t1 in zip(t[:-1],t[1:]):
            dPdt1 = lpm_pressure(Pm[-1]-P0, t0, a, b, q) 
            Pp = Pm[-1] + dPdt1*(t1-t0)
            dPdt2 = lpm_pressure(Pp-P0, t1, a, b, q)
            Pm.append(Pm[-1] + 0.5*(t1-t0)*(dPdt2+dPdt1))
        return np.interp(t, t, Pm)
    
    def q_future(num_cycles, inj_multiplier, average_production, W, O, S):
        ''' Returns an array of q values with prediction into the future.

            Parameters:
            -----------
            num_cycles : float
                Number of cycles to predict into the future
            inj_multiplier : float
                Amount by which to scale steam injection (500 tonnes = 1x)
            average_production : float
                The average extraction from the pilot project
            W : array-like
                Data of water production rate at tW from the study (kg/day).
            O : array-like
                Data for oil production rate at tO from the study (kg/day).
            S : array-like
                Data for steam injection rate at tS from the study (kg/day).
            
            Returns:
            --------
            q : array-like
                Combined mass rate including future prediction (kg/day).       

        '''
        # create future injection/production values
        inj_1cycle = np.full(60, inj_multiplier*500000)
        inj_1cycle = np.concatenate((inj_1cycle, np.zeros(90)))
        S_rest = np.full(240, 0)
        S_future = np.tile(inj_1cycle, num_cycles)
        S = np.concatenate((S, S_rest, S_future))

        prod_1cycle = np.full(90, inj_multiplier*average_production*(25/23)) # 25/23 = 500/460 tonnes
        prod_1cycle = np.concatenate((np.zeros(60), prod_1cycle))
        O_rest = np.full(240, 0)
        O_future = np.tile(prod_1cycle, num_cycles)      # q_water and q_oil together - ie extraction
        O = np.concatenate((O, O_rest, O_future))
        W = np.concatenate((W, O_rest, np.zeros(150*num_cycles)))

        q = W + O - S

        return q

    t_rest = np.arange(tq[-1], 457, 1)
    t_future = np.arange(457, 457+150*num_cycles, 1)
    tq = np.concatenate((tq, t_rest, t_future))
    
    # new time for model
    t_extrap = np.concatenate((t_step_P, np.arange(t_step_P[-1], 457, 1), np.linspace(457, 457+150*num_cycles)))

    # hard coded parameters from calibration
    ap = 1.68098207e-04
    bp = 5.30078898e-02
    P0 = 8.07784024e+02
 
    bT = 3.83846888e-02
    M0 = 5.71951305e+06
    T0 = 1.44183580e+02

    f,ax = plt.subplots(nrows=1,ncols=1)
    ax.plot(tP, P, 'bo', label='observations')

    multipliers = [0, 0.5, 1, 2] # the different scenarios
    color=iter(cm.rainbow(np.linspace(0,1,len(multipliers))))

    for multiplier in multipliers: # make a prediction for each scenario
        q = q_future(num_cycles, multiplier, average_production, W, O, S)
        P_predict = solve_lpm_pressure(t_extrap, ap, bp, P0)
        c=next(color)
        ax.plot(t_extrap, P_predict, '-', c=c, label='pump {:.0f} tonnes'.format(multiplier*500))
    
    Pm = solve_lpm_pressure(t_step_P, ap, bp, P0)
    ax.plot(t_step_P, Pm, 'k-', label='model')   

    ax.set_ylabel("pressure [kPa]",size=14); ax.set_xlabel("time [days]",size=14)
    ax.set_xlim([0, t_extrap[-1]])
    ax.legend(prop={'size':9})
    ax.set_title('aP={:2.1e},   bP={:2.1e},   P0={:2.1e}'.format(ap, bp, P0),size=14)

    f.set_size_inches(13.5, 6)
    plt.savefig("figures/model_prediction_pressure.png")
    # plt.show()



    def lpm_temp(Ti, t, b, M0, S):
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
        S : array-like
            steam injection rates

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
        for t0,t1 in zip(t[:-1],t[1:]):
            dTdt1 = lpm_temp(Tm[-1]-T0, t0, b, M0, S)
            Tp = Tm[-1] + dTdt1*(t1-t0)
            dTdt2 = lpm_temp(Tp-T0, t1, b, M0, S)
            Tm.append(Tm[-1] + 0.5*(t1-t0)*(dTdt2+dTdt1))
        return np.interp(t, t, Tm)

    def S_future(num_cycles, inj_multiplier, S):
        ''' Returns an array of S values with prediction into the future.

            Parameters:
            -----------
            num_cycles : float
                Number of cycles to predict into the future
            inj_multiplier : float
                Amount by which to scale steam injection (500 tonnes = 1x)
            S : array-like
                Data for steam injection rate at tS from the study (kg/day).
            
            Returns:
            --------
            S : array-like
                Steam injection rate including future prediction (kg/day).

        '''
        # create future q_steam values
        inj_1cycle = np.full(60, inj_multiplier*500000)
        inj_1cycle = np.concatenate((inj_1cycle, np.zeros(90)))
        S_rest = np.full(240, 0)
        S_future = np.tile(inj_1cycle, num_cycles)
        S = np.concatenate((S, S_rest, S_future))
        return S

    # new time for model
    t_extrap = np.concatenate((t_step_T, np.arange(t_step_T[-1], 457, 1), np.linspace(457, 457+150*num_cycles)))

    current_dir = os.getcwd()
    os.chdir(current_dir+os.sep+'data') # set working directory to retrieve from data folder

    f,ax = plt.subplots(nrows=1,ncols=1)
    ax.plot(tT, T, 'bo', label='observations')

    color=iter(cm.rainbow(np.linspace(0,1,len(multipliers))))
    for multiplier in multipliers:  # make a prediction for each scenario
        S = S_future(num_cycles, multiplier, S)
        T_predict = solve_lpm_temp(t_extrap, bT, M0, T0)
        c=next(color)
        ax.plot(t_extrap, T_predict, '-', c=c, label='pump {:.0f} tonnes'.format(multiplier*500))
        # reset S
        tS,S = np.genfromtxt('tr_steam.txt',delimiter=',',skip_header=1).T  
        S = S*1000
        S = np.interp(tq[:58], tS, S)
        
    os.chdir(current_dir)   # reset to current directory
    
    tq = tq[:58]    # reset tq
    Tm = solve_lpm_temp(t_step_T, bT, M0, T0)
    ax.plot(t_step_T, Tm, 'k-', label='model')   

    ax.set_ylabel("temperature [°C]",size=14); ax.set_xlabel("time [days]",size=14)
    ax.set_xlim([0, t_extrap[-1]])
    ax.legend(prop={'size':9})
    ax.set_title('b={:2.1e},   M0={:2.1e},   T0={:2.1e}'.format(bT, M0, T0),size=14)
    plt.hlines(240, 0, 760, colors='grey', linestyles='dashed')

    f.set_size_inches(13.5, 6)
    plt.savefig("figures/model_prediction_temp.png")
    # plt.show()