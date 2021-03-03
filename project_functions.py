# imports
import numpy as np
import math
from matplotlib import pyplot as plt
import os
from numpy.linalg import norm

def load_data():
    ''' Load the data from the text files.

        Parameters:
        -----------
        None

        Returns:
        --------
        t_oil : array-like
            Time array of oil production rate data (days).
        oil : array-like
            Data for oil production rate at tO from the study (kg/day).
        t_P : array-like
            Time array of pressure data (days).
        pressure : array-like
            Recorded pressure values during the study (kPA).
        t_steam : array-like
            Time array of steam injection rate data (days).
        steam : array-like
            Data for steam injection rate at tS from the study (kg/day).
        t_temp : array-like
            Time array of the temperature data (days).
        temp : array-like
            Recorded temperature values during the study (degrees C).
        t_water : array-like
            Time array of the water production rate data (days).
        water : array-like
            Data of water production rate at tW from the study (kg/day).

        Notes:
        ------
        File names hard coded.

    '''
    current_dir = os.getcwd()
    os.chdir(current_dir+os.sep+'data')

    t_oil,oil = np.genfromtxt('tr_oil.txt',delimiter=',',skip_header=1).T
    t_P,pressure = np.genfromtxt('tr_p.txt',delimiter=',',skip_header=1).T
    t_steam,steam = np.genfromtxt('tr_steam.txt',delimiter=',',skip_header=1).T
    t_temp,temp = np.genfromtxt('tr_T.txt',delimiter=',',skip_header=1).T
    t_water,water = np.genfromtxt('tr_water.txt',delimiter=',',skip_header=1).T

    os.chdir(current_dir)

    return t_oil,oil,t_P,pressure,t_steam,steam,t_temp,temp,t_water,water


def plot_data(tO,O,tP,P,tS,S,tT,T,tW,W):
    ''' Plot the data from the text files.

        Parameters:
        -----------
        t_oil : array-like
            Time array of oil production rate data (days).
        oil : array-like
            Data for oil production rate at tO from the study (kg/day).
        t_P : array-like
            Time array of pressure data (days).
        pressure : array-like
            Recorded pressure values during the study (kPA).
        t_steam : array-like
            Time array of steam injection rate data (days).
        steam : array-like
            Data for steam injection rate at tS from the study (kg/day).
        t_temp : array-like
            Time array of the temperature data (days).
        temp : array-like
            Recorded temperature values during the study (degrees C).
        t_water : array-like
            Time array of the water production rate data (days).
        water : array-like
            Data of water production rate at tW from the study (kg/day).

        Returns:
        --------
        None

    '''
    # initial plots of data
    f,ax1 = plt.subplots(nrows=1, ncols=1)
    ax2 = ax1.twinx()

    ax1.plot(tO, O, 'k.-', label='Production of bitumen (m^3/day)')
    ax2.plot(tS, S, 'g.-', label='Injection of steam\n(tonnes/day, at 260°C)')
    ax1.plot(tW, W, 'y.-', label='Production of water (m^3/day)')

    ax1.set_xlabel('time (days)', fontsize=14)
    ax1.set_ylabel('volume rate (m^3/day)', fontsize=14)
    ax2.set_ylabel('mass rate (tonnes/day)', fontsize=14)
    ax1.set_title('Summary of the pilot project\nSteam injection and oil-water mixture production rates', fontsize=14)
    ax1.legend(bbox_to_anchor=(0.753,1), fontsize=9)
    ax2.legend(bbox_to_anchor=(1,0.8))

    f.set_size_inches(13.5, 5)
    plt.savefig("figures/plot_data_1.png")
    # plt.show()

    f,ax3 = plt.subplots(nrows=1, ncols=1)
    ax4 = ax3.twinx()

    ax3.plot(tP, P, 'b.-', label='Pressure (kPa)')
    ax4.plot(tT, T, 'r.-', label='Temperature (°C)')

    ax3.set_xlabel('time (days)', fontsize=14)
    ax3.set_ylabel('pressure (kPa)', fontsize=14)
    ax4.set_ylabel('temperature (°C)', fontsize=14)
    ax3.set_title('Summary of the pilot project\nPressure and temperature levels at 350m depth in the well', fontsize=14)
    ax3.legend(bbox_to_anchor=(1,1))
    ax4.legend(bbox_to_anchor=(1,0.9))

    f.set_size_inches(13.5, 5)
    plt.savefig("figures/plot_data_2.png")
    # plt.show()


def solve_ode(f, t0, t1, dt, x0, q, pars):
    ''' Solve an ODE numerically.

        Parameters:
        -----------
        f : callable
            Function that returns dxdt given variable and parameter inputs.
        t0 : float
            Initial time of solution.
        t1 : float
            Final time of solution.
        dt : float
            Time step length.
        x0 : float
            Initial value of solution.
        q : array-like
            source/sink rate
        pars : array-like
            List of parameters passed to ODE function f.

        Returns:
        --------
        ts : array-like
            Independent variable solution vector.
        xs : array-like
            Dependent variable solution vector.

        Notes:
        ------
        ODE is solved using the Improved Euler Method. 

    '''
    # initialise
    nt = int(np.ceil((t1-t0)/dt))       # compute number of Euler steps to take
    ts = t0+np.arange(nt+1)*dt          # t array
    xs = 0.*ts                          # array to store solution
    xs[0] = x0                          # set initial value

    for i in range(nt):
            dydxk = f(ts[i], xs[i], q[i], *pars)                                # ODE function
            xs[i+1] = xs[i] + (dt/2)*(dydxk + f(ts[i], xs[i], q[i], *pars))     # take one IE step

    return ts, xs


def pressure_ode_model(t, p, q, a, b, p0):
    ''' Return the derivative dP/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable (time).
        p : float
            Dependent variable (pressure).
        q : float
            Source/sink rate.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        p0 : float
            Initial value of dependent variable (pressure).

        Returns:
        --------
        dPdt : float
            Derivative of dependent variable (pressure) with respect to independent variable.

        Examples:
        ---------
        >>> pressure_ode_model(3,4,2,7,5,2)
        -24

    '''
    dPdt = -(a*q + b*(p-p0))

    return dPdt


def temp_ode_model(t, T, q, p, P0, T0, M0, Ts, a, b, bt):
    ''' Return the derivative dT/dt at time, t, for given parameters.

        Parameters:
        -----------
        t : float
            Independent variable (time).
        T : float
            Dependent variable (temperature).
        q : float
            Source/sink rate.
        p : float
            Pressure value.
        P0 : float
            Initial value of pressure.
        T0 : float
            Initial value of temperature.
        M0 : float
            Mass recharge.
        Ts : float
            Steam temperature.
        a : float
            Source/sink strength parameter.
        b : float
            Recharge strength parameter.
        bt : float
            Conduction parameter.

        Returns:
        --------
        dTdt : float
            Derivative of dependent variable (temperature) with respect to independent variable.

        Examples:
        ---------
        >>> temperature_ode_model(3, 210, 1*10**-6, 1.5, 2, 200, 1*10**6, 260, 7, 5, 1)
        39.99999643

    '''
    if (p > P0):
        T_dash = T
    else:
        T_dash = T0

    dTdt = ((q/M0)*(Ts-T)) - ((b/(a*M0))*(p-P0)*(T_dash-T)) - (bt*(T-T0))

    return dTdt


def benchmark_pressure():
    ''' Plots of benchmark of both Numerical and Analytical solutions obtained for pressure.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

    '''
    # time vector
    t_interp = np.arange(2210)

    # NUMERICAL MODEL
    #q = 1, a = 1, b = 1
    a = 1
    b = 1
    q = np.ones(len(t_interp))
    t_model,pressure_model = solve_ode(pressure_ode_model, 0, 221, 0.1, 1291.76, q, [a, b, 1291.76])

    #ANALYTICAL MODEL
    #q = 1, a = 1, b = 1
    a=1
    b=1
    q=1
    #initialise arrays
    t_INT = np.linspace(0,221,2211)
    Analytic_P = np.zeros(2211)

    #equation
    for i in range(221):
        Analytic_P[i] = (((-a*q)/b)*(1-(math.exp(-b*t_INT[i])))) + 1291.76

    #plot both models on same figure
    f,ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(t_model, pressure_model, 'r*', label='numerical')
    ax.plot(t_INT, Analytic_P, 'b-', label='analytic')

    #plotting
    ax.set_xlabel('time, t[days]')
    ax.set_xlim([0.1, 10])
    ax.set_ylabel('pressure, P[kPa]')
    ax.set_ylim([1290.5, 1292])
    ax.set_title('Benchmark: a=1.0, b=1.0, q0=1.0')
    ax.legend(loc=1)

    plt.savefig("figures/benchmark_pressure.png")
    # plt.show()

def convergence_test_pressure():
    ''' A convergence is plot is obtained for different step sizes for pressure.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

    '''
    #Day 1 convergence test
    f,ax = plt.subplots(1,1)
    t_interp = np.arange(2211)

    inv_dt = []
    p_dt = []

    q = np.ones(len(t_interp))
    for g in range (10,101,1):
        t, x = solve_ode(pressure_ode_model, 0, 221, g/100, 1291.76, q, [1, 1, 1291.76])

        for i in range (1,len(t)):
            #day 1
            if t[i] <= 1.015 and t[i] >= 0.985:
                    inv_dt.append(1/(g/100))
                    p_dt.append(x[i])
        
        ax.plot(inv_dt, p_dt,'ro')

    # get final two step sizes and pressure values (to check for convergence and find suitable step size)
    h1 = inv_dt[0]
    h2 = inv_dt[1]
    P1 = p_dt[0]
    P2 = p_dt[1]
    
    #Labelling axis and title on the plot.
    ax.set_title('Step-size convergence plot')
    ax.set_ylabel('Day 1 Pressure Values at different step-sizes')
    ax.set_xlabel('1/h (step-size)')

    ax.set_ylim([1290.5, 1291.5])
    ax.set_xlim([0, 11])

    ax.text(9.1, 1291.17, 'h={:.2f}\nP={:.1f}'.format(1/h2,P2), ha='center', va='center', size=7)
    ax.text(10, 1291.04, 'h={:.2f}\nP={:.1f}'.format(1/h1,P1), ha='center', va='center', size=7)

    plt.savefig("figures/convergence_test_pressure.png")
    # plt.show()


def benchmark_temp():
    ''' Plots of benchmark of both Numerical and Analytical solutions obtained for temperature.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

    '''

    # time vector
    t_interp = np.arange(2210)

    # NUMERICAL MODEL
    M0 = 8.12*10**6
    Ts = 260
    a = 1
    b = 1
    P0 = 1291.76
    bt = 1
    T0 = 25
    q = np.ones(len(t_interp))*10**6
    p = 1

    temp_t_model,temp_model = solve_ode(temp_ode_model, 0, 221, 0.1, T0, q, [p, P0, T0, M0, Ts, a, b, bt])

    #ANALYTICAL MODEL

    #initialise arrays
    Analytic_T = np.zeros(2211)
    t = np.linspace(0,221,2211)
    #q is constant
    q= 1*10**6

    #equation
    for i in range(2211):
        Analytic_T[i] = ((q/M0)*math.exp(-t[i]*(q/M0 - (b*(p-P0)/(a*M0)) + bt))*(Ts*(math.exp(t[i]*(q/M0 - (b*(p-P0)/(a*M0)) + bt))-1)+T0) + (T0*(bt-(b*(p-P0)/(a*M0)))))/(q/M0 - (b*(p-P0)/(a*M0)) + bt)

    #plot both models on same figure
    f,ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(temp_t_model,temp_model, 'r*', label='numerical')
    ax.plot(t, Analytic_T, 'b-', label='analytic')

    #plotting
    ax.set_xlabel('time, t[days]')
    ax.set_xlim([0, 5])
    ax.set_ylabel('temp, [°C]')
    
    ax.set_title('Benchmark: a=1.0, b=1.0, bt=1.0,\nP0=1291.76, M0= 8.12*10^6, T0=25')
    ax.legend(loc=1)

    plt.savefig("figures/benchmark_temp.png")
    # plt.show()

def convergence_test_temperature():
    ''' A convergence is plot is obtained for different step sizes for temperature.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

    '''

    t_interp = np.arange(2210)
    M0 = 8.12*10**6
    Ts = 260
    a = 1
    b = 1
    P0 = 1.29176
    bt = 1
    T0 = 25
    q = np.ones(len(t_interp))*10**6
    p = 1

    # Day 1 convergence test
    f,ax = plt.subplots(1,1)

    inv_dt = []
    T_dt = []

    for g in range (10,100,1):
        t, x = solve_ode(temp_ode_model, 0, 221, g/100, T0, q, [p, P0, T0, M0, Ts, a, b, bt])

        for i in range (1,len(t)):
            #day 1
            if t[i] <= 1.015 and t[i] >= 0.985:
                    inv_dt.append(1/(g/100))
                    T_dt.append(x[i])
                    
        ax.plot(inv_dt, T_dt,'ro')

    # get final two step sizes and temperature values (to check for convergence and find suitable step size)
    h1 = inv_dt[0]
    h2 = inv_dt[1]
    T1 = T_dt[0]
    T2 = T_dt[1]

    #Labelling axis and title on the plot.
    ax.set_title('Temperature Step-size convergence plot')
    ax.set_ylabel('Day 1 Temperature Values at different step-sizes')
    ax.set_xlabel('1/h (step-size)')

    ax.text(9.1, 43.8, 'h={:.2f}\nT={:.1f}'.format(1/h2,T2), ha='center', va='center', size=7)
    ax.text(10, 43.8, 'h={:.2f}\nT={:.1f}'.format(1/h1,T1), ha='center', va='center', size=7)

    plt.savefig("figures/convergence_test_temperature.png")
    # plt.show()


def test_solve_ode():
    ''' Test to ensure the correct implementation of Improved Euler method in solve_ode.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

    '''

    def test_dydx(x,y,q,a,b):
        ''' Sample ODE function to be solved in solve_ode using Improved Euler method.

            Parameters:
            -----------
            x : float
                independent variable
            y : float
                dependent variable
            q : float
                constant parameter
            a : float
                constant parameter
            b : float
                constant parameter

            Returns:
            --------
            dydx : derivate of ODE function

        ''' 

        return (x*a*q) - y*b

    a = 2
    b = 3
    q = np.ones(3)*2
    x, y = solve_ode(test_dydx, 0, 2, 1, 1, q, [a,b])

    #hand-worked expected solution
    y_soln = [1,-2,8] 

    assert norm(y_soln-y) < 1.e-6

def test_pressure_ode_model():
    ''' Test to ensure correct output of pressure_ode_model.

        Parameters:
        -----------
        None

        Returns:
        --------
        None

    '''

    t= 3
    p= 4
    q= 2
    a = 7
    b = 5
    p0 = 2

    dPdt = pressure_ode_model(t, p, q, a, b, p0)
    #hand-worked expected solution
    dPdt_soln = -24
    
    assert norm(dPdt_soln-dPdt) < 1.e-6

def test_temperature_ode_model():
    ''' Test to ensure correct output of temp_ode_model

        Parameters:
        -----------
        None

        Returns:
        --------
        None
        
    '''
    #for p < P0, T_dash = T0
    t= 3
    T = 210
    q = 1*10**6
    p= 1.5
    P0 = 2
    T0 = 200
    M0 = 1*10**6
    Ts = 260
    a = 7
    b = 5
    bt = 1

    dTdt = temp_ode_model(t, T, q, p, P0, T0, M0, Ts, a, b, bt)
    ##hand-worked expected solution
    dTdt_soln = 39.99999643
    
    assert norm(dTdt_soln-dTdt) < 1.e-6

    #for p > p0, T_dash = T
    p = 3

    dTdt = temp_ode_model(t, T, q, p, P0, T0, M0, Ts, a, b, bt)
    dTdt_soln = 40
    
    assert norm(dTdt_soln-dTdt) < 1.e-6