from project_functions import *
from model_callibration_initial import *
from model_callibration import *
from model_prediction import *
from model_posterior import *

tO,O,tP,P,tS,S,tT,T,tW,W = load_data()

plot_data(tO,O,tP,P,tS,S,tT,T,tW,W)

# convert produced oil from m^3/day to kg/day
# density of bitumen = 1060kg/m^3 (https://atdmco.com/wiki-density+of+bitumen+60+70-328.html#:~:text=It%20means%20that%201.010kg,and%20the%20rest%20is%20aggregates.)
O = O*1060

# convert produced water from m^3/day to kg/day
# density of water = 997kg/m^3
W = W*997

# convert injected steam from tonnes/day to kg/day
S = S*1000

# compute q term (mass rate = water production + oil production - steam injection)
tq = np.sort(np.concatenate((tO, tS, tW)))
O = np.interp(tq, tO, O)
S = np.interp(tq, tS, S)
W = np.interp(tq, tW, W)
q = W + O - S

# benchmarking and convergence test for pressure model
benchmark_pressure()
convergence_test_pressure()

# benchmarking and convergence test for temperature model
benchmark_temp()
convergence_test_temperature()

# unit tests
test_solve_ode()
test_pressure_ode_model()
test_temperature_ode_model()

# initial model calibration with following assumptions:
# P0 = 1291.76kPa (initial pressure measured in given data)
# T0 = 25°C (ambient temperature of outside the system)
lpm_model_initial(tO,O,tP,P,tS,S,tT,T,tW,W,tq,q)

# refined model calibration with P0 and T0 as calibrated paramters
lpm_model(tO,O,tP,P,tS,S,tT,T,tW,W,tq,q)

# model prediction
lpm_model_prediction(tO,O,tP,P,tS,S,tT,T,tW,W,tq,q)

# model forecast
lpm_model_posterior(tO,O,tP,P,tS,S,tT,T,tW,W,tq,q)