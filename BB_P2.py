# This program calculates and plots the intensity
# curves for black-body radiation for a temperature
# provided by the user and for observed values in the range of wavelengths
# 0.1-5.0 mm.  It additionally calculates the rms error between 
# predicted values derived from a model and the observed data. Finally, it
# graphs the error against temperature and provides an estimate of the 
# temperature that minimizes the error


import numpy as np
import matplotlib.pyplot as plt

# Calculates the black-body spectrum  intensities 
# for a given wavelength and temperature
def intensity(wavelength, T):
  a=(2*_h*(_c**2))/(wavelength**5)
  b = (_h*_c)/(wavelength*_k*T)
  c=(np.exp(b)-1)**-1
  return a*c

# Calculates the root mean square error between the model prdictions 
# and the measured values of intesity 
def calculate_rms(measured_lambda_metres, measured_intensity, T):
  model_intensity=intensity(measured_lambda_metres, T)
  sq_dev=(measured_intensity-model_intensity)**2
  mean_sum=sum(sq_dev)/len(sq_dev)
  return mean_sum**0.5
  
# Some constants
_h=6.626E-34 # Boltzmann
_c=2.998E8 # Speed off Light
_k=1.381E-23 # Energy to temp conversion factor

# Measured data: wavelenths in mm
measured_lambda=np.array([4.405286344,3.676470588,3.144654088,\
2.754820937,2.450980392,2.004008016,1.834862385,1.377410468,\
0.881834215,0.468823254],float)

# Measured data: intensity in W m**-2 m**-1 sr**-1
measured_intensity=np.array([3.10085E-05,5.53419E-05,8.8836E-05,\
0.000129483,0.000176707,0.000284786,0.00034148,0.000531378,\
0.000561909,6.16936E-05],float)

# Convert wavelength to metres
measured_lambda_metres=measured_lambda*1e-3

# Define the wavelength range in metres
model_lambda_range=np.linspace(0.1,5.0,50,endpoint=True)
# and convert to metres
model_lambda_range_metres=model_lambda_range*1e-3


# Initialize lists to hold error and temperature data
errors=[]
temps=[]

temp=float(input("Please enter a value for T"))

while True:
  model_intensity=intensity(model_lambda_range_metres,temp)
  rms=calculate_rms(measured_lambda_metres, measured_intensity, temp)
  errors.append(rms)
  temps.append(temp)
 
  print "Error x 1e5 is",round(rms*1e5 ,2)
  # Likelihood is preferred to error here as it gives nore user friendly numbers
  print ""
  print "Smallest error so far is", round(min(errors)*1e5, 2), "for a temperature of", temps[errors.index(min(errors))]
  print ""
  

  temp=input("Enter another temperature or type q to finish")
  if temp=="q":
    break
  else:
    temp=float(temp)
