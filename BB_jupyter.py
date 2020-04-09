import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Uncomment to get graphs working right in Jupyter. Comment out in other environments
%matplotlib inline

# Calculates the radiaton intensity from the wavelength and temperature
# using the formula given in the notes.
def intensity(wavelength, T):
  a=(2*_h*(_c**2))/(wavelength**5)
  b = (_h*_c)/(wavelength*_k*T)
  c=(np.exp(b)-1)**-1
  return a*c

#Calcualtes the root mean squared error between the observed data and model predictions 
def calculate_rms(measured_lambda_metres, measured_intensity, T):
  model_intensity=intensity(measured_lambda_metres, T)
  sq_dev=(measured_intensity-model_intensity)**2
  mean_sum=sum(sq_dev)/len(sq_dev)
  return mean_sum**0.5
  
# Constants
_h=6.626E-34 #Planck
_c=2.998E8 #Speed of light
_k=1.381E-23 #temp to energy conversion factor.

measured_lambda=np.array([4.405286344,3.676470588,3.144654088,\
2.754820937,2.450980392,2.004008016,1.834862385,1.377410468,\
0.881834215,0.468823254],float)

# Measured data: intensity in W m**-2 m**-1 sr**-1
measured_intensity=np.array([3.10085E-05,5.53419E-05,8.8836E-05,\
0.000129483,0.000176707,0.000284786,0.00034148,0.000531378,\
0.000561909,6.16936E-05],float)

measured_lambda_metres=measured_lambda*1e-3

# Define the wavelength range in millimetres
model_lambda_range=np.linspace(0.1,5.0,50,endpoint=True)
# convert to metres
model_lambda_range_metres=model_lambda_range*1e-3

# Initialise a couple of lists to hold data
errors=[]
temps=[]

temp=float(input("Please enter a value for T")) #asks user for a temp

# main program loop
while True:
  model_intensity=intensity(model_lambda_range_metres,temp)
  rms=calculate_rms(measured_lambda_metres, measured_intensity, temp)
  errors.append(rms) #builds a list of errors
  temps.append(temp) #and associated temperatures
  model_intensity=intensity(model_lambda_range_metres, temp)
  best_so_far=temps[errors.index(min(errors))] #finds teh temp associated with the smallest error
  #outputs
  print("")
  print(f"Error (*1e5) is {round(rms*1e5,2)}" '\n')
  print(f"Smallest error so far is {round((min(errors)*1e5),2)} for a temperature of {best_so_far} K" '\n')
  # go round again or stop and finish up
  go_again=input("Enter another temperature or type q to finish")
  if go_again=="q":
    break
  else:
    temp=float(go_again)
    
# pyplot stuff
plt.clf() 
fig= plt.figure(figsize=(10,6))
plt.title("Intensity against wavelength")
plt.xlabel(r'$\mathrm{wavelength}\ \mathrm{/mm}$')
plt.ylabel(r'$Intensity\ /W\ m^{-2}\ m^{-1}\ sr^{-2}$')
plt.ticklabel_format(axis="y", style="sci", useOffset= True, scilimits=(0.01,1000))
plt.plot(measured_lambda,measured_intensity,'*', label="Observed")
plt.plot(model_lambda_range,model_intensity, label="Model: T= "+ str(temp))
plt.legend()
plt.show()
    
# Implements a 'brute force' algorithm for finding the temp that gives a minimum error.
# Works on this fairly small range but not really satisfactory. NB using mupy arrays gives
# a much faster speed than standard python lists but it still slows down pretty rapidly
# once we use any more than accuracy or range we have here.
    
temp_range=np.linspace(1,5,10000,endpoint=True)
error_array= array([])
    
for temperature in temp_range:
  t=calculate_rms(measured_lambda_metres, measured_intensity, temperature)
  error_array=np.append(error_array,t)
  
# Finds the smallest error in the error array and uses its index to return 
# the associated temp from the temp_range array
minimising_temp=temp_range[np.where(error_array==np.min(error_array))][0]

print(f"Your estimate for the temperature was {temp} K with an error of {round((errors[-1]*1e5),3)}")
print(f"The calculated temperature that gives the best fit to the obsered data is {round(minimising_temp,3)} K (4 s.f.)\
 with an error of {round(np.min(error_array)*1e5,3)}")

#with an error of {np.format_float_scientific(np.min(error_array),precision=3)} ")

# Plots graphs of error and likelihood for no real reason :)

#For sone reason trinket doesn't like this. Works OK in Jupyter though
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_title("Error and Likelihood by Temperature", fontsize=16)
ax1.set_xlabel("temp /K", fontsize=16)
ax1.set_yticks([])
ax1.plot(temp_range,error_array**-1, lw=1, color="blue", label="Error")
#ax1.set_ylabel(r"Likelihood", fontsize=16, color="blue")
for label in ax1.get_yticklabels():
 label.set_color("blue")

ax2 = ax1.twinx()
ax2.plot(temp_range,error_array, lw=1, color="red", label="Likelihood")
#ax2.set_ylabel(r"Error", fontsize=16, color="red")
ax2.set_yticks([])
#plt.legend([ax1, ax2], ['Line Up', 'Line Down'])
for label in ax2.get_yticklabels():
 label.set_color("red")

err_line=mlines.Line2D([],[], ls="-", color='red', label="Error")
like_line=mlines.Line2D([],[], ls="-", color='blue', label="Likelihood")
plt.legend(handles=[err_line, like_line])
