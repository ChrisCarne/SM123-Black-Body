import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['mathtext.default']='regular'

# Uncomment to get graphs working right in Jupyter
#%matplotlib inline

# Calculates the radiation intensity from the wavelength and temperature
# using the formula given in the notes.
def intensity(wavelength, T):
  a=(2*_h*(_c**2))/(wavelength**5)
  b = (_h*_c)/(wavelength*_k*T)
  c=(np.exp(b)-1)**-1
  return a*c

#Calcualtes teh root mean squared error between the observed data and model predictions 
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
model_intensity=np.array([],float)


# Initialse a couple of lists to hold data
errors=[]
temps=[]
temp=0

temp=float(input("Please enter a value for T")) #asks user for a temp

# main program loop
while True:
  model_intensity=intensity(model_lambda_range_metres,temp)
  rms=calculate_rms(measured_lambda_metres, measured_intensity, temp)
  errors.append(rms) #builds a list of errors
  temps.append(temp) #and associated temperatures
  model_intensity=intensity(model_lambda_range_metres, temp)
  best_so_far=temps[errors.index(min(errors))] #finds the temp associated with the smallest error
  #outputs
  # NB Error multiplied by 1e5 to give a more convenient figure for the user
  print(f"Error (*1e5) is {round(rms*1e5,2)}" '\n')
  print(f"Smallest error so far is {round((min(errors)*1e5),2)} for a temperature of {best_so_far} K" '\n')
    # go round again or stop and finish up
  go_again=input("Enter another temperature or type q to finish")
  if go_again=="q":
    break
  else:
    temp=float(go_again)
    

plt.clf() 
plt.title("Intensity against wavelength")
plt.xlabel(r'$\mathrm{wavelength}$  $\mathrm{/mm}$')
#plt.xlabel(r'$s(t) = \mathcal{A}\mathrm{sin}(2 \omega t)$')
plt.ylabel(r'$\mathrm{Intensity}$ $\mathrm{/W }$ $\mathrm{m}^{-2}$ $\mathrm{ m}^{-1}$ $\mathrm{sr}^{-2}$')
plt.ticklabel_format(axis="y", style="sci", useOffset= True, scilimits=(0.01,1000))
plt.plot(measured_lambda,measured_intensity,'*', label="Observed")
plt.plot(model_lambda_range,model_intensity, label="Model: T= "+ str(temp))
plt.legend()
plt.show()

# Implements a 'brute force' algorithm for finding the temp that gives a minimum error.
# Works on this fairly small range but not really satisfactory. NB using numpy arrays gives
# a much faster speed than standard python lists but it still slows down pretty rapidly
# once we require any more accuracy or range we have here due to the for loop.
    
temp_range=np.linspace(1,5,5000,endpoint=True)
error_array=np.array([])
    
# builds a great big array of the error for each temp
for temperature in temp_range:
 t=calculate_rms(measured_lambda_metres, measured_intensity, temperature)
 error_array=np.append(error_array,t)
  

# Finds the smallest error in the error array and uses its index to return 
# the associated temp from the temp_range array
minimising_temp=temp_range[np.where(error_array==np.min(error_array))][0]

print(f"The temperature that gives the closest match to the obsered data is {round(minimising_temp,4)} K (5 s.f.)")

