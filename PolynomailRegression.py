import numpy as np
from scipy.optimize import curve_fit
# Define the function for the quadratic model
def quadratic(x, a, b, c):
    return a*x**2 + b*x + c
# Define the data
x_data = np.array([3, 4, 5, 6, 7])
y_data = np.array([2.5, 3.2, 3.8, 6.5, 11.5])
# Fit the model to the data
popt, pcov = curve_fit(quadratic, x_data, y_data)
# Print the coefficients
print("a = {}, b = {}, c = {}".format(popt[0], popt[1], popt[2]))



#  Data Visualisation
import numpy
import matplotlib.pyplot as plt
x = [3,4,5,6,7]
y = [2.5,3.2,3.8,6.5,11.5]
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(1,10, 100)
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()


