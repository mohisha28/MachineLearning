import matplotlib.pyplot as plt
%matplotlib inline  ##or shd write plt.show()
import numpy as np

## examples 
x=np.arange(0,10)
y=np.arange(11,21)
a=np.arange(40,50)
b=np.arange(50,60)
## plotting using matplotlib
## plt scatter
plt.scatter(x,y,c='b') # c= colour
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.title('Graph in 2D')
plt.savefig('Test.ping')
y=x*x
## plt plot 
plt.plot(x,y,'r*--')

plt.plot(x,y,'r*',linestyle='dashed',linewidth=2,markersize=12)
plt.xlabel('x-axis')  
plt.ylabel('y-label')
plt.title('2d diagram')

## creating subplots
plt.subplot(2,2,1)
plt.plot(x,y,'r--')
plt.subplot(2,2,2)
plt.plot(x,y,'g*')
plt.subplot(2,2,3)
plt.plot(x,y,'b')
plt.subplot(2,2,4)
plt.plot(x,y,'r')

x=np.arange(1,11)
y=3*x+5
plt.title("matplotlib")
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.plot(x,y,'g')
plt.show()

np.pi
##compute the x and y coordinates for points on a sine curve
x=np.arange(0,4*np.pi,0.1) ##(from,to,stepsize)
y=np.sin(x)
plt.title("sine wave form")

#plot the points using matplotlib
plt.plot(x,y)
plt.show()

#subplot
#compute the x & y coordinates for points on sine and cosine curves
x = np.arange(0,5*np.pi,0.1)
y_sin=np.sin(x)
y_cos=np.cos(x)

# setup a subplot grid that has height 2 and width 1,
# and set the first such subplot as active 
plt.subplot(2,2,1)

#make the first plot
plt.plot(x,y_sin)
plt.title('Sine')

# set the secong subplot as active, and make the second plot.
plt.subplot(2,1,2)
plt.plot(x,y_cos)
plt.title('Cosine')

# show the figure 
plt.show()

# Bar plot
# example
x=[2,8,10]
y=[11,16,9]
x2=[3,9,11]
y2=[6,15,7]
plt.bar(x,y,align='center')
plt.bar(x2,y2,color='g')
plt.title('Bar graph')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()

# Histogram
# Example
a=np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
plt.hist(a)
plt.title("histogram")
plt.show()
## y-axis=density 

# Box Plot using Matplotlib
data = [np.random.normal(0,std,100) for std in range(1,4)]
#rectangular box plot
plt.boxplot(data,vert=True,patch_artist=False);
## can try vert=false(horizontal) and artist too
#round circle in diagram are called as outliers as we select randomly 

# Piechart
# Example
# data to plot
labels='Pyton','C++','Ruby','Java'
sizes = [215,130,245,210]
colors=['gold','yellowgreen','lightcoral','lightskyblue']
explode=(0.1,0.4,0,0) #explode 1st slice
#plot
plt.pie(sizes,explode=explode,labels=labels,colors=colors,
       autopct='%1.1f%%',shadow=True)
plt.axis('equal')
plt.show()
