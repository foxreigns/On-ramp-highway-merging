## This code implements the algorithm designed in the paper titled 
##### "Automated and Cooperative Vehicle Merging at Highway On-Ramps" 
## authored by Rios_Torres and Andreas Malikopoulos
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import random

plt.close('all')

## Simulation parameters
S = 30      # Merging zone length (m)
L = 400     # Control zone length (m)
v0 = 13.4   # Speed before entering control zone (m/s)
vf = v0
delta = 20  # Rear-end collision safe distance (m)

Mv = 1200   # Mass of the vehicles (kg)
Cd = 0.32   # Coefficient of drag
rho = 1.184 # Air density (kg/m^3)
Af = 2.5    # Frontal area (m^2)
"""
# Case 1) Coordination of 4 vehicles - 2 for each road
N = [0, 1, 2, 3] # Control zone queue (first-in first-out)
R1 = np.array([[0],[2]])
R2 = np.array([[1],[3]])
#Lset = np.array([[2],[3],[0],[1]])
#Cset = np.array([[1,3],[0,2],[1,3],[0,2]])
"""
#Case 2) Coordination of 30 vehicles - 15 each road - random initial positions
vNum = 30
N = set(list(range(0,vNum,1)))
NN = set(list(range(0,vNum,1)))
slen = (len(NN))/2
R1 = set(random.sample(NN,int(slen)))
NN -= R1
R2 = NN

N = list(N)
R1 = np.array(list(R1)).reshape((int(slen),1))
R2 = np.array(list(R2)).reshape((int(slen),1))

########### COMING SOON ###############
# Case 3) Coordination of 30 vehicles - 15 each road - random initial positions - different initial speeds for each road

########### COMING SOON ###############

# Case 4) Coordination at 29 m/s speed

########### COMING SOON ###############

# FUEL Consumption computation
def computeFuelConsumption(v, u):

    q = np.array([0.1569, 2.45*10**-2, -7.415*10**-4, 5.975*10**-5]) # Cruise coefficients
    r = np.array([0.07224, 9.681*10**-2, 1.075*10**-3]) # Acceleration coefficients
    
    fCruise = q @ np.array([1, v, v**2, v**3])

    fAccel = (r @ np.array([1, v, v**2]))*u

    return (fCruise + fAccel)

# Visualizing fuel consumption model
"""
vvv = range(0,25,1)
uuu = np.arange(0,2,0.1)
test = np.zeros((len(vvv),len(uuu)))

for iu,u in enumerate(uuu):
    for iv,v in enumerate(vvv):
        test[iv,iu] = computeFuelConsumption(v,u)

fig = plt.figure()
ax = fig.gca(projection = '3d')

vvv = np.reshape(vvv,(25,1))
uuu = np.reshape(uuu,(20,1))
vvv, uuu = np.meshgrid(np.array(vvv), uuu)
surf = ax.plot_surface(vvv, uuu, test.T, linewidth=2.5, cmap=cm.coolwarm, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
"""

## States and Dynamic equations
pi = np.zeros(shape = (len(N),1)) # Position
vi = np.zeros_like(pi) # Velocity
vi[:,0] = v0 # Setting initial velocity of all vehicles to v0
ui = np.zeros_like(vi) # Acceleration or Control input
fuel = np.zeros_like(vi)

ppi = pi.tolist()
pi = pi.tolist()
vvi = vi.tolist()
vi = vi.tolist()
ui = ui.tolist()
fuel = fuel.tolist()

############ Computing tf for all vehicles ##################
#t0 = np.array([[10],[10],[12.5],[12.5]]) # Case 1)
#t0 = np.array([[10],[10],[12.5],[12.5],[13],[14],[14.9],[15.5],[16],[16.8],[17],[17.4]]) # TEST

tMin = 10
tMax = 40
t0 = np.reshape(sorted(np.round(np.random.uniform(tMin,tMax,len(N)),1)),(len(N),1))

tf = np.zeros((len(N),1))
tm = np.zeros((len(N),1))
tm1 = (L)/v0
tm[0] = tm1
tf1 = (L+S)/v0 # final time
tf[0] = tf1

for ii in N:
    # In the paper, vi(tf) is assumed to be the same as v0
    if ii in R1:
        Lset = R1
        Cset = R2
    elif ii in R2:
        # vvi[ii] = [11.2]  Uncomment for Case (3)
        Lset = R2
        Cset = R1
    if ii-1 in Lset:
        tf[ii] = tf[ii-1] + delta/v0
        tm[ii] = tf[ii] - S/v0
    elif ii-1 in Cset:
        tf[ii] = tf[ii-1] + S/v0
        tm[ii] = tf[ii] - S/v0

#################################################################

## Solving the energy minimization problem (by computing the coefficients a,b,c,d)
def computeCoeff(tt,tff,pit,vit):

    Ti = np.zeros((4,4))
    Ti[0,:] = [(1/6)*tt**3, (1/2)*tt**2, tt, 1]
    Ti[1,:] = [(1/2)*tt**2, tt, 1, 0]
    Ti[2,:] = [(1/6)*tff**3, (1/2)*tff**2, tff, 1]
    Ti[3,:] = [(1/2)*tff**2, tff, 1, 0]

    # btilde = np.array([[ai],[bi],[ci],[di]])
    #if pit < L:
    #    qtilde = np.array([[pit],[vit],[L],[v0]])
    #elif pit >= L:
    #    qtilde = np.array([[pit],[vit],[L+S],[v0]])
    
    qtilde = np.array([[pit],[vit],[L],[vf]])
    try:
        btilde = np.linalg.inv(Ti) @ qtilde
    except np.linalg.LinAlgError:
        btilde = np.linalg.pinv(Ti) @ qtilde
    

    return btilde

step = 0.1
timePlot = []
for ii in N:
    
    timeRange = np.arange(t0[ii,0],tf[ii,0]+step,step)
    timePlot.append(timeRange)
    
    for t in timeRange:
        
        #if ppi[ii][-1] < L:
        #    ai,bi,ci,di = computeCoeff(t,tm[ii,0],ppi[ii][-1],vvi[ii][-1])
        #elif ppi[ii][-1] >= L:
        #    ai,bi,ci,di = computeCoeff(t,tf[ii,0],ppi[ii][-1],vvi[ii][-1])

        if t > tm[ii,0]:
            ui[ii].append(0)
        else:
            ai,bi,ci,di = computeCoeff(t,tm[ii,0],ppi[ii][-1],vvi[ii][-1])
            ui[ii].append((ai*t + bi)[0].tolist())
        #vi[ii].append(((1/2)*ai*t**2 + bi*t + ci)[0].tolist())
        vvi[ii].append(vvi[ii][-1] + step*ui[ii][-1])
        #pi[ii].append(((1/6)*ai*t**3 + (1/2)*bi*t**2 + ci*t + di)[0].tolist())
        ppi[ii].append(ppi[ii][-1] + step*vvi[ii][-1])
        fuel[ii].append(computeFuelConsumption(vvi[ii][-1], ui[ii][-1]))


## Plots

plt.figure(0)
# Plotting merging zone
xTicks = range(0,int(np.ceil(max(tf)))+5,5)
plt.plot(xTicks,L*np.ones(len(xTicks)), color = 'black', linestyle = '--',label = '_nolegend_')
plt.text(5,410,'Merging zone')
plt.plot(xTicks,(L+S)*np.ones(len(xTicks)), color = 'black', linestyle = '--',label = '_nolegend_')
plt.grid(b = True, linestyle = '--')


plt.figure(1)
plt.plot(xTicks,v0*np.ones(len(xTicks)), color = 'black', linestyle = '--')
plt.grid(b = True, linestyle = '--')

plt.figure(2)
plt.grid(b = True, linestyle = '--')

#plt.figure(3)
#plt.grid(b = True, linestyle = '--')


for ii in N:
    if ii in R1:
        plt.figure(0) # Position
        plt.plot(timePlot[ii], np.array(ppi[ii][:len(timePlot[ii])]),color = 'red',linestyle = '--')
        #plt.legend(['Road 1', 'Road 2'])
        plt.figure(1) # Velocity
        plt.plot(timePlot[ii], np.array(vvi[ii][:len(timePlot[ii])]),color = 'red', linestyle = '--')
        #plt.legend(['Road 1', 'Road 2'])
        plt.figure(2) # Acceleration/Deceleration or Control input
        plt.plot(timePlot[ii], np.array(ui[ii][:len(timePlot[ii])]),color = 'red', linestyle = '--')
        #plt.legend(['Road 1', 'Road 2'])
    else:
        plt.figure(0) # Position
        plt.plot(timePlot[ii], np.array(ppi[ii][:len(timePlot[ii])]),color = 'blue')
        plt.figure(1) # Velocity
        plt.plot(timePlot[ii], np.array(vvi[ii][:len(timePlot[ii])]),color = 'blue')
        plt.figure(2) # Acceleration/Deceleration or Control input
        plt.plot(timePlot[ii], np.array(ui[ii][:len(timePlot[ii])]),color = 'blue')


#plt.figure(3) # Fuel consumption
#sumFuel = np.sum(fuel, axis = 0)
#plt.plot(xTicks, np.array(fuel))
plt.show()