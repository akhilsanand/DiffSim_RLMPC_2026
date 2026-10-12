import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
 

plt.rcParams['text.usetex'] = True
FS = 40

Mu = {}
Std = {}
Wei = {}
Thresh = {}

Mu[ 'World'] = [np.array([-.25,-.25]),np.array([.25,.25])]
Std['World'] = [25*np.eye(2),25*np.eye(2)]
Wei['World'] = [0.3,.7]

Thresh['World'] = 0.1

Mu[ 'Model'] = [np.array([.15,.225])]
Std['Model'] = [20*np.array([[10,0],[0,2]])]
Wei['Model'] = [1]

Thresh['Model'] = 0.05

x = np.linspace(-1,1,500)
y = np.linspace(-1,1,499)
xv, yv = np.meshgrid(x, y, indexing='ij')

# Distribution
Distribution = {'World' : 0, 'Model' : 0}

for build in Distribution.keys():
    for k in range(0,len(Wei[build])):
        Expon = 0.5*Std[build][k][0,0]*(xv - Mu[build][k][0])**2 + 0.5*Std[build][k][1,1]*(yv - Mu[build][k][1])**2 + Std[build][k][0,1]*(yv - Mu[build][k][1])*(xv - Mu[build][k][0])
        Distribution[build] += Wei[build][k]*np.exp( - Expon )
        
    Distribution[build][ Distribution[build] < Thresh[build]-2e-3 ] = np.nan

plt.close('all')
figID = plt.figure(1,figsize=(12,6))
ax1 = figID.add_subplot(111)

Blues = plt.colormaps['Reds']
Blues.set_bad(color='w')
ax1.pcolormesh(xv, yv, Distribution['World'], cmap=Blues)#,vmin=1e-3,vmax = np.max(ModelPlot))#, vmax=positives)
cs = ax1.contour(xv, yv, Distribution['World'],colors='k',linestyles='solid',levels=[Thresh['World']+5e-3])

ax1.text(-.45,-.45,r'$\rho(.|\bf s,a)$',fontsize=20,verticalalignment='top',horizontalalignment='right',color='r')


ax1.set_aspect('equal', 'box')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel(r'$\bf s_{1}^+$',fontsize=20)
ax1.set_ylabel(r'$\bf s_{2}^+$',fontsize=20)


plt.show(block=False)

#sys.exit()

figID.savefig('World.tiff', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None)


Reds = plt.colormaps['Blues']
Reds.set_bad(color='none')
ax1.pcolormesh(xv, yv, Distribution['Model'], cmap=Reds)#,vmin=1e-3,vmax = np.max(ModelPlot))#, vmax=positives)
ax1.contour(xv, yv, Distribution['Model'],colors='k',linestyles='solid',levels=[Thresh['Model']+5e-3])

ax1.text(.1,.65,r'$\hat\rho(.|\bf s,a)$',fontsize=20,verticalalignment='bottom',horizontalalignment='left',color='b')


figID.savefig('Model.tiff', transparent=None, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None)





plt.show(block=False)

        
        
        

 
