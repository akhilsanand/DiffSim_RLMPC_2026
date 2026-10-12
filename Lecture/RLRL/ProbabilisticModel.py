import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import cm


plt.rcParams['text.usetex'] = True

Nstate = 2
N  = 40
FS = 15

std = 0.01

Start  = [0,0]
Finish = [1,1]
gain   = 0.1
inertia = 0.5
NMC     = 100

Mu  = [np.array([0,1]),np.array([0.5,.5])]
Std = [3*np.eye(2),3*np.eye(2)]
Wei = [.5,.5]

Time = [k for k in range(N)]

Finish = np.array(Finish).reshape(2,1)

C      = np.array([.5,.75]).reshape(2,1)
D      = 0.2

Angle  = np.linspace(0,2*np.pi,1000)

tau    = 20

Fgain  = 0.5


N = 200

BW = 1.1
LB = -0.1

def MakeBox(spred, CO, Time = [], LW = 3):
    
    Box = np.array([[np.inf,-np.inf],[np.inf,-np.inf]])
        
    for n in range(0,NMC):
        Box[0,0] = np.min([ Box[0,0], np.min(spred[n][0])  ])
        Box[0,1] = np.max([ Box[0,1], np.max(spred[n][0])  ])
        Box[1,0] = np.min([ Box[1,0], np.min(spred[n][1])  ])
        Box[1,1] = np.max([ Box[1,1], np.max(spred[n][1])  ])
    
    ax1.plot([Box[0,0],Box[0,1]],[Box[1,0],Box[1,0]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,1]],[Box[1,0],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,1],Box[0,0]],[Box[1,1],Box[1,1]],color=CO,linewidth=LW)
    ax1.plot([Box[0,0],Box[0,0]],[Box[1,1],Box[1,0]],color=CO,linewidth=LW)
    
    
    x = np.linspace(Box[0,0],Box[0,1],100)
    y = np.linspace(Box[1,0],Box[1,1],99)
    xv, yv = np.meshgrid(x, y, indexing='ij')

    # Distribution
    x = np.linspace(-1,1,1000)
    ModelPlot = 0
    for k in range(0,len(Wei)):
        Expon = 0.5*Std[k][0,0]*(xv - Mu[k][0])**2 + 0.5*Std[k][1,1]*(yv - Mu[k][1])**2 + Std[k][0,1]*(yv - Mu[k][1])*(xv - Mu[k][0])
        ModelPlot += Wei[k]*np.exp( - Expon )
    
    ax1.contour(xv,yv,ModelPlot,colors='k',linestyles='solid')
    return Box


Vec    = np.zeros(2).reshape(2,1)
Vecsys = np.zeros(2).reshape(2,1)

spred = []
for n in range(0,NMC):
    s0 = np.random.normal([0,0],0.035).reshape(2,1)
    spred.append(s0)

for time in Time:
    for n in range(0,NMC):
        Vec  = inertia*Vec + (1-inertia)*(Finish - spred[n][:,-1].reshape(2,1))
        
        Gradient = 0
        for k in range(0,len(Wei)):
            sa = spred[n][:,-1]
            Expon = 0.5*Std[k][0,0]*(sa[0] - Mu[k][0])**2 + 0.5*Std[k][1,1]*(sa[1] - Mu[k][1])**2 + Std[k][0,1]*(sa[1] - Mu[k][1])*(sa[0] - Mu[k][0])
            Gradient += -Wei[k]*np.exp( - Expon )*np.matmul(Std[k],np.array([sa[0] - Mu[k][0],sa[1] - Mu[k][1]]))

        
        Vec -= Fgain*Gradient.reshape(2,1)
        
        sps = spred[n][:,-1].reshape(2,1) + gain*Vec.reshape(2,1) + np.random.normal(0,std,2).reshape(2,1)#
        spred[n] = np.concatenate( (spred[n],sps), axis=1)
               
"""
#Flag Trajectories
Safe      = []
Violation = []
for k, traj in enumerate(spred):
    Err = traj-C
    DistTraj = np.min(np.sqrt(Err[0,:]**2 + Err[1,:]**2)-D)
    if DistTraj < 0:
        Violation.append(traj)
    else:
        Safe.append(traj)
"""
plt.close('all')

figID = plt.figure(1,figsize=(12,6))

ax1 = figID.add_subplot(111)
#ax1.fill(D*np.cos(Angle)+C[0],D*np.sin(Angle)+C[1],color='r')


        

EndMass = {'x': [], 'y' : []}
for traj in spred:
    ax1.plot(traj[0,:],traj[1,:],color='b',linewidth=1,marker='.')
    EndMass['x'].append(traj[0,-1])
    EndMass['y'].append(traj[1,-1])
    
MakeBox(spred,'k')
"""
ax1.plot([LB,LB],[LB,BW],color='r',linewidth=2)
ax1.plot([LB,BW],[BW,BW],color='r',linewidth=2)
ax1.plot([BW,BW],[BW,LB],color='r',linewidth=2)
ax1.plot([BW,LB],[LB,LB],color='r',linewidth=2)
"""

EndD = np.max([np.std(EndMass['x']),np.std(EndMass['y'])])
#ax1.plot(2*EndD*np.cos(Angle)+np.mean(EndMass['x']),2*EndD*np.sin(Angle)+np.mean(EndMass['y']),color='c',linewidth=2)

ax1.text(0,0,r'$\bf{s}_{0}$',horizontalalignment='left',verticalalignment='bottom',fontsize=40,color='r')
ax1.text(np.mean(EndMass['x']),np.mean(EndMass['y']),r'$\bf{s}_{N}$',horizontalalignment='center',verticalalignment='center',fontsize=40,color='r')

ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_axis_off()
ax1.set_aspect('equal', 'box')

plt.show(block=False)

figID.savefig('ProbabilisticModel.eps', transparent=True, dpi='figure', format=None,metadata=None, bbox_inches='tight', pad_inches=0.1,facecolor='auto', edgecolor='auto', backend=None) #, bbox_inches='tight'


















