import os
import sys
import numpy as np
from cell import Cell
from figures import *

###########################
###      PARAMETERS     ###
###########################
#--------- Environment -----------#
gamma=0.1    #drag coefficient (pN*sec/micron)
#----------- Pilus ---------------#
L0=0.001  #length of newborn pilus
t1=0.85
t2=0.04
Fd1=1.28
Fd2=33.8
Fs=180    #stall force
vr=2      #retracting speed (1 if anaerobic)
ve=2      #elongating speed
#----------- Cell ----------------#
n_WT=7       #average number of pili for wild strain
c_WT=9       #pili creation rate (such that <n>=n_WT=7)
a0_WT=2.4    #attachment rate (such that <L>=1 micron)
n=n_WT
c=9.0
a0=2.4
#---------- Trajectory ------------#
dt=0.0002    #simulation step (sec)
Ttransient=5 #transient time (sec)
T=40         #experiment time (sec)
res=0.1      #experimental resolution (sec), for snapshots
doSaveFrame = False
#-----------------------------------#

###########################
###      SIMULATION     ###
###########################
ntransient=int(Ttransient/dt)
nsteps=int(T/dt)
n_res=int(res/dt)

C=Cell(n,c,a0, L0, t1,t2,Fd1,Fd2,Fs, ve,vr)
C.resetEventCounters()
print(f"{sys.argv[0]}: Simulation of transient time STARTED ({ntransient} steps).")
# Transient time
for i in range(ntransient):
    C.step(gamma,dt)

C.resetEventCounters()
C.x=np.zeros(2) # start at the origin
Cs=[]
xs=[]
ys=[]
ns=[]
nBs=[]
Ls=[]
avgLs=[]
vxs=[]
vys=[]
print(f"{sys.argv[0]}: Simulation at equilibrium STARTED ({nsteps} steps).")
try:
    os.mkdir("frames")
except FileExistsError:
    print(f"{sys.argv[0]}: WARNING: folder frames/ exists.")

for i in range(nsteps):
    C.step(gamma,dt)
    if (i%n_res)==0:
        if C.Npili()>0:
            L=0
            for p in C.pili:
                Ls.append(p.L)
                L+=p.L
            avgLs.append(L/C.Npili())
        vxs.append(C.v[0])
        vys.append(C.v[1])
        ns.append(C.Npili())
        nBs.append(C.Nbound())
        xs.append(C.x[0])
        ys.append(C.x[1])
        if doSaveFrame:
            saveFrame(xs,ys,C,i//n_res,res)
    perc=int(100*(i+1)/float(nsteps))
    print(f"\r[{perc:d}%]", end='')

print("")
print(f"{sys.argv[0]}: Saving results.")
try:
    os.mkdir("results")
except FileExistsError:
    print(f"{sys.argv[0]}: WARNING: folder results/ exists.")
np.savetxt("results/position.dat", np.array([np.array(xs),np.array(ys)]).T)
np.savetxt("results/velocity.dat", np.array([np.array(vxs),np.array(vys)]).T)
np.savetxt("results/pili_length_all.dat", np.array(Ls))
np.savetxt("results/pili_length_average.dat", np.array(avgLs))
np.savetxt("results/pili_number.dat", np.array(ns), fmt='%d')
np.savetxt("results/pili_bound_number.dat", np.array(nBs), fmt='%d')

with open("results/summary.txt", "w") as f:
    f.write(f'# timestep: {dt} s\n')
    f.write(f'# transient time: {Ttransient} s\n')
    f.write(f'# experiment time: {T} s\n')
    f.write(f'# resolution: {res} s\n')
    f.write(f'# L<L0 events: {C.L0Events:d}\n')
    f.write(f'# bindings: {C.bindEvents:d}\n')
    f.write(f'# unbindings: {C.unbindEvents}\n')

print(f"{sys.argv[0]}: Simulation COMPLETED.")
if doSaveFrame:
    print(f"{sys.argv[0]}: Convert frames to video with [ ffmpeg -pattern_type glob -i 'frames/frame????.png' out.mp4 ]\n\n")

plot_pili_length_histogram()
plot_trajectory()
print("")