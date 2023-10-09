import numpy as np
from cell import Cell
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
plt.rcParams['figure.dpi'] = 200
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy.optimize import curve_fit

def saveFrame(xs,ys,cell : Cell,index, res):
   fig,ax = plt.subplots(figsize=(10,8))
   ax.set_xlabel('x [$\mu$m]')
   ax.set_ylabel('y [$\mu$m]')
   ax.set_title('Twitching motility (resolution: %1gs)'%(res))
   ax.set_xlim((-6,6))
   ax.set_ylim((-6,6))
   ax.set_aspect('equal')
   ax.plot(xs,ys,'gray',alpha=0.5)
   for p in cell.pili:
    x12=[xs[-1], xs[-1]+p.x()]
    y12=[ys[-1], ys[-1]+p.y()]
    if p.bound:
        col='r'
        lst='-'
    elif p.elong:
        col='g'
        lst='-'
    else:
        col='b'
        lst='--'
    ax.plot(x12,y12,color=col,linestyle=lst,alpha=0.9)
    ax.plot(xs[-1],ys[-1],'.k',alpha=0.9)
    fig.savefig(f'frames/frame{index:04d}.png')
    plt.close()
    return

def plot_pili_length_histogram():
   Ls = np.loadtxt("results/pili_length_all.dat")
   avgLs = np.loadtxt("results/pili_length_average.dat")
   def myExp(x,tau):
      return np.exp(-abs(x)/tau)/tau
   fig, axes = plt.subplots(1,2,figsize=(10,4))
   ax = axes[0]
   bin_heights,bin_borders,_=ax.hist(Ls,density=True, log=True)
   bin_centers = bin_borders[:-1] + np.diff(bin_borders) / 2
   popt, psig = curve_fit(myExp, bin_centers, bin_heights, p0=1)
   x_sample=np.linspace(0,np.max(bin_borders),100)
   ax.plot(x_sample,myExp(x_sample,popt[0]),'r', label='exponential fit\n$l_0$=%.3g$\pm$%.1g'%(popt[0],np.sqrt(psig[0])))
   ax.set_xlabel('pilus length per frame [$\mu$m]')
   ax.set_ylabel('density')
   ax.legend()
   ax = axes[1]
   ax.hist(avgLs,density=True)
   ax.set_ylabel('density')
   ax.set_xlabel('average pilus length per frame [$\mu$m]')

   for ax in axes:
      ax.tick_params(which='both', direction='in')
   fig.savefig("results/length.png")
   plt.close()
   print("Histograms of pili length saved into results/length.png")
   return

def plot_trajectory():
    xs,ys = np.loadtxt("results/position.dat", unpack=True)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.plot(xs,ys)
    bar_ = AnchoredSizeBar(ax.transData, 1, '1 $\mu$m',4,frameon=False,size_vertical=0.05)
    ax.add_artist(bar_)
    ax.tick_params( 
        axis='both',       # changes apply to both axis
        which='both',      # both major and minor ticks are affected
        bottom=False, top=False, labelbottom=False,
        right=False, left=False, labelleft=False
    )
    fig.savefig("results/trajectory.png")
    print("Trajectory micrograph saved into results/trajectory.png")
    return