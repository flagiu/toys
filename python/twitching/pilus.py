import numpy as np

class Pilus:
    
    def __init__(self, a0, L0, t1,t2,Fd1,Fd2,Fs, ve,vr):
        self.L0 = L0     # minimum length, for birth and death
        self.t1 = t1     # time scale 1  \______________
        self.t2 = t2     # time scale 2   |for unbinding \
        self.Fd1 = Fd1   # force scale 1  |___rate_______/
        self.Fd2 = Fd2   # force scale 2 /
        self.Fs = Fs     # stall force
        self.a0 = a0     # binding rate
        self.ve = ve     # elongation speed
        self.vr = vr     # retraction speed

        self.L=L0           #starting length
        self.a0=np.random.uniform(-np.pi,np.pi) #random starting angle
        self.F=0            #force modulus (nonzero only when bound to the surface)
        self.bound=False    #state: bound to the surface?
        self.elong=True     #state: elongating?
        self.angle=self.a0  #dynamical angle (varies during retraction)
        
    ### DETERMINISTIC FUNCTIONS ###
    
    def x(self):
        return self.L*np.cos(self.angle)
    def y(self):
        return self.L*np.sin(self.angle)
    
    def Fx(self):
        return self.F*np.cos(self.angle)
    def Fy(self):
        return self.F*np.sin(self.angle)
    
    def phi(self, gamma):
        # gamma = drag coeficient (pN*sec/micron)
        return gamma*self.vr
    def alpha(self, gamma):
        return self.phi(gamma)/self.Fs
    def beta(self, gamma):
        return 1.+self.alpha(gamma)
    
    def bind(self):
        # if binding takes place, then also retract
        self.bound=True
        self.elong=False
        return "bound"
    
    def unbinding_rate(self):
        tau1 = self.t1*np.exp(-abs(self.F)/self.Fd1)
        tau2 = self.t2*np.exp(-abs(self.F)/self.Fd2)
        return 1/(tau1+tau2)
        
    def unbind(self):
        #unbinding and keep retracting (??)
        self.bound=False
        self.angle=self.a0
        return "unbound"
    
    ### STOCHASTIC FUNCION ###
    def update_binding(self, dt):
        # if binded, use unbinding rate to unbind
        if self.bound:
            if (np.random.uniform(0,1)<self.unbinding_rate()*dt):
                return self.unbind()
        # if unbinded, use binding rates, but only if elongating
        elif self.elong:
            if (np.random.uniform(0,1)<self.a0*dt):
                return self.bind()
        return "nothing"