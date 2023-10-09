import numpy as np
from pilus import Pilus

class Cell:
    
    def __init__(self, n, c, a0, L0, t1,t2,Fd1,Fd2,Fs, ve,vr):
        """
         n: average number of pili
         c: pili creation rate
         a0: pili binding rate
        """
        self.c = c
        self.x=np.zeros(2) #x-y position
        self.v=np.zeros(2) #x-y speed
        self.pilus_params = (a0, L0, t1,t2,Fd1,Fd2,Fs, ve,vr)
        self.pili=[]       #list of pili
        for i in range(np.random.randint(1,n+1)):
            self.pili.append(Pilus(*self.pilus_params))
        self.bound_pili_idxs=[] #list of the indexes of bound pili
        self.resetEventCounters()
    
    def Npili(self):
        return len(self.pili)
    def Nbound(self):
        return len(self.bound_pili_idxs)
    def resetEventCounters(self):
        self.L0Events = 0
        self.bindEvents = 0
        self.unbindEvents = 0
        return
        
    ############## DETERMINISTIC FORCE CALCULATION, with matrices algebra #######################
    #also updates self.bound_pili_idxs
    
    def solve_forces(self,Bindexes,gamma):
        if len(Bindexes)==0:
            print("WARNING: Forces are None.")
            return [None,None]
        nB=len(Bindexes)
        C=np.zeros((nB,nB)) # matrix of cos(angle_i - angle-j)
        E=np.eye(nB)
        b=np.ones(nB)
        for i in range(nB):
            E[i,i]*=self.pili[Bindexes[i]].beta(gamma)
            b[i]*=self.pili[Bindexes[i]].phi(gamma)
            for j in range(i+1,nB):
                C[i][j]=np.cos(self.pili[Bindexes[i]].angle-self.pili[Bindexes[j]].angle)
        A= C + C.T + E
        F=np.linalg.solve(A,b)
        if (F<0).any():
            #Bindexes_active are the indexes of the pili which are effectively pulling
            #we do not count the ones for which the algebra gives negative force ('compressed')
            #we treat the latter as binded pili which do not contribute to the force balance (zero force)
            Bindexes_active=Bindexes[ F[i]>=0 ]
            if Bindexes_active is None: Bindexes_active=[]
            elif type(Bindexes_active)==int: Bindexes_active=[Bindexes_active]
            return self.solve_forces(Bindexes_active, gamma)
        return [F,Bindexes]
    
    def calc_forces(self, gamma):
        # gamma = drag coeficient (pN*sec/micron)
        self.bound_pili_idxs=[] #list of bound-pili indexes
        for idx,p in enumerate(self.pili):
            p.F=0
            if p.bound:
                self.bound_pili_idxs.append(idx)
        
        if self.Nbound()>0: #solve the linear system A*F=b for binded-pili forces F (see [1])
            Forces,Bpi_active=self.solve_forces(self.bound_pili_idxs, gamma)
            if Forces is not None:
                for F,Bi in zip(Forces,Bpi_active):
                    self.pili[Bi].F = F
    
        
    ############# STOCHASTIC FUNCTIONS ###########################
    def bindings(self, dt):
        for p in self.pili:
            outcome = p.update_binding(dt)
            if   outcome=="bound":     self.bindEvents+=1
            elif outcome=="unbound": self.unbindEvents+=1
    
    def add_pili(self, dt):
        #try to grow a new pilus
        if np.random.uniform(0,1)<self.c*dt:
            self.pili.append(Pilus(*self.pilus_params))
    
    ############### DETERMINISTIC MOVEMENT ##################
    # body: the speed is proportional to the sum of the pili forces (from v(F) relation)
    # free pili: uniform radial elongation
    # bound pili: update angle and length after cell's movement
    def move(self, gamma, dt):
        # gamma = drag coeficient (pN*sec/micron)
        #    dt = timestep (sec)
        self.v=np.zeros(2)
        for p in self.pili:
            #if bound, add the force
            if p.bound:
                self.v[0]+=p.Fx()
                self.v[1]+=p.Fy()
            #else, move the pilus, elongating or retracting
            elif p.elong:
                p.L+=p.ve*dt
            else:
                p.L-=p.vr*dt
        self.v/=gamma
        # Euler integration
        dx=self.v*dt
        self.x+=dx
        
        #update pili lengths and dynamical angles
        for bi in self.bound_pili_idxs:
            #xp,yp are the coordinates of the Bi-th binded-pilus,
            # with origin in the cell's new position x+dx.
            p=self.pili[bi]
            xp=p.x() - dx[0]
            yp=p.y() - dx[1]
            Lp = np.sqrt(xp*xp+yp*yp)
            p.L=Lp
            if Lp>=p.L0:
                p.a = np.arctan2(yp/Lp,xp/Lp)
        
        #check for zero-length events
        for j in range(self.Npili()):
            i=self.Npili()-j-1 #start from the last one
            if self.pili[i].L<self.pili[i].L0:
                self.L0Events += 1
                self.pili.pop(i)
                #update bound_pili_idxs after .pop(i):
                # the following indexes must shift backwards by 1;
                # the i-th pilus will be removed in the next call of calc_forces().
                for k in range(self.Nbound()):
                    if self.bound_pili_idxs[k]>i:
                        self.bound_pili_idxs[k]-=1
        
                
    
    ############### SIMULATION STEP ########################
    def step(self, gamma, dt):
        #stochastic events
        self.bindings(dt)
        self.add_pili(dt)
        #update force balance
        self.calc_forces(gamma)
        #deterministic movements
        self.move(gamma, dt)
        #update force balance
        self.calc_forces(gamma);