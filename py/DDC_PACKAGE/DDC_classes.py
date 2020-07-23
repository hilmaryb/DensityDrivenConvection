#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 11:24:20 2020

@author: hilmaryb
"""
import numpy as np
from DDC_PACKAGE.Field import Field
from scipy import sparse as sp
from scipy.sparse import linalg as splin
#import matplotlib.pyplot as plt
#from scipy import sparse as sp
#from scipy.sparse import linalg as splin
#from Field import *
#import scipy
#import time


"""
 - CLASSES - 
"""

class StreamSolver:
    # A class that handles solving a 5pt stenciled Poisson-like equation
    # for the stream function, granted a permeability variation can be introduced
    # Functions basically the same as the Laplacian class except there is an
    # Added gradF*gradC term, which allows for non-uniform permeabilities
    # The proposed fluctuations shall be defined on the cell centers
    def __init__(self,F_FIELD):
        self.Nx = F_FIELD.Nx -1
        self.Ny = F_FIELD.Ny -1
        self.hx = F_FIELD.hx
        self.hy = F_FIELD.hy

        #Needed operations on F, these are all N-1 compared to N for F        
        dFdx = ddx(F_FIELD)
        dFdy = ddy(F_FIELD)
        Fav = avg_grain(F_FIELD)
        
        hx = self.hx
        hy = self.hy
        a = np.ones(self.Nx); a[-1]=0
        u1_ones = np.tile(a,self.Ny)
        b = np.ones(self.Nx); b[0]=0
        l1_ones = np.tile(b,self.Ny)
        
        d = -2/Fav.vector*(1.0/hx**2 + 1.0/hy**2)
        u1 = u1_ones/(hx*hx*Fav.vector) - u1_ones*dFdx.vector/(hx*Fav.vector**2)
        l1 = l1_ones/(hx*hx*Fav.vector) + l1_ones*dFdx.vector/(hx*Fav.vector**2)
        uN = 1.0/(hy*hy*Fav.vector) - dFdy.vector/(hy*hy*Fav.vector)
        lN = 1.0/(hy*hy*Fav.vector) + dFdy.vector/(hy*hy*Fav.vector)

        self.spmat = sp.spdiags([np.roll(lN,-self.Nx), np.roll(l1,-1), d, np.roll(u1,1), np.roll(uN,self.Nx)], [-self.Nx,-1,0,1,self.Nx],self.Nx*self.Ny,self.Nx*self.Ny)
        
    def factorize(self):
        #SuperLU factorize
        self.f = splin.factorized(self.spmat.tocsc())
        
    def solve(self,RHS_field):
        #Solve using the SuperLU solver, for given RHS
        arr = self.f(RHS_field.vector)
        tmpField = Field(RHS_field.Nx,RHS_field.Ny,RHS_field.X,RHS_field.Y)
        tmpField.from_array(arr)
        return tmpField
        
        
class Laplacian:
    # A Laplacian class for solving 5pt stenciled Poisson equations.
    # Upon initializing and factorizing, the banded sparse matrix is
    # LU decomposed, and can return a <field> object by calling Laplacian.solve
    # using a rapid SuperLU algorithm,
    # for repeated solutions of the Poisson equation Ax=b for various b.
    # Nb: This is just a special case of StreamSolver for f=1, and is not better in any way.
    def __init__(self,Nx,Ny,X,Y):
        self.Nx = Nx
        self.Ny = Ny
        hx = X/(1.0+Nx)
        hy = Y/(1.0+Ny)
        
        d = np.ones(Nx*Ny)
        a = np.ones(Nx); a[-1]=0;
        xdiag = np.tile(a,Ny)
        self.spmat = sp.spdiags([d/(hy*hy),xdiag/(hx*hx),-2*(1/(hx*hx) + 1/(hy*hy))*d,np.roll(xdiag,1)/(hx*hx),d/(hy*hy)],[-Nx,-1,0,1,Nx],Nx*Ny,Nx*Ny)
        
    def factorize(self):
        self.f = splin.factorized(self.spmat)
    
    def solve(self,RHS_field):
        #Input a field. Take it's vector
        arr = self.f(RHS_field.vector)
        tmpField = Field(self.N)
        tmpField.from_array(arr)
        #Output a field type
        return tmpField
    
class FVM:
    #An object used to construct/update the time evolution matrix
    def __init__(self,C,dt):
        self.Nx = C.Nx
        self.Ny = C.Ny
        self.numel = C.numel
        self.hx = C.hx
        self.hy = C.hy
        self.dt = dt
        
    def set_dt(self,dt):
        #Change time step
        self.dt=dt
        
    def make_pt_lists(self):
        #For ease of use, make a (r,c) index of all point indices [i]
        self.r = np.empty(self.numel,dtype=int)
        self.c = np.empty(self.numel,dtype=int)
        for i in range(0,self.numel):
            self.r[i]=int(i/self.Nx)
            self.c[i]=int(i%self.Nx)

    def make_diff_op(self):
        #Make a diffusion operator with 0-flux BC's left/right/bot
        #And a dirichlet BC on top.
        
        d1 = np.ones(self.Nx)*(2/(self.hx**2) + 1/(self.hy**2));
        dtop = np.ones(self.Nx)*(2/(self.hx**2) + 4/(self.hy**2));
        dtop[0] = 1/(self.hx**2) + 4/(self.hy**2); dtop[-1] = 1/(self.hx**2) + 4/(self.hy**2)
        d = -np.concatenate([dtop, np.tile(1/(self.hy**2)+d1,self.Ny-2), d1])
        l_ones = np.ones(self.Nx)/(self.hx**2); l_ones[-1]=0
        l1 = np.tile(l_ones,self.Ny)
        u_ones = np.ones(self.Nx)/(self.hx**2); u_ones[0]=0
        u1 = np.tile(u_ones,self.Ny)
        lN = np.ones(self.numel)/(self.hy**2)
        uN = np.ones(self.numel)/(self.hy**2)
        uN[self.Nx:2*self.Nx] = 4/(3*self.hy**2)
        self.diffuse = sp.spdiags([lN,l1,d,u1,uN],[-self.Nx,-1,0,1,self.Nx],self.numel,self.numel)*self.dt
#        
    def make_diff_op_isolated(self):
        #Make a diffusion operator with 0-flux BC's on all edges.
        #Mainly for testing purposes
        d1 = np.ones(self.Nx)*(2/(self.hx**2) + 1/(self.hy**2));
        d1[0] = 1/(self.hx**2) + 1/(self.hy**2); d1[-1] = 1/(self.hx**2) + 1/(self.hy**2)
        d = -np.concatenate([d1, np.tile(1/(self.hy**2)+d1,self.Ny-2), d1])
        l_ones = np.ones(self.Nx)/(self.hx**2); l_ones[-1]=0
        l = np.tile(l_ones,self.Ny)
        u_ones = np.ones(self.Nx)/(self.hx**2); u_ones[0]=0
        u = np.tile(u_ones,self.Ny)
        b = np.ones(self.numel)/(self.hy**2)
        self.diffuse = sp.spdiags([b,l,d,u,b],[-self.Nx,-1,0,1,self.Nx],self.numel,self.numel)*self.dt
        
    def make_advect_op(self,PSI_FIELD):
        #Make an advection operator from a given stream function field
        #Calculate velocities
            #Pad the boundaries with 0's
        padded = np.zeros((self.Ny+1,self.Nx+1))
        padded[1:-1,1:-1] = PSI_FIELD.vector.reshape((self.Ny-1,self.Nx-1))
            #Differentiate to get velocity fields
        Ux = -np.diff(padded,axis=0)/self.hy
        Uy = np.diff(padded,axis=1)/self.hx
            #Can be accessed if needed
        self.U=[Ux,Uy]        
            #Note they are Ny x (Nx+1) and (Ny+1) x Nx respectively
            
        #Make the diagonal bands, which the matrix consists of
        d = np.zeros(self.numel)     #THIS CELL, [I,I]
        u1 = np.zeros(self.numel)    #RIGHT CELL [I,I+1]
        l1 = np.zeros(self.numel)    #LEFT CELL [I,I-1]
        uN = np.zeros(self.numel)    #SOUTH CELL [I,I+N]
        lN = np.zeros(self.numel)    #NORTH CELL [I,I-N]
        d = (Uy[self.r,self.c]- Uy[self.r+1,self.c])/(2*self.hy) + (Ux[self.r,self.c]- Ux[self.r,self.c+1])/(2*self.hx)
        u1 = -Ux[self.r,self.c+1]/(2*self.hx)
        l1 = Ux[self.r,self.c]/(2*self.hx)
        uN = -Uy[self.r+1,self.c]/(2*self.hy)
        lN = Uy[self.r,self.c]/(2*self.hy)
        #Roll them so the correct rows line up
        u1 = np.roll(u1,1)
        uN = np.roll(uN,self.Nx)
        l1 = np.roll(l1,-1)
        lN = np.roll(lN,-self.Nx)
        self.advect = sp.spdiags([lN,l1,d,u1,uN],[-self.Nx,-1,0,1,self.Nx],self.numel,self.numel)*self.dt
        
    def apply_dirichlet_top(self,C):
        for i in range(0,self.Nx):
            C[i] = C[i] + 8/(3*self.hy*self.hy)*self.dt
    def apply_dirichlet_top_f(self,C,f):
        for i in range(0,self.Nx):
            x = self.hx*(0.5+self.Nx)
            C[i] = C[i] + 8/(3*self.hy*self.hy)*f(x)*self.dt
    def apply_dirichlet_top_vec(self,C,v):
        for i in range(0,self.Nx):
            C[i] = C[i] + 8/(3*self.hy*self.hy)*v[i]*self.dt
            
class FVM_ADI_SP:
    #An object used to construct/update the time evolution matrix
    def __init__(self,C,dt):
        self.Nx = C.Nx
        self.Ny = C.Ny
        self.numel = C.numel
        self.hx = C.hx
        self.hy = C.hy
        self.dt = dt
        
    def set_dt(self,dt):
        #Change time step
        self.dt=dt
        
    def make_pt_lists(self):
        #For ease of use, make a (r,c) index of all point indices [i]
        self.r = np.empty(self.numel,dtype=int)
        self.c = np.empty(self.numel,dtype=int)
        for i in range(0,self.numel):
            self.r[i]=int(i/self.Nx)
            self.c[i]=int(i%self.Nx)

    def make_diff_op(self):
        #Make a diffusion operator with 0-flux BC's left/right/bot
        #And a dirichlet BC on top.
        
        d1 = np.ones(self.Nx)*(2/(self.hx**2) + 1/(self.hy**2));
        dtop = np.ones(self.Nx)*(2/(self.hx**2) + 4/(self.hy**2));
        dtop[0] = 1/(self.hx**2) + 4/(self.hy**2); dtop[-1] = 1/(self.hx**2) + 4/(self.hy**2)
        d = -np.concatenate([dtop, np.tile(1/(self.hy**2)+d1,self.Ny-2), d1])
        l_ones = np.ones(self.Nx)/(self.hx**2); l_ones[-1]=0
        l1 = np.tile(l_ones,self.Ny)
        u_ones = np.ones(self.Nx)/(self.hx**2); u_ones[0]=0
        u1 = np.tile(u_ones,self.Ny)
        lN = np.ones(self.numel)/(self.hy**2)
        uN = np.ones(self.numel)/(self.hy**2)
        uN[self.Nx:2*self.Nx] = 4/(3*self.hy**2)
        self.diffuse = sp.spdiags([lN,l1,d,u1,uN],[-self.Nx,-1,0,1,self.Nx],self.numel,self.numel)*self.dt
#        
    def make_diff_op_isolated(self):
        #Make a diffusion operator with 0-flux BC's on all edges.
        #Seperate for x and y direction, for ADI!
        #Mainly for testing purposes
        
        d1x = np.ones(self.Nx)*2/(self.hx**2);
        d1x[0] = 1/(self.hx**2); d1x[-1] = 1/(self.hx**2)
        dx = -np.tile(d1x,self.Ny)*self.dt
        l_ones = np.ones(self.Nx)/(self.hx**2); l_ones[-1]=0
        l1 = np.tile(l_ones,self.Ny)*self.dt
        u_ones = np.ones(self.Nx)/(self.hx**2); u_ones[0]=0
        u1 = np.tile(u_ones,self.Ny)*self.dt
        #self.diffuseX = sp.spdiags([l1,dx,u1],[-1,0,1],self.numel,self.numel)
        
        d1y = np.ones(self.Nx)*1/(self.hy**2);
        dy = -np.concatenate([d1y, np.tile(1/(self.hy**2)+d1y,self.Ny-2), d1y])*self.dt
        lN = np.ones(self.numel)/(self.hy**2)*self.dt
        uN = np.ones(self.numel)/(self.hy**2)*self.dt
        #self.diffuseY = sp.spdiags([lN,dy,uN],[-self.Nx,0,self.Nx],self.numel,self.numel)
        return [l1,dx,u1,lN,dy,uN]
        
    def make_advect_op(self,PSI_FIELD):
        #Make an advection operator from a given stream function field
        #Calculate velocities
            #Pad the boundaries with 0's
        padded = np.zeros((self.Ny+1,self.Nx+1))
        padded[1:-1,1:-1] = PSI_FIELD.vector.reshape((self.Ny-1,self.Nx-1))
            #Differentiate to get velocity fields
        Ux = -np.diff(padded,axis=0)/self.hy
        Uy = np.diff(padded,axis=1)/self.hx
            #Can be accessed if needed
        self.U=[Ux,Uy]        
            #Note they are Ny x (Nx+1) and (Ny+1) x Nx respectively
            
        #Make the diagonal bands, which the matrix consists of
        dx = np.zeros(self.numel)     #THIS CELL (x-dir), [I,I]
        dy = np.zeros(self.numel)     #THIS CELL (y-dir), [I,I]
        u1 = np.zeros(self.numel)    #RIGHT CELL [I,I+1]
        l1 = np.zeros(self.numel)    #LEFT CELL [I,I-1]
        uN = np.zeros(self.numel)    #SOUTH CELL [I,I+N]
        lN = np.zeros(self.numel)    #NORTH CELL [I,I-N]
        for i in range(0,self.numel):
            dy[i] = (Uy[self.r[i],self.c[i]]- Uy[self.r[i]+1,self.c[i]])/(2*self.hy)*self.dt
            dx[i] = (Ux[self.r[i],self.c[i]]- Ux[self.r[i],self.c[i]+1])/(2*self.hx)*self.dt
            u1[i] = -Ux[self.r[i],self.c[i]+1]/(2*self.hx)*self.dt
            l1[i] = Ux[self.r[i],self.c[i]]/(2*self.hx)*self.dt
            uN[i] = -Uy[self.r[i]+1,self.c[i]]/(2*self.hy)*self.dt
            lN[i] = Uy[self.r[i],self.c[i]]/(2*self.hy)*self.dt
        #Roll them so the correct rows line up
        
        u1 = np.roll(u1,1)
        uN = np.roll(uN,self.Nx)
        l1 = np.roll(l1,-1)
        lN = np.roll(lN,-self.Nx)
        #Can also be made into sparse mats
        self.advectX = sp.spdiags([l1,dx,u1],[-1,0,1],self.numel,self.numel)
        self.advectY = sp.spdiags([lN,dy,uN],[-self.Nx,0,self.Nx],self.numel,self.numel)
        
    def apply_dirichlet_top(self,C):
        for i in range(0,self.Nx):
            C[i] = C[i] + 8/(3*self.hy*self.hy)*self.dt
    def apply_dirichlet_top_f(self,C,f):
        for i in range(0,self.Nx):
            x = self.hx*(0.5+self.Nx)
            C[i] = C[i] + 8/(3*self.hy*self.hy)*f(x)*self.dt
    def apply_dirichlet_top_vec(self,C,v):
        for i in range(0,self.Nx):
            C[i] = C[i] + 8/(3*self.hy*self.hy)*v[i]*self.dt
            
class FVM_ADI:
    #An object used to construct/update the time evolution matrix
    def __init__(self,C,dt):
        self.Nx = C.Nx
        self.Ny = C.Ny
        self.numel = C.numel
        self.hx = C.hx
        self.hy = C.hy
        self.dt = dt
        
    def set_dt(self,dt):
        #Change time step
        self.dt=dt
        
    def make_pt_lists(self):
        # For ease of use, make a (r,c) index of all point indices [i]
        # Precalculate this to vectorize most other functions
        self.r = np.empty(self.numel,dtype=int)
        self.c = np.empty(self.numel,dtype=int)
        for i in range(0,self.numel):
            self.r[i]=int(i/self.Nx)
            self.c[i]=int(i%self.Nx)

    def make_diff_op(self):
        #Make a diffusion operator with 0-flux BC's left/right/bot
        #And a dirichlet BC on top.
        # The BC needs to be applied to the C Field aswell
        # - before solving (for explicit steps)
        # - after multiplying (for implicit steps)
        
        d1x = np.ones(self.Nx)*2/(self.hx**2);
        d1x[0] = 1/(self.hx**2); d1x[-1] = 1/(self.hx**2)
        dx = -np.tile(d1x,self.Ny)*self.dt
        l_ones = np.ones(self.Nx)/(self.hx**2); l_ones[-1]=0
        l1 = np.tile(l_ones,self.Ny)*self.dt
        u_ones = np.ones(self.Nx)/(self.hx**2); u_ones[0]=0
        u1 = np.tile(u_ones,self.Ny)*self.dt        
        d1y = np.ones(self.Nx)*1/(self.hy**2);
        dy = -np.concatenate([d1y+3/(self.hy**2), np.tile(1/(self.hy**2)+d1y,self.Ny-2), d1y])*self.dt
        lN = np.ones(self.numel)/(self.hy**2)*self.dt
        uN = np.ones(self.numel)/(self.hy**2)*self.dt
        uN[self.Nx:2*self.Nx] = 4/(3*self.hy**2)*self.dt
        return l1,dx,u1,lN,dy,uN
#        
    def make_diff_op_isolated(self):
        #Make a diffusion operator with 0-flux BC's on all edges.
        #Seperate for x and y direction, for ADI!
        #Mainly for testing purposes
        
        d1x = np.ones(self.Nx)*2/(self.hx**2);
        d1x[0] = 1/(self.hx**2); d1x[-1] = 1/(self.hx**2)
        dx = -np.tile(d1x,self.Ny)*self.dt
        l_ones = np.ones(self.Nx)/(self.hx**2); l_ones[-1]=0
        l1 = np.tile(l_ones,self.Ny)*self.dt
        u_ones = np.ones(self.Nx)/(self.hx**2); u_ones[0]=0
        u1 = np.tile(u_ones,self.Ny)*self.dt
        
        d1y = np.ones(self.Nx)*1/(self.hy**2);
        dy = -np.concatenate([d1y, np.tile(1/(self.hy**2)+d1y,self.Ny-2), d1y])*self.dt
        lN = np.ones(self.numel)/(self.hy**2)*self.dt
        uN = np.ones(self.numel)/(self.hy**2)*self.dt

        #Return bands
        return l1,dx,u1,lN,dy,uN
        
    def make_advect_op(self,PSI_FIELD):
        #Make an advection operator from a given stream function field
        #Calculate velocities
            #Pad the boundaries with 0's
        padded = np.zeros((self.Ny+1,self.Nx+1))
        padded[1:-1,1:-1] = PSI_FIELD.vector.reshape((self.Ny-1,self.Nx-1))
            #Differentiate to get velocity fields
        Ux = -np.diff(padded,axis=0)/self.hy
        Uy = np.diff(padded,axis=1)/self.hx
            #Can be accessed if needed
        #self.U=[Ux,Uy]        
            #Note they are Ny x (Nx+1) and (Ny+1) x Nx respectively
            
        #Make the diagonal bands, which the matrix consists of
        dx = np.zeros(self.numel)     #THIS CELL (x-dir), [I,I]
        dy = np.zeros(self.numel)     #THIS CELL (y-dir), [I,I]
        u1 = np.zeros(self.numel)    #RIGHT CELL [I,I+1]
        l1 = np.zeros(self.numel)    #LEFT CELL [I,I-1]
        uN = np.zeros(self.numel)    #SOUTH CELL [I,I+N]
        lN = np.zeros(self.numel)    #NORTH CELL [I,I-N]

        dy = (Uy[self.r,self.c] - Uy[self.r+1,self.c])
        dx = (Ux[self.r,self.c]- Ux[self.r,self.c+1])
        u1 = -Ux[self.r,self.c+1]
        l1 = Ux[self.r,self.c]
        uN = -Uy[self.r+1,self.c]
        lN = Uy[self.r,self.c]
        #Roll them so the correct rows line up
        dx *=self.dt/(2*self.hx)
        dy *=self.dt/(2*self.hy)
        u1 = np.roll(u1,1)/(2*self.hx)*self.dt
        uN = np.roll(uN,self.Nx)/(2*self.hy)*self.dt
        l1 = np.roll(l1,-1)/(2*self.hx)*self.dt
        lN = np.roll(lN,-self.Nx)/(2*self.hy)*self.dt
        
        #Return bands
        return [l1,dx,u1,lN,dy,uN]
        
    def apply_dirichlet_top(self,C):
        #Apply the C=1.0 boundary condition on the top row, y=0
        C[0:self.Nx] = C[0:self.Nx] + 8/(3*self.hy*self.hy)*self.dt
        
    def apply_dirichlet_top_f(self,C,f):
        #Apply a C=f(x) boundary condition on the top row, y=0
        for i in range(0,self.Nx):
            x = self.hx*(0.5+self.Nx)
            C[i] = C[i] + 8/(3*self.hy*self.hy)*f(x)*self.dt
            
    def apply_dirichlet_top_vec(self,C,v):
        #Apply a C(xj) = v_j boundary condition on the top row, y=0
        #v must have length FVM_ADI.Nx
        C[0:self.Nx] = C[0:self.Nx] + 8/(3*self.hy*self.hy)*v*self.dt
        
        

"""
 - FUNCTIONS - 
"""
        
def ddx(FIELD):
    # Differentiates a Field in the x-direction. Centers the derivative between 4 other points,
    # Averages 2 midpoint rules to yield a Nx-1 x Ny-1 sized ddx Field
    V = FIELD.vector
    Nx = FIELD.Nx
    Ny = FIELD.Ny
    hx = FIELD.hx
    d = np.zeros((Nx-1)*(Ny-1))
    for I in range(0,(Nx-1)*(Ny-1)):
        r = int(I/(Nx-1))
        c = int(I%(Nx-1))
        d[I] = (V[(r+1)*Nx+c+1] + V[r*Nx+c+1] - V[(r)*Nx+c] - V[(r+1)*Nx+c])/(2*hx)
    result = Field(Nx-1,Ny-1,FIELD.X,FIELD.Y)
    result.from_array(d)
    return result

def ddy(FIELD):
    # Differentiates a Field in the y-direction. Centers the derivative between 4 other points,
    # Averages 2 midpoint rules to yield a Nx-1 x Ny-1 sized ddy Field
    V = FIELD.vector
    Nx = FIELD.Nx
    Ny = FIELD.Ny
    hy = FIELD.hy
    d = np.zeros((Nx-1)*(Ny-1))
    for I in range(0,(Nx-1)*(Ny-1)):
        r = int(I/(Nx-1))
        c = int(I%(Nx-1))
        d[I] = (V[(r+1)*Nx+c+1] - V[r*Nx+c+1] - V[(r)*Nx+c] + V[(r+1)*Nx+c])/(2*hy)
    result = Field(Nx-1,Ny-1,FIELD.X,FIELD.Y)
    result.from_array(d)
    return result

def avg_grain(FIELD):
    # Coarse grains a Field,
    # Returns a Nx-1 x Ny-1 Field, in which every value is the average of the 4 cornering cells
    V = FIELD.vector
    Nx = FIELD.Nx
    Ny = FIELD.Ny
    A = np.zeros((Nx-1)*(Ny-1))
    for I in range(0,(Nx-1)*(Ny-1)):
        r = int(I/(Nx-1))
        c = int(I%(Nx-1))
        A[I] = (V[(r+1)*Nx+c+1] + V[r*Nx+c+1] + V[(r)*Nx+c] + V[(r+1)*Nx+c])/4
    result = Field(Nx-1,Ny-1,FIELD.X,FIELD.Y)
    result.from_array(A)
    return result

def ADI_TRI_X(l,d,u,RHS):
    # Solves a tridiagonal matrix with lower band l, diagonal d and upper band u for a given RHS
    numel = len(RHS)
    sol = np.empty_like(RHS)
    #Forward elim
    for i in range(1,numel):
        s = l[i-1]/d[i-1]
        d[i] -= u[i]*s
        RHS[i] -= RHS[i-1]*s
    #Backward subst.
    sol[-1] = RHS[-1]/d[-1]
    for i in range(numel-2,-1,-1):
        sol[i] = (RHS[i]-u[i+1]*sol[i+1])/d[i]
    return sol

def ADI_BAND_Y(l,d,u,O,RHS):
    # Solves a banded matrix with lower band l, diagonal d and upper band u for a given RHS
    # The l and u bands are offset by an amount O, which for ADI puroses is Nx
    numel = len(RHS)
    sol = np.empty_like(RHS)
    #Forward elim
    for i in range(O,numel):
        s = l[i-O]/d[i-O]
        d[i] -= u[i]*s
        RHS[i] -= RHS[i-O]*s
    #Backward subst.
    for i in range(numel-1,numel-O-1,-1):
        sol[i] = RHS[i]/d[i]
    for i in range(numel-O-1,-1,-1):
        sol[i] = (RHS[i]-u[i+O]*sol[i+O])/d[i]
    return sol

def ADI_TRI_X_VEC(l,d,u,RHS_FIELD):
    # Solves a tridiagonal matrix with lower band l, diagonal d and upper band u for a given RHS
    # Vectorized version of the normal Thomas algorithm, which utilizes the fact that
    # s=0 every Ny
    RHS = np.copy(RHS_FIELD.vector)
    sol = np.empty_like(RHS)
    I = np.arange(0,RHS_FIELD.Ny)*RHS_FIELD.Nx
    #Forward elim
    for i in range(1,RHS_FIELD.Nx):
        s = l[i+I-1]/d[i+I-1]
        d[i+I] -= u[i+I]*s
        RHS[i+I] -= RHS[i+I-1]*s
    #Backward subst.
    sol[-1-I] = RHS[-1-I]/d[-1-I]
    for i in range(RHS_FIELD.Nx-1,-1,-1):
        sol[i+I-1] = (RHS[i+I-1]-u[i+I]*sol[i+I])/d[i+I-1]
    return sol

def ADI_BAND_Y_VEC(l,d,u,RHS_FIELD):
    # Solves a banded matrix with lower band l, diagonal d and upper band u for a given RHS
    # The l and u bands are offset by an amount O, which for ADI puroses is Nx
    # Vectorized version of the normal Banded algorithm, which utilizes the fact that every
    # Nx points are independent.
    RHS = np.copy(RHS_FIELD.vector)
    numel = RHS_FIELD.numel
    O = RHS_FIELD.Nx
    sol = np.empty_like(RHS_FIELD.vector)
    I = np.arange(0,RHS_FIELD.Nx)
    #Forward elim
    for i in range(1,RHS_FIELD.Ny):
        s = l[(i-1)*RHS_FIELD.Nx+I]/d[(i-1)*RHS_FIELD.Nx+I]
        d[i*RHS_FIELD.Nx+I] -= u[i*RHS_FIELD.Nx+I]*s
        RHS[i*RHS_FIELD.Nx+I] -= RHS[(i-1)*RHS_FIELD.Nx+I]*s
    #Backward subst.
        sol[-1-I] = RHS[-1-I]/d[-1-I]
        
    for i in range(RHS_FIELD.Ny-2,-1,-1):
        sol[i*RHS_FIELD.Nx+I] = (RHS[i*RHS_FIELD.Nx+I]-u[(i+1)*RHS_FIELD.Nx+I]*sol[(i+1)*RHS_FIELD.Nx+I])/d[i*RHS_FIELD.Nx+I]
    return sol

def BAND_MUL(l,d,u,O,x):
    # Returns the vector b=A*x, where the matrix A
    # consists of bands l d and u, with l and u
    # offset by an amount O from the diagonal
    # Note: l[i], d[i], u[i] correspond to
    # matrix COLUMN A
    tmp = np.empty_like(x)
    
    for i in range(0,O):
        #First O elements, only d and u band
        tmp[i] = d[i]*x[i] + u[i+O]*x[i+O]
    for i in range(O,len(x)-O):
        tmp[i] = l[i-O]*x[i-O] + d[i]*x[i] + u[i+O]*x[i+O]
    for i in range(len(x)-O,len(x)):
        tmp[i] = l[i-O]*x[i-O] + d[i]*x[i]
    return tmp