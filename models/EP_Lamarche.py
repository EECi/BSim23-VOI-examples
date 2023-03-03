# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 13:33:21 2022
@author: bandeiraneto

Class implementing energy pile (borehole) behaviour as described
in Lamarche and Beauchamp (2007) [1].

[1] L. Lamarche and B. Beauchamp, “A new contribution to the finite
line-source model for geothermal boreholes,” Energy Build., vol. 39,
no. 2, pp. 188–198, 2007, doi: 10.1016/j.enbuild.2006.06.003.
"""
import numpy as np
import pandas as pd


def FLSM_G_Lamarche(ks,Rb,Cpvs,H,rb,t):  
    
    """ Simplified FLSM mean integral calculation from Lamarche and Beauchamp 2007

        ks = soil thermal conductivity (W/m.k)
        a = soil thermal diffusivity (m2/s)
        H = depth of the borehole (m)
        rb = borehole radius (m)
        T0 = undisturbed ground temp (degC)
        q = power applied to the GHE (W/m)
        Rb = GHE thermal resistance (m.K/W)
        t = time (hours)
    """
    
    import numpy as np
    from scipy.integrate import quad
    from scipy.special import erfc
    alpha = ks/Cpvs
    
    # Function to evaluate full improved FLSM at single time t
    def get_Temp(ks,Rb,alpha,H,rb,t):
        # Set up constants
        B = rb/H
        Fo = (alpha*t)/rb**2
        t_star = 9*B**2*Fo
        g = H/np.sqrt(4*alpha*t)
        
        # Define Da and Db components
        Da = ((np.sqrt(B**2 + 1)*erfc(g*np.sqrt(B**2 + 1))) - (B*erfc(g*B)) - 
              ((np.exp(-g**2 * (B**2 + 1)) - np.exp(-g**2 * B**2))/(g*np.sqrt(np.pi))))
        
        Db = (np.sqrt(B**2 + 1)*erfc(g*np.sqrt(B**2 + 1)) - 
              0.5*(B*erfc(g*B) + np.sqrt(B**2 + 4)*erfc(g*np.sqrt(B**2 + 4))) - 
              (np.exp(-g**2 * (B**2 + 1)) - 0.5*(np.exp(-g**2 * B**2) + np.exp(-g**2 * (B**2 + 4))))/(g*np.sqrt(np.pi)))
        
        # Define A, B integrals, they are the same, just different bounds
        def AB_int(z, B, g):
            return(erfc(g*z)/np.sqrt(z**2 - B**2))
        
        # Put it together in a g func
        Gfunc = (quad(AB_int, B, np.sqrt(B**2 + 1), args=(B, g))[0] - Da) - (quad(AB_int, np.sqrt(B**2 + 1), np.sqrt(B**2 + 4), args=(B, g))[0] + Db)
        
        return((1/(2*np.pi*ks))*Gfunc)
    
    t = t*3600 # convert t to seconds
    Temp = np.array(t.apply(lambda x: get_Temp(ks,Rb,alpha,H,rb,x))) + Rb

    return(Temp)



class EnergyPile:
    """
    The Energy Pile (EP) class receives the energy pile properties and annalyse its fluid 
    temperature over time under a certain thermal load
    
    Class initialisation parameters:
    """
    
    def __init__(self,H,T0,rb,ks,Cpvs,Rb,qrate):
        """ 
        Initial EP class parameters (all float type).
        H:      EP length (m)
        Cpvs:   Volumetric Specific heat capacity of the ground aroung GHE (J/m^3K)
        rb:     GHE radius (m)
        T0:     Undisturbed Ground Temperature (degC)
        ks:     Ground Thermal Conductivity (W/mK)
        Rb:     GHE Thermal Resistance (mK/W)
        qrate:  fluid flow rate during opeation (L/s)
        
        The respective parameters ara calculated from the information given.
        alpha:  Soil thermal diffusivity (m/s^2)
        AR:     Pile aspect ratio (-)
        """
    
        self.H=H
        self.Cpvs=Cpvs
        self.rb=rb
        self.T0=T0
        self.ks=ks
        self.Rb=Rb
        self.qrate = qrate
        self.AR=self.H/(self.rb*2)
        self.alpha=self.ks/self.Cpvs
        
    def thermal_design(self,data):
        """
        Import the thermal load that the pile will be sbjected to
        
        data:   pd.DataFrame object with two columns ['Hours','Load_W']
        """
        
        self.data = data.copy()
        self.data['Load_Wm']=self.data.Load_W/self.H
        
    def pipe_properties(self,n,ri,ro,kp):
        """
        Define circulation pipe propreties of the pile, required for spliting
        the pile thermal resistance Rb and calculate the Pile G-functions
        
        n:    number of piles in the cross section (-) (1 U loop = 2 pipes)
        ri:   inner radius of the pipe (m)
        ro:   outer radius of the pipe (m)
        kp:   thermal conductivity of the pipe wall (W/mK)
        """
        self.n = n
        self.ri=ri
        self.ro=ro
        self.kp=kp
        
    def Rb_split(self):
        """
        Split the pile thermal resistance in pipe and concrete components
        Required for calculation using Loveridge Pile G-functions
        """
        import numpy as np
        
        rho_w = 998       # water density
        visc_w = 0.00089  # water viscosity
        
        f_rate=self.qrate/1000    # converting flow rate from L/s to m3/s
        Pr = 7.56                 # Water Prandtl number
        lambdaf = 0.6             # water thermal conductivity
        Ap = np.pi*self.ri**2     # inner pipe area
        Re = f_rate*(self.ri*2)/(Ap*(visc_w/rho_w))   # Reynolds number
        hi = 0.023*(Re**0.8)*(Pr**0.35)*lambdaf/(2*self.ri) # Dittus-Boelter Eq.
        Rpconv = 1/(2*self.n*np.pi*self.ri*hi) # convective resistance
        Rpcond = np.log(self.ro/self.ri)/(2*self.n*np.pi*self.kp) # cond. resist.
        self.Rp = Rpcond + Rpconv
        self.Rc = self.Rb - self.Rp
        


    def FLSM_Gfunc(self,mode='Lamarche'):
        """ 
            Basic FLSM simulation function
            define mode accordingly to the calculation wanted
            
            mode = 'mid'
            assumes the temperature at H/2 equals the average temperature
            mode = 'intmean'
            calculates the mean temperature by integrating the full lenght
            mod = 'Lamarche'
        """
        # from Models import FLSM_G_Lamarche
        if mode=='Lamarche':
            temps=FLSM_G_Lamarche(self.ks,self.Rb,self.Cpvs,self.H,self.rb,
                                   self.data.Hours)
        else:
            print("Model used needs to be Lamarche.")
            return
        
        self.data['FLSM_'+mode+'_G']=temps
        
        return(temps)


    def Fluid_Temp(self,Model,load=None):
        """ Calculation of the average fluid temperature (degC)
            considering the selected analytical model.
            
            load variable is to be used on the loadfactor_estimate function
            
            Model options:
                ILSMlinreg
                ILSMei
                ICSM
                FLSM_mid
                FLSM_intmean
                FLSM_Lamarche
                PGfunc_<pile_boundary>_<pipe_placement>
                    pile_boundary = 'upper' or 'lower'
                    pipe_placement = 'edge' or 'center'
        
        """
        
        import numpy as np
        import pandas as pd
        from scipy.fft import fft, ifft
        
        def FourierSci(a):
            return fft(np.concatenate((a,np.zeros(a.size-1))))
        
        def invFourierSci(a):
            return np.real(ifft(a))[0:int((a.size+1)/2)]
        
        # transforming load values in a load step function
        if type(load) == type(None):
            q = self.data.Load_Wm.copy().to_numpy()
        else:
            q = load.copy()
        q[1:] -= np.roll(q,1)[1:]
        
        # getting the G_function from the selected analytical model
        G = self.data[Model+'_G']
        # print(G)
        
        # applying fourier transforms
        FFT_load = FourierSci(q)
        FFT_Gfunc = FourierSci(G)
        out=(invFourierSci(FFT_load*FFT_Gfunc))
        
        self.data[Model+"_T"] = out + self.T0
        
        return (out + self.T0)
            
    
    def loadfactor_estimate(self,Tmax,Tmin,Tinterv,precision,Model,mode):
        """ 
        Calculate a factor of reduction (or increment) of the given design 
        thermal load that the energy pile can support, given the temperature
        limits defined
        
        Tmax:          Maximum fluid temperature (degC)
        Tmin:          Minimum fluid temperature (degC)
        Tinterv:       Interval the limit fluid temperature should fall in (degC)
            E.G. if Tmax=40 and Tinterv=1 - Max fluid temperature will be
            between 39 and 40 degC
        precision:     Step of change on the load factor (-)
        Model:         base analytical model for calculation of the fluid temperature
        mode:          critical operation mode, either 'heating' or 'cooling'
                       or both
        """
        
        def fluidtemp(Model,factor):
            load = factor*self.data.Load_W.copy().to_numpy()
            q = load/self.H
            deltaT = load/(self.qrate*0.001*998*4185.5)
            Tavg = self.Fluid_Temp(Model,load=q)
            Tin = Tavg + deltaT/2
            Tout = Tavg - deltaT/2
            maxT = max(Tin.max(),Tout.max())
            minT = min(Tin.min(),Tout.min())
            return [maxT,minT]
        
        def Booldefine(Tmax,Tmin,Tinterv,mode):
            # defining upper and lower limit verifications per operation
            if mode == 'heating':
                Upfunc = lambda x,y: y<Tmin
                Lowfunc = lambda x,y: y>(Tmin+Tinterv)
            elif mode == 'cooling':
                Upfunc = lambda x,y: x>Tmax
                Lowfunc = lambda x,y: x<(Tmax-Tinterv)
            else:
                Upfunc = lambda x,y: (x>Tmax or y<Tmin)
                Lowfunc =lambda x,y: (x<(Tmax-Tinterv) or y>(Tmin+Tinterv))
            return [Upfunc,Lowfunc]
            
        # starting with factor = 1
        factor = 1
        
        # calculating fluid temperatures        
        maxT,minT = fluidtemp(Model,factor)
        
        # defining boolean functions
        Upfunc,Lowfunc = Booldefine(Tmax,Tmin,Tinterv,mode)
        UpBool = Upfunc(maxT,minT)

        while UpBool:
            factor -= precision
            if factor < 0:
                print('reduce your precision value')
                return None
            maxT,minT = fluidtemp(Model,factor)
            UpBool = Upfunc(maxT,minT)
        
        # Lower limit Bollean
        LowBool = Lowfunc(maxT,minT)
        
        while LowBool:
            factor += precision
            maxT,minT = fluidtemp(Model,factor)
            LowBool = Lowfunc(maxT,minT)
        
        # Repeat upper limit veification, in case UpBoll = True in the first moment
        UpBool = Upfunc(maxT,minT)

        while UpBool:
            factor -= precision
            if factor < 0:
                print('reduce your precision value')
                return None
            maxT,minT = fluidtemp(Model,factor)
            UpBool = Upfunc(maxT,minT)
        
        # final verification to check if LowBool still valid
        LowBool = Lowfunc(maxT,minT)
        if not LowBool:
            # print('successfully terminated: factor value  is',str(round(factor,5)),
            #       'MaxT is',str(round(maxT,5)),'degC and minT is',
            #       str(round(minT,5)),'degC')
            # return factor
            return factor, round(maxT,5), round(minT,5)
        else:
            print('No solution: change operation mode, reduce precision or increase Tinterv')
            return float('NaN'), float('NaN'), float('NaN')
        
        
        
    def change_H(self,H):
        self.H=H
        self.AR=self.H/(self.rb*2)
        self.data['Load_Wm']=self.data.Load_W/self.H
        
        
    def Hfactor_estimate(self,Tmax,Tmin,Tinterv,precision,Model,mode):
        """ 
        Calculate a factor of reduction (or increment) of the given pile depth,
        given the temperature limits defined
        
        Tmax:          Maximum fluid temperature (degC)
        Tmin:          Minimum fluid temperature (degC)
        Tinterv:       Interval the limit fluid temperature should fall in (degC)
            E.G. if Tmax=40 and Tinterv=1 - Max fluit temperature will be
            between 39 and 40 degC
        precision:     Step of change on the load factor (-)
        Model:         base analytical model for calculation of the fluid temperature
        mode:          critical operation mode, either 'heating' or 'cooling'
                       or both
        """
        
        def fluidtemp(Model,H_to_try):
            self.change_H(self,H_to_try)
            # TODO: Need to add re-creation of borehole properties etc. here to be able to  fully incorporate length
            # OR can re-write function outside here, by re-creating borehole for different Hs 
            
            
            load = 1.0*self.data.Load_W.copy().to_numpy()
            q = load/self.H
            deltaT = load/(self.qrate*0.001*998*4185.5)
            Tavg = self.Fluid_Temp(Model,load=q)
            Tin = Tavg + deltaT/2
            Tout = Tavg - deltaT/2
            maxT = max(Tin.max(),Tout.max())
            minT = min(Tin.min(),Tout.min())
            return [maxT,minT]
        
        def Booldefine(Tmax,Tmin,Tinterv,mode):
            # defining upper and lower limit verifications per operation
            if mode == 'heating':
                Upfunc = lambda x,y: y<Tmin
                Lowfunc = lambda x,y: y>(Tmin+Tinterv)
            elif mode == 'cooling':
                Upfunc = lambda x,y: x>Tmax
                Lowfunc = lambda x,y: x<(Tmax-Tinterv)
            else:
                Upfunc = lambda x,y: (x>Tmax or y<Tmin)
                Lowfunc =lambda x,y: (x<(Tmax-Tinterv) or y>(Tmin+Tinterv))
            return [Upfunc,Lowfunc]
            
        # starting with factor = 1
        H_to_try = self.H
            
        # calculating fluid temperatures        
        maxT,minT = fluidtemp(Model,H_to_try)
        
        # defining boolean functions
        Upfunc,Lowfunc = Booldefine(Tmax,Tmin,Tinterv,mode)
        UpBool = Upfunc(maxT,minT)

        while UpBool:
            H_to_try -= precision
            if H_to_try <= 0:
                print('reduce your precision value')
                return None
            maxT,minT = fluidtemp(Model,H_to_try)
            UpBool = Upfunc(maxT,minT)
        
        # Lower limit Bollean
        LowBool = Lowfunc(maxT,minT)
        
        while LowBool:
            H_to_try += precision
            maxT,minT = fluidtemp(Model,H_to_try)
            LowBool = Lowfunc(maxT,minT)
        
        # Repeat upper limit veification, in case UpBoll = True in the first moment
        UpBool = Upfunc(maxT,minT)

        while UpBool:
            H_to_try -= precision
            if H_to_try <= 0:
                print('reduce your precision value')
                return None
            maxT,minT = fluidtemp(Model,H_to_try)
            UpBool = Upfunc(maxT,minT)
        
        # final verification to check if LowBool still valid
        LowBool = Lowfunc(maxT,minT)
        if not LowBool:
            print('successfully terminated: factor value  is',str(round(H_to_try,5)),
                  'MaxT is',str(round(maxT,5)),'degC and minT is',
                  str(round(minT,5)),'degC')
            # return factor
            return H_to_try, round(maxT,5), round(minT,5)
        else:
            print('No solution: change operation mode, reduce precision or increase Tinterv')
            return float('NaN'), float('NaN'), float('NaN')