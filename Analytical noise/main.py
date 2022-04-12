import numpy as np 
import matplotlib.pyplot as plt 


#params#
dt = 1
t0 = 0
t1 = 10

x0 = 10
dx0 = 0
#------------------

#PID PARAMS
KP = 0.000001
KI = 0.0001
KD = 0.00001
#-----------------

class PID(object):
    """
        PID controller class.

        ...

        Attributes
        ----------
        kp : float
            the proptional error parameter

        ki :
            the integral error parameter
        
        kd :
            the derivative error parameter
        
        dt:
            the timestep
        
        setpoint : array
            the desired goal state to be reached
        
        error : array
            the error between the current state and the desired state
        
        last_error : array
            the error between the previous state and the desired state

        integral_error : array
            the approximation of the itegral error of the system at the current state
        
        derivative_error : array
            the approximation of the derivative error of the system at the current state

        Methods
        -------
        action(state:array):
            returns possible action to take given the current state
        
        sigmoid(x:float)
            returns a value between 0 and 1

    """
    
    def __init__(self,_kp,_ki,_kd,_dt,_setpoint):

        self.kp = _kp
        self.ki = _ki
        self.kd = _kd
        self.dt = _dt
        self.setpoint = _setpoint

        #initialsing errors#
        self.error = 0
        self.last_error = 0
        self.integral_error = 0
        self.derivative_error = 0
    
    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def action(self,_error):

        #updating error values#
        self.error = _error
        self.integral_error+= self.error * self.dt
        self.derivative_error = (self.error - self.last_error)/self.dt

        #updating previous error#
        self.last_error = self.error

        u = (self.kp * self.error) + (self.ki*self.integral_error) + (self.kd * self.derivative_error)
        
        return u
    
    def reset(self):
        self.integral_error = 0


def analytical(t):
    l1 = np.sqrt(2)
    l2 = -np.sqrt(2)
    u = (1+l1)*((dx0/(1+l1)) - (dx0-x0*l1)/((1+l1)*(l2-l1)))*np.exp(l1*t) + ((dx0-x0*l1)/(l2-l1))*np.exp(l2*t)
    return u 

def cost(x,u):
    return x**2 + u**2

def get_state(x0,u):
    dx = u - x0
    x = x0 + dx
    return x


def PID_sim():
    controller = PID(KP,KI,KD,dt,0)
    
    #intialising#
    x = x0
    t = 0
    error = 0
    

    #recording#
    error_list = []

    while t<t1:
        u = controller.action(error)
        error = cost(x,u)
        error_list.append(error)
        x = get_state(x, u)

        #updating time#
        t+=dt
    
    return error_list

def analytical_sim():
    #intialising#
    x = x0
    t = 0
    error = 0
    

    #recording#
    error_list = []

    while t<t1:
        u = analytical(t)
        error = cost(x,u)
        error_list.append(error)
        x = get_state(x, u)

        #updating time#
        t+=dt
    
    return error_list

if __name__ == '__main__':

    #PID_error = PID_sim()
    analytical_error = analytical_sim()
    #plt.plot(PID_error,label='PID')
    plt.plot(analytical_error,label='Analytical')
    #plt.ylim(0,1000)
    plt.legend(loc='best')
    plt.show()
        

