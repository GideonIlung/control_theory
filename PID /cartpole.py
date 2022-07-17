#IMPORTS
import gym
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from tikzplotlib import save as tikz_save
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

#CONSTANTS
TIME_STEP = 0.01
N_ITER = 500
SETPOINT = 0  # angle of pole must be zero
#-----------------

#PID PARAMS
KP = 0.5
KI = 1
KD = 1
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
    
    def action(self,angle,velocity):

        #updating error values#
        self.error = angle
        self.integral_error+= self.error * self.dt
        self.derivative_error = velocity

        #updating previous error#
        self.last_error = self.error

        u = (self.kp * self.error) + (self.ki*self.integral_error) + (self.kd * self.derivative_error)
        
        action = self.sigmoid(u) #returns value between 0  and 1#
        return int(np.round(action))
    
    def reset(self):
        self.integral_error = 0
    

def Simulate(n,h,param,setpoint):
    '''
    Simulates and attempts to solve the cart pole problem

            Parameters:
                    n        (int)   : number of iterations
                    h        (float) : the time-step. i.e the time between measurements
                    param    (list)  : list of 3 float values. parameters of the PID controller
                    setpoint (array) : the desired state
    '''

    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    state = env.reset()

    #creating PID controller#
    controller = PID(param[0],param[1],param[2],h,setpoint)

    for _ in range(n):
        action = controller.action(state[2],state[3])
        state,reward,done,info=env.step(action)
        env.render()
        time.sleep(h)

        if done == True:
            state = env.reset()
            controller.reset()

    env.close()


def results(rep=30):

    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    init_state = np.array([0.01,0.01,0.01,0.01])
    #creating PID controller#
    controller = PID(KP,KI,KD,TIME_STEP,SETPOINT)

    errors = []
    mean = []
    std = []
    time_data = []

    for _ in range(0,rep,1):
        
        controller.reset()
        state = env.reset()
        env.state = init_state
        state = env.state

        error = []

        for i in range(0,N_ITER,1):

            error.append(state[2])
            
            start = time.time()
            action = controller.action(state[2],state[3])
            end = time.time()
            time_length = end-start
            time_data.append(time_length)
            state,reward,done,info=env.step(action)


            if done == True:
                state = env.reset()
                controller.reset()

        errors.append(error)
    
    X = np.array(errors)
    M,N = X.shape

    for i in range(0,N,1):

        value = X[:,i].mean()
        std.append(X[:,i].std())
        mean.append(value)
    

    mean = np.array(mean)
    std = np.array(std)

    time_data = np.array(time_data)
    print("average response time: ",time_data.mean()*1000, " milliseconds")
    print("average error: ",np.abs(mean).mean())
    print("std:",np.abs(mean).std())

    t = np.arange(len(mean))
    plt.plot(mean,label='mean displacement')
    plt.fill_between(t,mean - std, mean + std, color='b', alpha=0.2)
    plt.ylabel(r'displacement $\theta$')
    plt.xlabel(r'time $t$')
    plt.legend(loc='best')
    #tikz_save('PID_plot.tikz',figureheight = '\\figureheight',figurewidth='\\figurewidth')
    #plt.savefig('PID_plot.pgf')
    plt.show()

def main():
    param = [KP,KI,KD]
    h = TIME_STEP
    n = N_ITER
    #Simulate(n, h, param,SETPOINT)
    results()

main()
