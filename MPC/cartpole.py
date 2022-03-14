import gym
import time
import numpy as np
import matplotlib.pyplot as plt

#CONSTANTS
TIME_STEP = 0.06
N_ITER = 100
SETPOINT = 0  # angle of pole must be zero
K = 3         # control horizion
#-----------------

#PARAMS
g = 9.8  #gravitational constant#
M = 1    #mass of cart
m = 0.1  #mass of pole
total = M + m 
l = 0.5  #position of centre of mass
#-----------------------




def get_state(x0,u,dt):
    """
        Predicts the next state based on the current state and control policy
            
            Parameters:
                x0  (array) : the current state
                u   (float) : the control policy for that given state
                dt  (float) : timestep

            Output:
                x   (array) : possible state at next iteration

    """



    #x0 = [x,xdot,theta,thetadot]

    C = np.cos(x0[2])
    S = np.sin(x0[2])

    A = 1/(total - (C**2)*m)
    temp = u + m * l * S * (x0[1]**2) - m*g*C*S

    x = [0,0,0,0]

    #horizontal acceleration#
    ddx = A*temp

    #angular acceleration#
    ddtheta = (g/l)*S - (C/l)*A*temp

    #updating state#X,
    x[0] = x0[0] + x0[1]*dt
    x[1] = x0[1] + ddx*dt 
    x[2] = x0[2] + x0[3] * dt
    x[3] = x0[3] + ddtheta * dt

    return x

def constraints(x):
    """
        Checks if all constraints are met, if not returns false

        Parameters:
            x  (array) : the state of the system
        
        Output:
            met (boolean) : returns true if all constraints are statisfied
    """

    match = True

    if not(-4.8 <x[0] < 4.8):
        match = False
    
    if not(-0.418 < x[2] < 0.418):
        match = False
    
    return match


def cost_function(X,k):

    ans = 0

    for i in range(0,k):
        ans+= X[i][2]**2
    
    return ans


def dynamic_MPC(x0,u,k,dt):

    if len(u) == k:

        X = []
        x = x0.copy()

        for i in range(0,k,1):
            temp = get_state(x,u[i],dt)

            if constraints(temp) == False:
                return np.inf,u    
            
            X.append(temp)
            x = temp.copy()
        
        value = cost_function(X, k)
        return value,u
    
    else:

        w = u.copy()
        v = u.copy()

        w.append(-10)
        v.append(10)

        f1,state1 = dynamic_MPC(x0, w, k,dt)
        f2,state2 = dynamic_MPC(x0, v, k,dt)

        if f1<=f2:
            return f1,state1
        else:
            return f2,state2


def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))


def MPC(x0,k,dt):
    u = []
    cost,policy = dynamic_MPC(x0,u,k,dt)
    action = sigmoid(policy[0])
    return int(np.round(action))



def Simulate(n,h,K,setpoint):
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

    error = []

    for _ in range(n):
        #plotting#
        error.append(np.abs(state[2]))

        action = MPC(state,K,h)
        state,reward,done,info=env.step(action)
        env.render()
        time.sleep(h)

        if done == True:
            state = env.reset()
            controller.reset()

    env.close()

    plt.plot(error)
    plt.show()

def main():
    h = TIME_STEP
    n = N_ITER
    Simulate(n, h,K,SETPOINT)

main()