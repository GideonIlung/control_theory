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


def results(param,rep=30):

    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    init_state = np.array([0.01,0.01,0.01,0.01])
    #creating PID controller#
    controller = PID(param[0],param[1],param[2],TIME_STEP,SETPOINT)

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
    
    time_data = np.array(time_data)
    response_time = time_data.mean()

    #######saving results to text file##################################
    resultsfile = open('PID_results_3.txt','w')

    lines = []

    #looping through mean values#
    string = str(mean[0])

    for i in range(1,len(mean),1):
        string = string + ',' + str(mean[i])
    
    string = string + '\n'
    resultsfile.writelines(string)

    #looping through std values#
    string = str(std[0])

    for i in range(1,len(std),1):
        string = string + ',' + str(std[i])
    string = string + '\n'
    resultsfile.writelines(string)
    resultsfile.writelines(str(response_time))
    resultsfile.close()
    ##################################################

    mean = np.array(mean)
    std = np.array(std)

    time_data = np.array(time_data)
    print("average response time: ",response_time, " milliseconds")
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

def results_noise(param,rep=30,std_div=0.1):

    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    init_state = np.array([0.01,0.01,0.01,0.01])
    #creating PID controller#
    controller = PID(param[0],param[1],param[2],TIME_STEP,SETPOINT)

    errors = []
    mean = []
    std = []
    time_data = []

    for _ in range(0,rep,1):
        
        controller.reset()
        state = env.reset()
        env.state = init_state
        state = env.state

        #will be used to measure performace of controller under noisy enviroments#
        noise = np.random.normal(loc=0.0,scale=std_div,size=500)
        error = []

        for i in range(0,N_ITER,1):

            error.append(state[2])
            
            start = time.time()
            new_state = state[2] + noise[i]
            action = controller.action(new_state,state[3])
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
    
    time_data = np.array(time_data)
    response_time = time_data.mean()

    #######saving results to text file##################################
    resultsfile = open('PID_noise_results_3.txt','w')

    lines = []

    #looping through mean values#
    string = str(mean[0])

    for i in range(1,len(mean),1):
        string = string + ',' + str(mean[i])
    
    string = string + '\n'
    resultsfile.writelines(string)

    #looping through std values#
    string = str(std[0])

    for i in range(1,len(std),1):
        string = string + ',' + str(std[i])
    string = string + '\n'
    resultsfile.writelines(string)
    resultsfile.writelines(str(response_time))
    resultsfile.close()
    ##################################################

    mean = np.array(mean)
    std = np.array(std)

    time_data = np.array(time_data)
    print("average response time: ",response_time, " milliseconds")
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

#################################PARAM_OPTIMISATION################################################

def single_run(x,n=500,h=0.01,setpoint=0):
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
    controller = PID(x[0],x[1],x[2],h,setpoint)
    total = 0

    for _ in range(n):
        action = controller.action(state[2],state[3])
        state,reward,done,info=env.step(action)
        total+=state[2]**2
        if done == True:
            break
    
    env.close()
    return total

    
def fitness(x,rep=3):
    """
        Determines the fitness of the parameters selected

        Parameters:
            x   (list) : candidate parameters
            rep (int)  : number of simulation runs
    """

    total = 0

    for _ in range(0,rep,1):
        total+=single_run(x)
    
    total/=rep
    return total

def init_chromosome(a,b):
    """
        Using real coded genetic algorithm, generates values between
        a_i and b_i
        Parameters:
        a (list) : lower boundaries of each axis
        b (list) : upper boundaries of each axis
        Outputs:
        x (list) : possible solution
    """
    n = len(a)
    x = []
    for i in range(0,n,1):
        r = np.random.uniform(0,1)
        #adding 1 to make it inclusive#
        value = a[i] + (b[i]-a[i])*r
        x.append(value)
    return x

def init_population(a,b,N):
    """
        Initialises the population of solutions

        Parameters:
            a     (list) : lower boundaries
            b     (list) : upper boundaries
            N     (int)  : population sizee
        
        Outputs:
            pop   (list) : population of possible solutions
            costs (list) : fitness values of each solution in population
    """

    pop = []
    costs = []

    for _ in range(0,N,1):
        x = init_chromosome(a,b)
        fx = fitness(x)

        pop.append(x)
        costs.append(fx)
    
    return pop,costs

def elitism(pop,costs,m):
    """
        returns the top N-m possible solutions
        where N is population size,m number of children created

        Parameters:
            pop   (list) : population of possible solutions
            costs (list) : fitness values of each solution in population
        
        Outputs:
            x     (list) : population of elite possible solutions
            fx    (list) : fitness values of each solution in elite population

    """
    x = pop.copy()
    fx = costs.copy()

    #sorting#
    for i in range(0,len(x)-1,1):
        for j in range(i,len(x),1):

            if fx[i]>fx[j]:
                f_temp = fx[i]
                x_temp = x[i].copy()

                fx[i] = fx[j]
                x[i] = x[j].copy()

                fx[j] = f_temp
                x[j] = x_temp.copy()
    
    #number of elite solutions to be kept#
    k = len(x) - m

    return x[0:k],fx[0:k]

def blend_crossover(p1,p2,alpha=0.5):
    """
        uses the blended crossover method to 
        generate children

        Parameters:
            p1    (list)  : first parent solution
            p1    (list)  : second parent solution
            alpha (float) : blended crossover parameter

        Outputs:
            x     (list)  : generated solutino
            fx    (float) : fitness of generated solution
    """

    x = []

    n = len(p1)

    for i in range(0,n,1):
        #getting indices#
        a = min(p1[i],p2[i])
        b = max(p1[i],p2[i])

        r = np.random.uniform(0,1)
        dist = b-a
        value = a - alpha*dist + (dist + 2*alpha*dist)*r
        x.append(value)
    
    fx = fitness(x)
    return x,fx

def tournament_selection(pop,costs):
    """
        Performs tournament selection to 
        determine the parents to be used in
        crossover

        Parameters:
            pop   (list) : population of possible solutions
            costs (list) : fitness values of each solution in population
        
        Outputs:
            p     (list) : parent to be used in crossover
    """
    
    i = 0
    j = 0

    while i ==j:
        i = np.random.randint(0,len(pop))
        j = np.random.randint(0,len(pop))
    
    if costs[i] < costs[j]:
        return pop[i]
    else:
        return pop[j]

def GA(N=30,m=25,max_iter=50,a=[-10,-10,-10],b=[10,10,10]):
    """
        performs th real coded genetic algorithm to determine
        the fitness of the function

        Parameters:
            N        (int)   : the population size
            m        (int)   : number of children created at each generation
            max_iter (int)   : maximum number of generations
            a        (list)  : lower boundaries
            b        (list)  : upper boundaries
        
        Outputs:
            x        (list)  : most min solution found
            fx       (float) : fitness of the solution x
    """

    pop,costs = init_population(a, b, N)

    for _ in range(0,max_iter,1):
        x,fx = elitism(pop, costs, m)

        #creating children using blended crossover#
        for _ in range(0,m,1):
            p1 = tournament_selection(pop, costs)
            p2 = tournament_selection(pop, costs)
            child,child_cost = blend_crossover(p1, p2)

            #appending to list#
            x.append(child)
            fx.append(child_cost)
        
        #updating population#
        pop = x.copy()
        costs = fx.copy()
    
    #getting solution#
    index = np.argmin(costs)
    return pop[index],costs[index]


if __name__ == '__main__':
    #param = [0.5,1,1]
    #param = [6.938717509576231, -1.5198859377033884, 0.38182199447522647]
    param = [10.482953744496049, -1.9495400012718576, 0.5857363525523501]
    h = TIME_STEP
    n = N_ITER
    #Simulate(n, h, param,SETPOINT)
    #results(param)
    results_noise(param)
    #PARAMETER OPTIMISATION
    # param,cost = GA()
    # print(param,cost)
