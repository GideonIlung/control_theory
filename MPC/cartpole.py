import gym
import time
import numpy as np
import matplotlib.pyplot as plt

#CONSTANTS
TIME_STEP = 0.1
N_ITER = 500
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


#MPC class#
class MPC(object):


    def __init__(self,_model,_constraints,_costfunction,_dt,_k):

        self.model = _model
        self.constraints = _constraints
        self.cost = _costfunction
        self.dt = _dt
        self.k = _k
    
    def action(self,x0,opti='dynamic',m=10,n=20,mu=1e-3,k=2,penalty=True):
        """
            determines the next action to be taken to control the system

            Parameters:
                x0      (array)  : the current state
                opt     (string) : which optimisation hueristic to use
                m       (int)    : (For GA) size of population
                n       (int)    : (For GA) number of generations
                mu      (float)  : (for GA) probability of mutation
                k       (int)    : (for GA) no of elite solutions to be handed over to next gen
                penalty (bool)   : penality added to function or not, if not penality violation values set to inf

        """
        if opti == 'dynamic':
            u = []
            fitness,policy = self.dynamic(x0, u)
            force = self.sigmoid(policy[0])
            return int(np.round(force))
        elif opti == 'GA':
            fitness,policy = self.GA(x0, m, n, mu, k,penalty=penalty)
            force = self.sigmoid(policy[0])
            return int(np.round(force))

    def sigmoid(self,x):
        return 1.0 / (1.0 + np.exp(-x))  
    
    def evaluate(self,x0,u,k,penalty=False):
        """
            determines the fitness of the policy u in question

            Parameters:
                x0      (array) : the current state
                u       (array) : the policy
                penalty (bool)  : penality added to function or not, if not penality violation values set to inf
        """
        X = []
        x = x0.copy()

        count = k
        violated = False

        for i in range(0,k,1):
            temp = self.model(x,u[i],self.dt)

            if (self.constraints(temp) == False) and (penalty == False):
                return np.inf,u    
            elif (self.constraints(temp) == False) and (violated == False):
                count = i
                violated = True
        
            X.append(temp)
            x = temp.copy()
        
        value = self.cost(X,k)

        if penalty == True:
            value-= count
        
        return value,u

###########################DYNAMIC PROGRAMMING##########################################
    def dynamic(self,x0,u):
        if len(u) == self.k:
            
            return self.evaluate(x0, u,self.k)
    
        else:

            w = u.copy()
            v = u.copy()

            w.append(-10)
            v.append(10)

            f1,state1 = self.dynamic(x0, w)
            f2,state2 = self.dynamic(x0, v)

            if f1<=f2:
                return f1,state1
            else:
                return f2,state2
####################################################################################

###########################GENETIC ALGORITHM########################################
    def init_chromosome(self,x0)->list:
        """
            randomly generates a possible solution

            Output:
                (array) : possible solution
        """
        L = self.k
        x = []

        for i in range(0,L,1):
            r = np.random.uniform(0,1)

            if r < 0.5:
                x.append(-10)
            else:
                x.append(10)
        
        return x
    
    def init_population(self,x0,m:int,_penalty = True)->list: #TODO: generate population that does not violate constraints#
        """
            randomly initialises the population

            Parameters:
                x0      (array) : the current state
                m       (int)   : the size of the population
                penalty (bool)  : penality added to function or not, if not penality violation values set to inf
        """
        pop = []
        fx = []

        for _ in range(0,m,1):
            
            u = self.init_chromosome(x0)
            y,_ = self.evaluate(x0, u,self.k,penalty=_penalty)

            pop.append(u)
            fx.append(y)
        
        return pop,fx
    
    def elitism(self,pop,costs,k):
        """
            returns the top k solutions to be used in the next generation

            Parameters :
                pop   (array) : the current population
                costs (array) : costs associated with each chromosome in population
        """
        
        x = pop.copy()
        fx = costs.copy()

        self.quickSort(x,fx) #sorting to determine#

        data = []
        fit = []

        for i in range(0,k,1):
            data.append(x[i])
            fit.append(fx[i])
        
        return data,fit
    

    def crossover(self,u,v,x0,mode=1,_penalty=True):
        """
            creates children using crossover

            Parameters:
                u       (list)  : the first parent
                v       (list)  : the second parent
                x0      (list)  : current state
                mode    (int)   : single point crossover or double point crossover
                penalty (bool)  : penality added to function or not, if not penality violation values set to inf
        """

        L = len(u)
        x = u.copy()
        y = v.copy()

        #single point crossover#
        if mode == 1:
            i = np.random.randint(0,L-2)
            x[i:L] = v[i:L].copy()
            y[i:L] = u[i:L].copy()
        else: #double point crossover#
            i = np.random.randint(0,L-1)
            j = np.random.randint(0,L-1)

            a = min(i,j)
            b = max(i,j)

            x[a:b+1] = v[a:b+1].copy()
            y[a:b+1] = u[a:b+1].copy()
        
        fx,_ = self.evaluate(x0,x,self.k,_penalty)
        fy,_ = self.evaluate(x0,y,self.k,_penalty)

        return [x,y],[fx,fy]
    
    def mutation(self,x0,u,mu,_penalty=True):
        """
            given probability of mu, the solution with mutate

            Parameters:
                x0      (list)  : the current state
                u       (list)  : possible solution
                mu      (float) : probability of mutation
                penalty (bool)  : penality added to function or not, if not penality violation values set to inf
        """

        x = u.copy()

        change = False

        for i in range(0,len(x),1):
            r = np.random.uniform(0,1)

            if (r<mu) and (x[i]==10):
                x[i] = -10
                change = True
            elif (r<mu) and (x[i]==-10):
                x[i] = 10
                change = True
        
        fx,_ = self.evaluate(x0, x,self.k,penalty=_penalty)
        return change,x,fx

    def new_pop(self,x0,pop,costs,mu,k,_penalty=True):
        """
            creates the solutions to be used in the next generation

            Parameters:
                x0      (list)  : current state
                pop     (array) : the current population
                costs   (array) : costs associated with each chromosome in population
                mu      (float) : probability of mutation
                k       (int)   : no of elite solutions to be handed over to next gen
                penalty (bool)  : penality added to function or not, if not penality violation values set to inf
        """

        #getting elite solutions#
        x,fx = self.elitism(pop, costs, k)

        #repeatedly create children#
        while len(x) < len(pop):

            #selecting parents using tournament selection#
            parents = []
            for _ in range(0,2,1):
                i = 0
                j = 0 

                while (i == j) and (len(pop)>1):
                    i = np.random.randint(0,len(pop)-1)
                    j = np.random.randint(0,len(pop)-1)
                
                if costs[i]< costs[j]:
                    parents.append(pop[i])
                else:
                    parents.append(pop[j])
            

            #creating children using crossover#
            child,f_child = self.crossover(parents[0],parents[1], x0,_penalty=_penalty)

            x = x + child
            fx = fx + f_child
        
        #mutation#
        for i in range(0,len(x),1):
            change,temp,temp_f = self.mutation(x0, x[i], mu,_penalty=_penalty)

            if change == True:
                x[i] = temp.copy()
                fx[i] = temp_f
        
        #if x larger than pop size remove worst solutions#
        self.quickSort(x, fx)
        while len(x) > len(pop):
            del x[-1]
            del fx[-1]
        
        #updating population#
        return x,fx


    def GA(self,x0,m,n,mu,k,penalty=True): #TODO#
        """
            optimises using binary Genetic algorithm

            Parameters:
                x0      (array) : current state
                m       (int)   : size of population
                n       (int)   : number of generations
                mu      (float) : probability of mutation
                k       (int)   : no of elite solutions to be handed over to next gen
                penalty (bool)  : penality added to function or not, if not penality violation values set to inf
        """
        
        #creating population#
        pop,cost = self.init_population(x0, m,_penalty=penalty)

        for _ in range(0,n,1):
            pop,cost = self.new_pop(x0, pop, cost, mu, k,_penalty=penalty)
        return cost[0],pop[0]

####################################################################################

##############################QUICKSORT#############################################
    def partition(self,x,fx,left:int,right:int):

        pivot = fx[right]
        i = left - 1

        for j in range(left,right,1):
        
            if fx[j] <= pivot:
                i+=1

                temp_x = x[i].copy()
                temp_f = fx[i]

                x[i] = x[j].copy()
                fx[i] = fx[j]

                x[j] = temp_x.copy()
                fx[j] = temp_f 
        
        temp_x = x[i+1].copy()
        temp_f = fx[i+1]

        x[i+1] = x[right].copy()
        fx[i+1] = fx[right]

        x[right] = temp_x.copy()
        fx[right] = temp_f

        return i+1
    
    def q_sort(self,x,fx,left:int,right:int):

        if left < right:
            part_index = self.partition(x, fx, left, right)
            self.q_sort(x, fx, left, part_index-1)
            self.q_sort(x, fx, part_index+1, right)
    
    def quickSort(self,x,fx):
        n = len(x)
        self.q_sort(x, fx,0,n-1)
####################################################################################     

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

def analysis(K,opti=['dynamic']):
    h = TIME_STEP
    n = N_ITER

    init_state = np.array([0.01,0.01,0.01,0.01])

    for i in range(0,len(K),1):
        Simulate(n, h,K[i],SETPOINT,init_state_bool=True,init_state=init_state,render=False,opti=opti[i])

    #plotting zero line#
    line = np.zeros(n)
    plt.ylabel(r'$\theta$ displacement')
    plt.xlabel('time')
    plt.plot(line,'k--')
    plt.legend(loc='best')    
    plt.show()

def Simulate(n,h,K,setpoint,init_state_bool = False,init_state=None,render=True,opti='dynamic'):
    '''
    Simulates and attempts to solve the cart pole problem

            Parameters:
                    n        (int)   : number of iterations
                    h        (float) : the time-step. i.e the time between measurements
                    param    (list)  : list of 3 float values. parameters of the PID controller
                    setpoint (array) : the desired state
                    render   (bool)  : visualisation of problem 
    '''

    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    controller = MPC(get_state,constraints,cost_function,h,K)

    state = env.reset()
    if init_state_bool != False:
        env.state = init_state
        state = env.state
        
    error = []

    for _ in range(n):
        #plotting#
        error.append(state[2])

        action = controller.action(state,opti)
        state,reward,done,info=env.step(action)

        if render == True:
            env.render()
            time.sleep(0.08)

        if done == True:
            state = env.reset()
            #controller.reset()

    env.close()

    plt.plot(error,label = opti + ":K = {}".format(K))

def info(rep=30):
    opti = "GA"
    h = TIME_STEP
    K = 10
    controller = MPC(get_state,constraints,cost_function,h,K)

    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    init_state = np.array([0.01,0.01,0.01,0.01])
    errors = []
    
    mean = []
    std = []
    time_data = []

    for i in range(0,rep,1):
        state = env.reset()
        env.state = init_state
        state = env.state
        error = []

        for _ in range(500):
        
            #plotting#
            error.append(state[2])
            start = time.time()
            action = controller.action(state,opti)
            end = time.time()
            time_data.append(end-start)
            state,reward,done,info=env.step(int(np.round(action)))
        
        errors.append(error)
    
    env.close()

    X = np.array(errors)
    M,N = X.shape

    for i in range(0,N,1):

        value = X[:,i].mean()
        std.append(X[:,i].std())
        mean.append(value)
    
    time_data = np.array(time_data)
    response_time = time_data.mean()*1000

    #######saving results to text file##################################
    resultsfile = open('GA_results.txt','w')

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

    print("average response time: ",response_time, " milliseconds")
    print("average error ",np.abs(mean).mean())
    print('average std error' ,np.abs(std).std())

    t = np.arange(len(mean))
    plt.plot(mean,label='mean displacement')
    plt.fill_between(t,mean - std, mean + std, color='b', alpha=0.2)
    plt.ylabel(r'displacement $\theta$')
    plt.xlabel(r'time $t$')
    plt.legend(loc='best')
    #tikz_save('NEAT_plot.tikz')
    plt.show()

if __name__ == '__main__':
    h = TIME_STEP
    n = N_ITER

    #Simulate(n, h,6,SETPOINT,opti='GA')
    analysis(K=[3,5,7,9],opti=['dynamic','dynamic','dynamic','dynamic'])
    #info()
