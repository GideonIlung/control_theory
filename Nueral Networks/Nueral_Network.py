#IMPORTS#
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import pandas as pd
import os
import sys
import gym
import time

#number of iterations each chromosome trains#
TRAIN_TIME = 500

#number of iterations for simulation#
TIME = 500

class NN:
    """
        Nueral Network that is optimised using hueristics
    """

    def __init__(self,shape,create=True):
        """
            Initialises the nueral network
        """
        self.W  = []
        self.shape = shape

        #initialises matrices if create is set to true#
        if create == True: 
            for i in range(1,len(shape),1):
                temp = self.create_weight(shape[i-1],shape[i])
                self.W.append(temp)
    
    def create_weight(self,u,v):
        """
            creates weight matrices including the bias
        """
        if v !=1:
            W = np.random.randn(v,u+1) * np.sqrt(2/(u+v))
        else:
            W = np.random.randn(u+1) * np.sqrt(2/(u+v))
    
        return W
    
    def flatten(self):
        """
            flattens nueral network
        """

        output = []
        shapes = []
        for x in self.W:
            
            if len(x.shape)!=1:
                u,v = x.shape
            else:
                u = 1
                v = len(x)
            
            shapes.append([u,v])

            temp = x.flatten()
            temp = temp.tolist()
            output = output + temp

        return output,shapes
    
    def reconstruct(self,data,shapes):
        """
            Reconstructs Neural network based on
            list of values provided and shapes of 
            weight matrices
        """
        
        #getting rid of all other weights#
        self.W = []
        self.shape = shapes
        
        index = 0

        for u in shapes:
            length = u[0]*u[1]
            x = data[index:index+length].copy()

            W = np.array(x)

            if (u[0]!=1):
                W = W.reshape(u[0],u[1])

            self.W.append(W)
            #updating index#
            index = length

    def save_model(self,filename='output'):
        
        name = filename + '.zip'
        output,shapes = self.flatten()

        #adding shapes to shapes file#
        shapesfile = open('shape.txt','w')

        lines = []

        for u in shapes:
            string = str(u[0])+ ' ' + str(u[1]) + ' \n'
            lines.append(string)

        shapesfile.writelines(lines)
        shapesfile.close()

        #saving weight values#
        #data = np.array(output)
        d = {'values':output}
        data = pd.DataFrame(d)
        data.to_csv('weight.csv')
        #np.savetxt('weight.csv',data,delimiter=",")
        
        #saving model to zip#
        outzip = zipfile.ZipFile(name,'w')
        outzip.write('shape.txt')
        outzip.write('weight.csv')
        os.remove('shape.txt')
        os.remove('weight.csv')
        outzip.close()
    
    def load_model(self,filename='output.zip'):

        with zipfile.ZipFile(filename,'r') as parent_file:
            #getting shape info#
            shapefile = parent_file.open('shape.txt')
            Lines = shapefile.readlines()

            shapes = []

            for line in Lines:
                sent = line.strip()
                data = sent.decode("utf-8")
                x = [int(e) for e in data.split(' ')]
                shapes.append(x)

            #getting weight values#
            W = pd.read_csv( parent_file.open('weight.csv'))
            W = W.to_numpy()[:,1]
            data = W.tolist()

            self.reconstruct(data,shapes)
    
######################################ACTIVATION_FUNCTIONS#################################################
    def sigmoid(self,x):
        
        value = 1/(1+np.exp(-x))
        return value
    
    def relu(self,x):
        value = np.maximum(x, 0)
        return value
    
    def activation_function(self,x,func='sigmoid'):
        """
            can choose which activation function to use
            (More to be added)

        """
        if func == 'sigmoid':
            return self.sigmoid(x)
        
        if func== 'relu':
            return self.relu(x)
        
#############################################################################################################
    def feedfoward(self,x0,round=True):

        #reshaping vector#
        x = np.copy(x0)
        x = np.array(x)
        x = x.reshape(len(x),1)

        for i in range(len(self.W)):

            #stacking#
            if len(x.shape)>1:
                v = np.ones((1,len(x[0,:])))
            else:
                v = np.ones(1)
            
            x = np.vstack((v,x))

            z = self.W[i] @ x

            #TODO: relu giving issues
            # if i != len(self.W)-1:
            #     a = self.activation_function(z,func='relu')
            # else:
            #     a = self.activation_function(z,func='sigmoid')

            a = self.activation_function(z,func='sigmoid')
            x = np.copy(a)

        if round==True:
            x = np.round(x)

        if len(x.shape) == 1 and (len(x)==1):
            return x[0]
        
        return x.flatten() 

   

    def sim_fitness(self):
        """
            one run of the simulation
        """

        #add in simulation here#
        env_name = 'CartPole-v1'
        env = gym.make(env_name)

        state = env.reset()

        score = 0

        for count in range(TRAIN_TIME):
            action = self.feedfoward(state)
            state,reward,done,info=env.step(int(action))
            #score+=reward - state[2]**2
            score+=reward

            if done == True:
                break
        
        env.close()
        return score

    def fitness(self,x,shapes,rep=3):
        """
            Determines fitness of the weights

            Parameters:
                x      (list)  : possible solution
                shapes (list)  : shape of weight matrices
            
            Outputs:
                score  (float) : fitness of chromosome x
        """

        self.reconstruct(x, shapes)
        score = 0

        for _ in range(0,rep,1):
            value = self.sim_fitness()
            score+=value
        
        score = score/rep
        return score
    
###################################REAL_CODED_GENETIC_ALGORITHM################################################# 
    def init_chromosome(self,a,b):
        """
            Initialises a possible solution randomly

            Parameters:
                a (list) : set of lower boundaries
                b (list) : set of upper boundaries
            
            Output:
                x (list) : possible solution
        """

        x = []
        n = len(a)

        for i in range(0,n,1):
            r = np.random.uniform(0,1)
            value = a[i] + (b[i]-a[i])*r
            x.append(value)
        
        return x
    
    def init_population(self,a,b,N,shapes):
        """
            Initalises the population of solutions

            Parameters:
                a      (list) : set of lower boundaries
                b      (list) : set of upper boundaries
                N      (int)  : size of population
                shapes (list) : shape of weight matrices
            
            Outputs
                pop    (list) : set of possible solutions
                costs  (list) : list of fitness values corresponding to each solution in pop
        """

        pop = []
        costs = []

        for _ in range(0,N,1):
            x = self.init_chromosome(a, b)
            fx = self.fitness(x, shapes)

            pop.append(x)
            costs.append(fx)
        
        return pop,costs
    
    def elitism(self,pop,costs,k):
        """
            returns the top k possible solutions

            Parameters:
                pop    (list) : set of possible solutions
                costs  (list) : list of fitness values corresponding to each solution in pop
                k      (int)  : number of top solutions to be returned

            Outputs:
                x      (list)  : top k chormosomes
                fx     (float) : fitness of values in x
        """

        x = pop.copy()
        fx = costs.copy()

        self.quickSort(x, fx)
        return x[0:k],fx[0:k]
    
    def blend_crossover(self,p1,p2,shapes,alpha=0.5):
        """
            Performs blended crossover to produce possible
            solution for next generation

            Parameters:
                p1     (list)  : first parent
                p2     (list)  : second parent
                shapes (list)  : shape of weight matrices
                alpha  (float) : blended crossover parameter
            
            Outputs:
                x     (list)  : new possible solution
                fx    (float) : fitness of solution
        """
        n = len(p1)

        x = []

        for i in range(0,n,1):
            a = min(p1[i],p2[i])
            b = max(p1[i],p2[i])

            r = np.random.uniform(0,1)
            dist = b-a
            value = a - alpha*dist + (dist + 2*alpha*dist)*r
            x.append(value)
        
        fx = self.fitness(x,shapes)
        return x,fx

    def tournament_selection(self,pop,costs):
        """
            Determines chromosomes to be used for crossover to produce
            better solutions using tournament selection

            Parameters:
                pop    (list) : set of possible solutions
                costs  (list) : list of fitness values corresponding to each solution in pop
            
            Outputs:
                p      (list) : parent to be used in crossover
        """

        i = 0
        j = 0

        while i==j:
            i = np.random.randint(0,len(pop))
            j = np.random.randint(0,len(pop))

        if costs[i] > costs[i]:
            return pop[i]
        else:
            return pop[j]
    

    def GA(self,a,b,shapes,N,k,m,mu,get_iter=False,learn_curve=False):
        """
            Optimises the Neural Network using 
            Genetic Alogrithm

            Parameters:
                a      (list)  : set of lower boundaries
                b      (list)  : set of upper boundaries
                shapes (list)  : shape of weight matrices
                N      (int)   : population size
                k      (int)   : number of elite solutions
                m      (int)   : number of generations
                mu     (int)   : probability of mutation
            
            Outputs:
                x      (list)  : best weights found
                fx     (float) : fitness of solution x
        """

        #number of new solutions to be created#
        n_children = N-k

        pop,costs = self.init_population(a, b, N, shapes)
        best = [np.max(costs)]
        
        for count in range(0,m,1):
            x,fx = self.elitism(pop, costs, k)

            #creating children#
            for _ in range(0,n_children,1):
                p1 = self.tournament_selection(pop, costs)
                p2 = self.tournament_selection(pop, costs)
                child,child_cost = self.blend_crossover(p1, p2, shapes)

                #appending to list#
                x.append(child)
                fx.append(child_cost)
            
            #TODO: Add Mutation#

            #updating generation#
            pop = x.copy()
            costs = fx.copy()
            best.append(np.max(costs))

            if get_iter==True and np.max(costs) == 500:
                return None, count+1


        if get_iter == True:
            return None,m
        
        if learn_curve == True:
            return None,best
        
        index = np.argmax(costs)
        return pop[index],costs[index]

    
################################ Particle Swarm Optimisation ###########################
    def init_agent(self,a,b):
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
    
    
    def PSO_init_population(self,a,b,N,shape):
        """
            Initialises the population of solutions

            Parameters:
                a            (list) : lower boundaries
                b            (list) : upper boundaries
                N            (int)  : population size
            
            Outputs:
                pop         (list)  : population of possible solutions
                local_best  (list)  : list of personal best solutions
                local_cost  (list)  : list of fitness values corresponding to local best
                global_best (list)  : current global best solution
                global_cost (float) : fitness of global best solution
                costs       (list)  : fitness values of each solution in population
        """

        #current location of agents#
        pop = []
        costs = []

        #personal best locations#
        local_best = []
        local_cost = []

        #global best position
        global_best = None
        global_cost = None

        for _ in range(0,N,1):
            x = self.init_agent(a,b)
            fx = self.fitness(x,shape)

            #adding to population#
            pop.append(x)
            costs.append(fx)

            #adding to local best#
            local_best.append(x)
            local_cost.append(fx)

            if global_best == None:
                global_best = x.copy()
                global_cost = fx
            elif fx > global_cost:
                global_best = x.copy()
                global_cost = fx
        
        return pop,costs,local_best,local_cost,global_best,global_cost

    def direction_vector(self,p_best,g_best,pos,c1,c2):
        """
            Computes the direction vector using line search

            Parameters:
                c1      (float) : parameter on first vector
                c2      (float) : parameter on second vector
                p_best  (list)  : personal best location
                g_best  (list)  : global best location
                pos     (list)  : current position
            
            Output
                d       (list)  : direction vector
        """

        #copying values#
        p = np.array(p_best.copy())
        g = np.array(g_best.copy())
        x = np.array(pos.copy())

        #random values#
        r1 = np.random.uniform(0,1)
        r2 = np.random.uniform(0,1)

        u = c1 * r1 * (p-x) 
        v = c2 * r2 * (g-x)

        d = u + v
        return d

    def update_pos(self,p_best,g_best,pos,shape,c1,c2,a,b):
        """
            determines next position using direction vector

            Parameters:
                c1      (float) : parameter on first vector
                c2      (float) : parameter on second vector
                p_best  (list)  : personal best location
                g_best  (list)  : global best location
                pos     (list)  : current position
                a       (list)  : lower boundaries
                b       (list)  : upper boundaries
            
            Output
                x       (list)  : updated position
        """
        x = np.copy(pos.copy())

        #update this for conjugate gradient#
        d = self.direction_vector(p_best,g_best,pos,c1=c1,c2=c2)

        xnew = x + d
        xnew = xnew.tolist()

        #making sure doesnt leave boundaries#
        for i in range(0,len(xnew),1):

            if (xnew[i] < a[i]) or (xnew[i]>b[i]):
                r = np.random.uniform(0,1)
                xnew[i] = a[i] + (b[i]-a[i])*r


        fx = self.fitness(xnew,shape)
        return xnew,fx

    def PSO(self,a,b,M,N,shape,c1=2,c2=2,get_iter=False,learn_curve = False):
        """
            Optimises function using Particle Swarm Optimisation

            Parameters:
                a      (list) : lower boundaries
                b      (list) : upper boundaries
                M      (int)  : population size
                N      (int)  : number of iterations
                shapes (list) : shapes of weight matrices
        """

        #initalising population#
        iteration = None
        pop,costs,local_best,local_cost,global_best,global_cost = self.PSO_init_population(a, b, M,shape)
        best = [global_best]

        for count in range(0,N,1):
            
            #updating agents#
            for i in range(0,M,1):

                #new position#
                x,fx = self.update_pos(local_best[i],global_best,pop[i],shape,c1, c2,a,b)

                pop[i] = x.copy()
                costs[i] = fx

                #updating personal best#
                if fx > local_cost[i]:
                    local_best[i] = x.copy()
                    local_cost[i] = fx
                
                #updating global best#
                if fx > global_cost:
                    global_best = x.copy()
                    global_cost = fx
            
            best.append(global_best)

            if (get_iter == True) and (global_cost == 500):
                return None,count+1
        
        if get_iter == True:
            return None,N

        if learn_curve == True:
            return None,best
        
        return global_best,global_cost
        

#################################QUICKSORT#############################################
    def partition(self,x,fx,left:int,right:int):

        pivot = fx[right]
        i = left - 1

        for j in range(left,right,1):
        
            if fx[j] >= pivot:
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
######################################################################################## 
#####################################Optimiser ########################################## 
    def optimise(self,N=100,k=30,m=200,mu=1e-3,opti="GA",get_iter=False,learn_curve=False):
        """
            Optimises the Neural Network using 
            Genetic Alogrithm

            Parameters:
                N   (int) : population size
                k   (int) : number of elite solutions
                m   (int) : number of generations
                mu  (int) : probability of mutation
        """
        
        a = [] #lower boundaries#
        b = [] #upper boundaries#
        shapes = []

        #getting info#
        for w in self.W:
            
            if len(w.shape)!=1:
                u,v = w.shape
            else:
                u = 1
                v = len(w)

            shapes.append([u,v])

            boundary = np.sqrt(2/(u+v))
            
            for _ in range(0,u*v,1):
                a.append(-boundary)
                b.append(boundary)
        

        if opti == "GA":
            weights,cost = self.GA(a, b, shapes,N,k,m,mu,get_iter=get_iter,learn_curve=learn_curve)
        elif opti == "PSO":
            weights,cost = self.PSO(a, b, N,m, shapes,get_iter=get_iter,learn_curve=learn_curve)
        
        #for analysis#
        if get_iter == True or learn_curve == True:
            return cost
        
        print(cost)
        self.reconstruct(weights, shapes)
        self.save_model()

######################################################################################################
def run(shape):
    model = NN(shape)
    #model.optimise(opti="PSO")

    model.load_model(filename='GA_model.zip')

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state = env.reset()

    error = []
    render = True

    for _ in range(TIME):
        
        #plotting#
        error.append(state[2])

        action = model.feedfoward(state)
        state,reward,done,info=env.step(int(action))

        if render == True:
            env.render()
            time.sleep(0.08)

        if done == True:
            state = env.reset()
    
    env.close()

    if render == False:
        plt.plot(error,label='error')
        plt.show()
    
def analysis(shape,name,rep = 30):
    model = NN(shape)
    model.load_model(filename=name)

    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    init_state = np.array([0.01,0.01,0.01,0.01])
    errors = []
    std = []
    mean = []

    time_data = []

    for i in range(0,rep,1):
        state = env.reset()
        env.state = init_state
        state = env.state
        error = []

        for _ in range(TIME):
        
            #plotting#
            error.append(state[2])

            start = time.time()
            action = model.feedfoward(state)
            end = time.time()
            time_data.append(end-start)
            state,reward,done,info=env.step(int(action))

            if done == True:
                break
        
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

    t = np.arange(len(mean))
    plt.plot(mean,label='mean displacement')
    plt.fill_between(t,mean - std, mean + std, color='b', alpha=0.2)
    plt.ylabel(r'displacement $\theta$')
    plt.xlabel(r'time $t$')
    plt.legend(loc='best')
    plt.show()


#############################LEARNING RATE COMPARISION##########################################
def learn_rate(shape,rep=30):
    output = []
    model = NN(shape)

    for _ in range(0,rep,1):
        temp = model.optimise(opti="GA",get_iter=True)
        print(temp)
        output.append(temp)

    d = {'values':output}
    data = pd.DataFrame(d)
    data.to_csv("GA_learn_rate.csv")
    print("mean: ",data['values'].mean())

def learn_curve(shape,rep=30):
    """
        Displays the learning curve of the algorithm

        Parameters:
            shape (list) : dimensions of network
            rep   (int)  : number of sampling cycles
    """
    model = NN(shape)

    data = []
    mean = []
    std = []

    for _ in range(0,rep,1):
        temp = model.optimise(opti="GA",learn_curve=True)
        data.append(temp)
    
    
    X = np.array(data)

    for i in range(0,N,1):

        value = X[:,i].mean()
        std.append(X[:,i].std())
        mean.append(value)

    #######saving results to text file##################################
    resultsfile = open('GA_learn_curve.txt','w')

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

    t = np.arange(len(mean))
    plt.plot(mean,label='learning curve')
    plt.fill_between(t,mean - std, mean + std, color='b', alpha=0.2)
    plt.ylabel('duration without constraint violation')
    plt.xlabel('iteration')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    shape = [4,4,1]
    # model = NN(shape)
    # #model.optimise(opti="PSO")
    # analysis(shape,name="GA_model.zip")
    # #run(shape)
    #learn_rate(shape)
    learn_curve(shape)
    
