#IMPORTS#
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import pandas as pd
import os

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
        value = np.max(0,x)
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
            v = np.ones((1,len(x[0,:])))
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

 ###################################REAL_CODED_GENETIC_ALGORITHM#################################################   

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
    

    def GA(self,a,b,shapes,N,k,m,mu):
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
        
        for _ in range(0,m,1):
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
        
        index = np.argmax(costs)
        return pop[index],costs[index]

    def optimise(self,N=100,k=30,m=200,mu=1e-3):
        """
            Optimises the Neural Network using 
            Genetic Alogrithm

            Parameters:
                acc (int) : number of decimals solution should be accurate to
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
        

        weights,cost = self.GA(a, b, shapes,N,k,m,mu)
        print(cost)
        self.reconstruct(weights, shapes)
        self.save_model()

        

    ##############################QUICKSORT#############################################
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
    ####################################################################################    

if __name__ == '__main__':
    shape = [4,4,1]
    model = NN(shape)
    #model.optimise()

    model.load_model()

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