#IMPORTS#
import numpy as np
import zipfile
import pandas as pd
import os

import gym
import time


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
    

    def convert_bits_to_values(self,x,bitlengths,matrix_bits,boundaries):
        """
            converts a genetic algorithm solution to values
        """

        values = []

        matrix_index = 0

        #looping through each weight matrix#
        for i in range(0,len(bitlengths),1):
            
            #weight matrix in bit form#
            bit_matrix = x[matrix_index:matrix_index + matrix_bits[i]].copy()

            #updating matrix index for next interation#
            matrix_index = matrix_index + matrix_bits[i]

            #getting each value#
            for j in range(0,len(bit_matrix),bitlengths[i]):
                data = bit_matrix[j:j+bitlengths[i]].copy()

                #binary value#
                binary = ''.join(str(e) for e in data)
                bin_value = int(binary,2)

                temp = -boundaries[i] +  2*boundaries[i] * (bin_value/(2**(bitlengths[i]) -1))
                values.append(temp)
        
        return values

    def sigmoid(self,x):
        value = 1/(1+np.exp(-x))
        return value
    
    def activation_function(self,x,func='sigmoid'):
        """
            can choose which activation function to use
            (More to be added)

        """
        if func == 'sigmoid':
            return self.sigmoid(x)

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
            a = self.activation_function(z)
            x = np.copy(a)

        if round==True:
            x = np.round(x)

        if len(x.shape) == 1 and (len(x)==1):
            return x[0]
        
        return x.flatten() 

    def GA_fitness(self,x,bitlengths,matrix_bits,boundaries,shapes):
        """
            Determines the fitness of the possible solution x
        """

        data = self.convert_bits_to_values(x,bitlengths,matrix_bits,boundaries)
        self.reconstruct(data,shapes)

        #add in simulation here#
        env_name = 'CartPole-v1'
        env = gym.make(env_name)

        state = env.reset()

        score = 0

        for count in range(TIME):
        #plotting#
            #error.append(state[2])

            action = self.feedfoward(state)
            state,reward,done,info=env.step(int(action))

            score+=reward

            if done == True:

                if count<TIME-1:
                    score-= 50
                
                state = env.reset()
                #controller.reset()
        
        env.close()
        return score
        ########################
    
    def init_chromosome(self,L):
        """
            Creates possible solution

            Parameters:
                L (int) : bitlength of solution
            
        """

        x = []

        for _ in range(L):
            r = np.random.uniform(0,1)

            if r<0.5:
                x.append(0)
            else:
                x.append(1)
        
        return x
    
    def init_population(self,N,L,bitlengths,matrix_bits,boundaries,shapes):
        """
            initialises the population
        """

        pop = []
        costs = []

        for _ in range(N):
            x = self.init_chromosome(L)
            fx = self.GA_fitness(x,bitlengths,matrix_bits,boundaries,shapes)
            pop.append(x)
            costs.append(fx)
        
        return pop,costs
    
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
    
    def crossover(self,u,v,bitlengths,matrix_bits,boundaries,shapes,mode=1):
        """
            creates children using crossover

            Parameters:
                u       (list)  : the first parent
                v       (list)  : the second parent
                x0      (list)  : current state
                mode    (int)   : single point crossover or double point crossover
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
        
        fx = self.GA_fitness(x,bitlengths,matrix_bits,boundaries,shapes)
        fy = self.GA_fitness(y,bitlengths,matrix_bits,boundaries,shapes)

        return [x,y],[fx,fy]
    

    def mutation(self,u,mu,bitlengths,matrix_bits,boundaries,shapes):
        """
            given probability of mu, the solution with mutate

            Parameters:
                u       (list)  : possible solution
                mu      (float) : probability of mutation
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
        
        fx = 0

        if change == True:
            fx = self.GA_fitness(x,bitlengths,matrix_bits,boundaries,shapes)
        
        return change,x,fx
    
    def new_pop(self,pop,costs,mu,k,bitlengths,matrix_bits,boundaries,shapes):
        """
            creates the solutions to be used in the next generation

            Parameters:
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
                
                if costs[i]> costs[j]:
                    parents.append(pop[i])
                else:
                    parents.append(pop[j])
            

            #creating children using crossover#
            child,f_child = self.crossover(parents[0],parents[1],bitlengths,matrix_bits,boundaries,shapes)

            x = x + child
            fx = fx + f_child
        
        #mutation#
        for i in range(0,len(x),1):
            change,temp,temp_f = self.mutation(x[i], mu,bitlengths,matrix_bits,boundaries,shapes)

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

    def GA(self,N,k,m,mu,L,bitlengths,matrix_bits,boundaries,shapes):
        
        #creating population#
        pop,costs = self.init_population(N,L,bitlengths,matrix_bits,boundaries,shapes)

        for _ in range(0,m,1):
            pop,costs = self.new_pop(pop, costs, mu, k,bitlengths,matrix_bits,boundaries,shapes)
        return pop[0],costs[0]


    def optimise(self,acc,N=20,k=12,m=30,mu=1e-3):
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

        bitlengths = []
        matrix_bits = []
        boundaries = []
        P_values = []

        shapes = []

        #total bitlength#
        L = 0

        #getting info#
        for w in self.W:
            
            if len(w.shape)!=1:
                u,v = w.shape
            else:
                u = 1
                v = len(w)

            shapes.append([u,v])

            boundary = np.sqrt(2/(u+v))
            P = 2*boundary * (10**acc)
            bit = int(np.ceil(np.log2(P)))
            matrix_bit = bit*u*v

            boundaries.append(boundary)
            bitlengths.append(bit)
            matrix_bits.append(matrix_bit)
            P_values.append(P)
            L+=matrix_bit
        
        binary_weights,cost = self.GA(N,k,m,mu,L,bitlengths,matrix_bits,boundaries,shapes)
        print(cost)
        weights = self.convert_bits_to_values(binary_weights, bitlengths, matrix_bits, boundaries)
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
   #model.optimise(acc=2)

    model.load_model(filename='smallModel.zip')

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state = env.reset()

    for _ in range(TIME):

        action = model.feedfoward(state)
        state,reward,done,info=env.step(int(action))

        env.render()
        time.sleep(0.08)

        if done == True:
            state = env.reset()
            #controller.reset()