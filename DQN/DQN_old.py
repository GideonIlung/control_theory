#IMPORTS#
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import pandas as pd
import os
import sys
import gym
import time
import random
from copy import deepcopy


################################################ NUERAL NETWORK CLASS ########################################################################################
class Layer:
    def __init__(self):
        """
            Creating the layer Class
        """
        self.input = None
        self.output = None
    
    def forward(self,X):
        """
            performs the feed forward algorithm

            Inputs:
                X (array) : the data to be propagated 
        """
        pass

    def backward(self,dE_dy,alpha):
        """
            updates the weights of the layer using backpropagation

            Inputs:
                dE_dy (array) : the derivative error of the output
                alpha (float) : the learning rate
            
            Outputs:
                dE_dx (array) : derivative error of the input
        """
        pass 

class Dense(Layer):
    def __init__(self,n,m,_type="dense"):
        """
            Creates a layer between fully connected nuerons

            Inputs:
                n (int) : number of inputs
                m (int) : number of outputs
        """
        
        self.W = np.random.randn(m,n)
        self.b = np.random.randn(m,1)

        #for saving models#
        self.n = n
        self.m = m
        self.type = _type
    
    def forward(self,X):
        """
            performs the feed forward algorithm

            Inputs:
                X (array) : the data to be propagated 
        """
        self.input = np.copy(X)
        self.output = np.dot(self.W,self.input) + self.b
        return self.output
    
    def backward(self,dE_dy,alpha,threshold=1):
        """
            updates the weights of the layer using backpropagation

            Inputs:
                dE_dy     (array) : the derivative error of the output
                alpha     (float) : the learning rate
                threshold (float) : the maximum allowed magnitude of the gradient vector
            
            Outputs:
                dE_dx (array) : derivative error of the input
        """
        
        #apply steepest descent on weight matrix#

        #checking if gradient magnitude is too large#
        if (np.linalg.norm(dE_dy) > threshold):
            dE_dy = dE_dy/np.linalg.norm(dE_dy)

        dE_dW = np.dot(dE_dy,self.input.T)

        if (np.linalg.norm(dE_dW) > threshold):
            dE_dW = dE_dy/np.linalg.norm(dE_dW)

        self.W = self.W - alpha*dE_dW

        #applying steepest decsent on the bias#
        dE_db = dE_dy
        self.b = self.b - alpha*dE_db

        #the derivative error of the input#
        dE_dX = np.dot(self.W.T,dE_dy)
        return dE_dX

class Activation(Layer):
    def __init__(self,f,f_x):
        """
            Creates an activation layer for the Network

            Inputs:
                f   (function) : the activation function
                f_x (function) : the derivative of activation function
        """

        self.f = f
        self.f_x = f_x
    
    def forward(self,X):
        """
            performs the feed forward algorithm

            Inputs:
                X (array) : the data to be propagated 
        """

        self.input = X
        self.output = self.f(self.input)
        return self.output
    
    def backward(self,dE_dy,alpha):
        """
            updates the weights of the layer using backpropagation

            Inputs:
                dE_dy (array) : the derivative error of the output
                alpha (float) : the learning rate
            
            Outputs:
                dE_dx (array) : derivative error of the input
        """
    
        return dE_dy * self.f_x(self.input)

######## Activation Functions ######################
class Tanh(Activation):
    def __init__(self):
        f = lambda x:np.tanh(x)
        f_x = lambda x : 1 - np.tanh(x)**2
        self.type = "tanh"
        super().__init__(f,f_x)

class Sigmoid(Activation):
    def __init__(self):
        f = lambda x: 1/(1+np.exp(-x))
        f_x = lambda x: (1/(1+np.exp(-x)))*(1 - (1/(1+np.exp(-x))))
        self.type = "sigmoid"
        super().__init__(f,f_x)

class RELU(Activation):
    def __init__(self):
        f = lambda x: np.maximum(x,0)
        f_x = lambda x: 1*(x>0)
        self.type="relu"
        super().__init__(f,f_x)

class SOFTMAX(Layer):
    def __init__(self):
        self.type = "softmax"
    def forward(self, X):
        self.input = X
        temp = np.exp(X)
        self.output = temp/np.sum(temp)
        return self.output
    def backward(self, dE_dy, alpha):
        n = np.size(self.output)
        temp = np.tile(self.output,n)
        return np.dot(temp*(np.eye(n)-np.transpose(temp)),dE_dy)


class NN:
    def __init__(self,n_input=0,n_output=0,layers=[]):

        """
            Creating Nueral network by combining the layers
        """
        self.layers = layers
        self.input = n_input
        self.output = n_output

    def predict(self,x0):

        x = np.reshape(x0,(len(x0),1))
        
        #forward propagation#
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        
        if self.output !=1:
            return output.flatten()
        else:
            return output.flatten()[0]

    def train(self,X,Y,epochs=1000,alpha=0.1):
        """
            trains the ANN
        """

        X = np.reshape(X,(X.shape[0],X.shape[1],1))
        Y = np.reshape(Y,(Y.shape[0],Y.shape[1],1))

        for e in range(epochs):

            for x,y in zip(X,Y):

                #forward propagation#
                output = x
                for layer in self.layers:
                    output = layer.forward(output)
                
                dE_dy = y
                for layer in reversed(self.layers):
                    dE_dy = layer.backward(dE_dy,alpha)
    
    
    def save_model(self,filename='output'):
        """
            Saves the model of network in zip file
        """

        weights = []
        bias = []
        types = []

        shapes = []

        for layer in self.layers:
            
            if layer.type == "dense":
                weights = weights + layer.W.flatten().tolist()
                bias = bias + layer.b.flatten().tolist()
                shapes.append([layer.n,layer.m])
            
            types.append(layer.type + "\n")
        
        name = filename + '.zip'

        #adding types to file#
        typesfile = open('types.txt','w')
        typesfile.writelines(types)
        typesfile.close()


        #adding shapes to shapes file#
        shapesfile = open('shape.txt','w')

        lines = []

        for u in shapes:
            string = str(u[0])+ ' ' + str(u[1]) + ' \n'
            lines.append(string)

        shapesfile.writelines(lines)
        shapesfile.close()

        #saving weight values#
        d = {'values':weights}
        data = pd.DataFrame(d)
        data.to_csv('weights.csv')

        #saving bias values#
        d = {'values':bias}
        data = pd.DataFrame(d)
        data.to_csv('bias.csv')

        #saving model to zip#
        outzip = zipfile.ZipFile(name,'w')
        outzip.write('shape.txt')
        outzip.write('types.txt')
        outzip.write('weights.csv')
        outzip.write('bias.csv')
        os.remove('shape.txt')
        os.remove('weights.csv')
        os.remove('types.txt')
        os.remove('bias.csv')
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
            
            #getting type info#
            types = []
            typefile = parent_file.open('types.txt')
            Lines = typefile.readlines()

            for line in Lines:
                sent = line.strip()
                data = sent.decode("utf-8")
                types.append(data)
            

            #getting weight values#
            W = pd.read_csv( parent_file.open('weights.csv'))
            W = W.to_numpy()[:,1]
            weights = W.tolist()

            #getting bias values#
            b = pd.read_csv( parent_file.open('bias.csv'))
            b = b.to_numpy()[:,1]
            bias = b.tolist()

            #reconstructing the network#
            weight_index = 0
            bias_index = 0
            shapes_index = 0

            layers = []

            for i in range(0,len(types),1):

                if types[i] == 'dense':

                    u = shapes[shapes_index]
                    w_length = u[0]*u[1]
                    b_length = u[1]

                    W = weights[weight_index:weight_index+w_length].copy()
                    b = bias[bias_index:bias_index+b_length].copy()

                    W = np.reshape(W,(u[1],u[0]))
                    b = np.reshape(b,(u[1],1))

                    layer = Dense(u[0],u[1])
                    layer.W = np.copy(W)
                    layer.b = np.copy(b)

                    layers.append(layer)

                    #updating information#
                    shapes_index+=1
                    weight_index+= w_length
                    bias_index+= b_length
                elif types[i] == 'tanh':
                    layers.append(Tanh())
                elif types[i] == 'sigmoid':
                    layers.append(Sigmoid())
                elif types[i] == 'relu':
                    layers.append(RELU())
            
            #updating layers#
            self.input = shapes[0][0]
            self.output = shapes[-1][1]
            self.layers = layers
    
    def model_copy(self,model):

        self.layers = []

        self.input = model.input
        self.output = model.output

        for layer in model.layers:

            if layer.type == 'dense':
                temp = Dense(layer.n,layer.m)
                temp.W = np.copy(layer.W)
                temp.b = np.copy(layer.b)
                self.layers.append(temp)
            elif layer.type == 'tanh':
                self.layers.append(Tanh())
            elif layer.type == 'sigmoid':
                self.layers.append(Sigmoid())
            elif layer.type == 'relu':
                self.layers.append(RELU())
            elif layer.type == 'softmax':
                self.layers.append(SOFTMAX())

################################################## REPLAY BUFFER ########################################################################################
class Memory():
    def __init__(self,capacity,batch_size):
        """
            Class which stores the SARS information the agent has collected
            through its simulation

            Parameters:
                capacity    (int) : the size of the memory bank
                batch_size  (int) : the amount of states to be sampled for training
        """

        #where all previous states will be stored#
        self.memory = []

        #the size of the memory bank#
        self.capacity = capacity

        #batch sample size#
        self.batch_size = batch_size
    
    def push(self,data):
        """
            adds new sampled transition data to the memory bank

            Parameters:
                data  (array) : the data to be added
        """

        if len(self.memory)<self.capacity:
            self.memory.append(data)
        else:
            self.memory.pop(0)
            self.memory.append(data)
    
    def sample(self):
        """
            returns randomly sampled transition data
        """
        
        data = random.sample(self.memory,self.batch_size)
        return data
    
    def full(self):
        """
            returns boolean wether memory at capacity or not
        """

        if len(self.memory) >= self.batch_size:
            return True
        else:
            return False

    
############################################################################################################################################################

def optimise_model(model,target_model,memory,gamma,alpha):
    """
        optimises the model nueral network

        Parameters:
            model        (ANN)    : the model to be optimised
            target_model (ANN)    : the target network
            memory       (object) : the memory bank of agent
            gamma        (float)  : the discount factor
            alpha        (float)  : the learning rate
    """

    #if theres not enough data to train model exit#
    if memory.full() == False:
        return 
    
    #getting training data#
    #state;action;reward;next_state;done
    data = memory.sample()

    #state#
    X = []

    #TD Error#
    Y = []

    for i in range(0,len(data),1):

        #getting transition information#
        state = data[i][0]
        action = data[i][1]
        reward = data[i][2]
        next_state = data[i][3]
        done = data[i][4]

        
        #getting current Q values#
        Q = model.predict(state)
        
        #target network Q values#
        target_Q = target_model.predict(next_state)

        if done == True:
            target = -10
        else:
            target = reward + gamma*np.max(target_Q)
        
        #getting TD error#
        TD_error = np.zeros(2)
        TD_error[action] = target - Q[action]

        #adding state and error to batch#
        X.append(state)
        Y.append(TD_error)
    
    #training model#
    X_data = np.array(X)
    Y_data = np.array(Y)
    model.train(X_data, Y_data,alpha=alpha,epochs=500)

        
        


def train_model(model:NN,num_episodes=1000,epsilon=0.01,gamma=1,alpha=0.5,Copy=10,capacity = 100,batch_size=30):
    """
        Trains the Nueral Network using the Q-learning algorithm

        Parameters:
            model        (NN)    : the model to be trained
            num_episodes (int)   : the number of episodes the algorithm is to be run
            epsilon      (float) : exploration factor
            gamma        (float) : the discount factor
            alpha        (float) : the learning rate 
            Copy         (int)   : Target network update rate
            capacity     (int)   : the size of the agents memory batch
            batch_size   (int)   : the size of the data to be used for training  (not used currently)
    """


    #creating target network#
    Target = NN()
    Target.model_copy(model)

    #creating enviroment#
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    #creating memory bank#
    memory  = Memory(capacity, batch_size)

    for i in range(0,num_episodes,1):

        state = env.reset()
        done = False

        count = 0

        while done == False:
            
            #using network to predict policy#
            policy = model.predict(state)

            #selecting action using greedy policy#
            r = np.random.uniform(0,1)

            if r<epsilon:
                action = np.random.randint(0,2)
            else:
                action = np.argmax(policy)
                
            
            #applying action and getting next state#
            next_state,reward,done,info=env.step(int(action))

            #state;action;reward;next_state;done
            data = [state,action,reward,next_state,done]

            #adding to memory bank#
            memory.push(data)
                
            #updating model#
            optimise_model(model, Target, memory, gamma, alpha)

            #updating state#
            state = np.copy(next_state)
            count+=1

        
        #number of samples#
        n = count
        print("episode {} length :".format(i),n)

        #updating target network#
        if i % Copy == 0:
            Target.model_copy(model)
    
    #saving model#
    model.save_model()


def run(model:NN,num_episodes=1000,epsilon=0.01,gamma=1,alpha=0.5,Copy=10,capacity = 100,batch_size=30):
    """
        Trains the Nueral Network using the Q-learning algorithm

        Parameters:
            model        (NN)    : the model to be trained
            num_episodes (int)   : the number of episodes the algorithm is to be run
            epsilon      (float) : exploration factor
            gamma        (float) : the discount factor
            alpha        (float) : the learning rate 
            Copy         (int)   : Target network update rate
            capacity     (int)   : the size of the agents memory batch
            batch_size   (int)   : the size of the data to be used for training  (not used currently)
    """


    #creating target network#
    # Target = NN()
    # Target.model_copy(model)

    #creating enviroment#
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    #creating memory bank#
    #memory  = Memory(capacity, batch_size)

    for i in range(0,num_episodes,1):

        state = env.reset()
        done = False

        count = 0

        while done == False:
            
            #using network to predict policy#
            policy = model.predict(state)
            action = np.argmax(policy)

            # #selecting action using greedy policy#
            # r = np.random.uniform(0,1)

            # if r<epsilon:
            #     action = np.random.randint(0,2)
            # else:
            #     action = np.argmax(policy)
                
            
            #applying action and getting next state#
            next_state,reward,done,info=env.step(int(action))

            #state;action;reward;next_state;done
            #data = [state,action,reward,next_state,done]

            #adding to memory bank#
            #memory.push(data)
                
            #updating model#
            #optimise_model(model, Target, memory, gamma, alpha)

            #updating state#
            state = np.copy(next_state)
            count+=1

        
        #number of samples#
        n = count
        print("episode {} length :".format(i),n)

        # #updating target network#
        # if i % Copy == 0:
        #     Target.model_copy(model)
    
    #saving model#
    #model.save_model()


if __name__ == '__main__':

    model = NN(n_input=4,n_output=2,layers=[Dense(4,3),Tanh(),Dense(3,2),SOFTMAX()])
    model.load_model()
    #train_model(model)
    run(model)

