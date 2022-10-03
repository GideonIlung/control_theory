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
from collections import deque

from keras.layers import Dense
from keras.optimizers import SGD,Adam
from keras.models import Sequential


#Building DQN#
class DQN:

    def __init__(self,input_size,output_size):

        self.input_size = input_size
        self.output_size = output_size

        #Hyperparameters#
        self.gamma = 0.9
        self.alpha = 0.001
        self.epsilon = 0.20
        self.batch_size = 30

        #memory bank#
        self.memory = deque(maxlen=2000)

        #building networks#
        self.model = self.build_network()
        self.target_model = self.build_network()

        #copying weights over#
        self.update_target_model()

    def build_network(self):
        """
            constructs the ANN
        """

        model = Sequential()
        model.add(Dense(24,input_dim=self.input_size,activation='relu'))
        model.add(Dense(24,activation='relu'))
        model.add(Dense(self.output_size,activation='linear'))
        model.summary()
        model.compile(loss='mse',optimizer=Adam())
        return model

    def get_target_Q(self,next_state):
        """
            returns the Q value of the next state obtained from 
            target network
        """

        Q = self.target_model.predict(next_state,verbose=0)[0]
        return np.max(Q)
    
    def update_target_model(self):
        """
            copies the networks weights to the target weights
        """

        #copying weights over#
        self.target_model.set_weights(self.model.get_weights())
    
    def experience_replay(self):
        """
            uses previously saved transition information 
            to train the nueral network
        """

        #theres not enough data in memory bank to train#
        if len(self.memory) < self.batch_size:
            return
        
        data = random.sample(self.memory,self.batch_size)

        X = []
        Y = []

        for state,action,reward,next_state,done in data:

            Q = self.model.predict(state,verbose=0)

            target_Q = self.get_target_Q(next_state)

            if done == True:
                Q[0][action] = reward
            else:
                Q[0][action] = reward + self.gamma*np.max(target_Q)

            X.append(state[0])
            Y.append(Q[0])
        
        #training model#
        self.model.fit(np.array(X),np.array(Y),batch_size=self.batch_size,epochs=1,verbose=0)
    
    def select_action(self,state,use_max=False):

        r = np.random.uniform()

        if r<= self.epsilon and use_max == False:
            action = np.random.randint(0,self.output_size)
        else:
            Q = self.model.predict(state,verbose=0)
            action = np.argmax(Q[0])
        
        return action
    
    def add_transition_data(self,state,action,reward,next_state,done):
        """
            adds transition data into memory bank
        """

        self.memory.append((state,action,reward,next_state,done))


if __name__ == "__main__":

    #creating enviroment#
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    #creating agent#
    model = DQN(input_size=4, output_size=2)

    score = []

    for e in range(0,100,1):
        
        state = env.reset()
        state = np.reshape(state,(1,len(state)))

        done = False
        count = 0

        score = []

        while done == False:
            #getting action#
            action = model.select_action(state)

            #applying action#
            next_state,reward,done,_ = env.step(action)

            next_state = np.reshape(next_state,(1,len(next_state)))

            #updating counter#
            count+=1

            if done == True:
                reward = -100
                model.update_target_model()
                score.append(count)
                print(count)
            
            #dding to memory bank#
            model.add_transition_data(state, action, reward, next_state, done)

            #training model#
            model.experience_replay()

            #updating state#
            state = np.copy(next_state)
    
    print(score)

