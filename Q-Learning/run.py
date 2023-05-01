import gym
import numpy as np
import matplotlib.pyplot as plt
from model import *

def test_q_learning(q_network, env_name='CartPole-v1', num_episodes=10, max_steps=500):
    env = gym.make(env_name)
    init_state = np.array([0.01, 0.01, 0.01, 0.01])
    angles = []
    for i in range(num_episodes):
        state = env.reset()
        env.state = init_state
        state = env.state
        angle_episode = []
        for j in range(max_steps):
            action = q_network.act(state)
            state, _, done, _ = env.step(action)
            angle_episode.append(state[2])
            if done:
                state = env.reset()
        angles.append(angle_episode)
    angles = np.array(angles)
    means = np.mean(angles, axis=0)
    stds = np.std(angles, axis=0)
    time_data = []
    for i in range(max_steps):
        time_data.append(1.0) # dummy value, as we don't record the time for each step
    # save results to text file
    results_file = open('Q_learning_results.txt', 'w')
    # write means to file
    string = str(means[0])
    for i in range(1, len(means)):
        string += ',' + str(means[i])
    string += '\n'
    results_file.writelines(string)
    # write stds to file
    string = str(stds[0])
    for i in range(1, len(stds)):
        string += ',' + str(stds[i])
    string += '\n'
    results_file.writelines(string)
    # write response time to file
    string = str(np.mean(time_data))
    results_file.writelines(string)
    results_file.close()
    # print results and plot
    print("Mean angle over {} episodes:".format(num_episodes))
    print(means)
    print("Standard deviation:")
    print(stds)
    plt.plot(means, label='mean angle')
    plt.fill_between(np.arange(len(means)), means-stds, means+stds, color='b', alpha=0.2)
    plt.ylabel('angle')
    plt.xlabel('time')
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":

    #creating enviroment#
    env_name='CartPole-v1'
    env = gym.make(env_name)

    #creating and training agent#
    agent = QLearningAgent(obs_size=4,act_size=2,hidden_size=20,gamma=0.7,epsilon=0.1,lr=0.01)
    agent.train(env,num_episodes=200)

    #checking results from agent#
    test_q_learning(agent.q_net)

    