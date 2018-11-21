import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

minimum_learning_rate=0.5
minimum_explore_rate=0.5

np.random.seed(999)

#Helper Function

def getAction(explore_rate,obs,partition,QTable):
    threshold=np.random.rand()
    if(threshold<=explore_rate):
        action=env.action_space.sample()
    else:
        action=getBestAction(obs,partition,QTable)
    return action

def getReward(prev_obs,curr_obs,action):
    if(curr_obs[0]-prev_obs[0])>=0:
        return (curr_obs[1]-prev_obs[1])
    else:
        return -1*(curr_obs[1]-prev_obs[1])
def getState(obs,partition):
    obs_dict=env.observation_space.__dict__
    max_obs=obs_dict["high"]
    min_obs=obs_dict["low"]
    obs_range=max_obs-min_obs
    
    state=[]
    
    for index,value in enumerate(obs):
        step=obs_range[index]/partition[index]
        threshold=min_obs[index]+step
        state_code=0
        for _ in range(1,partition[index]):
            if(value<=threshold):
                break
            else:
                state_code+=1
                threshold+=step
        state.append(state_code)
    
    return state
        
def generateQTable(shape):
    return np.zeros(np.prod(shape)).reshape(*shape)

def getQ(state,action,QTable):
    return QTable[state[0]][state[1]][action[0]]

def setQ(new_value,state,action,QTable):
    QTable[state[0]][state[1]][action[0]]=new_value

def getMaxQ(state,QTable):
    result=QTable
    for element in state:
        result=result[element]
    return np.max(result)

def getBestAction(obs,partition,QTable):
    state=getState(obs,partition)
    result=QTable
    for element in state:
        result=result[element]
    return list(result).index(np.max(result))

def updateQTable(obs, next_obs, action, partition, learning_rate,discount_rate,reward,QTable):
    state=getState(obs,partition)
    next_state=getState(next_obs,partition)
    current_q=getQ(state,action,QTable)
    max_future_q=getMaxQ(next_state,QTable)
    setQ(learning_rate*((-1*(current_q))+(discount_rate*max_future_q)+getReward(obs,next_obs,action)),state,action,QTable)

QTable=generateQTable([100,2,3])

for episode in range(1,100000000):
    explore_rate=0.05
##    rewards=200
    obs=env.reset()
    for step in range(200):
        action_=getAction(explore_rate,obs,[100,2],QTable)
        next_obs, reward, done, info = env.step(action_)
##        rewards+=reward
        #if(episode>9000):
        env.render()
        if(done):
            break
        else:
            updateQTable(obs, next_obs, [action_], [100,2], 0.5, 0.5, reward, QTable)
            obs=next_obs
    if(step<199):
        result="Successful"
        print("Episode {} : {}".format(episode,result))
    else:
        result="Unsuccessful"
        print("Episode {} : {}".format(episode,result))
