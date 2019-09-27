import gym
import tensorflow as tf
import random
import numpy as np
import time
from statistics import mean, median


env = gym.make('Breakout-v0').env
env.reset()
goal_steps = 2000
score_requirement = 200
initial_games = 10
loaded_model = tf.keras.models.load_model('Donkey.h5')
PTAS = 0

def model_data_preparation():
    training_data = []
    scores = []
    accepted_scores = []
    for game_index in range(initial_games):       
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(goal_steps):
            env.render()         
            if len(previous_observation) == 0:
                 action = random.randrange(0, 2)
                 observation, reward, done, info = env.step(action)
            else:
                 action = np.argmax(loaded_model.predict(observation.reshape(-1, len(observation)))[0])
                 observation, reward, done, info = env.step(action)
                 game_memory.append([previous_observation, action])            
            previous_observation = observation
            score += reward
            if done:
                env.reset
                break         
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output])       
        env.reset()
        scores.append(score)
    print(accepted_scores)  
    print(scores)
    PTAS = sum(scores)/len(scores)
    print('Pre Train Average Score: ', PTAS) 
    return training_data

model_data_preparation()