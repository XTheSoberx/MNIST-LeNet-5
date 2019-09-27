import gym
import tensorflow as tf
import random
import numpy as np
from statistics import mean, median

env = gym.make('CartPole-v0').env
env.reset()
goal_steps = 700
score_requirement = 60
initial_games = 10000

def model_data_preparation():
    training_data = []
    scores = []
    accepted_scores = []
    for game_index in range(initial_games):       
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(goal_steps):
            #env.render()         
            if len(previous_observation) == 0:
                 action = random.randrange(0, 2)
                 observation, reward, done, info = env.step(action)
            else:
                 action = random.randrange(0, 2)
                 #action = np.argmax(loaded_model.predict(observation.reshape(-1, len(observation)))[0])
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
    return training_data

def build_model(input_size, output_size):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(128, input_dim = input_size, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(output_size, activation=tf.nn.softmax))
    model.compile(loss= tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5))
    return model

def train_model(training_data):
    x = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    model = build_model(input_size=len(x[0]), output_size=len(y[0]))
    model.fit(x, y, epochs=10, callbacks=[tb])
    model.save('Cartpole.h5')
    return model

tb = tf.keras.callbacks.TensorBoard('./logs/OpenAI_Gym_Cartpole_Simple')    
training_data = model_data_preparation()
trained_model = train_model(training_data)
scores = []
choices = []
env.reset
for each_game in range(100):
    score = 0
    prev_obs = []
    for step_index in range(goal_steps):
        #env.render()
        if len(prev_obs)==0:
             action = random.randrange(0,2)
        else:
             action = np.argmax(trained_model.predict(prev_obs.reshape(-1, len(prev_obs)))[0])       
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        score+=reward
        if done:
            break
    env.reset()
    scores.append(score)
print(scores)
print('Average Score:', sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices))) 