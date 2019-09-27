import gym
import tensorflow as tf
import random
import numpy as np
import time
from statistics import mean, median

env = gym.make('CartPole-v0').env
env.reset()
goal_steps = 1000
score_requirement = 200
initial_games = 1000
loaded_model = tf.keras.models.load_model('Cartpole.h5')
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
            #env.render()         
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
        print ('episode '+ str(game_index + 1) + ' of ' + str(initial_games) + '-> score: ' + str(score))
    print(accepted_scores)  
    print(scores)
    PTAS = sum(scores)/len(scores)
    print('Pre Train Average Score: ', PTAS) 
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
    return model
    
tb = tf.keras.callbacks.TensorBoard('./logs/OpenAI_Gym_Cartpole_Selftrained'+ time.strftime("%Y%m%d%H%M") )    
training_data = model_data_preparation()
trained_model = train_model(training_data)
scores = []
choices = []
env.reset
for each_game in range(initial_games):
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
    print ('episode '+ str(each_game + 1) + ' of ' + str(initial_games) + '-> score: ' + str(score))
print(scores)
TAS = sum(scores)/len(scores)
print('Train Average Score:', TAS)
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
if PTAS <= TAS:
    print('Training Average Score is higher than Pre Training Average Score')
    print('Saving Model...')
    tf.keras.models.save_model(trained_model, filepath='Cartpole.h5')
else:
    print('Training Average Score is lower than Pre Training Average Score')
    print('Not Saving Model, set score requirement equal to Pre Training Average Score')