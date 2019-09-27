
# Import the gym module
import gym
import random

# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4')
env.reset()
goal_steps = 700
score_requirement = 4
initial_games = 1000
# Reset it, returns the starting frame
#frame = env.reset()
# Render
env.render()

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
                 action = random.randrange(0, 4)
                 observation, reward, done, info = env.step(action)
            else:
                 action = random.randrange(0, 4)
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
                if data[1] == 0:
                    output = [0]
                elif data[1] == 1:
                    output = [1]
                elif data[1] == 2:
                    output = [2]
                elif data[1] == 3:
                    output = [3]
                training_data.append([data[0], output])       
        env.reset()
        scores.append(score)
    print(accepted_scores)  
    print(scores) 
    return training_data

model_data_preparation()