import numpy as np
import gym
import random
import time
import game
import json
from ast import literal_eval
import sys

size = (5,5)

# env = gym.make("FrozenLake-v0")
g = game.Game(size=size,graphics=False)
# action_space_size = env.action_space.n
# state_space_size = env.observation_space.n

IMPORT = True
EXPORT = True

# q_table = np.zeros((state_space_size, action_space_size))
# q table will be of type:
"""
{state:
    {"action":
        val,
     "action2":
        val2,
        ...
    }
}
"""

max_steps_per_ep = 300

def train(q_table, EXPORT):
    num_ep = 1000000
    learning_rate = 0.1
    discount_rate = .99

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.000005#np.log(0.01)/(-num_ep)

    rewards_all_episodes = []
    print("0.00%", end="")
    # Q-learning algo
    for ep_num in range(num_ep):
        print(f"\r{100*(ep_num/num_ep):.2f}%", end="")
        state = g.reset(size=size)
        done = False
        rewards_curr_ep = 0

        for step in range(max_steps_per_ep):
            if state not in q_table:
                q_table[state] = {}

            rand = random.uniform(0, max_exploration_rate)
            if rand < exploration_rate:
                # time to explore
                action = g.sample_action_space()
                if str(action) not in q_table[state]:
                    q_table[state][str(action)] = 0
            else:
                # exploitaion time
                action = None
                max_key, max = None, None
                if state in q_table:
                    for key in q_table[state]:
                        if max_key == None:
                            max_key, max = key, q_table[state][key]
                        if q_table[state][key] > max:
                            max_key, max = key, q_table[state][key]
                if max_key:
                    action = literal_eval(max_key)
                else:
                    action = g.sample_action_space()
                    q_table[state][str(action)] = 0

            new_state, reward, done = g.move_snake(action)

            if new_state not in q_table:
                q_table[new_state] = {}

            max_key, max = None, None
            for key in q_table[new_state]:
                if max_key == None:
                    max_key, max = key, q_table[new_state][key]
                if q_table[new_state][key] > max:
                    max_key, max = key, q_table[new_state][key]
            if max == None:
                max = 0

            q_table[state][str(action)] = q_table[state][str(action)] * (1 - learning_rate) + \
                learning_rate * (reward + discount_rate * max)

            state = new_state
            rewards_curr_ep += reward

            if done == True:
                break

        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*ep_num)

        rewards_all_episodes.append(rewards_curr_ep)
    # Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_ep/1000)
    count = 1000
    print("\n********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000

    if EXPORT:
        with open('q_table.json', 'w') as fp:
            json.dump(q_table, fp)

def simulate(q_table):
    for episode in range(3):
        state = g.reset(size=size)

        done = False
        rewards_curr_ep = 0
        print("*****EPISODE ", episode+1, "*****\n\n\n\n")
        time.sleep(1)

        out = False

        for step in range(max_steps_per_ep):
            if state not in q_table:
                q_table[state] = {}
            g.draw_game()
            time.sleep(0.3)


            action = None
            max_key, max = None, None
            if state in q_table:
                for key in q_table[state]:
                    if max_key == None:
                        max_key, max = key, q_table[state][key]
                    if q_table[state][key] > max:
                        max_key, max = key, q_table[state][key]
            if max_key:
                action = literal_eval(max_key)
            else:
                action = g.sample_action_space()
                q_table[state][str(action)] = 0

            new_state, reward, done = g.move_snake(action)

            g.draw_game()

            if done:
                if reward>0:
                    print("****Completed!****")
                    time.sleep(3)
                else:
                    print("****Died!****")
                    time.sleep(3)
                out = True
                break
            state = new_state
        if not out:
            print("****Ran out of moves!****")
            time.sleep(3)


def main():
    q_table = {}
    batches = 1
    if len(sys.argv) > 1:
        if IMPORT:
            try:
                with open('q_table.json', 'r') as fp:
                    q_table = json.load(fp)
                print(len(q_table), "state entries loaded!")
            except:
                pass
        if sys.argv[1] == "train":
            if len(sys.argv) > 2:
                try:
                    batches = int(sys.argv)
                except:
                    pass
            for i in range(0,100):
                train(q_table, EXPORT)

main()
