from controller import Robot
import math
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import gym
import numpy as np
from rl import PPO
from env import RobotArmEnv
import torch
#如果有要指定障礙物點跟目標點的話這一段就不要註解
#-------------------參數--------------------#
goal = (600, 180, 250)
# start = (517.69, -122.49, 339.46)
obstacles_in_main = [(550, 0, 400, 55)]
#-------------------參數--------------------#

def train(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ppo_agent = PPO(state_dim, action_dim)

    max_episodes =10000
    max_steps = 500
    update_timestep = 2000
    timestep = 0

    states = []
    actions = []
    log_probs = []
    returns = []
    advantages = []

    for ep in range(max_episodes):
        state, target_point, obstacle_point = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            timestep += 1
            action = ppo_agent.get_action(state)
            next_state, reward, done ,_= env.step(action)
            env.render(next_state)
            # Accumulate data
            states.append(state)
            actions.append(action)
            log_prob, _, _ = ppo_agent.policy.evaluate(torch.tensor(state, dtype=torch.float32),
                                                       torch.tensor(action, dtype=torch.float32))
            log_probs.append(log_prob)  # Store log_prob tensor directly
            returns.append(reward)

            # Calculate advantage
            #advantage = reward - ppo_agent.policy.get_value(torch.tensor(state, dtype=torch.float32)).item()
            advantage = reward - torch.tensor(ppo_agent.policy.get_value(torch.tensor(state, dtype=torch.float32)), dtype=torch.float32).item()
            advantages.append(advantage)

            state = next_state
            episode_reward += reward

            if timestep % update_timestep == 0:
                # Convert lists to tensors
                states_tensor = torch.tensor(states, dtype=torch.float32)
                actions_tensor = torch.tensor(actions, dtype=torch.float32)
                log_probs_tensor = torch.stack(log_probs)  # Convert list of tensors to a tensor
                returns_tensor = torch.tensor(returns, dtype=torch.float32)
                advantages_tensor = torch.tensor(advantages, dtype=torch.float32)

                # Perform the update
                ppo_agent.update(states_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages_tensor)
                
                # Clear accumulated data
                states, actions, log_probs, returns, advantages = [], [], [], [], []

            if done:
                #print("done ",end="")
                break

        print(f"Episode: {ep + 1}, Reward: {episode_reward}")

    # Save the trained model
    torch.save(ppo_agent.policy.state_dict(), 'ppo_robotarm_model.pth')#, '../test_yt/ppo_robotarm_model.pth')
    print("Save")

def control_angle(robot, timestep, joint1, joint2, joint3, joint4, joint5, joint6,i_equal_length_of_best_path):
    list_motor = ['jointonemotor',
                  'jointtwomotor',
                  'jointthrmotor',
                  'jointfoumotor',
                  'jointfivmotor',
                  'jointsixmotor']
    list_ps = ['jointoneps',
               'jointtwops',
               'jointthrps',
               'jointfoups',
               'jointfivps',
               'jointsixps']
    
    # Adjust initial positions for Webots (if necessary)
    list_joints = [joint1 - 0.192,
                   joint2 - 0.191983,
                   joint3,
                   joint4,
                   -joint5,
                   joint6]
    #0.2 這是最準的速度
    speed=0.2 if i_equal_length_of_best_path else 0.7
    
    # Initialize motor positions and velocities
    # for i in range(len(list_motor)):
        # m = robot.getDevice(list_motor[i])
        # #m.setPosition(0.0)
        # m.setPosition(float('+inf'))
        # m.setVelocity(speed)  # Set initial velocity

    # Initialize GPS
    gps = robot.getDevice('gps')
    gps.enable(timestep)

    # Initialize Position Sensors
    pSensors = [robot.getDevice(ps) for ps in list_ps]
    for ps in pSensors:
        ps.enable(timestep)

    k=0
    gps_paths = []
    while robot.step(timestep) != -1:
        #k+=1
        #if k % 10 ==0:
        # Initialize motor positions and velocities
        for i in range(len(list_motor)):
            m = robot.getDevice(list_motor[i])
            #m.setPosition(0.0)
            m.setPosition(float('+inf'))
            m.setVelocity(speed)  
            
        gps_value = gps.getValues()
        gps_paths.append(gps_value) # Read GPS value
        #Set target positions for all motors
        for i in range(len(list_motor)):
            m = robot.getDevice(list_motor[i])
            m.setPosition(list_joints[i])
        # Check if all motors have reached their target positions
       
        if i_equal_length_of_best_path :
             #這個最準
            all_at_target = all(abs(ps.getValue() - list_joints[i]) < 0.00001 for i, ps in enumerate(pSensors))
        else:
            all_at_target = all(abs(ps.getValue() - list_joints[i]) < 0.1 for i, ps in enumerate(pSensors))
        
        if all_at_target:
            for m in [robot.getDevice(motor) for motor in list_motor]:
                m.setVelocity(0.0)
            break

    #print("the coordinate of end : ", gps_value)
    return gps_value,gps_paths
    
def execute(env,robot):
    timestep =64
    # try :
    #     from env import goal , obstacles_in_main
    #     env = gym.make('RobotArm-v0',goal,obstacles_in_main[0])
    # except :
    #     print("沒有目標跟障礙物點，所以隨機決定")
    #     env = gym.make('RobtArm-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ppo_agent = PPO(state_dim, action_dim)

    # Load the trained model
    ppo_agent.policy.load_state_dict(torch.load('ppo_robotarm_model.pth'))
    #print(obstacles_in_main[0])
    state,t,o = env.reset()
    done = False
    angle_list =[]
    time_step = 0
    while not done :
        #env.render()  # Render the environment
        action = ppo_agent.get_action(state)
        #print(action)
        state, reward, done, _ = env.step(action)
        angle_list.append(state)

        time_step += 1
        if time_step > 5000:
            print("Not reach the goal : ", time_step)
            break
    #print(angle_list)
    #length = len(angle_list)-1
    equal = False
    for angles in angle_list:
        joint1 = angles[0] if angles[0] < np.pi else -2*np.pi+angles[0]
        joint2 = angles[1] if angles[1] < np.pi else -2*np.pi+angles[1]
        joint3 = angles[2] if angles[2] < np.pi else -2*np.pi+angles[2]
        joint4 = angles[3] if angles[3] < np.pi else -2*np.pi+angles[3]
        print(joint1,joint2,joint3,joint4)
        gps_value,paths = control_angle(robot,timestep,joint1,joint2,joint3,joint4,1.57,0,equal)
        

    env.close()  # Close the environment when done

if __name__ == '__main__':
    robots = Robot()
    try :
        env = gym.make('RobotArm-v0', target_point=goal, obstacle_point=obstacles_in_main[0])
        print("有指定目標跟障礙物點")
    except:
        print("沒有指定目標跟障礙物點，所以隨機決定")
        env = gym.make('RobotArm-v0')
    train(env)
    #evaluate()
    #execute(env,robots)
    
