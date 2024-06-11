
import gym
import numpy as np
from rl import PPO , device
from env import RobotArmEnv
import torch
import matplotlib.pyplot as plt
#如果有要指定障礙物點跟目標點的話這一段就不要註解
#-------------------參數--------------------#
# goal = (600, 180, 250)
# start = (517.69, -122.49, 339.46)
# obstacles_in_main = [(550, 0, 400, 55)]
#-------------------參數--------------------#

def train(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ppo_agent = PPO(state_dim, action_dim)

    print(device)
    max_episodes =10000
    max_steps = 1000
    update_timestep = 2000
    timestep = 0

    states = []
    actions = []
    log_probs = []
    returns = []
    advantages = []

    episode_rewards = []
    episode_losses = []
    for ep in range(max_episodes):
        state, target_point, obstacle_point = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            timestep += 1
            action = ppo_agent.get_action(state)
            next_state, reward, done ,_= env.step(action)

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
                states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
                actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)
                log_probs_tensor = torch.stack(log_probs)  # Convert list of tensors to a tensor
                returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
                advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=device)

                # Perform the update
                ppo_agent.update(states_tensor, actions_tensor, log_probs_tensor, returns_tensor, advantages_tensor)

                # Calculate and record loss
                new_log_probs, _, values = ppo_agent.policy.evaluate(states_tensor, actions_tensor)
                ratio = torch.exp(new_log_probs - log_probs_tensor)
                clipped_ratio = torch.clamp(ratio, 1 - ppo_agent.epsilon_clip, 1 + ppo_agent.epsilon_clip)
                advantages_tensor = torch.unsqueeze(advantages_tensor, dim=1)
                policy_loss = -torch.min(ratio * advantages_tensor, clipped_ratio * advantages_tensor).mean()
                value_loss = 0.5 * (returns_tensor - values).pow(2).mean()
                entropy_loss = (torch.distributions.Normal(new_log_probs, ppo_agent.noise_std).entropy()).mean()
                loss = policy_loss + value_loss - 0.01 * entropy_loss

                episode_losses.append(loss.item())

                # Clear accumulated data
                states, actions, log_probs, returns, advantages = [], [], [], [], []

            if done:
                episode_rewards.append(episode_reward)
                #print(f"Episode: {ep + 1}, Reward: {episode_reward}")
                #print("done ",end="")
                break

        print(f"Episode: {ep + 1}, Reward: {episode_reward}")

    # Save the trained model
    torch.save(ppo_agent.policy.state_dict(), 'ppo_robotarm_model.pth')#, '../test_yt/ppo_robotarm_model.pth')
    print("Save")
    # Plotting loss and reward per episode
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(episode_losses)
    plt.title('Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(episode_rewards)
    plt.title('Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.tight_layout()
    plt.show()



def evaluate():
    env = gym.make('RobotArm-v0')
    goal = (600, 160, 250)
    start = (517.69, -122.49, 339.46)
    obstacles_in_main = [(550, 0, 400, 55)]

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ppo_agent = PPO(state_dim, action_dim)

    # Load the trained model
    ppo_agent.policy.load_state_dict(torch.load('../test_yt/ppo_robotarm_model.pth'))

    state = env.reset()
    done = False
    angle_list =[]
    while not done:
        #env.render()  # Render the environment
        action = ppo_agent.get_action(state[0])
        print(action)
        state, reward, done, _ = env.step(action)
        angle_list.append(state)
    print(angle_list)
    env.close()  # Close the environment when done

def execute(env):
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
        if time_step > 1000:
            print("Not reach the goal : ", time_step)
            break

    print(angle_list)
    env.close()  # Close the environment when done

if __name__ == '__main__':
    try :
        env = gym.make('RobotArm-v0', target_point=goal, obstacle_point=obstacles_in_main[0])
        print("有指定目標跟障礙物點")
    except:
        print("沒有指定目標跟障礙物點，所以隨機決定")
        env = gym.make('RobotArm-v0')
    train(env)
    #evaluate()
    execute(env)
