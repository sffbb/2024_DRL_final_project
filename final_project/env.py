
#只到達目標點
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import gym
from gym import spaces
from matplotlib.animation import FuncAnimation


# def random_point():
#     x = np.random.uniform(X_range[0], X_range[1])
#     y = np.random.uniform(Y_range[0], Y_range[1])
#     z = np.random.uniform(Z_range[0], Z_range[1])
#     return np.array([x,y,z])
#
# def get_point():
#     target_point = random_point()
#     while True:
#         obstacle_point = random_point()
#         if np.linalg.norm(obstacle_point - target_point) >= Safe_distance:
#             break
#     return [target_point , obstacle_point]


#-------------------參數--------------------#
from parameter import Safe_distance ,X_range , Y_range ,Z_range ,Points ,ENV_Name
#Points : get target_point , obstacle point


# X_range = [150, 800]
# Y_range = [-500, 500]
# Z_range = [50, 650]
#Points = get_point()
#ENV_Name = 'RobotArm-v0'
#-------------------參數--------------------#

class kinematics():
    def __init__(self):
        pass
    # -------------順向運動學-->使用Cregic DH matrix-------------
    def cregic_DH(self, alpha, a, di, th)  -> np.ndarray:
        """
        Calculate the DH transformation matrix.

        Parameters:
            alpha (float): DH parameter alpha.
            a (float): DH parameter a.
            di (float): DH parameter d (also denoted as di in some conventions).
            th (float): Joint angle theta in radius not degree.

        Returns:
            np.ndarray: Transformation matrix.
        """
        # th=math.radians(th)
        # print(f"theta is : {th}")
        T = np.array([
            [math.cos(th), -math.sin(th), 0, a],
            [(math.sin(th)) * math.cos(alpha), (math.cos(th)) * math.cos(alpha), -math.sin(alpha),
             di * (-(math.sin(alpha)))],
            [(math.sin(th)) * math.sin(alpha), (math.cos(th)) * math.sin(alpha), math.cos(alpha),
             di * (math.cos(alpha))],
            [0, 0, 0, 1]
        ])
        return T

    def mul_martix(self, m1, m2)-> np.ndarray:
        return_matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    return_matrix[i][j] += m2[i][k] * m1[k][j]
        return return_matrix

    def forward_kinematics(self, joint1, joint2, joint3, joint4 ,joint5=1.57,joint6=0)-> np.ndarray:
        pi = 3.1415926
        joint_1 = joint1
        joint_2 = joint2
        joint_3 = joint3
        joint_4 = joint4
        joint_5 = joint5
        joint_6 = joint6
        j1 = self.cregic_DH(0, 0, 145.2, joint_1)
        # 底座升高轉joint2
        j2 = self.cregic_DH(-pi / 2, 0, -146.0, joint_2)
        # joint2 轉換 joint2_up
        j2_up = self.cregic_DH(pi / 2, 0.0, 429.0, 0)
        # joint3
        j3 = self.cregic_DH(-pi / 2, 0, 0, joint_3)
        j3_rightshift = self.cregic_DH(pi, 0, -129.7, 0)
        j3_up = self.cregic_DH(-pi / 2, 0, 411.5, 0)
        # joint4
        j4 = self.cregic_DH(-pi / 2, 0, 0, joint_4)
        j4_leftshift = self.cregic_DH(0, 0, -106.0, 0)
        # joint5
        j5 = self.cregic_DH(pi / 2, 0, 0, joint_5)
        j5_up = self.cregic_DH(0, 0, 106.0, 0)
        # joint6
        j6 = self.cregic_DH(-pi / 2, 0, -113.15, joint_6)
        # transfer_matrix = j1 @ j2 @ j2_up @ j3 @ j3_rightshift @j3_up@ j4 @  j4_leftshift @ j5 @ j5_up @ j6
        # count  the transfer_matrix --> 只是單純的矩陣相乘
        transfer_matrix = self.mul_martix(j6, j5_up)
        transfer_matrix = self.mul_martix(transfer_matrix, j5)
        transfer_matrix = self.mul_martix(transfer_matrix, j4_leftshift)
        transfer_matrix = self.mul_martix(transfer_matrix, j4)
        transfer_matrix = self.mul_martix(transfer_matrix, j3_up)
        transfer_matrix = self.mul_martix(transfer_matrix, j3_rightshift)
        transfer_matrix = self.mul_martix(transfer_matrix, j3)
        transfer_matrix = self.mul_martix(transfer_matrix, j2_up)
        transfer_matrix = self.mul_martix(transfer_matrix, j2)
        transfer_matrix = self.mul_martix(transfer_matrix, j1)

        position = transfer_matrix[:3, 3]  # --> 表示位置
        orientation = transfer_matrix[:3, :3]  # --> 表示旋轉
        print("transfer matrix : \n", transfer_matrix)
        print(f"position:\n{position}\norientation:\n{orientation}")
        return position

    # -------------逆向運動學-------------
    def inverse_kinematics(self,target_point) :
        # joint 1
        x = target_point[0]
        y = target_point[1]
        z = target_point[2]
        if x * x + y * y > 21316:
            length_to_target = math.sqrt(x * x + y * y)
            phi = math.atan(y / x)
            joint1 = phi - math.asin(-122.3 / length_to_target)
            sin_joint1 = math.sin(joint1)
            cos_joint1 = math.cos(joint1)
            # 取c點
            target_x = x - 106 * (sin_joint1 + cos_joint1) - 16.3 * sin_joint1
            target_y = y - 106 * (sin_joint1 - cos_joint1) + 16.3 * cos_joint1

            target_z = z + 122 + 113.15 - 145.2
            LL = target_x * target_x + target_y * target_y + target_z * target_z
            L1 = 429
            L2 = 411.5
            numerator = LL - L1 * L1 - L2 * L2
            denominator = 2 * L1 * L2
            temp = numerator / denominator
            if temp > 1:
                temp = 1
            elif temp < -1:
                temp = -1
            joint3 = math.acos(temp)
            numerator2 = L2 * math.sin(joint3)
            denominator2 = L2 * math.cos(joint3) + L1
            gama = math.atan2(target_z, math.sqrt(target_x * target_x + target_y * target_y))
            th2 = gama + math.atan(numerator2 / denominator2)
            joint4 = th2 - joint3
            joint2 = 1.57 - th2
            return joint1, joint2, joint3, joint4
        else:
            print("too small")
            return 0,0,0,0

class RobotArm:
    def __init__(self, theta1, theta2, theta3, theta4, theta5, theta6=0):
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.theta4 = theta4
        self.theta5 = theta5
        self.theta6 = theta6
        # initial joint coordinate
        self.j1_start = np.array([0, 0, 0])
        self.j1_end = np.array([0, 0, 145.2])
        self.j1_horizon = np.array([0, -146, 145.2])
        self.j2 = np.array([0, -146, 574.2])
        self.j2_horizon = np.array([0, -16.3, 574.2])
        self.j3 = np.array([0, -16.3, 985.7])
        self.j3_horizon = np.array([0, -122.3, 985.7])
        self.j4 = np.array([0, -122.3, 1091.7])
        self.j4_horizon = np.array([0, -357.45, 1091.7])
    def calculate_joint_coordinates_first(self):
        # 只動joint 1
        theta = self.theta1
        j1_end = np.array([0, 0, 145.2])
        axis = j1_end
        axis = axis / np.linalg.norm(axis) # normalize
        j1_horizon = self.rotate_vector(self.j1_horizon - j1_end, axis, theta) + j1_end
        j2 = self.rotate_vector(self.j2 - j1_end, axis, theta) + j1_end
        j2_horizon = self.rotate_vector(self.j2_horizon - j1_end, axis, theta) + j1_end
        j3 = self.rotate_vector(self.j3 - j1_end, axis, theta) + j1_end
        j3_horizon = self.rotate_vector(self.j3_horizon - j1_end, axis, theta) + j1_end
        j4 = self.rotate_vector(self.j4 - j1_end, axis, theta) + j1_end
        j4_horizon = self.rotate_vector(self.j4_horizon - j1_end, axis, theta) + j1_end
        # print("1\n", j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon, "\n")
        return j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon

    def calculate_joint_coordinates_second(self):
        # 動joint2
        theta = self.theta2
        j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon = self.calculate_joint_coordinates_first()
        j1_end = np.array([0, 0, 145.2])
        axis = j1_end - j1_horizon
        axis = axis / np.linalg.norm(axis) # normalize
        j2 = self.rotate_vector(j2 - j1_horizon, axis, theta) + j1_horizon
        j2_horizon = self.rotate_vector(j2_horizon - j1_horizon, axis, theta) + j1_horizon
        j3 = self.rotate_vector(j3 - j1_horizon, axis, theta) + j1_horizon
        j3_horizon = self.rotate_vector(j3_horizon - j1_horizon, axis, theta) + j1_horizon
        j4 = self.rotate_vector(j4 - j1_horizon, axis, theta) + j1_horizon
        j4_horizon = self.rotate_vector(j4_horizon - j1_horizon, axis, theta) + j1_horizon
        # print("2\n", j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon, "\n")
        return j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon
    def calculate_joint_coordinates_third(self):
        theta = self.theta3
        j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon = self.calculate_joint_coordinates_second()
        axis = j2_horizon - j2
        axis = axis / np.linalg.norm(axis)
        j3 = self.rotate_vector(j3 - j2, axis, theta) + j2
        j3_horizon = self.rotate_vector(j3_horizon - j2, axis, theta) + j2
        j4 = self.rotate_vector(j4 - j2, axis, theta) + j2
        j4_horizon = self.rotate_vector(j4_horizon - j2, axis, theta) + j2
        # print("3\n", j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon, "\n")
        return j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon

    def calculate_joint_coordinates_fourth(self):
        theta = self.theta4
        j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon = self.calculate_joint_coordinates_third()
        axis = j3 - j3_horizon
        axis = axis / np.linalg.norm(axis)
        j4 = self.rotate_vector(j4 - j3_horizon, axis, theta) + j3_horizon
        j4_horizon = self.rotate_vector(j4_horizon - j3_horizon, axis, theta) + j3_horizon
        # print("4\n", j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon, "\n")
        return j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon

    def calculate_joint_coordinates_fifth(self):
        theta = self.theta5
        j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon = self.calculate_joint_coordinates_fourth()
        axis = j4 - j3_horizon
        axis = axis / np.linalg.norm(axis)
        # j4 = self.rotate_vector(j4-j3,axis,theta)+j3
        j4_horizon = self.rotate_vector(j4_horizon - j3_horizon, axis, theta) + j3_horizon
        # print("5\n", j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon, "\n")
        return j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon

    def rotate_vector(self, v, axis, theta):
        # v = np.asarray(v)
        # axis = np.asarray(axis) / np.linalg.norm(axis)  # Normalize k to a unit vector
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        term1 = v * cos_theta
        term2 = np.cross(axis, v) * sin_theta
        term3 = axis * np.dot(axis, v) * (1 - cos_theta)
        return term1 + term2 + term3

    def rotate_z_axis(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0],
                         [0, 0, 1]
                         ])

    def plot_arm(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Calculate joint coordinates up to the specified configuration
        j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon = self.calculate_joint_coordinates_fifth()
        print(j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon)
        # print("IN degrtees : ",j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon)
        # Plotting joints
        ax.scatter(*self.j1_start, c='b', marker='o', label='Joint 1 Start')
        ax.scatter(*self.j1_end, c='g', marker='o', label='Joint 1 End')
        ax.scatter(*j1_horizon, c='r', marker='o', label='Joint 1 Horizon')
        ax.scatter(*j2, c='m', marker='o', label='Joint 2')
        ax.scatter(*j2_horizon, c='y', marker='o', label='Joint 2 Horizon')
        ax.scatter(*j3, c='c', marker='o', label='Joint 3')
        ax.scatter(*j3_horizon, c='k', marker='o', label='Joint 3 Horizon')
        ax.scatter(*j4, c='orange', marker='o', label='Joint 4')
        ax.scatter(*j4_horizon, c='purple', marker='o', label='Joint 4 Horizon')

        # Connecting joints with lines
        ax.plot([self.j1_start[0], self.j1_end[0], j1_horizon[0], j2[0], j2_horizon[0], j3[0], j3_horizon[0], j4[0],
                 j4_horizon[0]],
                [self.j1_start[1], self.j1_end[1], j1_horizon[1], j2[1], j2_horizon[1], j3[1], j3_horizon[1], j4[1],
                 j4_horizon[1]],
                [self.j1_start[2], self.j1_end[2], j1_horizon[2], j2[2], j2_horizon[2], j3[2], j3_horizon[2], j4[2],
                 j4_horizon[2]], c='k', label='Arm')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.legend()
        ax.set_xlim(-1100, 1100)
        ax.set_ylim(-1100, 1100)
        ax.set_zlim(0, 1100)
        plt.show()



class RobotArmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,target_point=Points[0], obstacle_point=Points[1] ):
        super(RobotArmEnv, self).__init__()
        self.x_range = X_range
        self.y_range = Y_range
        self.z_range = Z_range
        self.initial_angle = [0,0,1.57,0,1.57]

        if target_point is None:
            self.target_point = self.random_point()
            #print("None : ",self.target_point)
        else:
            self.target_point = target_point
            #print("Not None : ",self.target_point)
        if obstacle_point is None:
            self.obstacle_point = self.generate_obstacle_points(self.target_point)
            #print("None : ",self.obstacle_point)
        else:
            self.obstacle_point = obstacle_point[:3]  # obstacle_point 有含半徑
            #print("Not None : ",self.obstacle_point)

        self.robot_arm = RobotArm(0, 0, 1.57, 0, 1.57, 0)
        self.kinematics = kinematics()
        self.dt = 0.1
        self.total_reward = 0
        has , las = self.calculate_action_space()

        # has = [np.pi , np.pi/3 , np.pi,np.pi,np.pi/2]
        # las = [-np.pi , -np.pi/3 , -np.pi,-np.pi,np.pi/2]
        low_action_space = np.array(las,dtype = 'float32')
        high_action_space = np.array(has,dtype = 'float32')
        self.action_space = spaces.Box(low = low_action_space,
                                       high = high_action_space,
                                       shape=(5,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(5,), dtype=np.float32) # 21 -> 5

    def calculate_action_space(self):
        k=kinematics()
        final_joint_angle_list = []
        j1 ,j2 ,j3,j4 = k.inverse_kinematics(self.target_point)

        joints= [j1,j2,j3,j4]
        print("joints : ",joints)
        low_action_space = []
        high_action_space = []
        for i in range(4):
            high_action_space.append(joints[i])
            low_action_space.append(-joints[i])

        #joint 5
        low_action_space.append(1.57)
        high_action_space.append(1.57)

        return  high_action_space , low_action_space
    def random_point(self):
        x = np.random.uniform(self.x_range[0], self.x_range[1])
        y = np.random.uniform(self.y_range[0], self.y_range[1])
        z = np.random.uniform(self.z_range[0], self.z_range[1])
        return np.array([x, y, z])

    def generate_obstacle_points(self, target_point):
        while True:
            obstacle_point = self.random_point()
            if np.linalg.norm(obstacle_point - target_point) >= 120:
                break
        return obstacle_point

    def reset(self):

        self.robot_arm = RobotArm(0, 0, 1.57, 0, 1.57, 0)

        state = np.concatenate(
            [[self.robot_arm.theta1, self.robot_arm.theta2, self.robot_arm.theta3, self.robot_arm.theta4, self.robot_arm.theta5]])


        return state , self.target_point,self.obstacle_point
    #隨機action
    def sample_action(self):
        return np.random.uniform(-0.1, 0.1, 5)


    def step(self, action):
        # print("In function step")
        # print("action : ",action)
        # print("theta : ", self.robot_arm.theta1, self.robot_arm.theta2, self.robot_arm.theta3, self.robot_arm.theta4,
        #       self.robot_arm.theta5)
        self.robot_arm.theta1 += action[0] #* self.dt
        self.robot_arm.theta2 += action[1] #* self.dt
        self.robot_arm.theta3 += action[2] #* self.dt
        self.robot_arm.theta4 += action[3] #* self.dt
        self.robot_arm.theta5 +=  0
        double_pi = 2 * np.pi
        # print("theta : ", self.robot_arm.theta1, self.robot_arm.theta2, self.robot_arm.theta3, self.robot_arm.theta4,
        #       self.robot_arm.theta5)

        # Normalize
        self.robot_arm.theta1 %= double_pi
        self.robot_arm.theta2 %= double_pi
        self.robot_arm.theta3 %= double_pi
        self.robot_arm.theta4 %= double_pi
        #self.robot_arm.theta5 %= double_pi
        #print("theta : ",self.robot_arm.theta1,self.robot_arm.theta2,self.robot_arm.theta3,self.robot_arm.theta4,self.robot_arm.theta5)

        j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon = self.robot_arm.calculate_joint_coordinates_fifth()
        self.edge = [j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon]

        reward, done = self.calculate_reward_and_done(self.edge)
        self.total_reward += reward

        state = np.concatenate([(self.robot_arm.theta1, self.robot_arm.theta2, self.robot_arm.theta3, self.robot_arm.theta4, self.robot_arm.theta5)])
        #print("state : ",state )#, "\nreward : ",reward )
        return state, reward, done,{}

    def calculate_reward_and_done(self, edge):
        target_point = np.array(self.target_point)
        obstacle_point = np.array(self.obstacle_point)
        end_effector_position = np.array(edge[-1])
        beta = 4e-4
        distances = [
            self.distance_point_to_line(edge[i], edge[i + 1], obstacle_point)
            for i in range(0, len(edge) - 1, 2)
        ]

        reward = 0
        done = False
        distance_to_target = self.distance_point_to_point(end_effector_position, target_point)
        # Ra : reach the point
        if distance_to_target <= 10:
            print("reach the goal")
            reward += 10.0
            done = True
        elif 10<distance_to_target<150:
            reward += 1/distance_to_target
        else:
            # Reward based on the distance to the target
            reward -= distance_to_target / 10000.0
        return reward, done
    def distance_point_to_line(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ab, ac = b - a, c - a
        cross_product = np.cross(ab, ac)
        ab_length = np.linalg.norm(ab)
        distance = np.linalg.norm(cross_product) / ab_length
        return distance

    def distance_point_to_point(self, v1, v2):
        v1, v2 = np.array(v1), np.array(v2)
        return np.linalg.norm(v1 - v2)

    def render(self, mode='human'):
        """
        Render the current state of the robotic arm environment.

        Parameters:
            mode (str): The mode to render with. Only 'human' is supported.

        Returns:
            None
        """
        # Create a new figure for rendering
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Set plot labels and limits
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_xlim(-1100, 1100)
        ax.set_ylim(-1100, 1100)
        ax.set_zlim(0, 1100)

        # Plot the target point and obstacle point
        target_point_plot, = ax.plot([], [], [], 'rx', label='Target Point')
        obstacle_point_plot, = ax.plot([], [], [], 'kx', label='Obstacle Point')

        # Initialize empty plots for the robotic arm
        joints_plot, = ax.plot([], [], [], 'ko', label='Joints')
        arm_line_plot, = ax.plot([], [], [], 'k-', label='Arm')

        def init():
            """Initialize the plot elements."""
            target_point_plot.set_data([], [])
            target_point_plot.set_3d_properties([])
            obstacle_point_plot.set_data([], [])
            obstacle_point_plot.set_3d_properties([])
            joints_plot.set_data([], [])
            joints_plot.set_3d_properties([])
            arm_line_plot.set_data([], [])
            arm_line_plot.set_3d_properties([])
            return target_point_plot, obstacle_point_plot, joints_plot, arm_line_plot

        def update(frame):
            """Update the plot for each frame."""
            action = self.sample_action()
            state, reward, done, _ = self.step(action)

            # Calculate the joint coordinates up to the current configuration
            j1_horizon, j2, j2_horizon, j3, j3_horizon, j4, j4_horizon = self.robot_arm.calculate_joint_coordinates_fifth()

            # Update the target and obstacle points
            target_point_plot.set_data([self.target_point[0]], [self.target_point[1]])
            target_point_plot.set_3d_properties([self.target_point[2]])
            obstacle_point_plot.set_data([self.obstacle_point[0]], [self.obstacle_point[1]])
            obstacle_point_plot.set_3d_properties([self.obstacle_point[2]])

            # Update the joints and arm line
            joint_coords = np.array(
                [self.robot_arm.j1_start, self.robot_arm.j1_end, j1_horizon, j2, j2_horizon, j3, j3_horizon, j4,
                 j4_horizon])
            joints_plot.set_data(joint_coords[:, 0], joint_coords[:, 1])
            joints_plot.set_3d_properties(joint_coords[:, 2])
            arm_line_plot.set_data(joint_coords[:, 0], joint_coords[:, 1])
            arm_line_plot.set_3d_properties(joint_coords[:, 2])

            return target_point_plot, obstacle_point_plot, joints_plot, arm_line_plot

        # Create the animation
        anim = FuncAnimation(fig, update, init_func=init, frames=range(100), interval=200, blit=True)

        # Show the plot
        plt.legend()
        plt.show()



gym.envs.registration.register(
    id=ENV_Name,
    entry_point=RobotArmEnv
)



if __name__ == "__main__":
    try :
        from main import goal , obstacles_in_main
        env = RobotArmEnv(goal,obstacles_in_main[0])
        #print("here")
    except:
        env = RobotArmEnv()

    for i in range (10):
        s,t,o = env.reset()
        #print("s : ",s)
        for j in range(10):
            env.step(env.sample_action())
            env.render()