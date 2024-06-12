import numpy as np
def random_point(X_range,Y_range,Z_range):
    x = np.random.uniform(X_range[0], X_range[1])
    y = np.random.uniform(Y_range[0], Y_range[1])
    z = np.random.uniform(Z_range[0], Z_range[1])
    return np.array([x,y,z])
def get_point(X_range,Y_range,Z_range):
    target_point = random_point(X_range,Y_range,Z_range)
    while True:
        obstacle_point = random_point(X_range,Y_range,Z_range)
        if np.linalg.norm(obstacle_point - target_point) >= Safe_distance:
            break
    return [target_point , obstacle_point]

#-------------------參數--------------------#
#與障礙物之間的安全距離
Safe_distance = 120
#機器手臂移動範圍
X_range = [150, 800]
Y_range = [-500, 500]
Z_range = [50, 650]
Points = get_point(X_range,Y_range,Z_range) #get target_point , obstacle point

ENV_Name = 'RobotArm-v0'

#如果有要指定障礙物點跟目標點的話這一段就不要註解
# goal = (600, 180, 250)
# start = (517.69, -122.49, 339.46)
# obstacles_in_main = [(550, 0, 400, 55)]

#-------------------參數--------------------#
