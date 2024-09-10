#######################################################################
# Name: ss_realistic_model.py
# Realistic signal strength communication model.
# References: https://hal.science/hal-03365129/document
#######################################################################

import numpy as np
import random
from skimage.draw import line

class SS_realistic_model:

    def __init__(self, P_T=-20, threshold_ss=-70, gamma=2, gamma_obst=4, dist_o=35, PL_o=31, X_g_min=0, X_g_max=0, K_min=0, K_max=0):
        self.P_T = P_T
        self.threshold_ss = threshold_ss
        self.gamma = gamma
        self.gamma_obst = gamma_obst
        self.dist_o = dist_o
        self.PL_o = PL_o

        # Randomize noise if min != max
        self.X_g = random.uniform(X_g_min, X_g_max) if X_g_min != X_g_max else X_g_min
        self.K = random.uniform(K_min, K_max) if K_min != K_max else K_min

    def is_within_signal_strength(self, robot_belief, robot_a_location, robot_b_location):
        """ Check if 2 locations are within signal strength threshold given free and obstacle cells in between """
        P_T = self.P_T
        X, Y = line(robot_a_location[0], robot_a_location[1], robot_b_location[0], robot_b_location[1])

        # Count the number of obstacles and free in the line
        num_obst = 0
        num_free = 0
        for (x, y) in zip(X[1:], Y[1:]):    # ignore first pose (source)
            if robot_belief[y, x] != 255:   # Obstacle & Unknown
                num_obst += 1
            else:
                num_free += 1
        total_dist = np.linalg.norm(robot_a_location - robot_b_location)
        dist_obst = num_obst * 1               # * self.map_resolution (Assume res = 1m/px)
        dist_free = total_dist - dist_obst

        # Compute path loss
        PL = self.PL_o 
        if dist_obst > 0:
            PL += ( 10 * self.gamma_obst * np.log10(dist_obst) + self.K )
        if dist_free >= self.dist_o:
            PL += ( 10 * self.gamma * np.log10(dist_free/self.dist_o) + self.X_g )

        # Compute received SS
        P_R = P_T - PL

        if P_R > self.threshold_ss:
            return True
        else:
            return False



