#######################################################################
# Name: node.py
# Initialize and update nodes in the coliision-free graph.
#######################################################################

import sys
if sys.modules['TRAINING']:
    from parameter import *
else:
    from test_parameter import *

import numpy as np


class Node():
    def __init__(self, coords, frontiers, robot_belief):
        self.coords = coords
        self.observable_frontiers = set()
        self.sensor_range = SENSOR_RANGE 
        self.initialize_observable_frontiers(frontiers, robot_belief)
        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def initialize_observable_frontiers(self, frontiers, robot_belief):
        """ Initialize observable frontiers from node position """
        dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
        frontiers_in_range = frontiers[dist_list < UTILITY_CALC_RANGE]      
        for point in frontiers_in_range:
            collision = self.check_collision(self.coords, point, robot_belief)
            if not collision:
                self.observable_frontiers.add(tuple(point))

    def get_node_utility(self):
        """ Exploration utility """
        return len(self.observable_frontiers)

    def update_observable_frontiers(self, observed_frontiers_set, new_frontiers, robot_belief):
        """ Update observable frontiers from node position """
        if len(observed_frontiers_set) > 0:
            self.observable_frontiers -= observed_frontiers_set

        if len(new_frontiers) > 0:
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < UTILITY_CALC_RANGE]     
            for point in new_frontiers_in_range:
                collision = self.check_collision(self.coords, point, robot_belief)
                if not collision:
                    self.observable_frontiers.add(tuple(point))

        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def reset_observable_frontiers(self, new_frontiers, robot_belief):
        """ Reset observable frontiers from node position """
        self.observable_frontiers = []

        if len(new_frontiers) > 0:
            dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
            new_frontiers_in_range = new_frontiers[dist_list < UTILITY_CALC_RANGE]    
            for point in new_frontiers_in_range:
                collision = self.check_collision(self.coords, point, robot_belief)
                if not collision:
                    self.observable_frontiers.add(tuple(point))

        self.utility = self.get_node_utility()
        if self.utility == 0:
            self.zero_utility_node = True
        else:
            self.zero_utility_node = False

    def set_visited(self):
        """ Set node to be visited """
        self.observable_frontiers = set()
        self.utility = 0
        self.zero_utility_node = True

    def frontiers_within_utility_calc_range(self, frontiers):
        """ Check frontiers only within specified threshold radius """
        dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
        return len(dist_list[dist_list < GLOBAL_NODES_TO_FRONTIER_AVOID_SPARSE_RAD]) > 0


    def check_collision(self, start, end, robot_belief):
        """ Bresenham line algorithm checking """
        collision = False
        map = robot_belief 

        x0 = start[0]
        y0 = start[1]
        x1 = end[0]
        y1 = end[1]
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
            k = map.item(int(y), int(x))
            if x == x1 and y == y1:
                break
            if k == 1:
                collision = True
                break
            if k == 127:
                collision = True
                break
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return collision

