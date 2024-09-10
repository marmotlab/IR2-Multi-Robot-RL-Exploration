#######################################################################
# Name: env.py
# Autonomous exploration environment.
#######################################################################

import sys
if sys.modules['TRAINING']:
    from parameter import *
else:
    from test_parameter import *

import copy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from time import time
from skimage import io
from skimage.measure import block_reduce
from sensor import *
from graph_generator import *
from node import *
from ss_realistic_model import SS_realistic_model



class Env():
    def __init__(self, map_index, n_agent, k_size=20, plot=False, test=False):
        self.n_agent = n_agent
        self.test = not sys.modules['TRAINING'] 
        if self.test:
            self.map_dir = TEST_SET_DIR
        else:
            self.map_dir = TRAIN_SET_DIR

        self.map_list = os.listdir(self.map_dir)
        self.map_list.sort(reverse=True)
        self.map_index = map_index % np.size(self.map_list)
        self.file_path = self.map_list[self.map_index]
        self.ground_truth, self.start_position = self.import_ground_truth(
            self.map_dir + '/' + self.map_list[self.map_index])
        self.ground_truth_size = np.shape(self.ground_truth) 

        self.resolution = 4
        self.sensor_range = SENSOR_RANGE     
        self.connectivity_rate = 0
        self.agents_connected_percentage = 0
        self.explored_rate = 0
        self.all_explored_rate = [0.0 for _ in range(self.n_agent)]
        self.all_rendezvous_utility_inputs = [None for _ in range(self.n_agent)]
        
        # Decentralized Map Merging
        self.agents_merged_belief = None
        self.downsampled_agents_merged_belief = None
        self.agents_merged_belief_frontiers = None
        self.all_robot_positions_belief = [[ self.start_position for _ in range(self.n_agent)] for _ in range(self.n_agent)]    # (N,N,2)
        self.all_robot_positions_step_updated = [[ 0 for _ in range(self.n_agent)] for _ in range(self.n_agent)]                # (N,N,1)
        self.all_robot_global_graph_belief = [[ [ [],[] ] for _ in range(self.n_agent)] for _ in range(self.n_agent)]           # (N,N,(H,O)) --> NxN belief of (Pose Hist, Pose Offshoot)
        self.all_robot_global_graph_step_updated = [[ 0 for _ in range(self.n_agent)] for _ in range(self.n_agent)]             # (N,N,1)
        self.all_robot_old_global_graph_belief_len = [[ [ 0,0 ] for _ in range(self.n_agent)] for _ in range(self.n_agent)]     # (N,N,(H,O)) --> NxN belief of (Pose Hist, Pose Offshoot)
        self.all_robot_positions_gt = None
        self.group_ids_list = None
        self.all_robot_map_belief_area_diff = [None for _ in range(self.n_agent)]
        self.all_robot_positions_missing_counts = [[ 0 for _ in range(self.n_agent)] for _ in range(self.n_agent)]    # (N,N,1)

        self.agents_comms_broken = []
        self.all_robot_belief, self.all_old_robot_belief, self.all_downsampled_belief = [], [], []
        self.all_robot_belief_step_updated = [[ 0 for _ in range(self.n_agent)] for _ in range(self.n_agent)]         # (N,N,1)
        for _ in range(self.n_agent):
            robot_belief = np.ones(self.ground_truth_size) * 127                          # unexplored 127
            self.all_robot_belief.append( [ robot_belief for _ in range(self.n_agent)] )  # (N,N,belief) - 2D
            self.all_old_robot_belief.append(copy.deepcopy(robot_belief))                 # (N,belief)   - 1D
            self.all_downsampled_belief.append(None)

        self.all_graph_generator, self.all_node_coords, self.all_graph, self.all_node_utility, self.all_guidepost, self.all_frontiers = [], [], [], [], [], []
        for id in range(self.n_agent):
            self.all_graph_generator.append(Graph_generator(robot_id=id, map_size=self.ground_truth_size, sensor_range=self.sensor_range, k_size=k_size, file_path=self.file_path, plot=plot))
            self.all_graph_generator[id].route_node.append(self.start_position)
            self.all_node_coords.append(None)
            self.all_graph.append(None)
            self.all_node_utility.append(None)
            self.all_guidepost.append(None)
            self.all_frontiers.append(None)
        
        self.all_curr_vertices = [None for id in range(self.n_agent)] 
        self.max_flock_size = -1
        self.plot = plot
        self.all_frame_files = [[] for _ in range(self.n_agent)]
        self.merged_frame_files = []

        # Signal Strength Comms Model
        if USE_SIGNAL_STRENGTH_NOT_PROXIMITY:
            self.ss_realistic_model = SS_realistic_model(P_T=SS_P_T, threshold_ss=SS_THRESH, gamma=SS_GAMMA, gamma_obst=SS_GAMMA_OBST, \
                                                         dist_o=SS_DIST_O, PL_o=SS_PL_O, X_g_min=SS_XG_MIN, X_g_max=SS_XG_MAX, K_min=SS_K_MIN, K_max=SS_K_MAX)
        else:
            # Proximity Comms Model
            self.max_comms_proximity = np.random.randint(PROXIMITY_COMMS_RANGE_MIN, PROXIMITY_COMMS_RANGE_MAX, 1)[0]  
        
        self.begin()


    def find_index_from_coords(self, position, agent_id):
        index = np.argmin(np.linalg.norm(self.all_node_coords[agent_id] - position, axis=1))
        return index


    def begin(self):
        """ Initialize key variables """

        for id in range(self.n_agent):
            self.all_robot_belief[id][id] = self.update_robot_belief(self.start_position, self.sensor_range, self.all_robot_belief[id][id], self.ground_truth)
            self.all_downsampled_belief[id] = block_reduce(self.all_robot_belief[id][id].copy(), block_size=(self.resolution, self.resolution), func=np.min)
            self.all_frontiers[id] = self.find_frontier(self.all_downsampled_belief[id])
            self.all_old_robot_belief[id] = copy.deepcopy(self.all_robot_belief[id][id])
            self.agents_merged_belief = self.merge_beliefs( [self.agents_merged_belief, self.all_robot_belief[id][id]] )

            node_coords, graph, node_utility, guidepost = self.all_graph_generator[id].generate_graph(self.start_position, self.all_robot_belief[id][id], self.all_frontiers[id])
            self.all_node_coords[id] = node_coords
            self.all_graph[id] = graph
            self.all_node_utility[id] = node_utility
            self.all_guidepost[id] = guidepost

        self.downsampled_agents_merged_belief = block_reduce(self.agents_merged_belief.copy(), block_size=(self.resolution, self.resolution), func=np.min)
        self.agents_merged_belief_frontiers = self.find_frontier(self.downsampled_agents_merged_belief)


    def single_robot_step(self, robot_id, all_robot_positions_gt, curr_eps, sim_step, dist_travelled): 
        """ Execute policy in environment """

        all_curr_frontiers = [[] for _ in range(self.n_agent)]
        all_num_new_frontiers = [[] for _ in range(self.n_agent)]
        self.all_robot_positions_gt = all_robot_positions_gt
        robot_position = all_robot_positions_gt[robot_id]

        self.all_graph_generator[robot_id].route_node.append(robot_position)
        self.all_robot_global_graph_step_updated[robot_id][robot_id] = sim_step                       
        self.all_robot_global_graph_belief[robot_id][robot_id][0].append(copy.deepcopy(robot_position))    # MOVED INTO UPDATE_GRAPH

        ### Update each agent's map belief ###
        next_node_index = self.find_index_from_coords(robot_position, agent_id=robot_id)
        self.all_graph_generator[robot_id].nodes_list[next_node_index].set_visited()
        self.all_robot_belief[robot_id][robot_id] = self.update_robot_belief(robot_position, self.sensor_range, self.all_robot_belief[robot_id][robot_id], self.ground_truth)
        self.all_downsampled_belief[robot_id] = block_reduce(self.all_robot_belief[robot_id][robot_id].copy(), block_size=(self.resolution, self.resolution), func=np.min)     
        
        ### Update global merged belief ###
        self.agents_merged_belief = self.merge_beliefs( [self.agents_merged_belief, self.all_robot_belief[robot_id][robot_id]] )
        self.downsampled_agents_merged_belief = block_reduce(self.agents_merged_belief.copy(), block_size=(self.resolution, self.resolution), func=np.min)
        curr_agents_merged_belief_frontiers = self.find_frontier(self.downsampled_agents_merged_belief)
        all_num_new_frontiers[robot_id] = self.calculate_num_observed_frontiers(self.agents_merged_belief_frontiers, curr_agents_merged_belief_frontiers)
        self.agents_merged_belief_frontiers = curr_agents_merged_belief_frontiers

        ### Compute agent's reward ###
        all_curr_frontiers[robot_id] = self.find_frontier(self.all_downsampled_belief[robot_id])
        new_pose_explore_util = self.all_node_utility[robot_id][next_node_index]
        new_pose_rendezvous_util = self.all_rendezvous_utility_inputs[robot_id][next_node_index].item()
        new_pose_guidepost_penalty = self.all_guidepost[robot_id][next_node_index].item()

        # NOTE: new_pose_rendezvous_util already normalized
        individual_reward = (all_num_new_frontiers[robot_id] / 25) + (new_pose_explore_util / 50) + (new_pose_rendezvous_util) - (dist_travelled / 512) #- (new_pose_guidepost_penalty / 10) #   

        ### Update each agent's graphs & utility ###
        success = self.update_graph(robot_id, all_curr_frontiers[robot_id], extend_global_graph_towards_fronters=True, eps=curr_eps, step=sim_step)
        if not success:
            return success, None, None


        ################################################
        # Connectivity Graph
        ################################################
        
        # Check if all agents are connected
        self.graph_dict = {}
        self.visited_dict = {}

        # Add graph vertices (bidirectional)
        closest_agent_proximity_list = [0.0 for _ in range(self.n_agent)]
        for i, _ in enumerate(all_robot_positions_gt):
            closest_agent_proximity = float('inf')

            for j, _ in enumerate(all_robot_positions_gt):
                if i != j:
                    vertex1 = (all_robot_positions_gt[i][0], all_robot_positions_gt[i][1])
                    vertex2 = (all_robot_positions_gt[j][0], all_robot_positions_gt[j][1])
                    
                    if vertex1 not in self.graph_dict:
                        self.graph_dict[vertex1] = []
                        self.visited_dict[vertex1] = False
                    if vertex2 not in self.graph_dict:
                        self.graph_dict[vertex2] = []
                        self.visited_dict[vertex2] = False

                    dist = np.linalg.norm(all_robot_positions_gt[i] - all_robot_positions_gt[j])

                    # Proximity / Signal-Strength Based (NOTE: Assume connected = bidirectional communication)
                    if (USE_SIGNAL_STRENGTH_NOT_PROXIMITY and self.ss_realistic_model.is_within_signal_strength(self.ground_truth, all_robot_positions_gt[i], all_robot_positions_gt[j])) \
                        or (not USE_SIGNAL_STRENGTH_NOT_PROXIMITY and dist < self.max_comms_proximity):

                        self.graph_dict[vertex1].append(vertex2)
                        self.graph_dict[vertex2].append(vertex1)

                    if dist < closest_agent_proximity:
                        closest_agent_proximity = dist

            closest_agent_proximity_list[i] = closest_agent_proximity

        # Derive sizes of all subconnected graphs
        unique_groups_list = self.unique_groups_list_from_connectivity_graph(self.graph_dict)

        # # If multiple largest flock - consider all broken (no majority)
        # # Else, consider everyone not in largest flock as broken (not majority)
        group_size_list = [len(group) for group in unique_groups_list]
        cur_max_flock_size = max(group_size_list)
        max_counts = group_size_list.count(cur_max_flock_size)
        self.agents_comms_broken = []
        
        if max_counts > 1:
            self.agents_comms_broken = list(self.graph_dict.keys())
        else:
            argmax_group_idx = group_size_list.index(cur_max_flock_size)
            for idx, group in enumerate(unique_groups_list):
                if idx != argmax_group_idx:
                    self.agents_comms_broken += group    # concat

        # Redefine unique_groups in terms of robot_ids
        self.group_ids_list = []
        for unique_group in unique_groups_list:
            group_ids = [id for id, pose in enumerate(all_robot_positions_gt) \
                                if (pose[0], pose[1]) in unique_group ]
            self.group_ids_list.append(group_ids)

        ################################################
        # Belief Propogation (hopping through graph)
        ################################################

        ### Map & Pose Belief Merger ###
        for group_ids in self.group_ids_list:
            if robot_id in group_ids:

                ### [Local Update] Merging map beliefs for agents that are connected ###
                merged_belief = self.merge_beliefs( [self.all_robot_belief[id][id] for id in group_ids] )
                for own_id in group_ids:
                    self.all_robot_belief[own_id][own_id] = merged_belief
                    self.all_robot_belief_step_updated[own_id][own_id] = sim_step
                    self.all_downsampled_belief[own_id] = block_reduce(self.all_robot_belief[own_id][own_id].copy(), block_size=(self.resolution, self.resolution), func=np.min)

                    for other_id_in_group in group_ids:

                        if own_id != other_id_in_group:
                            self.all_robot_belief[own_id][other_id_in_group] = merged_belief
                            self.all_robot_belief_step_updated[own_id][other_id_in_group] = sim_step
                
                        ### [Global Update] Merging map beliefs of other agents' beliefs of other agents ###
                        for other_id_out_group in range(self.n_agent):
                            if other_id_out_group not in group_ids:

                                own_step_updated = self.all_robot_belief_step_updated[own_id][other_id_out_group]
                                other_step_updated = self.all_robot_belief_step_updated[other_id_in_group][other_id_out_group]

                                # # Cond 1: Own belief is None, but other's belief is not None
                                # # Cond 2: Own belief is more outdated and other's belief not None
                                if own_step_updated < other_step_updated and \
                                    self.all_robot_belief[other_id_in_group][other_id_out_group] is not None:

                                    self.all_robot_belief[own_id][other_id_out_group] = \
                                            self.all_robot_belief[other_id_in_group][other_id_out_group]
                                    self.all_robot_belief_step_updated[own_id][other_id_out_group] = \
                                            self.all_robot_belief_step_updated[other_id_in_group][other_id_out_group]


                ### Merging position beliefs for agents that are connected, and agents' belief of other agents (if not as outdated) ###
                for own_id in group_ids:
                    
                    for other_id_in_group in group_ids:
                        
                        # [Local Update] Updating positions belief with agents in direct connectivity
                        if own_id != other_id_in_group:
                            self.all_robot_positions_belief[own_id][other_id_in_group] = all_robot_positions_gt[other_id_in_group]
                            self.all_robot_positions_step_updated[own_id][other_id_in_group] = sim_step

                        # [Global Update] Merging in belief of other agents' belief of other agents
                        for other_id_out_group in range(self.n_agent):
                            if other_id_out_group not in group_ids:
                                
                                own_step_updated = self.all_robot_positions_step_updated[own_id][other_id_out_group]
                                other_step_updated = self.all_robot_positions_step_updated[other_id_in_group][other_id_out_group]
                                
                                # # Cond 1: Own belief is None, but other's belief is not None
                                # # Cond 2: Own belief is more outdated and other's belief not None
                                if own_step_updated < other_step_updated and \
                                    self.all_robot_positions_belief[other_id_in_group][other_id_out_group] is not None:

                                    self.all_robot_positions_belief[own_id][other_id_out_group] = \
                                            self.all_robot_positions_belief[other_id_in_group][other_id_out_group]
                                    self.all_robot_positions_step_updated[own_id][other_id_out_group] = \
                                            self.all_robot_positions_step_updated[other_id_in_group][other_id_out_group]

                ### Merge route history belief of all agents 
                for own_id in group_ids:
                    
                    for other_id_in_group in group_ids:
                        
                        # [Local Update] Updating positions belief with agents in direct connectivity
                        if own_id != other_id_in_group:
                            self.all_robot_global_graph_belief[own_id][other_id_in_group] = copy.deepcopy(self.all_robot_global_graph_belief[other_id_in_group][other_id_in_group])
                            self.all_robot_global_graph_step_updated[own_id][other_id_in_group] = copy.deepcopy(sim_step)

                        # [Global Update] Merging in belief of other agents' belief of other agents
                        for other_id_out_group in range(self.n_agent):
                            if other_id_out_group not in group_ids:
                                
                                own_step_updated = self.all_robot_global_graph_step_updated[own_id][other_id_out_group]
                                other_step_updated = self.all_robot_global_graph_step_updated[other_id_in_group][other_id_out_group]
                                
                                # # Cond 1: Own belief is None, but other's belief is not None
                                # # Cond 2: Own belief is more outdated and other's belief not None
                                if own_step_updated < other_step_updated and \
                                    self.all_robot_global_graph_belief[other_id_in_group][other_id_out_group] is not None:

                                    self.all_robot_global_graph_belief[own_id][other_id_out_group] = \
                                            copy.deepcopy(self.all_robot_global_graph_belief[other_id_in_group][other_id_out_group])
                                    self.all_robot_global_graph_step_updated[own_id][other_id_out_group] = \
                                            copy.deepcopy(self.all_robot_global_graph_step_updated[other_id_in_group][other_id_out_group])


                    ### Update of essential params after map update ### 
                    if own_id == robot_id: 
                        all_curr_frontiers[robot_id] = self.find_frontier(self.all_downsampled_belief[robot_id])
                        success = self.update_graph(robot_id, all_curr_frontiers[robot_id], eps=curr_eps, step=sim_step)
                        if not success:
                            return success, None, None

        ###################################################################

        ### Removing agents' pose belief if belief within comms range, but cannot comms that agent  ###
        for other_id in range(len(self.all_robot_positions_belief[robot_id])):
            if robot_id != other_id and self.all_robot_positions_belief[robot_id][other_id] is not None:

                if USE_SIGNAL_STRENGTH_NOT_PROXIMITY:
                    belief_in_comms_range = self.ss_realistic_model.is_within_signal_strength(self.ground_truth, self.all_robot_positions_belief[robot_id][robot_id], self.all_robot_positions_belief[robot_id][other_id])
                    gt_in_comms_range = self.ss_realistic_model.is_within_signal_strength(self.ground_truth, self.all_robot_positions_gt[robot_id], self.all_robot_positions_gt[other_id])
                else:
                    belief_in_comms_range = (np.linalg.norm(self.all_robot_positions_belief[robot_id][other_id] - self.all_robot_positions_belief[robot_id][robot_id]) < self.max_comms_proximity)
                    gt_in_comms_range = (np.linalg.norm(self.all_robot_positions_gt[other_id] - self.all_robot_positions_gt[robot_id]) < self.max_comms_proximity)
                if belief_in_comms_range and not gt_in_comms_range:
                    self.all_robot_positions_missing_counts[robot_id][other_id] += 1
                elif (belief_in_comms_range and gt_in_comms_range) or (not belief_in_comms_range and gt_in_comms_range):
                    self.all_robot_positions_missing_counts[robot_id][other_id] = 0

                if self.all_robot_positions_missing_counts[robot_id][other_id] >= REMOVE_POSE_BELIEF_MISSING_COUNT:
                    self.all_robot_positions_belief[robot_id][other_id] = None
                    self.all_robot_belief[robot_id][other_id] = None
                    self.all_robot_positions_missing_counts[robot_id][other_id] = 0

        ### Done only if all agents have explored most of the map ###
        done = self.check_done()

        # ### Store for tensorboard logs ###
        self.all_explored_rate[robot_id] = self.evaluate_exploration_rate(agent_id=robot_id)

        success = True
        return success, individual_reward, done


    def update_graph(self, robot_id, curr_frontiers, extend_global_graph_towards_fronters=False, eps=None, step=None):
        """ Update graph based on newly explored map """
        success, node_coords, graph, node_utility, guidepost = self.all_graph_generator[robot_id].update_graph(self.all_robot_belief[robot_id][robot_id], \
                                                                                                                curr_frontiers, self.all_frontiers[robot_id], \
                                                                                                                self.all_robot_positions_belief[robot_id], self.all_robot_global_graph_belief[robot_id], \
                                                                                                                self.all_robot_old_global_graph_belief_len[robot_id], \
                                                                                                                extend_global_graph_towards_fronters=extend_global_graph_towards_fronters, \
                                                                                                                eps=eps, step=step)
        self.all_node_coords[robot_id] = node_coords
        self.all_graph[robot_id] = graph
        self.all_node_utility[robot_id] = node_utility
        self.all_guidepost[robot_id] = guidepost
        self.all_old_robot_belief[robot_id] = copy.deepcopy(self.all_robot_belief[robot_id][robot_id])
        self.all_frontiers[robot_id] = curr_frontiers
        self.all_robot_old_global_graph_belief_len[robot_id] = \
              [[len(inner_list) for inner_list in robot_global_graph_belief] for robot_global_graph_belief in self.all_robot_global_graph_belief[robot_id]]
        
        return success
    

    def update_env_and_get_team_rewards(self):
        """ Evaluate team performance and rewards """
        self.agents_connected_percentage = 1 - (len(self.agents_comms_broken) / self.n_agent)
        self.connectivity_rate = (len(self.agents_comms_broken) == 0)
        self.explored_rate = self.evaluate_team_exploration_rate()

        team_reward = 0
        done = self.check_done()
        if done:
            team_reward += 40
        return team_reward

    ########################

    def merge_beliefs(self, beliefs_to_merge):
        """ Merge map beliefs together"""
        merged_belief = np.ones_like(self.ground_truth) * 127  # unknown
        for belief in beliefs_to_merge:
            merged_belief[belief == 1] = 1   # Obstacle
            merged_belief[belief == 255] = 255   # Free
        return merged_belief


    def check_area_discovered_count(self, new_belief, prior_belief):
        """ Number of pixels for newly discovered area """
        free_area_count = np.count_nonzero( (new_belief - prior_belief) > 0)
        return free_area_count
    

    def compute_map_belief_area_diff(self, robot_id):
        """ Number of pixels for newly discovered area (all robots) """
        map_area_diff = np.zeros((self.n_agent))
        for id in range(len(map_area_diff)):
            if id != robot_id:
                if self.all_robot_belief[robot_id][id] is not None:
                    map_area_diff[id] = self.check_area_discovered_count(self.all_robot_belief[robot_id][robot_id], self.all_robot_belief[robot_id][id])
                else:
                    map_area_diff[id] = None
        return map_area_diff


    def unique_groups_list_from_connectivity_graph(self, graph):
        """ Recursively group connected subgraphs """
        unique_groups = []
        for agent_pose in list(graph.keys()):
            if not any(agent_pose in group for group in unique_groups):
                group_members = []
                self.flock_neighbours_recurse_connectivity_graph(agent_pose, group_members)
                unique_groups.append(group_members)

        return unique_groups


    def flock_neighbours_recurse_connectivity_graph(self, node, group):
        """ Depth-first search recursion in connectivity graph """
        self.visited_dict[node] = True
        group.append(node)

        if node not in self.graph_dict:
            self.graph_dict[node] = []

        for child_node in self.graph_dict[node]:
            if not self.visited_dict[child_node]:
                self.flock_neighbours_recurse_connectivity_graph(child_node, group)


    def visualize_flock_recurse_connectivity_graph(self, node, iter=0):
        """ Depth-first search recursion in connectivity graph (Visualization) """
        self.visited_dict[node] = True
            
        if node not in self.graph_dict:
            self.graph_dict[node] = []

        for child_node in self.graph_dict[node]:
            if not self.visited_dict[child_node]:
                plt.plot([node[0], child_node[0]], [node[1], child_node[1]], c='grey', linewidth='2.0', zorder=88)
                self.visualize_flock_recurse_connectivity_graph(child_node, iter)


    def generate_rendezvous_utility_layer(self, robot_id, eps):
        """ Generate map-delta utility layer for robot observation """

        rendezvous_utility_inputs = np.zeros((len(self.all_node_coords[robot_id]), 1))
        max_map_area = self.ground_truth_size[0] * self.ground_truth_size[1]
        min_map_area = max_map_area * MIN_MAP_DELTA_MAP_RATIO   
        map_delta_unnormalized = self.compute_map_belief_area_diff(robot_id) 
        current = self.all_robot_positions_belief[robot_id][robot_id]

        if self.group_ids_list is not None:
            for group_ids in self.group_ids_list:
                if robot_id not in group_ids:
                    for other_id in group_ids:
                        if map_delta_unnormalized[other_id] < min_map_area: # don't gen path if map_delta is too small ...
                            continue

                        destination = self.all_robot_positions_belief[robot_id][other_id]
                        if destination is not None:

                            map_delta = (map_delta_unnormalized[other_id] * MAP_DELTA_NORM_FACTOR / max_map_area) # Add constant later

                            # (1) Find A* path to neighbour
                            dist, route = self.all_graph_generator[robot_id].find_shortest_path(current, destination, self.all_node_coords[robot_id], self.all_graph_generator[robot_id].graph) 
                                                        
                            # Attempt to run A* reversed since 1st attempt failed
                            if route is None:

                                # # Ensure all graph edges are bidirectional 
                                t0 = time()
                                temp_graph = copy.deepcopy(self.all_graph_generator[robot_id].graph)
                                for node in temp_graph.nodes:
                                    for edge in temp_graph.edges[tuple(node)].values():
                                        temp_graph.add_edge(edge.to_node, node, edge.length)
                                # print(YELLOW, "[Eps {} | Robot {} | Step {}] A* path is none for rendezvous util. Redefining all graph edges to be bi-directional! ({:.2f}s) ".format(eps, robot_id+1, step, time()-t0), NC)
                                dist, route = self.all_graph_generator[robot_id].find_shortest_path(current, destination, self.all_node_coords[robot_id], temp_graph)

                                if route is None:
                                    t1 = time()
                                    self.all_graph_generator[robot_id].edge_clear_all_nodes()
                                    self.all_graph_generator[robot_id].find_k_neighbor_all_nodes(self.all_robot_belief[robot_id][robot_id], update_dense=True, \
                                                                                                 global_graph=self.all_graph_generator[robot_id].global_graph, global_graph_knn_dist_max=10*SENSOR_RANGE, global_graph_knn_dist_min=0)   # Emphasis on global graph edges to prevent broken graph
                                    self.all_graph[robot_id] = copy.deepcopy(self.all_graph_generator[robot_id].graph.edges)

                                    temp_graph = copy.deepcopy(self.all_graph_generator[robot_id].graph)
                                    for node in temp_graph.nodes:
                                        for edge in temp_graph.edges[tuple(node)].values():
                                            temp_graph.add_edge(edge.to_node, node, edge.length)
                                    # print(RED, "[Eps {} | Robot {} | Step {}] A* path is none for rendezvous util. Regen Graph, then redefining all graph edges to be bi-directional! \
                                    #             Time taken to regen all graph edges: {:.2f}s".format(eps, robot_id+1, step, time()-t1), NC)
                                    dist, route = self.all_graph_generator[robot_id].find_shortest_path(current, destination, self.all_node_coords[robot_id], temp_graph)


                            # (2) Backtrack A* path, starting from destination (i.e. agents' position). 
                            # Decay magnitude of map-delta linearly, based on path len. Min magnitude = MAP_DELTA_MIN_CONST.
                            if route is not None and route != []:

                                route = [np.array(coord) for coord in route]    

                                # Densify A* route if too sparse
                                coords_to_insert = {}
                                for i, node in enumerate(route):
                                    if i+1 < len(route):
                                        dist = np.linalg.norm(route[i] - route[i+1])
                                        num_coords_to_insert = int(dist // RENDEZVOUS_ASTAR_DENSIFY_PATH_RAD)

                                        if num_coords_to_insert >= 1:
                                            for j in range(1, num_coords_to_insert+1):
                                                partial_frac = j / (num_coords_to_insert+1)     
                                                x = route[i][0] + partial_frac * (route[i+1][0] - route[i][0])
                                                y = route[i][1] + partial_frac * (route[i+1][1] - route[i][1])
                                                coords_to_insert.setdefault(i+1, []).append(np.array([round(x), round(y)]))

                                num_inserted = 0
                                for idx, coords in sorted(coords_to_insert.items()):
                                    route[(idx+num_inserted):(idx+num_inserted)] = coords       # Merge additional nodes into route list
                                    num_inserted += len(coords)

                                # Set neighboring dense node coords with same map-delta values
                                knn = NearestNeighbors(radius=RENDEZVOUS_ASTAR_MAP_DELTA_INFLATION_RAD)
                                knn.fit(self.all_node_coords[robot_id])
                                map_delta_decay_rate = map_delta / len(route)

                                for i, curr_coord in enumerate(reversed(route)):
                                    _, indices = knn.radius_neighbors(curr_coord.reshape(1,2))

                                    for index in indices[0]:
                                        neighbor_coord = self.all_node_coords[robot_id][index]
                                        
                                        if not self.all_graph_generator[robot_id].check_collision(curr_coord, neighbor_coord, self.all_robot_belief[robot_id][robot_id]):
                                            index = self.find_index_from_coords(neighbor_coord, robot_id)

                                            new_map_delta = map_delta - (i*map_delta_decay_rate) + MAP_DELTA_MIN_CONST
                                            if new_map_delta > rendezvous_utility_inputs[index]:
                                                rendezvous_utility_inputs[index] = new_map_delta

                            elif route is None:
                                success = False
                                print(RED, "Astar path is None, for map-delta utility generation! Skipping Episode{}! ".format(eps), NC)
                                return map_delta_unnormalized, rendezvous_utility_inputs, success

            # Set neighboring coords around robot to be 0
            knn = NearestNeighbors(radius=RENDEZVOUS_OWN_POSE_NO_UTIL_RAD)
            knn.fit(self.all_node_coords[robot_id])
            _, indices = knn.radius_neighbors(current.reshape(1,2))
            for index in indices[0]:
                rendezvous_utility_inputs[index] = 0
                
        success = True
        return map_delta_unnormalized, rendezvous_utility_inputs, success

    ########################

    def import_ground_truth(self, map_index):
        """ Import map (occupied 1, free 255, unexplored 127) """
        try:
            ground_truth = (io.imread(map_index, 1)).astype(int)
            if np.all(ground_truth == 0):
                ground_truth = (io.imread(map_index, 1) * 255).astype(int)
        except:
            new_map_index = self.map_dir + '/' + self.map_list[0]
            ground_truth = (io.imread(new_map_index, 1)).astype(int)
            print('could not read the map_path ({}), hence skipping it and using ({}).'.format(map_index, new_map_index))

        robot_location = np.nonzero(ground_truth == 208)
        robot_location = np.array([np.array(robot_location)[1, 127], np.array(robot_location)[0, 127]])
        ground_truth = (ground_truth > 150)
        ground_truth = ground_truth * 254 + 1
        return ground_truth, robot_location


    def update_robot_belief(self, robot_position, sensor_range, robot_belief, ground_truth):
        """ Expand map belief based on sensor dynamics """
        robot_belief = sensor_work(robot_position, sensor_range, robot_belief, ground_truth)
        return robot_belief


    def check_done(self):
        """ Check if all agents to have explored most of the ground truth map """
        done = True
        for idx in range(self.n_agent):
            if np.sum(self.all_robot_belief[idx][idx] == 255) / np.sum(self.ground_truth == 255) < 0.99:
                done = False
        return done


    def calculate_num_observed_frontiers(self, old_frontiers, frontiers):
        """ Number of frontiers observed from previous to current step """
        frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
        pre_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
        frontiers_num = np.intersect1d(frontiers_to_check, pre_frontiers_to_check).shape[0]
        pre_frontiers_num = pre_frontiers_to_check.shape[0]
        delta_num = pre_frontiers_num - frontiers_num

        return delta_num


    def evaluate_exploration_rate(self, agent_id):
        """ Evaluate exploration rate currently """
        rate = np.sum(self.all_robot_belief[agent_id][agent_id] == 255) / np.sum(self.ground_truth == 255)
        return rate


    def evaluate_team_exploration_rate(self):
        """ Evaluate averaged team exploration rate """
        avg_rate = 0
        for agent_id in range(self.n_agent):
            self.all_explored_rate[agent_id] = self.evaluate_exploration_rate(agent_id)
            avg_rate += self.all_explored_rate[agent_id]
        avg_rate /= self.n_agent
        return avg_rate    


    def find_frontier(self, downsampled_belief):
        """ Returns frontiers on current map belief """
        y_len = downsampled_belief.shape[0]
        x_len = downsampled_belief.shape[1]
        mapping = downsampled_belief.copy()
        belief = downsampled_belief.copy()
        # 0-1 unknown area map
        mapping = (mapping == 127) * 1
        mapping = np.lib.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
        fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + \
                  mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,
                                                                                                      2:] + \
                  mapping[:y_len][:, :x_len]
        ind_free = np.where(belief.ravel(order='F') == 255)[0]
        ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]
        ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
        ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
        ind_to = np.intersect1d(ind_free, ind_fron)

        map_x = x_len
        map_y = y_len
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T

        f = points[ind_to]
        f = f.astype(int)

        f = f * self.resolution

        return f


    def plot_env(self, n, path, step, travel_dist, robots_route, robot_id):
        """ Plot robot's belief (given communication constraints) """

        # plt.rcParams.update({'font.size': 10})
        color_list = ["r", "g", "c", "m", "y", "k"]
        color_list_text = ["Red", "Green", "Blue", "Purple", "Yellow", "Black"]
        
        plt.switch_backend('agg')
        plt.cla()
        plt.imshow(self.all_robot_belief[robot_id][robot_id], cmap='gray')
        plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))

        if VIZ_GRAPH_EDGES:
            x_coords, y_coords = [], []
            for node in self.all_graph_generator[robot_id].graph.nodes:
                for edge in self.all_graph_generator[robot_id].graph.edges[tuple(node)].values():
                    x_coords.extend([node[0], edge.to_node[0], None])  # 'None' to break the line segment
                    y_coords.extend([node[1], edge.to_node[1], None])
            plt.plot(x_coords, y_coords, c='tan', linewidth=1, zorder=1)

        plt.scatter(self.all_frontiers[robot_id][:, 0], self.all_frontiers[robot_id][:, 1], c='r', s=2, zorder=3)

        # Visualize Utility
        # plt.scatter(self.all_node_coords[robot_id][:, 0], self.all_node_coords[robot_id][:, 1], s=2, c=self.all_node_utility[robot_id], zorder=999)  # grid pattern
        plt.scatter(self.all_node_coords[robot_id][:, 0], self.all_node_coords[robot_id][:, 1], s=2, c=self.all_rendezvous_utility_inputs[robot_id], zorder=999)  # grid pattern
        # plt.scatter(self.all_node_coords[robot_id][:, 0], self.all_node_coords[robot_id][:, 1], s=2, c=self.all_guidepost[robot_id], zorder=999)  # grid pattern

        ### Visualize other robot's belief position ###
        for i, position in enumerate(self.all_robot_positions_belief[robot_id]):
            if position is not None:    # 'None' when outdated position belief that have been verified to no longer be there
                robot_marker_color = color_list[i % len(color_list)]
                if robots_route[-1][0][-1] == position[0] and robots_route[-1][1][-1] == position[1]:
                    plt.plot(position[0], position[1], markersize=8, zorder=9999, marker="D", ls="-", c=robot_marker_color, mec="black") 
                else:
                    plt.plot(position[0], position[1], markersize=8, zorder=9999, marker="^", ls="-", c=robot_marker_color, mec="black") 

        # Visualize global graph nodes
        global_nodes = self.all_graph_generator[robot_id].global_graph_nodes
        if global_nodes is not None and len(global_nodes) > 0:
            global_graph_nodes_set = set(map(tuple, global_nodes))
            for i in range(self.n_agent):
                own_global_graph_nodes = self.all_robot_global_graph_belief[i][i][0] + self.all_robot_global_graph_belief[i][i][1]    # route_hist + route_offshoots
                own_global_graph_nodes = np.array(list(set(map(tuple, own_global_graph_nodes)).intersection(global_graph_nodes_set)))
                if len(own_global_graph_nodes) > 0:
                    robot_marker_color = color_list[i % len(color_list)]
                    plt.scatter(own_global_graph_nodes[:, 0], own_global_graph_nodes[:, 1], s=20, c=robot_marker_color, zorder=4)

        # # Visualize Connectivity Graph
        for group_ids in self.group_ids_list:
            if robot_id in group_ids:
                for id in group_ids:
                    agent_pose = (self.all_robot_positions_gt[id][0], self.all_robot_positions_gt[id][1])
                    self.visited_dict = dict.fromkeys(self.visited_dict, False)
                    self.visualize_flock_recurse_connectivity_graph(agent_pose)

        # # Visualize frontier centers
        if self.all_graph_generator[robot_id].frontier_centers is not None:
            centers = self.all_graph_generator[robot_id].frontier_centers
            dummy_vals = np.ones((centers.shape[0]))
            plt.scatter(centers[:,0], centers[:,1], c=dummy_vals, s=400, alpha=0.5, zorder=4) 


        plt.suptitle('Explored: {:.1f}%  Distance: {:.1f}\n(Robot{} Belief - {})'.format(self.all_explored_rate[robot_id]*100, travel_dist, robot_id + 1, color_list_text[robot_id]))
        plt.tight_layout()
        plt.savefig('{}/eps{}_step{}_robot{}.png'.format(path, n, step, robot_id+1, dpi=150))
        # plt.show()
        frame = '{}/eps{}_step{}_robot{}.png'.format(path, n, step, robot_id+1)
        self.all_frame_files[robot_id].append(frame)
        plt.close()


    def plot_env_ground_truth(self, n, path, step, travel_dist, robots_route):
        """ Plot combined belief (given no communication constraints) """

        # plt.rcParams.update({'font.size': 14})
        # plt.figure(figsize=(12, 10))
        color_list_label = ["Robot1", "Robot2", "Robot3", "Robot4", "Robot5", "Robot6"]
        color_list = ["r", "g", "c", "m", "y", "k"]

        plt.switch_backend('agg')
        plt.cla()
        plt.imshow(self.agents_merged_belief, cmap='gray')
        plt.axis((0, self.ground_truth_size[1], self.ground_truth_size[0], 0))
        for robot_id in range(self.n_agent):
            if VIZ_GRAPH_EDGES_GROUND_TRUTH:
                for i in range(len(self.all_graph_generator[robot_id].x)):
                    plt.plot(self.all_graph_generator[robot_id].x[i], self.all_graph_generator[robot_id].y[i], 'tan', zorder=1)

        plt.scatter(self.agents_merged_belief_frontiers[:, 0], self.agents_merged_belief_frontiers[:, 1], c='r', s=2, zorder=3)

        # Visualize Utility
        free_coords = self.all_graph_generator[0].generate_coords_from_map(self.agents_merged_belief)
        dummy_vals = np.ones((free_coords.shape[0], 1))
        plt.scatter(free_coords[:, 0], free_coords[:, 1], s=5.0, c=dummy_vals, zorder=5)  

        # Visualize Routes
        for i, route in enumerate(robots_route):
            xPoints = route[0]
            yPoints = route[1]
            robot_marker_color = color_list[i % len(color_list)]
            plt.plot(xPoints, yPoints, c=robot_marker_color, linewidth=3, zorder=6)
            plt.plot(xPoints[0], yPoints[0], c=robot_marker_color, marker="o", markersize=8, zorder=6)

        ### Visualize other robot's belief position ###
        for i, position in enumerate(self.all_robot_positions_gt):
            robot_marker_color = color_list[i % len(color_list)]
            plt.plot(position[0], position[1], markersize=12, zorder=9999, marker="D", ls="-", c=robot_marker_color, mec="black")
        
        # Visualize Connectivity Graph
        for agent_pose in list(self.graph_dict.keys()):
            self.visited_dict = dict.fromkeys(self.visited_dict, False)
            self.visualize_flock_recurse_connectivity_graph(agent_pose)
        
        # # Add legend
        # patches = [mpatches.Patch(color=color_list[i], label=color_list_label[i]) for i in range(len(self.all_robot_positions_gt))]
        # plt.legend(handles=patches, bbox_to_anchor=(1.2, 0.7), title="Robots", loc="upper right",  title_fontsize='large')       # fontsize='x-large',  title_fontsize='xx-large'  

        plt.suptitle('Total Explored: {:.1f}%  Max Distance: {:.1f}\n(No Communication Constraints)'.format(self.all_explored_rate[robot_id]*100, travel_dist, robot_id + 1))
        plt.tight_layout()
        plt.savefig('{}/eps{}_step{}_merged.png'.format(path, n, step, dpi=150))
        # plt.show()
        frame = '{}/eps{}_step{}_merged.png'.format(path, n, step)
        self.merged_frame_files.append(frame)
        plt.close()
