#######################################################################
# Name: multi_robot_worker.py
# Interact with environment and collect episode experience.
#######################################################################

from parameter import *
import copy
import os
import imageio
import numpy as np
import torch
from env import Env
from robot import Robot


class Worker:
    def __init__(self, meta_agent_id, n_agent, policy_net, q_net, global_step, device='cuda', greedy=False, save_image=False):
        self.device = device
        self.greedy = greedy
        self.n_agent = n_agent
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image

        self.env = Env(map_index=self.global_step, n_agent=self.n_agent, k_size=self.k_size, plot=save_image)
        self.local_policy_net = policy_net
        self.local_q_net = q_net

        # Distribute starting positions (NOTE: Every's belief is different)
        self.robot_list = []
        self.all_robot_positions = []
        for i in range(self.n_agent):
            iter = min(copy.deepcopy(i), len(self.env.all_node_coords[i])-1 )    # In case idx out of bounds
            robot_position = self.env.all_node_coords[i][iter]     
            robot = Robot(robot_id=i, position=robot_position, plot=save_image)
            self.robot_list.append(robot)
            self.all_robot_positions.append(robot_position)

        self.perf_metrics = dict()
        self.episode_buffer = []
        for i in range(15):
            self.episode_buffer.append([])

        self.max_node_coords = 0


    def run_episode(self, curr_episode):
        """ Run simulation episode for multiple robots """
        done = False
        astar_unsuccessful = False

        ### Run episode ###
        for step in range(MAX_EPS_STEPS):
            reward_list = []
            travel_dist_list = []

            for robot_id, deciding_robot in enumerate(self.robot_list):

                ### Update each agent's graphs & utility (if map updated from map_merge) ###
                success = self.env.update_graph(robot_id, self.env.find_frontier(self.env.all_downsampled_belief[robot_id]), eps=self.global_step, step=step)
                if not success: astar_unsuccessful = True; break
            
                deciding_robot.observations, success = self.get_observations(deciding_robot.robot_position, robot_id, curr_episode, step, plot=True)
                if not success: astar_unsuccessful = True; break
                deciding_robot.save_observations(deciding_robot.observations)

                ### Forward pass through policy to get next position ###
                next_position, action_index = self.select_node(deciding_robot.observations, robot_id)
                deciding_robot.save_action(action_index)

                ### Take Action ###
                dist_travelled = np.linalg.norm(next_position - deciding_robot.robot_position)
                deciding_robot.travel_dist += dist_travelled
                deciding_robot.robot_position = next_position

                ### Log results of action (e.g. distance travelled) ###
                travel_dist_list.append(deciding_robot.travel_dist)
                self.all_robot_positions[robot_id] = next_position                    
                self.env.all_robot_positions_belief[robot_id][robot_id] = next_position     
                self.env.all_robot_positions_step_updated[robot_id][robot_id] = step

                ### Execute step in env
                success, reward, done = self.env.single_robot_step(robot_id, self.all_robot_positions, self.global_step, step, dist_travelled)
                if not success: astar_unsuccessful = True; break
                reward_list.append(reward)

                ### Update observations + rewards from action ###
                deciding_robot.observations, success = self.get_observations(deciding_robot.robot_position, robot_id, curr_episode, step, plot=True)
                if not success: astar_unsuccessful = True; break
                deciding_robot.save_next_observations(deciding_robot.observations)

                ### Save a frame to generate gif of robot trajectories ###
                if self.save_image:
                    deciding_robot.save_robot_position()    
                    robots_route = []
                    robots_route.append([deciding_robot.xPoints, deciding_robot.yPoints])
                    robot_gifs_path = copy.deepcopy(GIFS_DIR) + "/robot_{}".format(robot_id+1)
                    if not os.path.exists(robot_gifs_path):
                        os.makedirs(robot_gifs_path)
                    self.env.plot_env(self.global_step, robot_gifs_path, step, travel_dist_list[robot_id], robots_route, robot_id)

            if astar_unsuccessful:
                break

            team_reward = self.env.update_env_and_get_team_rewards()
            for i in range(len(reward_list)):
                reward_list[i] += team_reward
                self.robot_list[i].save_reward_done(reward_list[i], done)

            ### [Ground Truth] Save a frame to generate gif of robot trajectories ###
            if self.save_image:
                robots_route = []
                for robot in self.robot_list:
                    robots_route.append([robot.xPoints, robot.yPoints])
                for _ in range(self.n_agent):
                    robot_gifs_path = copy.deepcopy(GIFS_DIR) + "/merged"
                if not os.path.exists(robot_gifs_path):
                    os.makedirs(robot_gifs_path)
                self.env.plot_env_ground_truth(self.global_step, robot_gifs_path, step, max(travel_dist_list), robots_route)
            
            if done:
                break

        for robot in self.robot_list:
            for i in range(15):
                self.episode_buffer[i] += robot.episode_buffer[i]

        if astar_unsuccessful:
            return False

        self.perf_metrics['travel_dist'] = max(travel_dist_list)
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['connectivity_rate'] = self.env.connectivity_rate
        self.perf_metrics['agents_connected_percentage'] = self.env.agents_connected_percentage

        # save merged gif
        if self.save_image:
            for robot_id in range(self.n_agent):
                robot_gifs_path = copy.deepcopy(GIFS_DIR) + "/robot_{}".format(robot_id+1)
                self.make_gif(robot_gifs_path, curr_episode, robot_id)
            robot_gifs_path = copy.deepcopy(GIFS_DIR) + "/merged"
            self.make_gif_ground_truth(robot_gifs_path, curr_episode)

        num_node_coords = len(max(self.env.all_node_coords, key=len))
        if self.max_node_coords < num_node_coords:
            self.max_node_coords = num_node_coords
        print(YELLOW, f"[Eps {curr_episode} Completed] Steps: {step}, Node Coords: {num_node_coords}, Max Dist: {max(travel_dist_list):.2f}", NC)
        return True


    def get_observations(self, robot_position, robot_id, eps, step, plot=True):
        """ Get robot's observation of environment (neural network inputs) """
    
        # Rendezvous Utility Layer (Gen first because it involves A* path planning, and potentially regen graph if no path found)
        map_delta_unnormalized, rendezvous_utility_inputs, success = self.env.generate_rendezvous_utility_layer(robot_id, eps)
        if not success:
            return [], False

        self.env.all_robot_map_belief_area_diff[robot_id] = map_delta_unnormalized
        self.env.all_rendezvous_utility_inputs[robot_id] = rendezvous_utility_inputs

        node_coords = copy.deepcopy(self.env.all_node_coords[robot_id])
        graph = copy.deepcopy(self.env.all_graph[robot_id])
        node_utility = copy.deepcopy(self.env.all_node_utility[robot_id])
        guidepost = copy.deepcopy(self.env.all_guidepost[robot_id])

        current_node_index = self.env.find_index_from_coords(robot_position, robot_id)
        current_index = torch.tensor([current_node_index]).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,1)
        n_nodes = node_coords.shape[0]

        node_coords = node_coords * NODE_COORDS_SCALING_FACTOR 
        node_utility = node_utility * NODE_UTILITY_SCALING_FACTOR 

        node_utility_inputs = node_utility.reshape((n_nodes, 1))

        # Augment with all agents' positions
        occupied_node = np.zeros((n_nodes, 1))
        all_robot_positions_belief = copy.deepcopy(self.env.all_robot_positions_belief[robot_id])

        for i, position in enumerate(all_robot_positions_belief):
            if position is not None:    # 'None' when outdated position belief that have been verified to no longer be there
                index = self.env.find_index_from_coords(position, robot_id)
                if index == current_index.item():
                    occupied_node[index] = -1
                else:
                    occupied_node[index] = 1

        # Collate final augmented node_coords inputs
        # node_inputs = np.concatenate((node_coords, node_utility_inputs, guidepost, occupied_node, nodes_ss), axis=1)  # ABLATION
        node_inputs = np.concatenate((node_coords, node_utility_inputs, rendezvous_utility_inputs, guidepost, occupied_node), axis=1)  
        node_inputs = torch.FloatTensor(node_inputs).unsqueeze(0).to(self.device)  # (1, node_padding_size+1, 3)

        if node_coords.shape[0] >= self.node_padding_size:
            print(RED, "[Eps {} | Robot {} | Step {}] node_coords.shape[0] >= self.node_padding_size ({} >= {}). Skipping eps.".format(eps, robot_id+1, step, node_coords.shape[0], self.node_padding_size))
            return [], False
        
        assert node_coords.shape[0] < self.node_padding_size
        padding = torch.nn.ZeroPad2d((0, 0, 0, self.node_padding_size - node_coords.shape[0]))
        node_inputs = padding(node_inputs)

        node_padding_mask = torch.zeros((1, 1, node_coords.shape[0]), dtype=torch.int64).to(self.device)
        node_padding = torch.ones((1, 1, self.node_padding_size - node_coords.shape[0]), dtype=torch.int64).to(
            self.device)
        node_padding_mask = torch.cat((node_padding_mask, node_padding), dim=-1)

        # Order wrt self.node_coords indices
        graph = list(graph.values())
        edge_inputs = []
        for coord in self.env.all_node_coords[robot_id]:
            node_edges = self.env.all_graph[robot_id][tuple(coord)].values()
            node_edges = [int(self.env.find_index_from_coords(np.array(edge.to_node), robot_id)) for edge in node_edges]
            edge_inputs.append(node_edges)

        bias_matrix = self.calculate_edge_mask(edge_inputs)
        edge_mask = torch.from_numpy(bias_matrix).float().unsqueeze(0).to(self.device)

        assert len(edge_inputs) < self.node_padding_size
        padding = torch.nn.ConstantPad2d(
            (0, self.node_padding_size - len(edge_inputs), 0, self.node_padding_size - len(edge_inputs)), 1)
        edge_mask = padding(edge_mask)

        if current_index >= len(edge_inputs):
            print(RED, "[Eps {} | Robot {} | Step {}] current_index > len(edge_inputs) ({} >= {}). Skipping eps.".format(eps, robot_id+1, step, current_index, len(edge_inputs)))
            return [], False
        edge = edge_inputs[current_index]
        if plot:
            self.env.all_curr_vertices[robot_id] = [self.env.all_node_coords[robot_id][e] if e != 0 else None for e in edge]  

        while len(edge) < self.k_size:
            edge.append(0)

        edge_inputs = torch.tensor(edge).unsqueeze(0).unsqueeze(0).to(self.device)  # (1, 1, k_size)
        edge_padding_mask = torch.zeros((1, 1, K_SIZE), dtype=torch.int64).to(self.device)
        one = torch.ones_like(edge_padding_mask, dtype=torch.int64).to(self.device)
        if not (edge_inputs.shape == one.shape == edge_padding_mask.shape):
            print(RED, "[Eps {} | Robot {} | Step {}] Not (edge_inputs.shape = one.shape == edge_padding_mask.shape) not (edge_inputs.shape == one.shape == edge_padding_mask.shape). Skipping eps.".format(eps, robot_id+1, step))
            return [], False

        edge_padding_mask = torch.where(edge_inputs == 0, one, edge_padding_mask)

        observations = node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask
        return observations, True   # success


    def select_node(self, observations, robot_id):
        """ Forward pass through policy to get next position to go """
        node_inputs, edge_inputs, current_index, node_padding_mask, edge_padding_mask, edge_mask = observations
        with torch.no_grad():
            logp_list = self.local_policy_net(node_inputs, edge_inputs, current_index, node_padding_mask,
                                              edge_padding_mask, edge_mask)

        if self.greedy:
            action_index = torch.argmax(logp_list, dim=1).long()
        else:
            action_index = torch.multinomial(logp_list.exp(), 1).long().squeeze(1)

        next_node_index = edge_inputs[0, 0, action_index.item()]    
        next_position = self.env.all_node_coords[robot_id][next_node_index]

        return next_position, action_index
    

    def work(self, currEpisode):
        """ Interacts with the environment """
        success = self.run_episode(currEpisode)
        return success


    def calculate_edge_mask(self, edge_inputs):
        """ Generates 2D graph connectivity matrix """
        size = len(edge_inputs)
        bias_matrix = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                if j in edge_inputs[i]:
                    bias_matrix[i][j] = 0
        return bias_matrix


    def make_gif(self, path, n, robot_id):
        """ Generate a gif given list of images """
        with imageio.get_writer('{}/eps{}_robot{}_explored_rate_{:.4g}.gif'.format(path, n, robot_id+1, self.env.all_explored_rate[robot_id]), mode='I',
                                duration=0.5) as writer:
            for frame in self.env.all_frame_files[robot_id]:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.all_frame_files[robot_id][:-1]:
            os.remove(filename)


    def make_gif_ground_truth(self, path, n):
        """ Generate a gif given list of images (Combined, no communication constraints) """
        with imageio.get_writer('{}/eps{}_merged_explored_rate_{:.4g}.gif'.format(path, n, self.env.explored_rate), mode='I',
                                duration=0.5) as writer:
            for frame in self.env.merged_frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)
        print('gif complete\n')

        # Remove files
        for filename in self.env.merged_frame_files[:-1]:
            os.remove(filename)
