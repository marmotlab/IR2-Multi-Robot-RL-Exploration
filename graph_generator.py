#######################################################################
# Name: graph_generator.py
# Generate and update the collision-free graph.
#######################################################################

import sys
if sys.modules['TRAINING']:
    from parameter import *
else:
    from test_parameter import *

import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from node import Node
from graph import Graph, a_star
from time import time
from scipy.spatial import KDTree
from scipy.ndimage import label


class Graph_generator:
    def __init__(self, robot_id, map_size, k_size, sensor_range, file_path, plot=False):
        self.robot_id = robot_id
        self.k_size = k_size
        self.graph = Graph()
        self.node_coords = None
        self.plot = plot
        self.x = []
        self.y = []
        self.map_x = map_size[1]
        self.map_y = map_size[0]
        self.uniform_points = self.generate_uniform_points()
        self.sensor_range = sensor_range
        self.route_node = []
        self.nodes_list = []
        self.node_utility = None
        self.guidepost = None
        self.file_path = file_path
        self.nodes_not_to_merge = None
        self.frontier_centers = None
        self.global_graph = Graph()
        self.global_graph_nodes = []


    def edge_clear_all_nodes(self):
        """ Re-init graphs """
        self.graph = Graph()
        self.x = []
        self.y = []

    def edge_clear(self, coords):
        """ Clear specific graph edge """
        self.graph.clear_edge(tuple(coords))

    def node_clear(self, coords, remove_bidirectional_edges=False):
        """ Clear specific graph node """
        self.graph.clear_node(tuple(coords), remove_bidirectional_edges=remove_bidirectional_edges)


    def generate_graph(self, robot_location, robot_belief, frontiers):
        """ Initialize graphs of map belief """

        self.edge_clear_all_nodes()
        free_area = self.free_area(robot_belief)

        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        node_coords = self.uniform_points[candidate_indices]
        node_coords = np.concatenate((robot_location.reshape(1, 2), node_coords))
        self.node_coords = node_coords

        self.find_k_neighbor_all_nodes(robot_belief, update_dense=True)

        self.node_utility = []
        for coords in self.node_coords:
            node = Node(coords, frontiers, robot_belief)
            self.nodes_list.append(node)
            utility = node.utility
            self.node_utility.append(utility)

        self.node_utility = np.array(self.node_utility)
        self.guidepost = np.zeros((self.node_coords.shape[0], 1))
        x = self.node_coords[:,0] + self.node_coords[:,1]*1j
        for node in self.route_node:
            index = self.find_closest_index_from_coords(self.node_coords, node)     
            self.guidepost[index] += 1    # = 1

        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost


    def update_graph(self, robot_belief, frontiers, old_frontiers, robot_location_belief, robot_global_graph_belief, robot_old_global_graph_belief_len, extend_global_graph_towards_fronters=False, eps=None, step=None):
        """ Update graphs of map belief """

        # Update route_hist and route_offshoots based on newest nodes ONLY
        new_route_hist, new_route_offshoot = [], []
        for i in range(len(robot_global_graph_belief)):
            route_hist = robot_global_graph_belief[i][0]
            old_route_hist_len = robot_old_global_graph_belief_len[i][0]
            if route_hist is not None:
                if old_route_hist_len > 0 and len(route_hist)-old_route_hist_len > 0:
                    for node in route_hist[old_route_hist_len:]:
                        if node is not None:
                            new_route_hist.append(node)
                elif old_route_hist_len == 0:
                    new_route_hist += route_hist

            route_offshoot = robot_global_graph_belief[i][1]
            old_route_offshoot_len = robot_old_global_graph_belief_len[i][1]
            if route_offshoot is not None:
                if old_route_offshoot_len > 0 and len(route_offshoot)-old_route_offshoot_len > 0:
                    for node in route_offshoot[old_route_offshoot_len:]:
                        if node is not None:
                            new_route_offshoot.append(node)
                elif old_route_offshoot_len == 0:
                    new_route_offshoot += route_offshoot

        new_global_graph_nodes = copy.deepcopy(new_route_hist) + copy.deepcopy(new_route_offshoot)
        self.global_graph_nodes += new_global_graph_nodes
        if len(self.global_graph_nodes) > 0:
            self.global_graph_nodes = list(self.unique_coords(np.array(self.global_graph_nodes)).reshape(-1, 2))

        # # Reconstruct new global_graph
        if len(new_global_graph_nodes) > 0:
            global_graph_nodes_np = np.array(self.global_graph_nodes)
            new_node_idx_to_update = []
            for coords in new_global_graph_nodes:
                neighbor_indices = self.find_k_neighbor_custom(global_graph_nodes_np, np.array(coords), robot_belief, self.global_graph, global_graph_knn_dist_min=0.0, max_edge_len=GLOBAL_GRAPH_KNN_RAD)
                new_node_idx_to_update += neighbor_indices
            new_node_idx_to_update = set(new_node_idx_to_update)
            for index in new_node_idx_to_update:
                coords = global_graph_nodes_np[index]
                self.global_graph.clear_edge(tuple(coords))
                self.find_k_neighbor_custom(global_graph_nodes_np, np.array(coords), robot_belief, self.global_graph, global_graph_knn_dist_min=0.0, max_edge_len=GLOBAL_GRAPH_KNN_RAD)

            # Ensure bi-directional edges
            for node in self.global_graph.nodes:
                for edge in self.global_graph.edges[tuple(node)].values():
                    self.global_graph.add_edge(edge.to_node, node, edge.length)

        ##################################################################################################################################
        # GRAPH SPARSIFICATION
        ##################################################################################################################################

        # Generate frontier centers if used
        frontier_centers = np.array([])
        if len(self.global_graph.nodes) > 0 and \
            (PRUNE_GLOBAL_GRAPH):
            frontier_centers = self.extract_frontier_centers_new(robot_belief, robot_location_belief)
            self.frontier_centers = frontier_centers    

        # Graph Merger 
        if MERGE_GLOBAL_GRAPH and len(self.global_graph.nodes) > 0 and step > 0 and \
            (step % MERGE_GLOBAL_GRAPH_EVERY == 0 or len(self.global_graph.nodes) > GLOBAL_GRAPH_NODE_COORDS_THRESH): # 0  
            self.merge_global_graph(robot_belief, frontiers, robot_location_belief, global_graph_unique_radius=GLOBAL_GRAPH_UNIQUE_RAD)    
        
        # GLOBAL GRAPH PRUNING
        if PRUNE_GLOBAL_GRAPH and len(self.global_graph.nodes) > 0 and len(frontier_centers) > 0 and step > 0 and \
            (step % PRUNE_GLOBAL_GRAPH_EVERY == 0 or len(self.global_graph.nodes) > GLOBAL_GRAPH_NODE_COORDS_THRESH): # 0  

            success = self.prune_global_graph(robot_belief, robot_location_belief, frontier_centers, eps=eps)
            if not success:
                return success, None, None, None, None


        ##################################################################################################################################
        # LOCAL DENSE GRAPH FORMULATION
        ##################################################################################################################################
            
        ## Add node coords around own robot position (CUR_AGENT_KNN_RAD) and other robot positions (OTHER_AGENT_KNN_RAD)
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        robots_local_nodes = [set() for _ in range(len(robot_location_belief))]
        for id, position in enumerate(robot_location_belief):
            if position is not None:
                knn_rad = CUR_AGENT_KNN_RAD if id == self.robot_id else OTHER_AGENT_KNN_RAD
                
                height, width = robot_belief.shape
                x0, x1 = max(0, position[0] - knn_rad), min(width, position[0] + knn_rad)
                y0, y1 = max(0, position[1] - knn_rad), min(height, position[1] + knn_rad)
                filtered_belief = np.zeros_like(robot_belief)
                filtered_belief[y0:y1, x0:x1] = robot_belief[y0:y1, x0:x1]

                new_filtered_area = self.free_area(filtered_belief)
                filtered_area_to_check = new_filtered_area[:, 0] + new_filtered_area[:, 1] * 1j
                _, _, candidate_indices = np.intersect1d(filtered_area_to_check, uniform_points_to_check, return_indices=True)
                candidate_node_coords = self.uniform_points[candidate_indices]

                # Retrieve all connected components in map to robot's position
                padded_labeled_map = np.full_like(robot_belief, -99)         # Impossble for -99 to interfere with ndimage labelling
                labeled_map, _ = label(robot_belief[y0:y1, x0:x1] == 255)    # Obstacles = 1, free = 255
                padded_labeled_map[y0:y1, x0:x1] = labeled_map
                local_occupancy_map = padded_labeled_map[candidate_node_coords[:,1], candidate_node_coords[:,0]]
                robot_location = self.node_coords[self.find_index_from_coords(self.node_coords, position)]
                pose_idx = self.find_index_from_coords(candidate_node_coords, robot_location)  # robot_location guaranteed to be in local_occupancy_map
                connected_coords = np.argwhere(local_occupancy_map == local_occupancy_map[pose_idx])
                connected_coords = candidate_node_coords[connected_coords[:,0]]
                robots_local_nodes[id].update([tuple(coord) for coord in connected_coords])

        # # Combine all pose filtered node idx
        robots_local_nodes_combined = [node for robot_local_nodes in robots_local_nodes for node in robot_local_nodes]
        robots_local_nodes_combined = np.array(list(set(robots_local_nodes_combined)))

        robot_locations = [position for position in robot_location_belief if position is not None]
        old_node_coords = copy.deepcopy(self.node_coords)
        if len(self.global_graph_nodes) > 0:
            self.node_coords = np.concatenate((self.global_graph_nodes, robot_locations, robots_local_nodes_combined))  
        else:
            self.node_coords = np.concatenate((robot_locations, robots_local_nodes_combined))  
        self.node_coords = self.unique_coords(self.node_coords).reshape(-1, 2)


        ##################################################################################################################################
        # GRAPH COMBINATION
        ##################################################################################################################################
            
        # Add in new nodes 
        coords_old_not_in_new = set(map(tuple, old_node_coords)) - set(map(tuple, self.node_coords))
        coords_new_not_in_old = set(map(tuple, self.node_coords)) - set(map(tuple, old_node_coords))
        coords_old_not_in_new_tuples = [tuple(coords) for coords in coords_old_not_in_new]
        self.node_coords = [coord for coord in old_node_coords if tuple(coord) not in coords_old_not_in_new_tuples]
        self.node_coords += list(coords_new_not_in_old)
        self.node_coords = np.array(self.node_coords)
        self.nodes_list = [node for node in self.nodes_list if tuple(node.coords) not in coords_old_not_in_new_tuples] 

        # Update node utility in self.nodes_list
        old_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
        new_frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
        observed_frontiers_index = np.where(
            np.isin(old_frontiers_to_check, new_frontiers_to_check, assume_unique=True) == False)
        new_frontiers_index = np.where(
            np.isin(new_frontiers_to_check, old_frontiers_to_check, assume_unique=True) == False)
        observed_frontiers = old_frontiers[observed_frontiers_index]
        new_frontiers = frontiers[new_frontiers_index]

        observed_frontiers_set = set(map(tuple, observed_frontiers)) 

        for node in self.nodes_list:
            dist_new_frontiers = np.linalg.norm((new_frontiers - np.array(node.coords)), axis=1)
            close_new_frontiers = new_frontiers[dist_new_frontiers < UTILITY_CALC_RANGE]
            dist_old_frontiers = np.linalg.norm((old_frontiers - np.array(node.coords)), axis=1)
            close_old_frontiers = old_frontiers[dist_old_frontiers < UTILITY_CALC_RANGE]
            no_changed_frontiers = (len(close_new_frontiers) == 0 and len(close_old_frontiers) == 0)
            if node.zero_utility_node is True or no_changed_frontiers:  
                pass
            else:
                node.update_observable_frontiers(observed_frontiers_set, new_frontiers, robot_belief)

        # Add new nodes to self.nodes_list
        self.nodes_list += [Node(coord, frontiers, robot_belief) for coord in coords_new_not_in_old]

        # Consolidate new nodes added to graph
        final_nodes_added = list(coords_new_not_in_old)


        ##################################################################################################################################
        # GRAPH RECONSTRUCTION
        ##################################################################################################################################
            
        # Redefine graph edges based on new set of node coords
        graph_coords_old_not_in_new = set(map(tuple, self.graph.nodes)) - set(map(tuple, self.node_coords))
        graph_coords_new_not_in_old = set(map(tuple, self.node_coords)) - set(map(tuple, self.graph.nodes))

        # Redefine graph edges for node coords to be REMOVED    
        graph_coords_old_not_in_new = np.array(list(graph_coords_old_not_in_new))
        old_nodes_to_update = []
        if len(graph_coords_old_not_in_new) > 0:
            for coords in graph_coords_old_not_in_new:
                neighbor_coords = [edge.to_node for edge in self.graph.edges[tuple(coords)].values()]
                old_nodes_to_update += neighbor_coords
            old_nodes_to_update = np.array(list(set(map(tuple, old_nodes_to_update)) - set(map(tuple, graph_coords_old_not_in_new))))
            
            for coords in graph_coords_old_not_in_new:
                self.node_clear(coords, remove_bidirectional_edges=True)
        
            for coords in old_nodes_to_update:
                self.edge_clear(coords)
                node_coords = self.node_coords                         
                self.find_k_neighbor(node_coords, np.array(coords), robot_belief, global_graph=self.global_graph)     

            for coords in self.global_graph_nodes:
                if np.linalg.norm(coords - robot_location_belief[self.robot_id]) <= 2 * SENSOR_RANGE:
                    self.edge_clear(coords)
                    node_coords = self.node_coords                         
                    self.find_k_neighbor(node_coords, np.array(coords), robot_belief, global_graph=self.global_graph)     

        # Redefine graph edges for node coords to be ADDED
        graph_coords_new_not_in_old = np.array(list(graph_coords_new_not_in_old))
        if len(graph_coords_new_not_in_old) > 0:
            new_node_idx_to_update = []
            node_coords = self.node_coords
            for coords in graph_coords_new_not_in_old:
                neighbor_indices = self.find_k_neighbor(node_coords, np.array(coords), robot_belief, global_graph=self.global_graph)
                new_node_idx_to_update += neighbor_indices
            new_node_idx_to_update = set(new_node_idx_to_update)
            for index in new_node_idx_to_update:
                coords = node_coords[index]
                self.edge_clear(coords)
                self.find_k_neighbor(node_coords, np.array(coords), robot_belief, global_graph=self.global_graph)


        ##################################################################################################################################
        # SPARSE GLOBAL GRAPH FORMULATION (OFFSHOOTS)
        ##################################################################################################################################
            
        # Find top K nodes with highest utility that are GLOBAL_GRAPH_UNIQUE_RAD apart from each other
        if extend_global_graph_towards_fronters and len(self.global_graph_nodes) > 0:
            local_coords_util, local_coords_zero_util = [], []
            own_local_coords = np.array(list(robots_local_nodes[self.robot_id]))  # own local coords
            own_local_coords = own_local_coords[np.linalg.norm(own_local_coords - robot_location_belief[self.robot_id], axis=-1) <= GLOBAL_GRAPH_OFFSHOOT_MAX_RAD]  # Dist filter

            for coords in own_local_coords:
                util_index = self.find_index_from_coords(self.node_coords, coords)
                local_coords_util.append(self.nodes_list[util_index].utility)
                local_coords_zero_util.append(self.nodes_list[util_index].zero_utility_node)

            # sort unique_coords_util by utility
            own_util = self.nodes_list[self.find_index_from_coords(self.node_coords, robot_location_belief[self.robot_id])].utility
            sorted_utility_index = np.argsort(local_coords_util)[::-1]  # descending...
            sorted_local_coords_zero_util = np.array(local_coords_zero_util)[sorted_utility_index]
            sorted_local_coords = np.array(own_local_coords)[sorted_utility_index]

            # Find high utility offshoot nodes (and path to it, if not line of sight)
            global_nodes_added, final_global_nodes_added, final_local_nodes_added = [], [], []
            robot_location = self.node_coords[self.find_index_from_coords(self.node_coords, robot_location_belief[self.robot_id])]
            for i, node in enumerate(sorted_local_coords):
                if sorted_utility_index[i] > own_util and not sorted_local_coords_zero_util[i]:      # Higher than own utility
                    num_within_rad = np.count_nonzero(np.linalg.norm(self.global_graph_nodes - node, axis=-1) < GLOBAL_GRAPH_OFFSHOOT_UNIQUE_RAD)
                    if num_within_rad == 0:
                        if not self.check_collision(robot_location, node, robot_belief):
                            global_nodes_added.append(node)
                        else: # A* to check if node is reachable
                            _, route = self.find_shortest_path(robot_location_belief[self.robot_id], node, self.node_coords, self.graph)
                            if route is not None:
                                global_nodes_added += route
                        
                        global_nodes_set = set(map(tuple, self.global_graph_nodes))
                        for node_added in global_nodes_added:
                            if tuple(node_added) not in global_nodes_set:
                                final_global_nodes_added.append(node_added)
                                self.global_graph_nodes.append(node_added)
                                global_nodes_set.add(tuple(node_added))
                        
                    if len(global_nodes_added) >= GLOBAL_GRAPH_OFFSHOOT_FRONTIER_NODES:
                        break

            # concat global_nodes_added
            robot_global_graph_belief[self.robot_id][1] += final_global_nodes_added  # Add back to route offshoot belief

            # Redefine graph edges for node coords to be ADDED (GLOBAL GRAPH)
            if len(final_global_nodes_added) > 0:
                global_graph_nodes_np = np.array(self.global_graph_nodes)
                new_node_idx_to_update = []
                for coords in final_global_nodes_added:
                    neighbor_indices = self.find_k_neighbor_custom(global_graph_nodes_np, np.array(coords), robot_belief, self.global_graph, global_graph_knn_dist_min=0.0, max_edge_len=GLOBAL_GRAPH_KNN_RAD)
                    new_node_idx_to_update += neighbor_indices
                new_node_idx_to_update = set(new_node_idx_to_update)
                for index in new_node_idx_to_update:
                    coords = global_graph_nodes_np[index]
                    self.global_graph.clear_edge(tuple(coords))
                    self.find_k_neighbor_custom(global_graph_nodes_np, np.array(coords), robot_belief, self.global_graph, global_graph_knn_dist_min=0.0, max_edge_len=GLOBAL_GRAPH_KNN_RAD)


        ##################################################################################################################################

        # Define outputs
        self.node_utility = []
        for i, coords in enumerate(self.node_coords):
            utility = self.nodes_list[i].utility
            self.node_utility.append(utility)
        self.node_utility = np.array(self.node_utility)
        self.guidepost = np.zeros((self.node_coords.shape[0], 1))
        x = self.node_coords[:, 0] + self.node_coords[:, 1] * 1j

        for node in self.route_node:
            index = self.find_closest_index_from_coords(self.node_coords, node)     
            self.guidepost[index] += 1      # = 1

        success = True
        return success, self.node_coords, self.graph.edges, self.node_utility, self.guidepost


    def merge_global_graph(self, robot_belief, frontiers, robot_location_belief, global_graph_unique_radius):
        """ Merge different robots' global graphs """

        temp_graph = copy.deepcopy(self.global_graph)

        # Graph Merge algorithm: Merge in new nodes in global graph
        if RAYTRACE_ZERO_UTIL_GLOBAL_NODES_TO_SPARSIFY:
            self.nodes_not_to_merge = set([tuple(node.coords) for node in self.nodes_list if not node.zero_utility_node]) # NOTE: Node utility not updated at nodes closest to frontiers...
        else:
            self.nodes_not_to_merge = set([tuple(node.coords) for node in self.nodes_list if node.frontiers_within_utility_calc_range(frontiers)])  # Won't sparse even if frontier not LOS 
        
        # Don't remove global nodes around robots
        curr_global_graph_nodes = np.array(list(self.global_graph.nodes))
        for loc in robot_location_belief:
            if loc is not None:
                dist_list = np.linalg.norm((curr_global_graph_nodes - np.array(loc)), axis=1)
                neighboring_global_nodes = curr_global_graph_nodes[dist_list < global_graph_unique_radius]        
                self.nodes_not_to_merge.update([tuple(coord) for coord in neighboring_global_nodes])
        
        merged_nodes = set(self.nodes_not_to_merge) # Don't remove non-zero utility nodes
        for local_iter, curr_node in enumerate(np.array(list(self.global_graph.nodes))):     # separate copy (affected_global_nodes)
            if tuple(curr_node) in merged_nodes:
                continue

            curr_global_graph_nodes = np.array(list(temp_graph.nodes))

            # Get nearest neighbors within RAD
            dist_list = np.linalg.norm((curr_global_graph_nodes-curr_node), axis=1)
            closest_neighbors = curr_global_graph_nodes[dist_list < global_graph_unique_radius]        
            closest_neighbors_to_merge = [neighbor for neighbor in closest_neighbors if tuple(neighbor) != tuple(curr_node) and tuple(neighbor) not in merged_nodes]


            # Check if merging nearest neighbors will maintain graph connectivity
            if len(closest_neighbors_to_merge) > 0:   

                # Find all affected neighbors' coords due to merging
                affected_graph_edges_copied = {}
                coords_with_edges_to_save = copy.deepcopy(closest_neighbors_to_merge)
                for neighbor in closest_neighbors_to_merge:
                    coords_with_edges_to_save += [tuple(edge.to_node) for edge in temp_graph.edges[tuple(neighbor)].values()]
                coords_with_edges_to_save += [tuple(edge.to_node) for edge in temp_graph.edges[tuple(curr_node)].values()]
                coords_with_edges_to_save = set(map(tuple, coords_with_edges_to_save))
                for edge_to_save in coords_with_edges_to_save:
                    affected_graph_edges_copied[tuple(edge_to_save)] = copy.deepcopy(temp_graph.edges[tuple(edge_to_save)])

                # Clear all edges associated with curr and neighbor nodes
                for neighbor in closest_neighbors_to_merge:
                    temp_graph.clear_node(tuple(neighbor), remove_bidirectional_edges=True)
                temp_graph.clear_edge(tuple(curr_node))

                # Find all affected neighbors' coords due to KNN of curr_node
                graph_nodes_np = np.array(list(temp_graph.nodes))
                neighbor_index_list = self.find_k_neighbor_custom(graph_nodes_np, tuple(curr_node), robot_belief, temp_graph, modify_graph=False, global_graph_knn_dist_min=0.0, max_edge_len=GLOBAL_GRAPH_KNN_RAD)
                for node_idx in neighbor_index_list:
                    if affected_graph_edges_copied is not None and tuple(graph_nodes_np[node_idx]) not in affected_graph_edges_copied:
                        affected_graph_edges_copied[tuple(graph_nodes_np[node_idx])] = copy.deepcopy(temp_graph.edges[tuple(graph_nodes_np[node_idx])])

                # Redefine edges for curr_node (given old neighboring nodes alr removed)
                self.find_k_neighbor_custom(graph_nodes_np, tuple(curr_node), robot_belief, temp_graph, modify_graph=True, global_graph_knn_dist_min=0.0, max_edge_len=GLOBAL_GRAPH_KNN_RAD)

                # Ensure bi-directional edges
                for node in temp_graph.nodes:
                    for edge in temp_graph.edges[tuple(node)].values():
                        temp_graph.add_edge(edge.to_node, node, edge.length)

                graph_is_connected, visited_nodes = temp_graph.is_connected_bfs(tuple(curr_node), criteria=coords_with_edges_to_save)

                # If graph is connected, continue to use temp_graph. Else, undo changes...
                if graph_is_connected: 
                    merged_nodes.add(tuple(curr_node))
                    merged_nodes.update([tuple(coord) for coord in closest_neighbors_to_merge])
                else:
                    for node, edges in affected_graph_edges_copied.items():
                        temp_graph.add_node(node)
                        temp_graph.edges[node] = edges

        self.global_graph = copy.deepcopy(temp_graph)
        self.global_graph_nodes = list(self.global_graph.nodes)


    def prune_global_graph(self, robot_belief, robot_location_belief, centers, eps=None):
        """ Prune useless graph branches that does not lead to frontier centers """

        ### Perform A* between Agents-2-Agents and Agents-2-Frontiers
        route_nodes = set()
        outer_loop_centers = centers if SPARSIFY_RETAIN_FRONTIER_TO_FRONTIER_ASTAR else robot_location_belief
        global_graph_nodes_np = np.array(list((self.global_graph.nodes)))

        for outer_loop_center in outer_loop_centers:
            if outer_loop_center is not None:
                path_start = outer_loop_center
                for center in centers:
                    _, route = self.find_shortest_path(path_start, center, global_graph_nodes_np, self.global_graph) 
                                            
                    # Attempt to run A* with edges forced bidirectional - if 1st attempt failed
                    # # Ensure all graph edges are bidirectional 
                    if route is None:
                        t0 = time()
                        temp_graph = copy.deepcopy(self.global_graph)
                        for node in temp_graph.nodes:
                            for edge in temp_graph.edges[tuple(node)].values():
                                temp_graph.add_edge(edge.to_node, node, edge.length)

                        # print(YELLOW, "[Eps {} | Robot {} | Step {}] A* path is none for graph pruning. Redefining all graph edges to be bi-directional! ({:.2f}s) ".format(eps, self.robot_id, step, time()-t0), NC)
                        _, route = self.find_shortest_path(path_start, center, global_graph_nodes_np, temp_graph)

                        if route is None:
                            success = False
                            print(RED, "Astar path is None, for prune_global_graph! Skipping Episode {}! ".format(eps), NC)
                            return success
                    
                    # List of tuples --> list of np.array
                    route = [np.array(coord) for coord in route]    

                    # Densify A* route if too sparse
                    coords_to_insert = {}
                    for i, node in enumerate(route):
                        if i+1 < len(route):
                            dist = np.linalg.norm(route[i] - route[i+1])
                            num_coords_to_insert = int(dist // GLOBAL_GRAPH_UNIQUE_RAD)

                            if num_coords_to_insert >= 1:
                                for j in range(1, num_coords_to_insert+1):
                                    partial_frac = j / (num_coords_to_insert+1)     # (0,1)
                                    x = route[i][0] + partial_frac * (route[i+1][0] - route[i][0])
                                    y = route[i][1] + partial_frac * (route[i+1][1] - route[i][1])
                                    coords_to_insert.setdefault(i+1, []).append(np.array([round(x), round(y)]))

                    num_inserted = 0
                    for idx, coords in sorted(coords_to_insert.items()):
                        route[(idx+num_inserted):(idx+num_inserted)] = coords       # Merge additional nodes into route list
                        num_inserted += len(coords)

                    # Locate original global graph paths along A* paths
                    knn = NearestNeighbors(radius=2*GLOBAL_GRAPH_UNIQUE_RAD)    
                    knn.fit(global_graph_nodes_np)

                    for i, curr_coord in enumerate(reversed(route)):
                        _, indices = knn.radius_neighbors(curr_coord.reshape(1,2))
                        for index in indices[0]:
                            node = global_graph_nodes_np[index]
                            if tuple(node) in route_nodes:
                                continue
                            elif not self.check_collision(curr_coord, node, robot_belief):
                                route_nodes.add(tuple(node))
                    
        # Reconstruct pruned global graph
        route_nodes = np.array(list(route_nodes)).reshape(-1, 2)
        self.global_graph = Graph()
        self.find_k_neighbor_all_nodes_custom(robot_belief, route_nodes, self.global_graph, global_graph_knn_dist_max=2*GLOBAL_GRAPH_KNN_RAD, global_graph_knn_dist_min=0.0)

        self.global_graph_nodes = list(route_nodes)
        success = True
        return success


    def generate_uniform_points(self):
        """ Generate uniform grid in free space of map belief """
        x = np.linspace(0, self.map_x - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int)
        y = np.linspace(0, self.map_y - 1, NUM_DENSE_COORDS_WIDTH).round().astype(int) 
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points


    def free_area(self, robot_belief):
        """ Identify free space in map belief """
        index = np.where(robot_belief == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def unique_coords(self, coords):
        """ Remove duplicates in node coords """
        x = coords[:, 0] + coords[:, 1] * 1j
        indices = np.unique(x, return_index=True)[1]
        coords = np.array([coords[idx] for idx in sorted(indices)])
        return coords

    def find_k_neighbor(self, node_coords, coords, robot_belief, global_graph=None, global_graph_knn_dist_max=GLOBAL_GRAPH_KNN_RAD, global_graph_knn_dist_min=CUR_AGENT_KNN_RAD):
        """ Find nearest k neighbors to specified coords """
        dist_list = np.linalg.norm((node_coords-coords), axis=1)
        sorted_index = np.argsort(dist_list)
        k = 0
        neighbor_index_list, topk_global_graph_nodes = [], []
        count = 0

        # Append global graph edges to each node first (to ensure connectivity)
        num_global_neighbours = 0
        if global_graph is not None and tuple(coords) in global_graph.edges:
            global_graph_edges = global_graph.edges[tuple(coords)].values()
            global_graph_nodes = np.array([edge.to_node for edge in global_graph_edges])
            global_graph_dist = np.array([edge.length for edge in global_graph_edges])
            filtered_global_graph_idx = (global_graph_dist <= global_graph_knn_dist_max) & (global_graph_dist > global_graph_knn_dist_min)      
            filtered_global_graph_nodes = global_graph_nodes[filtered_global_graph_idx]
            filtered_global_graph_dist = global_graph_dist[filtered_global_graph_idx]
            num_global_neighbours = len(filtered_global_graph_nodes) if len(filtered_global_graph_nodes) < self.k_size else self.k_size
            topk_global_graph_nodes = filtered_global_graph_nodes[np.argsort(filtered_global_graph_dist)[:num_global_neighbours]]
            topk_global_graph_nodes = set(map(tuple, topk_global_graph_nodes))  

            for neighbour_node in topk_global_graph_nodes:
                self.graph.add_node(tuple(coords))
                self.graph.add_edge(tuple(coords), tuple(neighbour_node), np.linalg.norm(coords-neighbour_node))
        
        max_neighbours = self.k_size - num_global_neighbours
        num_neighbours = len(node_coords) if len(node_coords) < max_neighbours else max_neighbours

        for neighbor_index in sorted_index:
            neighbor_index_list.append(neighbor_index)
            dist = dist_list[k]
            start = coords
            end = node_coords[neighbor_index]
            if tuple(end) in topk_global_graph_nodes:   # Don't consider global nodes already added
                continue

            if not self.check_collision(start, end, robot_belief):
                self.graph.add_node(tuple(start))
                self.graph.add_edge(tuple(start), tuple(end), np.linalg.norm(start-end))

                if self.plot:
                    self.x.append([start[0], end[0]])
                    self.y.append([start[1], end[1]])
                count += 1
            k += 1
            if k >= num_neighbours:
                break
        return neighbor_index_list
    

    def find_k_neighbor_all_nodes(self, robot_belief, update_dense=True, global_graph=None, global_graph_knn_dist_max=GLOBAL_GRAPH_KNN_RAD, global_graph_knn_dist_min=CUR_AGENT_KNN_RAD):
        """ Find nearest k neighbors to all coords """

        kd_tree = KDTree(self.node_coords)
        for i, p in enumerate(self.node_coords):

            # Append global graph edges to each node first (to ensure connectivity)
            num_global_neighbours = 0
            if global_graph is not None and tuple(p) in global_graph.edges:
                global_graph_edges = global_graph.edges[tuple(p)].values()
                global_graph_nodes = np.array([edge.to_node for edge in global_graph_edges])
                global_graph_dist = np.array([edge.length for edge in global_graph_edges])
                filtered_global_graph_idx = (global_graph_dist <= global_graph_knn_dist_max) & (global_graph_dist > global_graph_knn_dist_min)   
                filtered_global_graph_nodes = global_graph_nodes[filtered_global_graph_idx]
                filtered_global_graph_dist = global_graph_dist[filtered_global_graph_idx]
                num_global_neighbours = len(filtered_global_graph_nodes) if len(filtered_global_graph_nodes) < self.k_size else self.k_size
                topk_global_graph_nodes = filtered_global_graph_nodes[np.argsort(filtered_global_graph_dist)[:num_global_neighbours]]
                topk_global_graph_nodes = set(map(tuple, topk_global_graph_nodes))  

                for neighbour_node in topk_global_graph_nodes:
                    self.graph.add_node(tuple(p))
                    self.graph.add_edge(tuple(p), tuple(neighbour_node), np.linalg.norm(p-neighbour_node))
            
            max_neighbours = self.k_size - num_global_neighbours
            num_neighbours = len(self.node_coords) if len(self.node_coords) < max_neighbours else max_neighbours
            if num_neighbours > 0:
                _, indices = kd_tree.query(p, k=num_neighbours)
                if np.isscalar(indices):
                    indices = np.array([indices])
                for j, neighbour in enumerate(self.node_coords[indices]):                
                    start = p
                    end = neighbour
                    if not self.check_collision(start, end, robot_belief):
                        if update_dense:
                            self.graph.add_node(tuple(start))
                            self.graph.add_edge(tuple(start), tuple(end), np.linalg.norm(start-end))
                        if self.plot:
                            self.x.append([p[0], neighbour[0]])
                            self.y.append([p[1], neighbour[1]])


    def find_k_neighbor_custom(self, node_coords, coords, robot_belief, graph, global_graph=None, global_graph_knn_dist_max=GLOBAL_GRAPH_KNN_RAD, global_graph_knn_dist_min=CUR_AGENT_KNN_RAD, modify_graph=True, max_edge_len=0):     
        """ Find nearest k neighbors to specified coords (with more options) """

        dist_list = np.linalg.norm((node_coords-coords), axis=1)
        sorted_index = np.argsort(dist_list)
        k = 0
        neighbor_index_list, topk_global_graph_nodes = [], []
        count = 0

        # Append global graph edges to each node first (to ensure connectivity)
        num_global_neighbours = 0
        if global_graph is not None and tuple(coords) in global_graph.edges:
            global_graph_edges = global_graph.edges[tuple(coords)].values()
            global_graph_nodes = np.array([edge.to_node for edge in global_graph_edges])
            global_graph_dist = np.array([edge.length for edge in global_graph_edges])
            filtered_global_graph_idx = (global_graph_dist <= global_graph_knn_dist_max) & (global_graph_dist > global_graph_knn_dist_min)   
            filtered_global_graph_nodes = global_graph_nodes[filtered_global_graph_idx]
            filtered_global_graph_dist = global_graph_dist[filtered_global_graph_idx]
            num_global_neighbours = len(filtered_global_graph_nodes) if len(filtered_global_graph_nodes) < self.k_size else self.k_size
            topk_global_graph_nodes = filtered_global_graph_nodes[np.argsort(filtered_global_graph_dist)[:num_global_neighbours]]
            topk_global_graph_nodes = set(map(tuple, topk_global_graph_nodes))  

            for neighbour_node in topk_global_graph_nodes:
                if modify_graph:
                    graph.add_node(tuple(coords))
                    graph.add_edge(tuple(coords), tuple(neighbour_node), np.linalg.norm(coords-neighbour_node))
        
        max_neighbours = self.k_size - num_global_neighbours
        num_neighbours = len(node_coords) if len(node_coords) < max_neighbours else max_neighbours

        for neighbor_index in sorted_index:
            neighbor_index_list.append(neighbor_index)
            start = coords
            end = node_coords[neighbor_index]
            if tuple(end) in topk_global_graph_nodes:   # Don't consider global nodes alr added
                continue

            edge_len = np.linalg.norm(start-end)
            if edge_len > max_edge_len: 
                k += 1  
                continue

            if not self.check_collision(start, end, robot_belief):
                if modify_graph:
                    graph.add_node(tuple(start))
                    graph.add_edge(tuple(start), tuple(end), edge_len)
                count += 1
            k += 1
            if k >= num_neighbours:
                break
        return neighbor_index_list


    def find_k_neighbor_all_nodes_custom(self, robot_belief, node_coords, graph, global_graph=None, global_graph_knn_dist_max=GLOBAL_GRAPH_KNN_RAD, global_graph_knn_dist_min=CUR_AGENT_KNN_RAD):
        """ Find nearest k neighbors to all coords (with more options) """
        
        kd_tree = KDTree(node_coords)
        for i, p in enumerate(node_coords):

            # Append global graph edges to each node first (to ensure connectivity)
            num_global_neighbours = 0
            if global_graph is not None and tuple(p) in global_graph.edges:
                global_graph_edges = global_graph.edges[tuple(p)].values()
                global_graph_nodes = np.array([edge.to_node for edge in global_graph_edges])
                global_graph_dist = np.array([edge.length for edge in global_graph_edges])
                filtered_global_graph_idx = (global_graph_dist <= global_graph_knn_dist_max) & (global_graph_dist > global_graph_knn_dist_min)     
                filtered_global_graph_nodes = global_graph_nodes[filtered_global_graph_idx]
                filtered_global_graph_dist = global_graph_dist[filtered_global_graph_idx]
                num_global_neighbours = len(filtered_global_graph_nodes) if len(filtered_global_graph_nodes) < self.k_size else self.k_size
                topk_global_graph_nodes = filtered_global_graph_nodes[np.argsort(filtered_global_graph_dist)[:num_global_neighbours]]
                topk_global_graph_nodes = set(map(tuple, topk_global_graph_nodes))  

                for neighbour_node in topk_global_graph_nodes:
                    graph.add_node(tuple(p))
                    graph.add_edge(tuple(p), tuple(neighbour_node), np.linalg.norm(p-neighbour_node))

            max_neighbours = self.k_size - num_global_neighbours
            num_neighbours = len(node_coords) if len(node_coords) < max_neighbours else max_neighbours
            if num_neighbours > 0:
                _, indices = kd_tree.query(p, k=num_neighbours)
                if np.isscalar(indices):
                    indices = np.array([indices])
                for j, neighbour in enumerate(node_coords[indices]):     
                    start = p
                    end = neighbour
                    if not self.check_collision(start, end, robot_belief):
                        graph.add_node(tuple(start))
                        graph.add_edge(tuple(start), tuple(end), np.linalg.norm(start-end))


    def find_index_from_coords(self, node_coords, p):
        if len(np.where(np.linalg.norm(node_coords - p, axis=1) < 1e-1)[0]) == 0:
            return -1
        else:
            return np.where(np.linalg.norm(node_coords - p, axis=1) < 1e-1)[0][0]

    def find_closest_index_from_coords(self, node_coords, p):
        return np.argmin(np.linalg.norm(node_coords - p, axis=1))

    def check_collision(self, start, end, robot_belief):

        # # Bresenham line algorithm checking
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

    def find_shortest_path(self, current, destination, node_coords, graph):
        t1 = time()
        start_node = tuple(node_coords[self.find_closest_index_from_coords(node_coords, current)])        
        end_node = tuple(node_coords[self.find_closest_index_from_coords(node_coords, destination)])      
        route, dist, _, _ = a_star(start_node, end_node, graph) 

        if start_node != end_node:
            assert route != []

        elif route is not None:
            route = list(map(tuple, route))
        return dist, route

    def extract_frontier_centers_new(self, robot_belief, robot_location_belief):
        
        global_nodes = np.array([node.coords for node in self.nodes_list if tuple(node.coords) in self.global_graph.nodes])
        global_node_utility = np.array([node.utility for node in self.nodes_list if tuple(node.coords) in self.global_graph.nodes])

        if len(global_nodes > 0):

            center_indices = np.argwhere(np.array(global_node_utility) > MAX_UTILITY_TO_SPARSE)[:, 0].tolist()
            sorted_center_indices = sorted(center_indices, key=lambda idx: global_node_utility[idx], reverse=True)
            centers = global_nodes[sorted_center_indices]

            ### Sparsify centers derivedd from 'non_zero_utility_node_indices' (if enough centers)
            if centers.shape[0] >= MIN_CENTERS_BEFORE_SPARSIFY:
                knn = NearestNeighbors(radius=SPARSIFICATION_CENTERS_KNN_RAD)
                knn.fit(centers)
                key_center_indices = []
                coverd_center_indices = []
                for i, center in enumerate(centers):
                    if i in coverd_center_indices:
                        pass
                    else:
                        _, indices = knn.radius_neighbors(center.reshape(1,2))
                        key_center_indices.append(i)
                        for index in indices[0]:
                            node = centers[index]
                            if not self.check_collision(center, node, robot_belief):
                                coverd_center_indices.append(index)

                center_indices = [self.find_closest_index_from_coords(global_nodes, centers[i]) for i in key_center_indices]
                center_indices = list(set(center_indices))
                centers = global_nodes[center_indices]
            
            if SPARSIFY_RETAIN_AGENT_TO_AGENT_ASTAR:
                center_indices += [self.find_closest_index_from_coords(global_nodes, pose) for pose in robot_location_belief if pose is not None]       
                center_indices = list(set(center_indices))
                centers = global_nodes[center_indices]

            return centers


    def generate_coords_from_map(self, map):
        new_free_area = self.free_area(map)
        free_area_to_check = new_free_area[:, 0] + new_free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        return self.uniform_points[candidate_indices]

