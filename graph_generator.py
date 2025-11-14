#######################################################################
# Name: simplified_graph_generator.py
# 簡化版Graph生成器 - 完全替代graph_generator.py
# 使用方式: 
#   from simplified_graph_generator import Graph_generator
#######################################################################

import sys
if sys.modules['TRAINING']:
    from parameter import *
else:
    from test_parameter import *

import copy
import numpy as np
from sklearn.neighbors import NearestNeighbors
from node import Node
from graph import Graph, a_star
from time import time
from scipy.spatial import KDTree


class Graph_generator:
    """
    簡化版Graph生成器 (類名保持Graph_generator以兼容原接口)
    
    主要簡化:
    1. 使用更稀疏的網格 (grid_resolution=16)
    2. 移除複雜的全局/局部圖合併
    3. 移除圖剪枝和稀疏化
    4. 簡化K-NN連接邏輯
    5. 限制節點數量上限
    
    保持接口完全兼容原版
    """
    
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
        
        # 使用適中密度的網格 (減少 grid_resolution 以增加節點數)
        self.grid_resolution = 8  # 每8像素一個節點 (增加節點密度以改善覆蓋)
        self.uniform_points = self.generate_uniform_points()
        
        self.sensor_range = sensor_range
        self.route_node = []
        self.nodes_list = []
        self.node_utility = None
        self.guidepost = None
        self.file_path = file_path
        self.nodes_not_to_merge = None
        self.frontier_centers = None
        
        # 簡化版:不使用複雜的全局圖
        self.global_graph = Graph()
        self.global_graph_nodes = []


    def edge_clear_all_nodes(self):
        """Re-init graphs"""
        self.graph = Graph()
        self.x = []
        self.y = []

    def edge_clear(self, coords):
        """Clear specific graph edge"""
        self.graph.clear_edge(tuple(coords))

    def node_clear(self, coords, remove_bidirectional_edges=False):
        """Clear specific graph node"""
        self.graph.clear_node(tuple(coords), remove_bidirectional_edges=remove_bidirectional_edges)


    def generate_graph(self, robot_location, robot_belief, frontiers):
        """
        Initialize graphs of map belief
        簡化版:只生成局部稀疏圖
        """
        print(f"[SimplifiedGraphGen Robot {self.robot_id}] Generating initial graph...")
        
        self.edge_clear_all_nodes()
        free_area = self.free_area(robot_belief)

        # 使用更稀疏的網格
        free_area_to_check = free_area[:, 0] + free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        node_coords = self.uniform_points[candidate_indices]
        
        # 限制節點數量 (增加上限以改善地圖覆蓋)
        if len(node_coords) > 800:
            indices = np.random.choice(len(node_coords), 800, replace=False)
            node_coords = node_coords[indices]
        
        node_coords = np.concatenate((robot_location.reshape(1, 2), node_coords))
        self.node_coords = node_coords

        # 簡化版K-NN連接
        self.find_k_neighbor_all_nodes_simplified(robot_belief)

        # 計算節點效用
        self.node_utility = []
        for coords in self.node_coords:
            node = Node(coords, frontiers, robot_belief)
            self.nodes_list.append(node)
            utility = node.utility
            self.node_utility.append(utility)

        self.node_utility = np.array(self.node_utility)
        self.guidepost = np.zeros((self.node_coords.shape[0], 1))
        
        for node in self.route_node:
            index = self.find_closest_index_from_coords(self.node_coords, node)     
            self.guidepost[index] += 1

        print(f"[SimplifiedGraphGen Robot {self.robot_id}] Generated {len(self.node_coords)} nodes")
        return self.node_coords, self.graph.edges, self.node_utility, self.guidepost


    def update_graph(self, robot_belief, frontiers, old_frontiers, robot_location_belief, 
                     robot_global_graph_belief, robot_old_global_graph_belief_len, 
                     extend_global_graph_towards_fronters=False, eps=None, step=None):
        """
        Update graphs of map belief
        簡化版:移除複雜的全局圖合併和稀疏化
        """
        
        # 簡化版:只更新局部節點
        robot_location = robot_location_belief[self.robot_id]
        
        # 生成當前機器人周圍的局部節點 - FIX: Convert to integers
        height, width = robot_belief.shape
        x0, x1 = max(0, int(robot_location[0]) - CUR_AGENT_KNN_RAD), min(width, int(robot_location[0]) + CUR_AGENT_KNN_RAD)
        y0, y1 = max(0, int(robot_location[1]) - CUR_AGENT_KNN_RAD), min(height, int(robot_location[1]) + CUR_AGENT_KNN_RAD)
        
        filtered_belief = np.zeros_like(robot_belief)
        filtered_belief[y0:y1, x0:x1] = robot_belief[y0:y1, x0:x1]
        
        local_free_area = self.free_area(filtered_belief)
        
        if len(local_free_area) == 0:
            # 如果沒有自由空間,保持原有節點
            success = True
            return success, self.node_coords, self.graph.edges, self.node_utility, self.guidepost
        
        local_free_to_check = local_free_area[:, 0] + local_free_area[:, 1] * 1j
        uniform_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(local_free_to_check, uniform_to_check, return_indices=True)
        local_node_coords = self.uniform_points[candidate_indices]
        
        # 限制節點數 (增加上限以改善探索效率)
        if len(local_node_coords) > 500:
            indices = np.random.choice(len(local_node_coords), 500, replace=False)
            local_node_coords = local_node_coords[indices]
        
        # 合併舊節點和新節點
        old_node_coords = copy.deepcopy(self.node_coords)
        self.node_coords = self.unique_coords(np.vstack([robot_location.reshape(1, 2), local_node_coords]))
        
        # 找出新增和刪除的節點
        coords_old_not_in_new = set(map(tuple, old_node_coords)) - set(map(tuple, self.node_coords))
        coords_new_not_in_old = set(map(tuple, self.node_coords)) - set(map(tuple, old_node_coords))
        
        # 更新nodes_list
        coords_old_not_in_new_tuples = [tuple(coords) for coords in coords_old_not_in_new]
        
        # FIXED: Use numpy array operations to properly filter
        old_node_coords_list = list(old_node_coords)
        new_node_coords_list = []
        new_nodes_list = []
        
        for i, coord in enumerate(old_node_coords_list):
            if tuple(coord) not in coords_old_not_in_new_tuples:
                new_node_coords_list.append(coord)
                new_nodes_list.append(self.nodes_list[i])
        
        # Add new nodes
        for coord in coords_new_not_in_old:
            new_node_coords_list.append(list(coord))
        
        self.node_coords = np.array(new_node_coords_list)
        self.nodes_list = new_nodes_list
        
        # 更新已有節點的效用
        if len(old_frontiers) > 0 and len(frontiers) > 0:
            observed_frontiers_to_check = old_frontiers[:, 0] + old_frontiers[:, 1] * 1j
            new_frontiers_to_check = frontiers[:, 0] + frontiers[:, 1] * 1j
            observed_frontiers_index = np.where(
                np.isin(observed_frontiers_to_check, new_frontiers_to_check, assume_unique=True) == False)
            new_frontiers_index = np.where(
                np.isin(new_frontiers_to_check, observed_frontiers_to_check, assume_unique=True) == False)
            observed_frontiers = old_frontiers[observed_frontiers_index]
            new_frontiers = frontiers[new_frontiers_index]
            observed_frontiers_set = set(map(tuple, observed_frontiers))
            
            for node in self.nodes_list:
                if len(new_frontiers) > 0:
                    dist_new = np.linalg.norm((new_frontiers - np.array(node.coords)), axis=1)
                    close_new = new_frontiers[dist_new < UTILITY_CALC_RANGE]
                else:
                    close_new = []
                
                if len(old_frontiers) > 0:
                    dist_old = np.linalg.norm((old_frontiers - np.array(node.coords)), axis=1)
                    close_old = old_frontiers[dist_old < UTILITY_CALC_RANGE]
                else:
                    close_old = []
                
                no_changed = (len(close_new) == 0 and len(close_old) == 0)
                
                if node.zero_utility_node or no_changed:
                    pass
                else:
                    node.update_observable_frontiers(observed_frontiers_set, new_frontiers, robot_belief)
        
        # 添加新節點
        self.nodes_list += [Node(coord, frontiers, robot_belief) for coord in coords_new_not_in_old]
        
        # 重建圖edges - 先計算哪些節點需要更新
        graph_coords_old_not_in_new = set(map(tuple, self.graph.nodes)) - set(map(tuple, self.node_coords))
        graph_coords_new_not_in_old = set(map(tuple, self.node_coords)) - set(map(tuple, self.graph.nodes))
        
        # 清除舊的，為所有當前節點重建
        if len(graph_coords_new_not_in_old) > 0 or len(graph_coords_old_not_in_new) > 0:
            # Clear old graph completely
            self.edge_clear_all_nodes()
            
            # Rebuild K-NN connections for all nodes
            self.find_k_neighbor_all_nodes_simplified(robot_belief)
            
            # Verify all nodes in node_coords have corresponding entries in graph
            for coord in self.node_coords:
                coord_tuple = tuple(coord)
                if coord_tuple not in self.graph.edges:
                    # Add node to graph even if it has no edges
                    self.graph.add_node(coord_tuple)
                    self.graph.edges[coord_tuple] = dict()
        
        # 輸出
        self.node_utility = []
        for i, coords in enumerate(self.node_coords):
            utility = self.nodes_list[i].utility
            self.node_utility.append(utility)
        self.node_utility = np.array(self.node_utility)
        
        self.guidepost = np.zeros((self.node_coords.shape[0], 1))
        for node in self.route_node:
            index = self.find_closest_index_from_coords(self.node_coords, node)
            self.guidepost[index] += 1
        
        success = True
        return success, self.node_coords, self.graph.edges, self.node_utility, self.guidepost


    def find_k_neighbor_all_nodes_simplified(self, robot_belief):
        """
        簡化版K-NN連接
        只連接最近的k個鄰居,不考慮全局圖
        """
        if len(self.node_coords) < 2:
            # Still add the single node to graph
            if len(self.node_coords) == 1:
                self.graph.add_node(tuple(self.node_coords[0]))
                self.graph.edges[tuple(self.node_coords[0])] = dict()
            return
        
        kd_tree = KDTree(self.node_coords)
        
        for i, p in enumerate(self.node_coords):
            # Always add node to graph
            self.graph.add_node(tuple(p))
            if tuple(p) not in self.graph.edges:
                self.graph.edges[tuple(p)] = dict()
            
            num_neighbors = min(self.k_size, len(self.node_coords))
            
            if num_neighbors > 1:
                _, indices = kd_tree.query(p, k=num_neighbors)
                if np.isscalar(indices):
                    indices = np.array([indices])
                
                count = 0
                for j in indices:
                    if i != j:
                        neighbor = self.node_coords[j]
                        
                        # 檢查碰撞
                        if not self.check_collision(p, neighbor, robot_belief):
                            self.graph.add_edge(tuple(p), tuple(neighbor), np.linalg.norm(p - neighbor))
                            count += 1
                            
                            if self.plot:
                                self.x.append([p[0], neighbor[0]])
                                self.y.append([p[1], neighbor[1]])

    def find_k_neighbor_all_nodes(self, robot_belief, update_dense=False, global_graph=None, 
                                   global_graph_knn_dist_max=None, global_graph_knn_dist_min=None):
        """
        兼容性包裝器：調用簡化版的 K-NN 方法
        原版有很多複雜參數，簡化版忽略這些參數，只做基本連接
        """
        print(f"[SimplifiedGraphGen Robot {self.robot_id}] Called find_k_neighbor_all_nodes (using simplified version)")
        # 忽略所有額外參數，直接調用簡化版
        return self.find_k_neighbor_all_nodes_simplified(robot_belief)


    def generate_uniform_points(self):
        """
        Generate uniform grid in free space of map belief
        使用更稀疏的網格
        """
        x = np.linspace(0, self.map_x - 1, self.map_x // self.grid_resolution).round().astype(int)
        y = np.linspace(0, self.map_y - 1, self.map_y // self.grid_resolution).round().astype(int)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points


    def free_area(self, robot_belief):
        """Identify free space in map belief"""
        index = np.where(robot_belief == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def unique_coords(self, coords):
        """Remove duplicates in node coords"""
        x = coords[:, 0] + coords[:, 1] * 1j
        indices = np.unique(x, return_index=True)[1]
        coords = np.array([coords[idx] for idx in sorted(indices)])
        return coords

    def find_index_from_coords(self, node_coords, p):
        matches = np.where(np.linalg.norm(node_coords - p, axis=1) < 1e-1)[0]
        if len(matches) == 0:
            return -1
        else:
            return matches[0]

    def find_closest_index_from_coords(self, node_coords, p):
        return np.argmin(np.linalg.norm(node_coords - p, axis=1))

    def check_collision(self, start, end, robot_belief):
        """Bresenham line algorithm checking"""
        collision = False
        map = robot_belief 

        x0 = int(start[0])
        y0 = int(start[1])
        x1 = int(end[0])
        y1 = int(end[1])
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
        """
        A*路徑規劃
        如果原graph失敗,使用簡化的直接搜索
        """
        start_node = tuple(node_coords[self.find_closest_index_from_coords(node_coords, current)])
        end_node = tuple(node_coords[self.find_closest_index_from_coords(node_coords, destination)])
        
        route, dist, _, _ = a_star(start_node, end_node, graph)
        
        # 如果A*失敗,嘗試確保圖連通
        if route is None:
            print(f"[SimplifiedGraphGen Robot {self.robot_id}] A* failed, trying bidirectional edges...")
            # 嘗試添加雙向edges
            temp_graph = copy.deepcopy(graph)
            for node in temp_graph.nodes:
                if tuple(node) in temp_graph.edges:
                    for edge in temp_graph.edges[tuple(node)].values():
                        temp_graph.add_edge(edge.to_node, node, edge.length)
            
            route, dist, _, _ = a_star(start_node, end_node, temp_graph)
            
            if route is not None:
                print(f"[SimplifiedGraphGen Robot {self.robot_id}] Path found with bidirectional edges!")
        
        if start_node != end_node and route is not None:
            route = list(map(tuple, route))
        
        return dist, route

    def generate_coords_from_map(self, map):
        """從地圖生成座標點"""
        new_free_area = self.free_area(map)
        if len(new_free_area) == 0:
            return np.array([]).reshape(0, 2)
        free_area_to_check = new_free_area[:, 0] + new_free_area[:, 1] * 1j
        uniform_points_to_check = self.uniform_points[:, 0] + self.uniform_points[:, 1] * 1j
        _, _, candidate_indices = np.intersect1d(free_area_to_check, uniform_points_to_check, return_indices=True)
        return self.uniform_points[candidate_indices]

    # 以下是保持接口兼容但簡化實現的方法
    
    def merge_global_graph(self, robot_belief, frontiers, robot_location_belief, global_graph_unique_radius):
        """簡化版:不執行複雜的圖合併"""
        print(f"[SimplifiedGraphGen Robot {self.robot_id}] Skipping global graph merge (simplified version)")
        pass

    def prune_global_graph(self, robot_belief, robot_location_belief, centers, eps=None):
        """簡化版:不執行圖剪枝"""
        print(f"[SimplifiedGraphGen Robot {self.robot_id}] Skipping global graph pruning (simplified version)")
        success = True
        return success

    def extract_frontier_centers_new(self, robot_belief, robot_location_belief):
        """簡化版:返回空"""
        return np.array([])