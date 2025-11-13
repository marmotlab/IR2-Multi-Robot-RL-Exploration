##############################################################################
# Name: test_driver_custom_start.py
# [Inference] Driver for testing with custom start points
# Allows testing multiple starting positions for multi-robot exploration
###############################################################################

from test_parameter import *
import numpy as np
import os
import torch
import csv
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import PolicyNet
from test_multi_robot_worker import TestWorker
from datetime import datetime
from env import Env
from robot import Robot
import time
import copy

# 每10步保存一次图片
SAVE_IMAGE_INTERVAL = 10


class RobotIndividualMapTracker:
    """
    追蹤並紀錄兩個機器人的個人探索地圖（只包含自己探索的區域）
    基於 Env 的 robot_belief 來追踪
    """

    def __init__(self, env, robot_list, save_dir='robot_individual_maps'):
        """
        初始化追蹤器

        參數:
            env: 環境實例（包含所有地圖信息）
            robot_list: 機器人列表
            save_dir: 保存地圖的目錄
        """
        self.env = env
        self.robot_list = robot_list
        self.n_robots = len(robot_list)
        self.save_dir = save_dir

        # 確保保存目錄存在
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 初始化每個機器人的個人地圖（只有自己探索的區域）
        self.robot_individual_maps = []

        # 記錄地圖的歷史
        self.robots_maps_history = []

        # 是否正在追蹤
        self.is_tracking = False

    def start_tracking(self):
        """開始追蹤個人地圖"""
        self.is_tracking = True

        # 初始化個人地圖為未知區域 (值為127)
        map_shape = self.env.ground_truth.shape
        self.robot_individual_maps = []
        for i in range(self.n_robots):
            individual_map = np.ones(map_shape) * 127
            self.robot_individual_maps.append(individual_map)

        # 清空地圖歷史
        self.robots_maps_history = [[] for _ in range(self.n_robots)]

    def update(self):
        """更新所有機器人的個人探索地圖"""
        if not self.is_tracking:
            return

        # 檢查是否已經初始化
        if not self.robot_individual_maps:
            self.start_tracking()

        # 更新每個機器人的個人地圖
        for robot_id in range(self.n_robots):
            # 從 env 獲取機器人的 belief（這是機器人自己觀察到的地圖）
            robot_belief = self.env.all_robot_belief[robot_id][robot_id].copy()

            # 更新個人地圖：只記錄這個機器人自己探索的部分
            # robot_belief 中不為 127 的部分就是該機器人探索過的區域
            explored_mask = (robot_belief != 127)
            self.robot_individual_maps[robot_id][explored_mask] = robot_belief[explored_mask]

            # 保存到歷史記錄
            self.robots_maps_history[robot_id].append(self.robot_individual_maps[robot_id].copy())

    def calculate_overlap(self):
        """計算兩個機器人探索區域的重疊程度"""
        if self.n_robots != 2 or not self.robot_individual_maps:
            return 0

        # 計算兩個機器人都探索過的區域
        robot1_explored = (self.robot_individual_maps[0] != 127)
        robot2_explored = (self.robot_individual_maps[1] != 127)

        overlap = np.sum(robot1_explored & robot2_explored)

        # 計算任一機器人探索過的區域
        any_explored = np.sum(robot1_explored | robot2_explored)

        # 計算重疊比例
        overlap_ratio = overlap / any_explored if any_explored > 0 else 0

        return overlap_ratio

    def plot_coverage_over_time(self):
        """
        繪製覆蓋率隨時間變化的圖表
        """
        try:
            print(f"{GREEN}[調試] plot_coverage_over_time 被調用{NC}")
            print(f"[調試] n_robots={self.n_robots}, history長度={len(self.robots_maps_history[0]) if self.robots_maps_history else 0}")

            if self.n_robots != 2 or not self.robots_maps_history[0]:
                print(f"{YELLOW}[調試] 條件不滿足，退出plot_coverage_over_time{NC}")
                return
        except Exception as e:
            print(f"{RED}[錯誤] plot_coverage_over_time 初始檢查失敗: {e}{NC}")
            import traceback
            traceback.print_exc()
            return

        try:
            # 計算每個時間點的覆蓋率指標
            time_steps = range(len(self.robots_maps_history[0]))
            robot1_coverage = []
            robot2_coverage = []
            intersection_coverage = []
            union_coverage = []

            # 計算全局地圖的可探索區域總數（255 = 可探索空間）
            total_explorable = np.sum(self.env.ground_truth == 255)

            if total_explorable == 0:
                total_explorable = self.env.ground_truth.size

            print(f"[調試] 開始計算覆蓋率數據，共 {len(time_steps)} 個時間點")

            for i in time_steps:
                # 獲取每個時間點的地圖
                robot1_map = self.robots_maps_history[0][i]
                robot2_map = self.robots_maps_history[1][i]

                # 計算已探索區域（值不為127的區域）
                robot1_explored = (robot1_map != 127)
                robot2_explored = (robot2_map != 127)

                # 計算交集（兩個機器人都探索的區域）
                intersection = np.logical_and(robot1_explored, robot2_explored)

                # 計算聯集（至少一個機器人探索的區域）
                union = np.logical_or(robot1_explored, robot2_explored)

                # 計算覆蓋率
                robot1_ratio = np.sum(robot1_explored) / total_explorable
                robot2_ratio = np.sum(robot2_explored) / total_explorable
                intersection_ratio = np.sum(intersection) / total_explorable
                union_ratio = np.sum(union) / total_explorable

                # 保存數據
                robot1_coverage.append(robot1_ratio)
                robot2_coverage.append(robot2_ratio)
                intersection_coverage.append(intersection_ratio)
                union_coverage.append(union_ratio)

            print(f"[調試] 覆蓋率數據計算完成")

            # 創建圖表
            plt.figure(figsize=(12, 8))

            # 繪製各條曲線
            plt.plot(time_steps, robot1_coverage, 'b-', linewidth=2, label='Robot 1')
            plt.plot(time_steps, robot2_coverage, 'r-', linewidth=2, label='Robot 2')
            plt.plot(time_steps, intersection_coverage, 'g-', linewidth=2, label='intersection')
            plt.plot(time_steps, union_coverage, 'k-', linewidth=2, label='union')

            # 添加標籤和標題
            plt.xlabel('time(steps)', fontsize=14)
            plt.ylabel('coverage', fontsize=14)
            plt.title('time-coverage', fontsize=16)

            # 添加網格和圖例
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)

            # 設置y軸範圍
            plt.ylim(0, 1.05)

            print(f"[調試] 圖表繪製完成，準備保存...")

            # 保存圖片
            coverage_plot_path = os.path.join(self.save_dir, 'coverage_over_time.png')
            plt.savefig(coverage_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"{GREEN}[調試] 已保存圖表: {coverage_plot_path}{NC}")

            # 保存數據到CSV文件
            csv_path = os.path.join(self.save_dir, 'coverage_data.csv')
            df = pd.DataFrame({
                'Time': time_steps,
                'Robot1_Coverage': robot1_coverage,
                'Robot2_Coverage': robot2_coverage,
                'Intersection': intersection_coverage,
                'Union': union_coverage
            })
            df.to_csv(csv_path, index=False)
            print(f"{GREEN}[調試] 已保存CSV: {csv_path}{NC}")

        except Exception as e:
            print(f"{RED}[錯誤] 生成圖表時出錯: {e}{NC}")
            import traceback
            traceback.print_exc()

def find_nearest_free_space(ground_truth, position, max_search_radius=50):
    """找到距離給定位置最近的自由空間

    參數:
        ground_truth: 地圖真實數據 (255=自由, 1=障礙物, 127=未知)
        position: 起始位置 [x, y]
        max_search_radius: 最大搜索半徑

    返回:
        nearest_free: 最近的自由空間位置 [x, y]
    """
    x, y = int(position[0]), int(position[1])
    map_height, map_width = ground_truth.shape

    # 如果當前位置就是自由空間，直接返回
    if ground_truth[y, x] == 255:
        return position.copy()

    # 使用螺旋搜索找最近的自由空間
    for radius in range(1, max_search_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                # 只檢查當前半徑圓周上的點
                if abs(dx) != radius and abs(dy) != radius:
                    continue

                nx, ny = x + dx, y + dy

                # 檢查是否在地圖範圍內
                if 0 <= nx < map_width and 0 <= ny < map_height:
                    # 檢查是否為自由空間
                    if ground_truth[ny, nx] == 255:
                        print(f"{YELLOW}找到最近的自由空間: ({nx}, {ny})，距離原位置 {np.sqrt(dx**2 + dy**2):.1f} pixels{NC}")
                        return np.array([float(nx), float(ny)])

    # 如果沒找到，返回原位置並警告
    print(f"{RED}警告: 在 {max_search_radius} 像素範圍內未找到自由空間，使用原位置{NC}")
    return position.copy()


def create_custom_env_and_robots(map_index, n_agent, k_size, custom_start_pos=None, plot=False):
    """創建具有自定義起始位置的環境和機器人

    參數:
        map_index: 地圖索引
        n_agent: 機器人數量
        k_size: K大小
        custom_start_pos: 自定義起始位置 [x, y]，兩個機器人將從同一位置開始
        plot: 是否繪圖

    返回:
        env: 環境實例
        robot_list: 機器人列表
        all_robot_positions: 所有機器人位置列表
    """
    # 創建環境
    env = Env(map_index=map_index, n_agent=n_agent, k_size=k_size, plot=plot)

    # 如果提供了自定義起始位置，修改環境的起始位置並重新初始化
    if custom_start_pos is not None:
        custom_start_pos = np.array(custom_start_pos, dtype=np.float64)

        # 驗證位置是否在地圖範圍內
        map_height, map_width = env.ground_truth_size
        custom_start_pos[0] = np.clip(custom_start_pos[0], 0, map_width-1)
        custom_start_pos[1] = np.clip(custom_start_pos[1], 0, map_height-1)

        print(f"{YELLOW}[調試] 檢查起始位置 {custom_start_pos}...{NC}")
        print(f"{YELLOW}[調試] 地圖值: {env.ground_truth[int(custom_start_pos[1]), int(custom_start_pos[0])]}{NC}")

        # 驗證位置是否為自由空間 (255 = 自由空間)
        if env.ground_truth[int(custom_start_pos[1]), int(custom_start_pos[0])] != 255:
            print(RED, f"警告: 指定位置 {custom_start_pos} 不是自由空間 (值={env.ground_truth[int(custom_start_pos[1]), int(custom_start_pos[0])]})", NC)
            print(YELLOW, "正在尋找最近的自由空間...", NC)

            # 找到最近的自由空間
            custom_start_pos = find_nearest_free_space(env.ground_truth, custom_start_pos)
            print(GREEN, f"已移動到最近的自由空間: {custom_start_pos}", NC)
        else:
            print(GREEN, f"起始位置驗證通過，位於自由空間", NC)

        # 設置所有機器人的起始位置為相同位置
        env.start_position = custom_start_pos
        for i in range(n_agent):
            env.all_robot_positions_belief[i] = [custom_start_pos.copy() for _ in range(n_agent)]
            env.all_graph_generator[i].route_node = [custom_start_pos.copy()]

        # 重新初始化環境以使用新的起始位置
        env.begin()

        print(GREEN, f"使用自定義起始位置: {custom_start_pos}", NC)

    # 創建機器人列表
    robot_list = []
    all_robot_positions = []

    for i in range(n_agent):
        if custom_start_pos is not None:
            # 如果有自定義起始位置，所有機器人都從相同位置開始
            robot_position = custom_start_pos.copy()
        else:
            # 否則使用環境默認的node_coords
            iter = min(i, len(env.all_node_coords[i])-1)   # In case idx out of bounds
            robot_position = env.all_node_coords[i][iter]

        robot = Robot(robot_id=i, position=robot_position, plot=plot)
        robot_list.append(robot)
        all_robot_positions.append(robot_position)

    return env, robot_list, all_robot_positions


class CustomTestWorker(TestWorker):
    """繼承TestWorker並允許自定義環境"""

    def __init__(self, meta_agent_id, n_agent, policy_net, global_step, device='cuda',
                 greedy=False, save_image=False, custom_start_pos=None, output_dir=None, map_index=None):
        self.device = device
        self.greedy = greedy
        self.n_agent = n_agent
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image
        self.output_dir = output_dir

        # 創建輸出目錄（如果需要保存圖片）
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 使用自定義起始位置創建環境和機器人
        # 如果指定了map_index，使用指定的地圖；否則使用global_step作為地圖索引
        actual_map_index = map_index if map_index is not None else self.global_step
        self.env, self.robot_list, self.all_robot_positions = create_custom_env_and_robots(
            map_index=actual_map_index,
            n_agent=self.n_agent,
            k_size=self.k_size,
            custom_start_pos=custom_start_pos,
            plot=save_image
        )

        self.local_policy_net = policy_net
        self.perf_metrics = dict()
        self.max_node_coords = 0

        # 初始化個人地圖追蹤器
        self.map_tracker = None
        if self.n_agent == 2 and self.output_dir:  # 只在2個機器人且需要保存時啟用
            tracker_dir = os.path.join(self.output_dir, 'individual_maps')
            self.map_tracker = RobotIndividualMapTracker(
                self.env,
                self.robot_list,
                save_dir=tracker_dir
            )
            print(f"{GREEN}[調試] 已創建map_tracker，保存目錄: {tracker_dir}{NC}")
        else:
            print(f"{YELLOW}[調試] 未創建map_tracker (n_agent={self.n_agent}, output_dir存在={self.output_dir is not None}){NC}")

    def save_current_state(self, step, travel_dist_list):
        """保存當前狀態的圖片"""
        if not self.save_image or not self.output_dir:
            return

        # 保存每個機器人的視圖
        for robot_id, robot in enumerate(self.robot_list):
            robot.save_robot_position()
            robots_route = [[robot.xPoints, robot.yPoints]]

            robot_img_dir = os.path.join(self.output_dir, f'robot_{robot_id+1}')
            if not os.path.exists(robot_img_dir):
                os.makedirs(robot_img_dir)

            self.env.plot_env(
                self.global_step,
                robot_img_dir,
                step,
                travel_dist_list[robot_id],
                robots_route,
                robot_id
            )

        # 保存合併視圖
        robots_route = []
        for robot in self.robot_list:
            robots_route.append([robot.xPoints, robot.yPoints])

        merged_dir = os.path.join(self.output_dir, 'merged')
        if not os.path.exists(merged_dir):
            os.makedirs(merged_dir)

        self.env.plot_env_ground_truth(
            self.global_step,
            merged_dir,
            step,
            max(travel_dist_list),
            robots_route
        )

    def run_episode(self, curr_episode):
        """運行episode - 每10步保存圖片"""
        done = False
        astar_unsuccessful = False

        if self.save_image:
            print(f"\n{GREEN}{'='*70}{NC}")
            print(f"{GREEN}開始Episode {curr_episode}{NC}")
            print(f"  保存間隔: 每 {SAVE_IMAGE_INTERVAL} 步")
            print(f"  輸出目錄: {self.output_dir}")
            print(f"{GREEN}{'='*70}{NC}\n")

        # 啟動個人地圖追蹤
        if self.map_tracker is not None:
            self.map_tracker.start_tracking()
            print(f"{GREEN}[調試] 已啟動map_tracker追蹤{NC}")

        start_time = time.time()
        last_save_time = start_time

        step = 0
        while not done:
            reward_list = []
            travel_dist_list = []

            for robot_id, deciding_robot in enumerate(self.robot_list):
                # 更新圖和效用
                success = self.env.update_graph(
                    robot_id,
                    self.env.find_frontier(self.env.all_downsampled_belief[robot_id]),
                    eps=self.global_step,
                    step=step
                )
                if not success:
                    astar_unsuccessful = True
                    break

                deciding_robot.observations, success = self.get_observations(
                    deciding_robot.robot_position,
                    robot_id,
                    curr_episode,
                    step,
                    plot=self.save_image
                )
                if not success:
                    astar_unsuccessful = True
                    break

                # 檢查是否需要新目標
                need_new_target = False
                if deciding_robot.target_position is None:
                    need_new_target = True
                else:
                    distance_to_target = np.linalg.norm(deciding_robot.target_position - deciding_robot.robot_position)
                    if distance_to_target < 2.0:
                        need_new_target = True

                # 獲取新目標
                if need_new_target:
                    next_target_position, action_index = self.select_node(deciding_robot.observations, robot_id)
                    target_dist = np.linalg.norm(next_target_position - deciding_robot.robot_position)
                    if target_dist > 1.0:
                        deciding_robot.target_position = next_target_position
                    else:
                        if deciding_robot.target_position is None:
                            deciding_robot.target_position = deciding_robot.robot_position

                # 移動
                direction = deciding_robot.target_position - deciding_robot.robot_position
                distance_to_target = np.linalg.norm(direction)
                step_size = 2.0

                if distance_to_target > step_size:
                    normalized_direction = direction / distance_to_target
                    next_position = deciding_robot.robot_position + normalized_direction * step_size
                    dist_travelled = step_size
                else:
                    next_position = deciding_robot.target_position
                    dist_travelled = distance_to_target

                deciding_robot.travel_dist += dist_travelled
                deciding_robot.robot_position = next_position

                travel_dist_list.append(deciding_robot.travel_dist)
                self.all_robot_positions[robot_id] = next_position
                self.env.all_robot_positions_belief[robot_id][robot_id] = next_position
                self.env.all_robot_positions_step_updated[robot_id][robot_id] = step

                # 執行環境步驟
                success, reward, done = self.env.single_robot_step(
                    robot_id,
                    self.all_robot_positions,
                    self.global_step,
                    step,
                    dist_travelled
                )
                if not success:
                    astar_unsuccessful = True
                    break
                reward_list.append(reward)

                deciding_robot.observations, success = self.get_observations(
                    deciding_robot.robot_position,
                    robot_id,
                    curr_episode,
                    step,
                    plot=self.save_image
                )
                if not success:
                    astar_unsuccessful = True
                    break

            if astar_unsuccessful:
                if self.save_image:
                    print(f"{RED}A*路徑規劃失敗，在步數 {step}{NC}")
                break

            team_reward = self.env.update_env_and_get_team_rewards()
            for i in range(len(reward_list)):
                reward_list[i] += team_reward
                self.robot_list[i].save_reward_done(reward_list[i], done)

            # 更新個人地圖追蹤
            if self.map_tracker is not None:
                self.map_tracker.update()

            # 每10步保存一次圖片並顯示進度
            if self.save_image and (step % SAVE_IMAGE_INTERVAL == 0 or step == 0):
                self.save_current_state(step, travel_dist_list)

                current_time = time.time()
                elapsed_total = current_time - start_time
                elapsed_interval = current_time - last_save_time
                last_save_time = current_time

                print(f"{GREEN}步數 {step:4d} | "
                      f"探索率: {self.env.explored_rate:6.2%} | "
                      f"最大距離: {max(travel_dist_list):7.2f} | "
                      f"用時: {elapsed_total:6.1f}s | "
                      f"間隔: {elapsed_interval:4.1f}s{NC}")

            if done:
                if self.save_image:
                    print(f"\n{GREEN}✓ Episode完成！總步數: {step}{NC}")
                    # 保存最終狀態
                    self.save_current_state(step, travel_dist_list)
                break

            step += 1

            # 安全上限：避免無限循環
            if step >= MAX_EPS_STEPS:
                if self.save_image:
                    print(f"{YELLOW}達到最大步數限制 {MAX_EPS_STEPS}，停止{NC}")
                break

        if astar_unsuccessful:
            return False

        # 保存性能指標
        self.perf_metrics['travel_dist'] = max(travel_dist_list)
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done
        self.perf_metrics['connectivity_rate'] = self.env.connectivity_rate
        self.perf_metrics['agents_connected_percentage'] = self.env.agents_connected_percentage
        self.perf_metrics['travel_steps'] = step + 1

        # 計算並保存overlap ratio
        if self.map_tracker is not None:
            overlap_ratio = self.map_tracker.calculate_overlap()
            self.perf_metrics['overlap_ratio'] = overlap_ratio
            print(f"{GREEN}[調試] 計算出overlap_ratio: {overlap_ratio:.2%}{NC}")

            # 生成覆蓋率圖表
            if self.save_image:
                print(f"{GREEN}[調試] 正在生成覆蓋率圖表...{NC}")
                self.map_tracker.plot_coverage_over_time()
                print(f"{GREEN}[調試] 覆蓋率圖表生成完成{NC}")
            else:
                print(f"{YELLOW}[調試] 跳過圖表生成 (save_image={self.save_image}){NC}")
        else:
            self.perf_metrics['overlap_ratio'] = 0
            print(f"{YELLOW}[調試] map_tracker為None，無法計算overlap_ratio{NC}")

        if self.save_image:
            elapsed = time.time() - start_time
            print(f"\n{GREEN}{'='*70}{NC}")
            print(f"{GREEN}Episode {curr_episode} 完成統計:{NC}")
            print(f"  總步數: {step + 1}")
            print(f"  探索率: {self.env.explored_rate:.2%}")
            print(f"  最大距離: {max(travel_dist_list):.2f}")
            print(f"  成功完成: {done}")
            print(f"  Overlap Ratio: {self.perf_metrics.get('overlap_ratio', 0):.2%}")
            print(f"  總用時: {elapsed:.1f}s")
            print(f"  保存圖片數: {(step // SAVE_IMAGE_INTERVAL) + 1}")
            print(f"{GREEN}{'='*70}{NC}")

        return True


def run_test_with_custom_start_points(global_network, device, start_points_list, output_dir='results_custom_start', map_index=None):
    """使用多個自定義起始點運行測試

    參數:
        global_network: 訓練好的策略網絡
        device: 計算設備
        start_points_list: 起始點列表，每個元素是 [x, y] 坐標
        output_dir: 輸出目錄
        map_index: 指定使用的地圖索引，如果為None則每次測試使用不同地圖
    """

    # 確保輸出目錄存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 創建CSV文件記錄數據
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_file_name = f"custom_start_data_{current_datetime}.csv"
    csv_file_path = os.path.join(output_dir, csv_file_name)

    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ['start_point_idx', 'start_x', 'start_y', 'eps', 'num_robots',
                     'max_dist', 'steps', 'explored', 'success', 'connectivity', 'overlap_ratio']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    weights = global_network.state_dict()

    # 為每個起始點運行測試
    all_results = []

    for start_idx, start_pos in enumerate(start_points_list):
        print(f"\n{GREEN}===== 測試起始點 {start_idx+1}/{len(start_points_list)} ====={NC}")
        print(f"{GREEN}起始位置: {start_pos}{NC}")

        # 創建當前起始點的輸出目錄
        current_output_dir = os.path.join(output_dir, f'start_point_{start_idx+1}')
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)

        # 為當前起始點運行多次測試
        start_point_results = []

        for test_run in range(NUM_TEST):
            n_agent = np.random.randint(NUM_ROBOTS_MIN, NUM_ROBOTS_MAX+1, 1)[0]

            # 創建使用自定義起始點的worker
            worker = CustomTestWorker(
                meta_agent_id=0,
                n_agent=n_agent,
                policy_net=global_network,
                global_step=test_run,
                device=device,
                save_image=SAVE_GIFS and test_run == 0,  # 只保存第一次運行的GIF
                greedy=True,
                custom_start_pos=start_pos,
                output_dir=current_output_dir,  # 傳遞輸出目錄用於圖片保存
                map_index=map_index  # 傳遞指定的地圖索引
            )

            # 運行測試
            print(f"{YELLOW}[調試] 開始運行測試 {test_run+1}...{NC}")
            success = worker.work(test_run)
            print(f"{YELLOW}[調試] 測試 {test_run+1} 完成，結果: {success}{NC}")

            if success:
                perf_metrics = worker.perf_metrics

                # 記錄數據
                result = {
                    'start_point_idx': start_idx + 1,
                    'start_x': start_pos[0],
                    'start_y': start_pos[1],
                    'eps': test_run,
                    'num_robots': n_agent,
                    'max_dist': perf_metrics['travel_dist'],
                    'steps': perf_metrics['travel_steps'],
                    'explored': perf_metrics['explored_rate'],
                    'success': perf_metrics['success_rate'],
                    'connectivity': perf_metrics['connectivity_rate'],
                    'overlap_ratio': perf_metrics.get('overlap_ratio', 0)
                }

                start_point_results.append(result)

                # 寫入CSV
                with open(csv_file_path, mode='a', newline='') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writerow(result)

                print(GREEN, f"[起始點 {start_idx+1} | 測試 {test_run+1}] " +
                      f"步數: {perf_metrics['travel_steps']}, " +
                      f"探索率: {perf_metrics['explored_rate']:.2%}, " +
                      f"最大距離: {perf_metrics['travel_dist']:.2f}", NC)
            else:
                print(RED, f"[起始點 {start_idx+1} | 測試 {test_run+1}] 測試失敗", NC)

        all_results.append({
            'start_point_idx': start_idx + 1,
            'start_pos': start_pos,
            'results': start_point_results
        })

        # 計算當前起始點的統計數據
        if start_point_results:
            avg_explored = np.mean([r['explored'] for r in start_point_results])
            avg_steps = np.mean([r['steps'] for r in start_point_results])
            avg_max_dist = np.mean([r['max_dist'] for r in start_point_results])
            success_rate = np.mean([r['success'] for r in start_point_results])

            print(f"\n{GREEN}起始點 {start_idx+1} 統計數據:{NC}")
            print(f"  平均探索率: {avg_explored:.2%}")
            print(f"  平均步數: {avg_steps:.0f}")
            print(f"  平均最大距離: {avg_max_dist:.2f}")
            print(f"  成功率: {success_rate:.2%}")

    # 生成對比圖表
    generate_comparison_charts(all_results, output_dir)

    print(f"\n{GREEN}===== 完成所有起始點的測試 ====={NC}")
    print(f"結果儲存在: {output_dir}")
    print(f"數據儲存在: {csv_file_path}")

    return all_results


def generate_comparison_charts(all_results, output_dir):
    """生成對比圖表

    參數:
        all_results: 所有測試結果
        output_dir: 輸出目錄
    """

    if not all_results:
        print("沒有結果可以生成圖表")
        return

    # 準備數據
    start_points = []
    avg_explored_rates = []
    avg_steps = []
    avg_max_dists = []
    success_rates = []

    for result_set in all_results:
        results = result_set['results']
        if results:
            start_points.append(f"Point {result_set['start_point_idx']}\n({result_set['start_pos'][0]:.0f},{result_set['start_pos'][1]:.0f})")
            avg_explored_rates.append(np.mean([r['explored'] for r in results]))
            avg_steps.append(np.mean([r['steps'] for r in results]))
            avg_max_dists.append(np.mean([r['max_dist'] for r in results]))
            success_rates.append(np.mean([r['success'] for r in results]))

    # 圖表1: 探索率對比
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(start_points)), avg_explored_rates, color='steelblue', alpha=0.8)
    plt.xlabel('Start Points', fontsize=12)
    plt.ylabel('Average Explored Rate', fontsize=12)
    plt.title('Average Exploration Rate Comparison Across Different Start Points', fontsize=14)
    plt.xticks(range(len(start_points)), start_points, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)

    # 在柱狀圖上添加數值標籤
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg_explored_rates[i]:.1%}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exploration_rate_comparison.png'), dpi=300)
    plt.close()

    # 圖表2: 平均步數對比
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(start_points)), avg_steps, color='coral', alpha=0.8)
    plt.xlabel('Start Points', fontsize=12)
    plt.ylabel('Average Steps', fontsize=12)
    plt.title('Average Steps Comparison Across Different Start Points', fontsize=14)
    plt.xticks(range(len(start_points)), start_points, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    # 在柱狀圖上添加數值標籤
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg_steps[i]:.0f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_steps_comparison.png'), dpi=300)
    plt.close()

    # 圖表3: 平均最大距離對比
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(start_points)), avg_max_dists, color='lightgreen', alpha=0.8)
    plt.xlabel('Start Points', fontsize=12)
    plt.ylabel('Average Max Distance', fontsize=12)
    plt.title('Average Max Distance Comparison Across Different Start Points', fontsize=14)
    plt.xticks(range(len(start_points)), start_points, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    # 在柱狀圖上添加數值標籤
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg_max_dists[i]:.1f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'max_distance_comparison.png'), dpi=300)
    plt.close()

    # 圖表4: 成功率對比
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(start_points)), success_rates, color='mediumpurple', alpha=0.8)
    plt.xlabel('Start Points', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Success Rate Comparison Across Different Start Points', fontsize=14)
    plt.xticks(range(len(start_points)), start_points, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)

    # 在柱狀圖上添加數值標籤
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{success_rates[i]:.1%}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_comparison.png'), dpi=300)
    plt.close()

    # 圖表5: 綜合對比（所有指標）
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 探索率
    ax1.bar(range(len(start_points)), avg_explored_rates, color='steelblue', alpha=0.8)
    ax1.set_ylabel('Explored Rate', fontsize=11)
    ax1.set_title('Exploration Rate', fontsize=12)
    ax1.set_xticks(range(len(start_points)))
    ax1.set_xticklabels(start_points, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis='y', alpha=0.3)

    # 平均步數
    ax2.bar(range(len(start_points)), avg_steps, color='coral', alpha=0.8)
    ax2.set_ylabel('Average Steps', fontsize=11)
    ax2.set_title('Average Steps', fontsize=12)
    ax2.set_xticks(range(len(start_points)))
    ax2.set_xticklabels(start_points, rotation=45, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # 平均最大距離
    ax3.bar(range(len(start_points)), avg_max_dists, color='lightgreen', alpha=0.8)
    ax3.set_ylabel('Max Distance', fontsize=11)
    ax3.set_title('Average Max Distance', fontsize=12)
    ax3.set_xticks(range(len(start_points)))
    ax3.set_xticklabels(start_points, rotation=45, ha='right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)

    # 成功率
    ax4.bar(range(len(start_points)), success_rates, color='mediumpurple', alpha=0.8)
    ax4.set_ylabel('Success Rate', fontsize=11)
    ax4.set_title('Success Rate', fontsize=12)
    ax4.set_xticks(range(len(start_points)))
    ax4.set_xticklabels(start_points, rotation=45, ha='right', fontsize=9)
    ax4.set_ylim(0, 1.0)
    ax4.grid(axis='y', alpha=0.3)

    plt.suptitle('Comprehensive Performance Comparison Across Different Start Points',
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_comparison.png'), dpi=300)
    plt.close()

    print(f"{GREEN}已生成所有對比圖表{NC}")


def main():
    """主函數"""

    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    # 載入模型
    if not os.path.exists(MODEL_PATH):
        print(f"{RED}錯誤: 在 {MODEL_PATH} 找不到模型檔案{NC}")
        return

    if device == torch.device('cuda'):
        checkpoint = torch.load(MODEL_PATH)
    else:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    global_network.load_state_dict(checkpoint['policy_model'])

    print(f"{GREEN}已載入模型: {MODEL_PATH}{NC}")

    # ============================================================
    # 配置區域：修改這裡來自定義測試參數
    # ============================================================

    # 指定使用的地圖索引 (設為None則每次測試使用不同地圖)
    # 例如: MAP_INDEX = 0 表示所有測試都使用地圖0
    MAP_INDEX = None  # 設為 None 使用隨機地圖，或設為特定數字(如 0, 1, 2...)使用固定地圖

    # 定義多個自定義起始點 [x, y]
    # 兩個機器人將從相同位置開始
    start_points_list = [
        [100, 100], 
        [520, 120], 
        [250, 250], 
        [250, 130],
        [250, 100],
        [400, 120], 
        [140, 410], 
        [110, 590], 
        [90, 300], 
        [260, 200], 
    ]

    # ============================================================

    print(f"\n{GREEN}將測試 {len(start_points_list)} 個不同的起始點{NC}")
    print(f"{GREEN}每個起始點將運行 {NUM_TEST} 次測試{NC}")
    if MAP_INDEX is not None:
        print(f"{GREEN}使用固定地圖: 地圖索引 {MAP_INDEX}{NC}")
    else:
        print(f"{YELLOW}使用隨機地圖: 每次測試使用不同地圖{NC}")

    # 設置輸出目錄
    output_dir = 'results_custom_start'

    # 運行測試
    results = run_test_with_custom_start_points(
        global_network=global_network,
        device=device,
        start_points_list=start_points_list,
        output_dir=output_dir,
        map_index=MAP_INDEX
    )

    print(f"\n{GREEN}測試完成！{NC}")


if __name__ == '__main__':
    print(f"{GREEN}歡迎使用 IR2-MARL 自定義起始點探索測試系統！{NC}")
    main()