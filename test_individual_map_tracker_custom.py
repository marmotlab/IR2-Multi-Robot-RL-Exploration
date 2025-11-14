#!/usr/bin/env python3
#######################################################################
# Name: test_individual_map_tracker_custom.py
# Test script with custom map and multiple start positions
# 支持指定地图和10個起始點
#######################################################################

import sys
sys.modules['TRAINING'] = True

from parameter import *
import torch
import numpy as np
from model import PolicyNet, QNet
from env import Env
from robot import Robot
import copy
import os
from skimage import io


class CustomWorker:
    """支持自定義地圖和起始點的 Worker"""

    def __init__(self, map_path, start_positions, n_agent, policy_net, q_net,
                 episode_number, device='cuda', save_image=True, track_individual_maps=True):
        """
        初始化自定義 Worker

        參數:
            map_path: 地圖文件路徑
            start_positions: 起始位置列表 [(x1,y1), (x2,y2), ...]
            n_agent: 機器人數量
            policy_net: 策略網絡
            q_net: Q網絡
            episode_number: Episode編號
            device: 計算設備
            save_image: 是否保存圖像
            track_individual_maps: 是否追蹤個人地圖
        """
        self.device = device
        self.n_agent = n_agent
        self.episode_number = episode_number
        self.save_image = save_image
        self.track_individual_maps = track_individual_maps

        # 加載自定義地圖
        self.ground_truth = self.load_custom_map(map_path)
        self.ground_truth_size = np.shape(self.ground_truth)

        # 驗證起始位置
        if len(start_positions) < n_agent:
            raise ValueError(f"起始位置數量 ({len(start_positions)}) 少於機器人數量 ({n_agent})")

        self.start_positions = [np.array(pos) for pos in start_positions[:n_agent]]

        # 創建環境（不使用默認地圖）
        self.env = self.create_custom_env()

        # 創建機器人
        self.robot_list = []
        self.all_robot_positions = []
        for i in range(n_agent):
            robot_position = self.start_positions[i]
            robot = Robot(robot_id=i, position=robot_position, plot=save_image)
            self.robot_list.append(robot)
            self.all_robot_positions.append(robot_position)

        self.local_policy_net = policy_net
        self.local_q_net = q_net

        print(f"✓ 自定義地圖已加載: {map_path}")
        print(f"✓ 地圖大小: {self.ground_truth_size}")
        print(f"✓ 機器人數量: {n_agent}")
        print(f"✓ 起始位置:")
        for i, pos in enumerate(self.start_positions):
            print(f"    Robot {i+1}: {pos}")

    def load_custom_map(self, map_path):
        """加載自定義地圖"""
        try:
            ground_truth = (io.imread(map_path, 1)).astype(int)
            if np.all(ground_truth == 0):
                ground_truth = (io.imread(map_path, 1) * 255).astype(int)
        except Exception as e:
            raise ValueError(f"無法加載地圖 {map_path}: {e}")

        # 將地圖轉換為標準格式 (1=障礙物, 255=自由空間)
        ground_truth = (ground_truth > 150)
        ground_truth = ground_truth * 254 + 1
        return ground_truth

    def create_custom_env(self):
        """創建使用自定義地圖的環境"""
        # 創建一個臨時環境以獲取基本配置
        temp_env = Env(
            map_index=0,
            n_agent=self.n_agent,
            k_size=K_SIZE,
            plot=self.save_image,
            track_individual_maps=self.track_individual_maps
        )

        # 替換為自定義地圖和起始位置
        temp_env.ground_truth = self.ground_truth
        temp_env.ground_truth_size = self.ground_truth_size
        temp_env.start_position = self.start_positions[0]  # 默認使用第一個起始位置

        # 重新初始化環境
        temp_env.agents_merged_belief = None
        temp_env.downsampled_agents_merged_belief = None
        temp_env.agents_merged_belief_frontiers = None

        temp_env.all_robot_belief = []
        temp_env.all_old_robot_belief = []
        temp_env.all_downsampled_belief = []

        for i in range(self.n_agent):
            robot_belief = np.ones(self.ground_truth_size) * 127
            temp_env.all_robot_belief.append([robot_belief for _ in range(self.n_agent)])
            temp_env.all_old_robot_belief.append(copy.deepcopy(robot_belief))
            temp_env.all_downsampled_belief.append(None)

        # 重新初始化每個機器人的圖生成器
        temp_env.all_graph_generator = []
        temp_env.all_node_coords = []
        temp_env.all_graph = []
        temp_env.all_node_utility = []
        temp_env.all_guidepost = []
        temp_env.all_frontiers = []

        from graph_generator import Graph_generator
        from skimage.measure import block_reduce

        for id in range(self.n_agent):
            start_pos = self.start_positions[id]

            temp_env.all_graph_generator.append(
                Graph_generator(
                    robot_id=id,
                    map_size=self.ground_truth_size,
                    sensor_range=temp_env.sensor_range,
                    k_size=K_SIZE,
                    file_path=f"custom_map_{id}",
                    plot=self.save_image
                )
            )
            temp_env.all_graph_generator[id].route_node.append(start_pos)

            # 更新機器人在起始位置的觀測
            temp_env.all_robot_belief[id][id] = temp_env.update_robot_belief(
                start_pos, temp_env.sensor_range,
                temp_env.all_robot_belief[id][id], temp_env.ground_truth
            )
            temp_env.all_downsampled_belief[id] = block_reduce(
                temp_env.all_robot_belief[id][id].copy(),
                block_size=(temp_env.resolution, temp_env.resolution),
                func=np.min
            )
            temp_env.all_frontiers[id] = temp_env.find_frontier(temp_env.all_downsampled_belief[id])
            temp_env.all_old_robot_belief[id] = copy.deepcopy(temp_env.all_robot_belief[id][id])
            temp_env.agents_merged_belief = temp_env.merge_beliefs([
                temp_env.agents_merged_belief, temp_env.all_robot_belief[id][id]
            ])

            node_coords, graph, node_utility, guidepost = temp_env.all_graph_generator[id].generate_graph(
                start_pos, temp_env.all_robot_belief[id][id], temp_env.all_frontiers[id]
            )
            temp_env.all_node_coords.append(node_coords)
            temp_env.all_graph.append(graph)
            temp_env.all_node_utility.append(node_utility)
            temp_env.all_guidepost.append(guidepost)

            # 更新位置信念
            temp_env.all_robot_positions_belief[id][id] = start_pos

        temp_env.downsampled_agents_merged_belief = block_reduce(
            temp_env.agents_merged_belief.copy(),
            block_size=(temp_env.resolution, temp_env.resolution),
            func=np.min
        )
        temp_env.agents_merged_belief_frontiers = temp_env.find_frontier(
            temp_env.downsampled_agents_merged_belief
        )

        # 開始追蹤個人地圖
        if temp_env.track_individual_maps and temp_env.individual_map_tracker is not None:
            temp_env.individual_map_tracker.start_tracking()
            for id in range(self.n_agent):
                temp_env.individual_map_tracker.update_robot_map(
                    id, self.start_positions[id], temp_env.ground_truth
                )

        return temp_env


def test_custom_map_and_positions():
    """
    測試自定義地圖和起始點
    """
    print("=" * 70)
    print("自定義地圖和起始點測試")
    print("=" * 70)

    # ==================== 配置區域 ====================

    # 1. 指定地圖路徑
    map_path = "DungeonMaps/test/complex/100.png"  # 修改為你的地圖路徑

    # 2. 指定10個起始點 (x, y)
    start_positions = [
        (100, 100),  # Robot 1
        (200, 100),  # Robot 2
        (300, 100),  # Robot 3
        (100, 200),  # Robot 4
        (200, 200),  # Robot 5
        (300, 200),  # Robot 6
        (100, 300),  # Robot 7
        (200, 300),  # Robot 8
        (300, 300),  # Robot 9
        (400, 300),  # Robot 10
    ]

    # 3. 指定使用多少個機器人（從起始點列表中選擇前N個）
    n_agent = 2  # 可以改成 2-10 之間的任意數字

    # ==================================================

    episode_number = 1

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用設備: {device}")

    policy_net = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    q_net = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    # 創建自定義 Worker
    print(f"\n初始化中...")
    try:
        from multi_robot_worker import Worker

        # 修改 Worker 以支持自定義環境
        worker = Worker(
            meta_agent_id=0,
            n_agent=n_agent,
            policy_net=policy_net,
            q_net=q_net,
            global_step=episode_number,
            device=device,
            save_image=True,
            greedy=True,
            track_individual_maps=True
        )

        # 替換為自定義地圖和起始位置
        custom_worker = CustomWorker(
            map_path=map_path,
            start_positions=start_positions,
            n_agent=n_agent,
            policy_net=policy_net,
            q_net=q_net,
            episode_number=episode_number,
            device=device,
            save_image=True,
            track_individual_maps=True
        )

        # 使用自定義環境和機器人
        worker.env = custom_worker.env
        worker.robot_list = custom_worker.robot_list
        worker.all_robot_positions = custom_worker.all_robot_positions

        print(f"\n開始運行 Episode {episode_number}...")
        print("=" * 70)

        # 運行 episode
        success = worker.run_episode(episode_number)

        if success:
            print("\n" + "=" * 70)
            print("✓ 測試完成！")
            print("=" * 70)

            # 顯示結果
            if worker.env.individual_map_tracker is not None:
                print("\n最終統計:")

                exploration_ratios = worker.env.individual_map_tracker.get_exploration_ratio(
                    worker.env.ground_truth
                )
                print("\n各機器人探索比例:")
                for robot_id, ratio in enumerate(exploration_ratios):
                    print(f"  Robot {robot_id+1} (起始點: {start_positions[robot_id]}): {ratio*100:.2f}%")

                overlap_stats = worker.env.individual_map_tracker.calculate_overlap()
                print(f"\n總體重疊比例: {overlap_stats['overall_overlap_ratio']*100:.2f}%")

                if overlap_stats['pairwise_overlaps']:
                    print("\n兩兩機器人重疊比例:")
                    for pair, ratio in overlap_stats['pairwise_overlaps'].items():
                        print(f"  {pair}: {ratio*100:.2f}%")
        else:
            print("\n✗ 測試失敗！")

    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_custom_map_and_positions()
