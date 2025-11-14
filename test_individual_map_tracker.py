#!/usr/bin/env python3
#######################################################################
# Name: test_individual_map_tracker.py
# Test script for RobotIndividualMapTracker
# Demonstrates how to track individual robot local maps without communication
#######################################################################

import sys
sys.modules['TRAINING'] = True

from test_parameter import *
import torch
import numpy as np
from model import PolicyNet, QNet
from multi_robot_worker import Worker


def test_individual_map_tracking(custom_map_path=None, custom_start_positions=None):
    """
    測試個人地圖追蹤功能

    這個函數演示如何：
    1. 啟用個人地圖追蹤
    2. 運行一個測試 episode
    3. 生成覆蓋率分析和可視化

    參數:
        custom_map_path: 自定義地圖路徑（可選）
        custom_start_positions: 自定義起始位置列表 [(x1,y1), (x2,y2), ...] （可選）
    """
    print("=" * 70)
    print("個人地圖追蹤測試")
    print("=" * 70)

    # 設置參數
    n_agent = 2  # 機器人數量
    episode_number = 1
    meta_agent_id = 0

    # 如果提供了自定義起始位置，調整機器人數量
    if custom_start_positions is not None:
        n_agent = min(n_agent, len(custom_start_positions))
        print(f"使用自定義起始位置，機器人數量: {n_agent}")
        for i, pos in enumerate(custom_start_positions[:n_agent]):
            print(f"  Robot {i+1}: {pos}")

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    policy_net = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
    q_net = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    # 創建 Worker，啟用個人地圖追蹤
    print(f"\n創建 Worker（{n_agent} 個機器人）...")
    worker = Worker(
        meta_agent_id=meta_agent_id,
        n_agent=n_agent,
        policy_net=policy_net,
        q_net=q_net,
        global_step=episode_number,
        device=device,
        save_image=True,  # 保存圖像
        greedy=True,  # 使用貪婪策略便於測試
        track_individual_maps=True  # 啟用個人地圖追蹤
    )

    print(f"環境地圖大小: {worker.env.ground_truth_size}")
    print(f"傳感器範圍: {worker.env.sensor_range}")

    # 檢查追蹤器狀態
    print(f"\n追蹤器狀態檢查:")
    print(f"  - worker.track_individual_maps: {worker.track_individual_maps}")
    print(f"  - worker.env.track_individual_maps: {worker.env.track_individual_maps}")
    print(f"  - worker.env.individual_map_tracker: {worker.env.individual_map_tracker}")
    if worker.env.individual_map_tracker is not None:
        print(f"  - tracker.is_tracking: {worker.env.individual_map_tracker.is_tracking}")
        print(f"  ✓ 個人地圖追蹤已啟用")
    else:
        print(f"  ✗ 警告: 追蹤器未初始化！")

    # 運行測試 episode
    print(f"\n開始運行測試 episode {episode_number}...")
    success = worker.run_episode(episode_number)

    if success:
        print("\n" + "=" * 70)
        print("測試完成！")
        print("=" * 70)

        # 顯示生成的文件位置
        print("\n生成的文件:")
        print(f"  - 覆蓋率圖表: robot_individual_maps/coverage_over_time.png")
        print(f"  - 覆蓋率數據: robot_individual_maps/coverage_data.csv")
        print(f"  - 地圖歷史: robot_individual_maps/history/")

        # 顯示最終統計信息
        if worker.env.individual_map_tracker is not None:
            print("\n最終統計:")

            # 探索比例
            exploration_ratios = worker.env.individual_map_tracker.get_exploration_ratio(
                worker.env.ground_truth
            )
            print("\n各機器人探索比例:")
            for robot_id, ratio in enumerate(exploration_ratios):
                print(f"  Robot {robot_id+1}: {ratio*100:.2f}%")

            # 重疊統計
            overlap_stats = worker.env.individual_map_tracker.calculate_overlap()
            print(f"\n總體重疊比例: {overlap_stats['overall_overlap_ratio']*100:.2f}%")

            if overlap_stats['pairwise_overlaps']:
                print("\n兩兩機器人重疊比例:")
                for pair, ratio in overlap_stats['pairwise_overlaps'].items():
                    print(f"  {pair}: {ratio*100:.2f}%")

            print(f"\n探索區域聯集: {overlap_stats['union_area']} pixels")
            print(f"探索區域交集: {overlap_stats['intersection_area']} pixels")
    else:
        print("\n測試失敗！")

    print("\n" + "=" * 70)


def main():
    """主函數"""

    # ==================== 配置區域 ====================
    # 1. 指定自定義地圖（設為 None 使用默認地圖）
    CUSTOM_MAP = None  # 例如: "DungeonMaps/test/complex/100.png"

    # 2. 指定10個起始點（設為 None 使用默認起始點）
    CUSTOM_START_POSITIONS = None
    # 示例：取消下面的註釋以使用自定義起始點
    # CUSTOM_START_POSITIONS = [
    #     (100, 100),  # Robot 1
    #     (200, 100),  # Robot 2
    #     (300, 100),  # Robot 3
    #     (100, 200),  # Robot 4
    #     (200, 200),  # Robot 5
    #     (300, 200),  # Robot 6
    #     (100, 300),  # Robot 7
    #     (200, 300),  # Robot 8
    #     (300, 300),  # Robot 9
    #     (400, 300),  # Robot 10
    # ]
    # ==================================================

    try:
        test_individual_map_tracking(
            custom_map_path=CUSTOM_MAP,
            custom_start_positions=CUSTOM_START_POSITIONS
        )
    except Exception as e:
        print(f"\n錯誤: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
