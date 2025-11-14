#!/usr/bin/env python3
"""
快速診斷腳本 - 檢查個人地圖追蹤器狀態
"""

import sys
sys.modules['TRAINING'] = True

from test_parameter import *
import torch
from model import PolicyNet, QNet
from multi_robot_worker import Worker

print("="*70)
print("個人地圖追蹤器診斷工具")
print("="*70)

# 初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy_net = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
q_net = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)

print(f"\n1. 測試設備: {device}")

# 測試 1: 不啟用追蹤
print(f"\n2. 測試創建 Worker (track_individual_maps=False)...")
worker_no_track = Worker(
    meta_agent_id=0,
    n_agent=2,
    policy_net=policy_net,
    q_net=q_net,
    global_step=1,
    device=device,
    save_image=False,
    track_individual_maps=False  # 關閉
)

print(f"   ✓ Worker 創建成功")
print(f"   - track_individual_maps: {worker_no_track.track_individual_maps}")
print(f"   - env.track_individual_maps: {worker_no_track.env.track_individual_maps}")
print(f"   - env.individual_map_tracker: {worker_no_track.env.individual_map_tracker}")

# 測試 2: 啟用追蹤
print(f"\n3. 測試創建 Worker (track_individual_maps=True)...")
worker_with_track = Worker(
    meta_agent_id=0,
    n_agent=2,
    policy_net=policy_net,
    q_net=q_net,
    global_step=1,
    device=device,
    save_image=False,
    track_individual_maps=True  # 開啟
)

print(f"   ✓ Worker 創建成功")
print(f"   - track_individual_maps: {worker_with_track.track_individual_maps}")
print(f"   - env.track_individual_maps: {worker_with_track.env.track_individual_maps}")
print(f"   - env.individual_map_tracker: {worker_with_track.env.individual_map_tracker}")

if worker_with_track.env.individual_map_tracker is not None:
    print(f"   - tracker.is_tracking: {worker_with_track.env.individual_map_tracker.is_tracking}")
    print(f"   - tracker.n_agent: {worker_with_track.env.individual_map_tracker.n_agent}")

print(f"\n{'='*70}")
print("診斷結果:")
print(f"{'='*70}")

if worker_with_track.env.individual_map_tracker is None:
    print("❌ 問題: 追蹤器未正確初始化！")
    print("   請檢查 env.py 中的 RobotIndividualMapTracker 導入")
else:
    print("✅ 追蹤器可以正常初始化")
    print(f"\n如果看不到 Individual 輸出，請確認:")
    print(f"   1. 測試腳本中設置了 track_individual_maps=True")
    print(f"   2. Worker 初始化時正確傳遞了該參數")
    print(f"\n示例:")
    print(f"   worker = Worker(")
    print(f"       ...,")
    print(f"       track_individual_maps=True  # ← 確保這裡是 True")
    print(f"   )")

print(f"\n{'='*70}")
