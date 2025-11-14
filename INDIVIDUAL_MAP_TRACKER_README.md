# Robot Individual Map Tracker 使用說明

## 概述

`RobotIndividualMapTracker` 是一個用於追蹤和分析多機器人系統中每個機器人獨立探索地圖的工具。它可以記錄每個機器人在**沒有與其他機器人通訊**的情況下，僅依靠自己的傳感器所觀察到的局部地圖（local map）。

## 主要功能

1. **追蹤個人地圖**：記錄每個機器人獨立探索的區域（未經通訊合併）
2. **計算重疊比例**：分析機器人之間探索區域的重疊程度
3. **覆蓋率分析**：生成隨時間變化的覆蓋率圖表
4. **可視化**：保存地圖歷史和生成分析圖表

## 快速開始

### 1. 基本使用

在創建 `Worker` 時，設置 `track_individual_maps=True`：

```python
from multi_robot_worker import Worker
from model import PolicyNet, QNet
import torch

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy_net = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)
q_net = QNet(INPUT_DIM, EMBEDDING_DIM).to(device)

# 創建 Worker，啟用個人地圖追蹤
worker = Worker(
    meta_agent_id=0,
    n_agent=2,  # 2個機器人
    policy_net=policy_net,
    q_net=q_net,
    global_step=1,
    device=device,
    save_image=True,
    track_individual_maps=True  # 啟用個人地圖追蹤
)

# 運行 episode
worker.run_episode(episode_number=1)
```

### 2. 運行測試腳本

我們提供了一個測試腳本來演示功能：

```bash
python test_individual_map_tracker.py
```

這將：
- 創建一個 2 機器人的測試環境
- 運行一個 episode
- 生成覆蓋率圖表和統計信息
- 保存地圖歷史

## 生成的輸出

### 文件結構

```
robot_individual_maps/
├── coverage_over_time.png      # 覆蓋率隨時間變化圖表
├── coverage_data.csv            # 覆蓋率原始數據
└── history/                     # 地圖歷史快照
    ├── individual_maps_history_0000.png
    ├── individual_maps_history_0010.png
    └── ...
```

### 覆蓋率圖表說明

`coverage_over_time.png` 包含以下曲線：

- **Robot 1, Robot 2, ...**: 每個機器人的獨立探索覆蓋率
- **Intersection**: 所有機器人都探索過的區域（交集）
- **Union**: 至少一個機器人探索過的區域（聯集）

### 統計信息

在 episode 結束時，會打印以下統計信息：

```
各機器人探索比例:
  Robot 1: 0.4532
  Robot 2: 0.4125

總體重疊比例: 0.2341
兩兩機器人重疊比例:
  robot1_robot2: 0.3156

探索區域聯集: 125000 pixels
探索區域交集: 45000 pixels
```

## API 參考

### RobotIndividualMapTracker

#### 初始化

```python
tracker = RobotIndividualMapTracker(
    n_agent=2,                      # 機器人數量
    ground_truth_size=(480, 640),   # 地圖大小
    sensor_range=80,                # 傳感器範圍
    save_dir='robot_individual_maps' # 保存目錄
)
```

#### 主要方法

- `start_tracking()`: 開始追蹤
- `update_robot_map(robot_id, position, ground_truth)`: 更新單個機器人的地圖
- `save_current_maps(robot_positions)`: 保存當前所有機器人的地圖快照
- `get_exploration_ratio(ground_truth)`: 獲取每個機器人的探索比例
- `calculate_overlap()`: 計算重疊統計信息
- `plot_coverage_over_time(ground_truth)`: 生成覆蓋率圖表
- `save_map_history(interval=10)`: 保存地圖歷史

## 與現有代碼的集成

### 在 env.py 中

`Env` 類新增了 `track_individual_maps` 參數：

```python
env = Env(
    map_index=1,
    n_agent=2,
    k_size=20,
    plot=True,
    track_individual_maps=True  # 啟用追蹤
)
```

### 在 multi_robot_worker.py 中

`Worker` 類新增了 `track_individual_maps` 參數：

```python
worker = Worker(
    meta_agent_id=0,
    n_agent=2,
    policy_net=policy_net,
    q_net=q_net,
    global_step=1,
    device=device,
    track_individual_maps=True  # 啟用追蹤
)
```

## 技術細節

### 追蹤時機

個人地圖的更新發生在：

1. **初始化時**：在 `env.begin()` 中，記錄每個機器人在起始位置的初始觀察
2. **每步更新時**：在 `env.single_robot_step()` 中，當機器人移動後：
   - 先更新機器人自己的信念
   - **立即記錄到 tracker**（在與其他機器人通訊合併之前）
   - 然後才進行機器人間的地圖合併

這樣確保了 tracker 中記錄的是每個機器人**純粹依靠自己傳感器**觀察到的地圖。

### 地圖值說明

- `255`: 已探索的自由空間
- `127`: 未探索區域
- `1`: 障礙物
- `76`: 機器人當前位置標記

## 應用場景

1. **通訊效率分析**：比較有無通訊時的探索效率
2. **協作評估**：分析機器人之間的協作程度（通過重疊比例）
3. **策略優化**：評估探索策略是否能有效減少重複探索
4. **可視化演示**：生成直觀的探索過程動畫

## 常見問題

### Q: 如何只保存最後的結果而不保存中間過程？

A: 設置 `save_image=False`，這樣只會在 episode 結束時生成最終的覆蓋率圖表：

```python
worker = Worker(
    ...,
    save_image=False,
    track_individual_maps=True
)
```

### Q: 如何調整地圖歷史保存的間隔？

A: 修改 `multi_robot_worker.py` 中的 `interval` 參數：

```python
self.env.individual_map_tracker.save_map_history(interval=20)  # 每20步保存一次
```

### Q: 如何在訓練過程中只在特定 episode 啟用追蹤？

A: 在 `runner.py` 中動態設置：

```python
# 只在能被 100 整除的 episode 啟用追蹤
track_maps = (episode_number % 100 == 0)
worker = Worker(..., track_individual_maps=track_maps)
```

## 性能考慮

- 啟用追蹤會增加少量的計算和內存開銷
- 保存地圖歷史會消耗磁盤空間
- 建議在訓練時關閉追蹤，只在評估時啟用

## 示例輸出

運行測試腳本後，你將看到類似以下的輸出：

```
======================================================================
個人地圖追蹤測試
======================================================================
使用設備: cuda

創建 Worker（2 個機器人）...
環境地圖大小: (480, 640)
傳感器範圍: 80

開始運行測試 episode 1...
正在生成個人地圖追蹤分析...
探索區域聯集: 125234 pixels
探索區域交集: 45123 pixels
總體重疊比例: 0.3601
兩兩機器人重疊比例:
  robot1_robot2: 0.3601
各機器人探索比例:
  Robot 1: 0.4532
  Robot 2: 0.4125
覆蓋率圖表已保存到 robot_individual_maps/coverage_over_time.png
覆蓋率數據已保存到 robot_individual_maps/coverage_data.csv
已保存個人地圖歷史，共 15 幀

======================================================================
測試完成！
======================================================================
```

## 貢獻

如有問題或建議，歡迎提出 Issue 或 Pull Request。

## 授權

本項目遵循與主項目相同的授權協議。
