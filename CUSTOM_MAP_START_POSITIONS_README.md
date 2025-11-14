# 自定義地圖和起始點使用說明

## 📋 概述

本項目提供兩種方式來測試 RobotIndividualMapTracker 功能：

1. **簡單測試**：使用默認地圖和起始點
2. **自定義測試**：指定特定地圖和多個起始點（支持最多10個起始點）

---

## 🚀 方式一：簡單測試（推薦初次使用）

### 使用默認配置

```bash
python test_individual_map_tracker.py
```

### 修改配置

編輯 `test_individual_map_tracker.py`，在 `main()` 函數中修改：

```python
def main():
    # ==================== 配置區域 ====================
    # 1. 指定自定義地圖
    CUSTOM_MAP = "DungeonMaps/test/complex/100.png"  # 你的地圖路徑

    # 2. 指定起始點（取消註釋以使用）
    CUSTOM_START_POSITIONS = [
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
    # ==================================================
```

---

## 🎯 方式二：完整自定義測試（支持複雜配置）

### 使用完整自定義腳本

```bash
python test_individual_map_tracker_custom.py
```

### 詳細配置

編輯 `test_individual_map_tracker_custom.py`：

```python
def test_custom_map_and_positions():
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
```

---

## 📝 配置說明

### 地圖文件

- **格式**：PNG 圖像文件
- **位置**：通常在 `DungeonMaps/test/` 目錄下
- **值含義**：
  - 黑色/深色 (值 < 150)：障礙物
  - 白色/亮色 (值 > 150)：自由空間

### 起始點坐標

- **格式**：`(x, y)` 元組
- **坐標系**：
  - `x`：水平方向（從左到右）
  - `y`：垂直方向（從上到下）
  - 原點 (0, 0) 在地圖左上角

### 機器人數量

- **範圍**：2 到 10 個機器人
- **起始點**：必須至少提供與機器人數量相等的起始點
- **選擇**：程序會使用列表中前 N 個起始點（N = 機器人數量）

---

## 📊 輸出示例

運行後會看到類似輸出：

```
======================================================================
自定義地圖和起始點測試
======================================================================
使用設備: cuda

初始化中...
✓ 自定義地圖已加載: DungeonMaps/test/complex/100.png
✓ 地圖大小: (480, 640)
✓ 機器人數量: 2
✓ 起始位置:
    Robot 1: [100 100]
    Robot 2: [200 100]

開始運行 Episode 1...
======================================================================
[Eps 1 | Step   0] Explored: 5.23% | Max Dist: 0.0 | Individual: R1:2.31%, R2:2.92%
[Eps 1 | Step  10] Explored: 12.45% | Max Dist: 45.2 | Individual: R1:6.12%, R2:6.33%
...

正在生成個人地圖追蹤分析...
探索區域聯集: 125234 pixels
探索區域交集: 45123 pixels
總體重疊比例: 0.3601
兩兩機器人重疊比例:
  robot1_robot2: 0.3601
各機器人探索比例:
  Robot 1 (起始點: (100, 100)): 45.32%
  Robot 2 (起始點: (200, 100)): 41.25%

✓ 測試完成！
======================================================================
```

---

## 🛠️ 進度監控

程序運行時會每10步打印一次進度：

```
[Eps 1 | Step   0] Explored: 5.23% | Max Dist: 0.0 | Individual: R1:2.31%, R2:2.92%
[Eps 1 | Step  10] Explored: 12.45% | Max Dist: 45.2 | Individual: R1:6.12%, R2:6.33%
[Eps 1 | Step  20] Explored: 18.67% | Max Dist: 89.4 | Individual: R1:9.23%, R2:9.44%
```

**顯示信息：**
- `Explored`：總體探索率（通訊後）
- `Max Dist`：最大移動距離
- `Individual`：每個機器人的獨立探索率（未通訊）

---

## 📁 生成的文件

運行完成後會生成：

```
robot_individual_maps/
├── coverage_over_time.png      # 覆蓋率隨時間變化圖表
├── coverage_data.csv            # 覆蓋率原始數據
└── history/                     # 地圖歷史快照
    ├── individual_maps_history_0000.png
    ├── individual_maps_history_0010.png
    └── ...
```

---

## 💡 使用技巧

### 1. 選擇合適的起始點

確保起始點：
- 在地圖的自由空間內（不在障礙物上）
- 彼此分散，以便觀察重疊行為
- 距離障礙物有一定距離

### 2. 驗證起始點

可以使用 Python 快速驗證：

```python
from skimage import io
import numpy as np

# 加載地圖
map_data = io.imread("DungeonMaps/test/complex/100.png", 1)

# 檢查某個點是否在自由空間
x, y = 100, 100
value = map_data[y, x]  # 注意：數組是 [row, col] = [y, x]

if value > 150:
    print(f"位置 ({x}, {y}) 是自由空間 ✓")
else:
    print(f"位置 ({x}, {y}) 是障礙物 ✗")
```

### 3. 調整機器人數量

測試不同數量的機器人：

```python
# 測試 2 個機器人
n_agent = 2

# 測試 5 個機器人
n_agent = 5

# 測試 10 個機器人
n_agent = 10
```

---

## ⚠️ 常見問題

### Q1: 起始點在障礙物上怎麼辦？

**A**: 程序可能會失敗或行為異常。請確保起始點在自由空間內。

### Q2: 機器人數量超過起始點數量？

**A**: 程序會自動調整機器人數量為起始點的數量。

### Q3: 地圖文件找不到？

**A**: 確認：
- 文件路徑正確
- 文件格式為 PNG
- 文件權限允許讀取

### Q4: 如何快速測試多種配置？

**A**: 創建一個腳本循環測試：

```python
# 測試不同的配置
configs = [
    ("DungeonMaps/test/complex/100.png", 2),
    ("DungeonMaps/test/complex/100.png", 5),
    ("DungeonMaps/test/complex/100.png", 10),
]

for map_path, n_robots in configs:
    print(f"\n測試: {map_path} with {n_robots} robots")
    # 運行測試...
```

---

## 📚 相關文檔

- [RobotIndividualMapTracker 完整文檔](INDIVIDUAL_MAP_TRACKER_README.md)
- [項目主 README](README.md)

---

## 🤝 貢獻

如有問題或建議，歡迎提出 Issue 或 Pull Request。
