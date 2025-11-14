#######################################################################
# Name: robot_individual_map_tracker.py
# Track individual robot's local maps (without inter-robot communication)
# Calculate overlap ratio between robots' exploration areas
#######################################################################

import numpy as np
import matplotlib.pyplot as plt
import os
from sensor import sensor_work


class RobotIndividualMapTracker:
    """
    追蹤並記錄多個機器人的個人探索地圖（只包含自己探索的區域，未經機器人間通訊）
    地圖大小與 ground_truth 相同
    考慮障礙物遮擋（使用 sensor_work 函數）
    支持任意數量的機器人
    """

    def __init__(self, n_agent, ground_truth_size, sensor_range, save_dir='robot_individual_maps'):
        """
        初始化追蹤器

        參數:
            n_agent: 機器人數量
            ground_truth_size: 地圖大小 (height, width)
            sensor_range: 傳感器範圍
            save_dir: 保存地圖的目錄
        """
        self.n_agent = n_agent
        self.ground_truth_size = ground_truth_size
        self.sensor_range = sensor_range
        self.save_dir = save_dir

        # 確保保存目錄存在
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # 初始化每個機器人的個人地圖（只有自己探索的區域）
        self.individual_maps = []
        for _ in range(n_agent):
            # 初始化為未知區域 (值為127)
            individual_map = np.ones(ground_truth_size) * 127
            self.individual_maps.append(individual_map)

        # 記錄地圖的歷史
        self.maps_history = [[] for _ in range(n_agent)]

        # 是否正在追蹤
        self.is_tracking = False

        # 可視化相關設置
        self.fig = None
        self.axes = None

        print(f"RobotIndividualMapTracker 已初始化，追蹤 {n_agent} 個機器人")

    def start_tracking(self):
        """開始追蹤個人地圖"""
        self.is_tracking = True

        # 重新初始化個人地圖為未知區域 (值為127)
        for i in range(self.n_agent):
            self.individual_maps[i] = np.ones(self.ground_truth_size) * 127

        # 清空地圖歷史
        self.maps_history = [[] for _ in range(self.n_agent)]

        print("開始追蹤機器人個人探索地圖")

    def stop_tracking(self):
        """停止追蹤個人地圖"""
        self.is_tracking = False
        total_frames = len(self.maps_history[0]) if self.maps_history else 0
        print(f"停止追蹤機器人個人探索地圖，共記錄了 {total_frames} 個時間點")

    def update_robot_map(self, robot_id, robot_position, ground_truth):
        """
        更新單個機器人的個人探索地圖（使用 sensor_work 函數）

        參數:
            robot_id: 機器人ID
            robot_position: 機器人當前位置
            ground_truth: 真實地圖
        """
        if not self.is_tracking:
            return

        # 使用 sensor_work 函數更新機器人的個人地圖
        self.individual_maps[robot_id] = sensor_work(
            robot_position,
            self.sensor_range,
            self.individual_maps[robot_id],
            ground_truth
        )

    def save_current_maps(self, robot_positions):
        """
        保存當前所有機器人的個人探索地圖到歷史記錄

        參數:
            robot_positions: 所有機器人的位置列表
        """
        if not self.is_tracking:
            print(f"[DEBUG] save_current_maps: is_tracking={self.is_tracking}, 跳過保存")
            return

        # 為每個機器人創建帶位置標記的地圖副本並保存
        for robot_id in range(self.n_agent):
            map_with_robot = self._get_map_with_robot(
                self.individual_maps[robot_id],
                robot_positions[robot_id]
            )
            self.maps_history[robot_id].append(map_with_robot.copy())

        # 調試信息
        if len(self.maps_history[0]) <= 2:
            print(f"[DEBUG] save_current_maps 完成: maps_history 長度 = {len(self.maps_history[0])}")

    def _get_map_with_robot(self, map_data, position):
        """
        在地圖上標記機器人位置

        參數:
            map_data: 地圖數據
            position: 機器人位置

        返回:
            帶有機器人標記的地圖副本
        """
        # 創建副本避免修改原始地圖
        map_copy = map_data.copy()

        # 標記機器人位置
        x, y = int(position[0]), int(position[1])
        robot_size = 3  # 標記大小

        # 確保不超出地圖邊界
        min_x = max(0, x - robot_size)
        max_x = min(map_copy.shape[1] - 1, x + robot_size)
        min_y = max(0, y - robot_size)
        max_y = min(map_copy.shape[0] - 1, y + robot_size)

        # 標記機器人位置為76
        map_copy[min_y:max_y+1, min_x:max_x+1] = 76

        return map_copy

    def save_current_frame(self, step):
        """
        保存當前的個人探索地圖為圖片

        參數:
            step: 當前步數

        返回:
            bool: 是否成功保存
        """
        if not self.is_tracking:
            print(f"[DEBUG] save_current_frame step {step}: 追蹤未啟動 (is_tracking={self.is_tracking})")
            return False

        if not self.maps_history[0]:
            print(f"[DEBUG] save_current_frame step {step}: 地圖歷史為空 (maps_history 長度={len(self.maps_history[0]) if self.maps_history else 'N/A'})")
            return False

        # 計算需要的子圖佈局
        n_cols = min(3, self.n_agent)  # 每行最多3個
        n_rows = (self.n_agent + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if self.n_agent == 1:
            axes = np.array([axes])
        axes = axes.flatten() if self.n_agent > 1 else axes

        for robot_id in range(self.n_agent):
            if self.maps_history[robot_id]:
                robot_map = self.maps_history[robot_id][-1]

                ax = axes[robot_id]
                im = ax.imshow(robot_map, cmap='gray', vmin=0, vmax=255)
                ax.set_title(f'Robot{robot_id+1} Individual Exploration')
                plt.colorbar(im, ax=ax, label='Map Value')
                ax.axis('off')

        # 隱藏多餘的子圖
        for i in range(self.n_agent, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f'individual_maps_step_{step:04d}.png')
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        return True

    def get_exploration_ratio(self, global_ground_truth):
        """
        獲取每個機器人的個人探索比例

        參數:
            global_ground_truth: 全局真實地圖

        返回:
            列表: 每個機器人的探索比例
        """
        if not self.is_tracking:
            return [0] * self.n_agent

        # 計算全局地圖的可探索區域總數
        total_explorable = np.sum(global_ground_truth == 255)

        # 計算每個機器人的探索比例
        ratios = []
        for robot_id in range(self.n_agent):
            robot_explored = np.sum(self.individual_maps[robot_id] == 255)
            ratio = robot_explored / total_explorable if total_explorable > 0 else 0
            ratios.append(ratio)

        return ratios

    def calculate_overlap(self):
        """
        計算所有機器人探索區域的重疊程度

        返回:
            dict: 包含重疊統計信息的字典
        """
        if not self.is_tracking:
            return {}

        # 計算每個機器人探索的區域
        explored_masks = []
        for robot_id in range(self.n_agent):
            explored_mask = (self.individual_maps[robot_id] == 255)
            explored_masks.append(explored_mask)

        # 計算聯集（至少一個機器人探索過的區域）
        union = np.zeros_like(explored_masks[0], dtype=bool)
        for mask in explored_masks:
            union = union | mask

        union_count = np.sum(union)

        # 計算交集（所有機器人都探索過的區域）
        intersection = np.ones_like(explored_masks[0], dtype=bool)
        for mask in explored_masks:
            intersection = intersection & mask

        intersection_count = np.sum(intersection)

        # 計算兩兩機器人之間的重疊
        pairwise_overlaps = {}
        for i in range(self.n_agent):
            for j in range(i+1, self.n_agent):
                overlap = np.sum(explored_masks[i] & explored_masks[j])
                pair_union = np.sum(explored_masks[i] | explored_masks[j])
                overlap_ratio = overlap / pair_union if pair_union > 0 else 0
                pairwise_overlaps[f'robot{i+1}_robot{j+1}'] = overlap_ratio

        # 計算總體重疊比例
        overall_overlap_ratio = intersection_count / union_count if union_count > 0 else 0

        return {
            'union_area': union_count,
            'intersection_area': intersection_count,
            'overall_overlap_ratio': overall_overlap_ratio,
            'pairwise_overlaps': pairwise_overlaps
        }

    def plot_coverage_over_time(self, global_ground_truth):
        """
        繪製覆蓋率隨時間變化的圖表，包括：
        - 每個機器人的覆蓋率
        - 所有機器人探索區域的交集
        - 所有機器人探索區域的聯集

        參數:
            global_ground_truth: 全局真實地圖
        """
        if not self.maps_history[0]:
            print("沒有記錄的地圖歷史，無法生成圖表")
            return

        # 計算每個時間點的覆蓋率指標
        time_steps = range(len(self.maps_history[0]))
        robot_coverage = [[] for _ in range(self.n_agent)]
        intersection_coverage = []
        union_coverage = []

        # 計算全局地圖的可探索區域總數
        total_explorable = np.sum(global_ground_truth == 255)

        for t in time_steps:
            # 獲取每個時間點每個機器人的地圖
            explored_masks = []
            for robot_id in range(self.n_agent):
                robot_map = self.maps_history[robot_id][t]
                robot_explored = (robot_map == 255)
                explored_masks.append(robot_explored)

                # 計算個別機器人的覆蓋率
                ratio = np.sum(robot_explored) / total_explorable if total_explorable > 0 else 0
                robot_coverage[robot_id].append(ratio)

            # 計算交集（所有機器人都探索的區域）
            intersection = np.ones_like(explored_masks[0], dtype=bool)
            for mask in explored_masks:
                intersection = intersection & mask

            # 計算聯集（至少一個機器人探索的區域）
            union = np.zeros_like(explored_masks[0], dtype=bool)
            for mask in explored_masks:
                union = union | mask

            # 計算覆蓋率
            intersection_ratio = np.sum(intersection) / total_explorable if total_explorable > 0 else 0
            union_ratio = np.sum(union) / total_explorable if total_explorable > 0 else 0

            intersection_coverage.append(intersection_ratio)
            union_coverage.append(union_ratio)

        # 創建圖表
        plt.figure(figsize=(12, 8))

        # 繪製每個機器人的覆蓋率曲線
        colors = ['b', 'r', 'g', 'c', 'm', 'y']
        for robot_id in range(self.n_agent):
            color = colors[robot_id % len(colors)]
            plt.plot(time_steps, robot_coverage[robot_id],
                    color=color, linestyle='-', linewidth=2,
                    label=f'Robot {robot_id+1}')

        # 繪製交集和聯集曲線
        plt.plot(time_steps, intersection_coverage, 'k--', linewidth=2, label='Intersection')
        plt.plot(time_steps, union_coverage, 'k-', linewidth=2, label='Union')

        # 添加標籤和標題
        plt.xlabel('Time (steps)', fontsize=14)
        plt.ylabel('Coverage', fontsize=14)
        plt.title('Coverage Over Time - Individual Robot Maps', fontsize=16)

        # 添加網格和圖例
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12, loc='best')

        # 設置y軸範圍
        plt.ylim(0, 1.05)

        # 保存圖片
        coverage_plot_path = os.path.join(self.save_dir, 'coverage_over_time.png')
        plt.savefig(coverage_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 保存數據到CSV文件以便後續分析
        try:
            import pandas as pd
            data_dict = {'Time': time_steps}
            for robot_id in range(self.n_agent):
                data_dict[f'Robot{robot_id+1}_Coverage'] = robot_coverage[robot_id]
            data_dict['Intersection'] = intersection_coverage
            data_dict['Union'] = union_coverage

            df = pd.DataFrame(data_dict)
            df.to_csv(os.path.join(self.save_dir, 'coverage_data.csv'), index=False)
            print(f"覆蓋率數據已保存到 {os.path.join(self.save_dir, 'coverage_data.csv')}")
        except ImportError:
            print("未安裝 pandas，跳過 CSV 文件生成")

        print(f"覆蓋率圖表已保存到 {coverage_plot_path}")

    def save_map_history(self, interval=10):
        """
        保存所有記錄的個人地圖歷史（每隔interval步保存一次）

        參數:
            interval: 保存間隔
        """
        if not self.maps_history[0]:
            print("沒有記錄的地圖歷史")
            return

        history_dir = os.path.join(self.save_dir, 'history')
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)

        # 計算需要的子圖佈局
        n_cols = min(3, self.n_agent)
        n_rows = (self.n_agent + n_cols - 1) // n_cols

        # 保存地圖歷史
        num_saved = 0
        for i in range(0, len(self.maps_history[0]), interval):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            if self.n_agent == 1:
                axes = np.array([axes])
            axes = axes.flatten() if self.n_agent > 1 else axes

            for robot_id in range(self.n_agent):
                if i < len(self.maps_history[robot_id]):
                    robot_map = self.maps_history[robot_id][i]

                    ax = axes[robot_id]
                    im = ax.imshow(robot_map, cmap='gray', vmin=0, vmax=255)
                    ax.set_title(f'Robot{robot_id+1} - Step {i}')
                    plt.colorbar(im, ax=ax, label='Map Value')
                    ax.axis('off')

            # 隱藏多餘的子圖
            for idx in range(self.n_agent, len(axes)):
                axes[idx].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(history_dir, f'individual_maps_history_{i:04d}.png'), dpi=150)
            plt.close(fig)
            num_saved += 1

        print(f"已保存個人地圖歷史，共 {num_saved} 幀")

    def cleanup(self):
        """清理資源"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = None

        # 釋放記憶體
        self.maps_history = [[] for _ in range(self.n_agent)]
        self.individual_maps = []
        self.is_tracking = False
        print("RobotIndividualMapTracker 已清理")
