##############################################################################
# Name: test_driver_single_start.py
# [Inference] Driver for testing with single custom start point and individual map tracking
# 簡化版：單一起始點測試
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

# 每10步保存一次圖片
SAVE_IMAGE_INTERVAL = 10


def find_nearest_free_space(ground_truth, position, max_search_radius=50):
    """找到距離給定位置最近的自由空間"""
    x, y = int(position[0]), int(position[1])
    map_height, map_width = ground_truth.shape

    if ground_truth[y, x] == 255:
        return position.copy()

    for radius in range(1, max_search_radius + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if abs(dx) != radius and abs(dy) != radius:
                    continue

                nx, ny = x + dx, y + dy

                if 0 <= nx < map_width and 0 <= ny < map_height:
                    if ground_truth[ny, nx] == 255:
                        print(f"{YELLOW}找到最近的自由空間: ({nx}, {ny}){NC}")
                        return np.array([float(nx), float(ny)])

    print(f"{RED}警告: 未找到自由空間{NC}")
    return position.copy()


def create_custom_env_and_robots(map_index, n_agent, k_size, custom_start_pos=None, plot=False, track_individual_maps=False):
    """創建具有自定義起始位置的環境和機器人"""
    env = Env(map_index=map_index, n_agent=n_agent, k_size=k_size, plot=plot, track_individual_maps=track_individual_maps)

    if custom_start_pos is not None:
        custom_start_pos = np.array(custom_start_pos, dtype=np.float64)

        map_height, map_width = env.ground_truth_size
        custom_start_pos[0] = np.clip(custom_start_pos[0], 0, map_width-1)
        custom_start_pos[1] = np.clip(custom_start_pos[1], 0, map_height-1)

        print(f"{YELLOW}[調試] 檢查起始位置 {custom_start_pos}...{NC}")
        print(f"{YELLOW}[調試] 地圖值: {env.ground_truth[int(custom_start_pos[1]), int(custom_start_pos[0])]}{NC}")

        if env.ground_truth[int(custom_start_pos[1]), int(custom_start_pos[0])] != 255:
            print(RED, f"警告: 指定位置 {custom_start_pos} 不是自由空間", NC)
            print(YELLOW, "正在尋找最近的自由空間...", NC)

            custom_start_pos = find_nearest_free_space(env.ground_truth, custom_start_pos)
            print(GREEN, f"已移動到最近的自由空間: {custom_start_pos}", NC)
        else:
            print(GREEN, f"起始位置驗證通過，位於自由空間", NC)

        env.start_position = custom_start_pos
        for i in range(n_agent):
            env.all_robot_positions_belief[i] = [custom_start_pos.copy() for _ in range(n_agent)]
            env.all_graph_generator[i].route_node = [custom_start_pos.copy()]

        env.begin()

        print(GREEN, f"使用自定義起始位置: {custom_start_pos}", NC)

    robot_list = []
    all_robot_positions = []

    for i in range(n_agent):
        if custom_start_pos is not None:
            robot_position = custom_start_pos.copy()
        else:
            iter = min(i, len(env.all_node_coords[i])-1)
            robot_position = env.all_node_coords[i][iter]

        robot = Robot(robot_id=i, position=robot_position, plot=plot)
        robot_list.append(robot)
        all_robot_positions.append(robot_position)

    return env, robot_list, all_robot_positions


class CustomTestWorkerWithTracker(TestWorker):
    """繼承TestWorker並允許自定義環境，支持個人地圖追蹤"""

    def __init__(self, meta_agent_id, n_agent, policy_net, global_step, device='cuda',
                 greedy=False, save_image=False, custom_start_pos=None, output_dir=None,
                 map_index=None, track_individual_maps=False):
        self.device = device
        self.greedy = greedy
        self.n_agent = n_agent
        self.metaAgentID = meta_agent_id
        self.global_step = global_step
        self.node_padding_size = NODE_PADDING_SIZE
        self.k_size = K_SIZE
        self.save_image = save_image
        self.output_dir = output_dir
        self.track_individual_maps = track_individual_maps

        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        actual_map_index = map_index if map_index is not None else self.global_step
        self.env, self.robot_list, self.all_robot_positions = create_custom_env_and_robots(
            map_index=actual_map_index,
            n_agent=self.n_agent,
            k_size=self.k_size,
            custom_start_pos=custom_start_pos,
            plot=save_image,
            track_individual_maps=track_individual_maps
        )

        self.local_policy_net = policy_net
        self.perf_metrics = dict()
        self.max_node_coords = 0

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
        """運行episode - 每10步保存圖片，支持個人地圖追蹤"""
        done = False
        astar_unsuccessful = False

        if self.save_image:
            print(f"\n{GREEN}{'='*70}{NC}")
            print(f"{GREEN}開始Episode {curr_episode}{NC}")
            print(f"  保存間隔: 每 {SAVE_IMAGE_INTERVAL} 步")
            print(f"  輸出目錄: {self.output_dir}")
            if self.track_individual_maps:
                print(f"  個人地圖追蹤: 已啟用")
                print(f"  停止條件: Union coverage >= 99.5%")
            print(f"{GREEN}{'='*70}{NC}\n")

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

            # 保存個人地圖追蹤器快照
            if self.track_individual_maps and self.env.individual_map_tracker is not None:
                self.env.individual_map_tracker.save_current_maps(self.all_robot_positions)

            # 每10步保存一次圖片
            if self.save_image and (step % SAVE_IMAGE_INTERVAL == 0 or step == 0):
                self.save_current_state(step, travel_dist_list)

            # 每10步保存 individual maps 圖片（無論 save_image 設置如何）
            if self.track_individual_maps and self.env.individual_map_tracker is not None:
                if step % SAVE_IMAGE_INTERVAL == 0 or step == 0:
                    success = self.env.individual_map_tracker.save_current_frame(step)
                    if success:
                        save_dir = self.env.individual_map_tracker.save_dir
                        print(f"{YELLOW}[Individual Maps] 已保存步數 {step} 的圖片到: {save_dir}/individual_maps_step_{step:04d}.png{NC}")

            # 每10步顯示進度（不管是否保存圖片）
            if step % SAVE_IMAGE_INTERVAL == 0 or step == 0:
                current_time = time.time()
                elapsed_total = current_time - start_time
                elapsed_interval = current_time - last_save_time
                last_save_time = current_time

                # 顯示個人地圖追蹤信息和聯集覆蓋率
                individual_info = ""
                union_info = ""
                union_coverage = 0.0

                if self.track_individual_maps and self.env.individual_map_tracker is not None:
                    exploration_ratios = self.env.individual_map_tracker.get_exploration_ratio(self.env.ground_truth)
                    individual_info = " | Individual: " + ", ".join([f"R{i+1}:{r:.2%}" for i, r in enumerate(exploration_ratios)])

                    # 計算聯集覆蓋率
                    explored_masks = []
                    for robot_id in range(self.n_agent):
                        explored_mask = (self.env.individual_map_tracker.individual_maps[robot_id] == 255)
                        explored_masks.append(explored_mask)

                    union = np.zeros_like(explored_masks[0], dtype=bool)
                    for mask in explored_masks:
                        union = union | mask

                    total_explorable = np.sum(self.env.ground_truth == 255)
                    union_coverage = np.sum(union) / total_explorable if total_explorable > 0 else 0
                    union_info = f" | Union: {union_coverage:.2%}"

                    # 檢查停止條件: Union coverage >= 99.5%
                    if union_coverage >= 0.995:
                        done = True
                        print(f"{GREEN}✓ 聯集覆蓋率達到 99.5%，停止探索 (Union: {union_coverage:.2%}){NC}")

                print(f"{GREEN}步數 {step:4d} | "
                      f"探索率: {self.env.explored_rate:6.2%} | "
                      f"最大距離: {max(travel_dist_list):7.2f} | "
                      f"用時: {elapsed_total:6.1f}s | "
                      f"間隔: {elapsed_interval:4.1f}s{individual_info}{union_info}{NC}")

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

        # 如果啟用了個人地圖追蹤，保存統計信息
        if self.track_individual_maps and self.env.individual_map_tracker is not None:
            # 生成覆蓋率圖表
            tracker_output_dir = os.path.join(self.output_dir, 'individual_maps')
            if not os.path.exists(tracker_output_dir):
                os.makedirs(tracker_output_dir)

            # 臨時修改 tracker 的保存目錄
            original_save_dir = self.env.individual_map_tracker.save_dir
            self.env.individual_map_tracker.save_dir = tracker_output_dir

            # 生成分析
            self.env.individual_map_tracker.plot_coverage_over_time(self.env.ground_truth)

            # 計算重疊統計
            overlap_stats = self.env.individual_map_tracker.calculate_overlap()
            self.perf_metrics['overlap_ratio'] = overlap_stats['overall_overlap_ratio']
            self.perf_metrics['union_area'] = overlap_stats['union_area']
            self.perf_metrics['intersection_area'] = overlap_stats['intersection_area']

            # 獲取探索比例
            exploration_ratios = self.env.individual_map_tracker.get_exploration_ratio(self.env.ground_truth)
            for i, ratio in enumerate(exploration_ratios):
                self.perf_metrics[f'robot{i+1}_individual_explored'] = ratio

            # 保存地圖歷史
            if self.save_image:
                self.env.individual_map_tracker.save_map_history(interval=10)

            # 恢復原來的保存目錄
            self.env.individual_map_tracker.save_dir = original_save_dir

        if self.save_image:
            elapsed = time.time() - start_time
            print(f"\n{GREEN}{'='*70}{NC}")
            print(f"{GREEN}Episode {curr_episode} 完成統計:{NC}")
            print(f"  總步數: {step + 1}")
            print(f"  探索率: {self.env.explored_rate:.2%}")
            print(f"  最大距離: {max(travel_dist_list):.2f}")
            print(f"  成功完成: {done}")
            print(f"  總用時: {elapsed:.1f}s")
            print(f"  保存圖片數: {(step // SAVE_IMAGE_INTERVAL) + 1}")

            if self.track_individual_maps:
                print(f"\n{GREEN}個人地圖追蹤統計:{NC}")
                print(f"  重疊比例: {self.perf_metrics.get('overlap_ratio', 0):.2%}")
                print(f"  各機器人獨立探索率:")
                for i in range(self.n_agent):
                    ratio = self.perf_metrics.get(f'robot{i+1}_individual_explored', 0)
                    print(f"    Robot {i+1}: {ratio:.2%}")

            print(f"{GREEN}{'='*70}{NC}")

        return True


def run_test():
    """運行測試"""

    # ============================================================
    # 配置區域：修改這裡來自定義測試參數
    # ============================================================

    # 自定義起始位置 [x, y]（設為 None 使用隨機位置）
    CUSTOM_START_POS = [100, 100]

    # 指定使用的地圖索引（設為None則每次測試使用不同地圖）
    MAP_INDEX = None

    # 是否啟用個人地圖追蹤
    TRACK_INDIVIDUAL_MAPS = True

    # 輸出目錄
    OUTPUT_DIR = 'results_single_start'

    # ============================================================

    # 創建輸出目錄
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 創建 CSV 文件記錄數據
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    csv_file_name = f"data_{current_datetime}.csv"
    csv_file_path = os.path.join(OUTPUT_DIR, csv_file_name)

    # 定義CSV欄位
    fieldnames = ['eps', 'num_robots', 'max_dist', 'steps', 'explored', 'success', 'connectivity']

    if TRACK_INDIVIDUAL_MAPS:
        fieldnames.extend(['overlap_ratio', 'union_area', 'intersection_area'])
        for i in range(NUM_ROBOTS_MAX):
            fieldnames.append(f'robot{i+1}_individual_explored')

    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    # 載入模型
    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    if not os.path.exists(MODEL_PATH):
        print(f"{RED}錯誤: 在 {MODEL_PATH} 找不到模型檔案{NC}")
        return

    if device == torch.device('cuda'):
        checkpoint = torch.load(MODEL_PATH)
    else:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

    global_network.load_state_dict(checkpoint['policy_model'])

    print(f"{GREEN}已載入模型: {MODEL_PATH}{NC}")
    print(f"{GREEN}起始位置: {CUSTOM_START_POS if CUSTOM_START_POS else '隨機'}{NC}")
    print(f"{GREEN}個人地圖追蹤: {'已啟用' if TRACK_INDIVIDUAL_MAPS else '未啟用'}{NC}")
    print(f"{GREEN}將運行 {NUM_TEST} 次測試{NC}\n")

    dist_history = []
    eps_skipped = []

    for test_run in range(NUM_TEST):
        n_agent = np.random.randint(NUM_ROBOTS_MIN, NUM_ROBOTS_MAX+1, 1)[0]

        print(f"\n{GREEN}===== 測試 {test_run+1}/{NUM_TEST} ====={NC}")
        print(f"{GREEN}機器人數量: {n_agent}{NC}")

        # 創建 Worker
        worker = CustomTestWorkerWithTracker(
            meta_agent_id=0,
            n_agent=n_agent,
            policy_net=global_network,
            global_step=test_run,
            device=device,
            save_image=SAVE_GIFS and test_run == 0,  # 只保存第一次運行的圖片
            greedy=True,
            custom_start_pos=CUSTOM_START_POS,
            output_dir=OUTPUT_DIR,
            map_index=MAP_INDEX,
            track_individual_maps=TRACK_INDIVIDUAL_MAPS
        )

        # 運行測試
        success = worker.work(test_run)

        if success:
            perf_metrics = worker.perf_metrics
            dist_history.append(perf_metrics['travel_dist'])

            # 記錄數據
            result = {
                'eps': test_run,
                'num_robots': n_agent,
                'max_dist': perf_metrics['travel_dist'],
                'steps': perf_metrics['travel_steps'],
                'explored': perf_metrics['explored_rate'],
                'success': perf_metrics['success_rate'],
                'connectivity': perf_metrics['connectivity_rate']
            }

            # 添加個人地圖追蹤數據
            if TRACK_INDIVIDUAL_MAPS:
                result['overlap_ratio'] = perf_metrics.get('overlap_ratio', 0)
                result['union_area'] = perf_metrics.get('union_area', 0)
                result['intersection_area'] = perf_metrics.get('intersection_area', 0)

                for i in range(NUM_ROBOTS_MAX):
                    result[f'robot{i+1}_individual_explored'] = perf_metrics.get(f'robot{i+1}_individual_explored', 0)

            # 寫入CSV
            with open(csv_file_path, mode='a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow(result)

            tracker_info = ""
            if TRACK_INDIVIDUAL_MAPS:
                tracker_info = f", 重疊: {perf_metrics.get('overlap_ratio', 0):.2%}"

            print(GREEN, f"[測試 {test_run+1}] 步數: {perf_metrics['travel_steps']}, " +
                  f"探索率: {perf_metrics['explored_rate']:.2%}, " +
                  f"最大距離: {perf_metrics['travel_dist']:.2f}" +
                  tracker_info, NC)
        else:
            eps_skipped.append(test_run)
            print(RED, f"[測試 {test_run+1}] 測試失敗", NC)

    # 排序 CSV 文件
    df = pd.read_csv(csv_file_path)
    sorted_df = df.sort_values(by='eps')
    sorted_df.to_csv(csv_file_path, index=False)

    # 顯示統計
    print(f"\n{GREEN}{'='*70}{NC}")
    print(f"{GREEN}測試完成統計:{NC}")
    print(f"  總測試數: {NUM_TEST}")
    print(f"  成功測試數: {len(dist_history)}")
    print(f"  失敗測試數: {len(eps_skipped)}")
    if dist_history:
        print(f"  平均(最大)長度: {np.array(dist_history).mean():.2f}")
        print(f"  長度標準差: {np.array(dist_history).std():.2f}")
    if eps_skipped:
        print(f"  跳過的episode: {eps_skipped}")
    print(f"  數據儲存在: {csv_file_path}")
    print(f"{GREEN}{'='*70}{NC}")


if __name__ == '__main__':
    print(f"{GREEN}歡迎使用 IR2-MARL 單一起始點探索測試系統！{NC}")
    run_test()
