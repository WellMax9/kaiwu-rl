#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

"""

import numpy as np
import math
import queue
from agent_target_dqn.feature.definition import RelativeDistance, RelativeDirection, DirectionAngles, reward_process

def norm(v, max_v, min_v=0):
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)

def get_neighbors(pos):
    neighbors = []
    i = pos[0]
    j = pos[1]
    if i >= 5 and i < 10:
        neighbors.append([i+1, j])
    if i <= 5 and i > 0:
        neighbors.append([i-1, j])
        
    if j >= 5 and j < 10:
        neighbors.append([i, j+1])
    if j <= 5 and j > 0:
        neighbors.append([i,j-1])
        
    return neighbors
            
        
    
def get_obstacle(map_info):
    q = queue.Queue()
    start = [5, 5]
    q.put(start)
    
    while(not q.empty()):
        pos = q.get()
        i = pos[0]
        j = pos[1]
        if map_info[i]["values"][j] == 0:
            return pos
        else:
            for neighbor in get_neighbors(pos):
                q.put(neighbor)
    return None
                
    
        
    

class Preprocessor:
    def __init__(self) -> None:
        self.move_action_num = 8
        self.feature_treasure_pos = np.zeros(6)
        self.reset()

    def reset(self):
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array((0, 0))
        self.end_pos = None
        self.is_end_pos_found = False
        self.history_pos = []
        self.bad_move_ids = set()
        self.exploration_tracker = BlockExplorationTracker()
        self.no_explore_steps = 0  # 重置连续未探索步数
        self.current_exploration_status = False
        self.found_obstacle = False
        self.found_treasure = False
        self.feature_end_pos = np.zeros(6)

    def _get_pos_feature(self, found, cur_pos, target_pos):
        relative_pos = tuple(y - x for x, y in zip(cur_pos, target_pos))
        dist = np.linalg.norm(relative_pos)
        target_pos_norm = norm(target_pos, 128, -128)
        feature = np.array(
            (
                found,
                norm(relative_pos[0] / max(dist, 1e-4), 1, -1),
                norm(relative_pos[1] / max(dist, 1e-4), 1, -1),
                target_pos_norm[0],
                target_pos_norm[1],
                norm(dist, 1.41 * 128),
            ),
        )
        return feature

    def pb2struct(self, frame_state, last_action):
        obs, _ = frame_state
        self.step_no = obs["frame_state"]["step_no"]

        hero = obs["frame_state"]["heroes"][0]
        self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])

        #更新探索区域
        self.feature_exploration = self._get_exploration_features()
        is_new_exploration = self.exploration_tracker.mark_explored(*self.cur_pos)
        
        # 更新连续未探索步数
        if is_new_exploration:
            self.no_explore_steps = 0  # 重置计数器
            self.current_exploration_status = True
        else:
            self.no_explore_steps += 1  # 增加未探索步数
            self.current_exploration_status = False
        
        # History position
        # 历史位置
        self.history_pos.append(self.cur_pos)
        if len(self.history_pos) > 10:
            self.history_pos.pop(0)

        # End position
        # 终点位置
        for organ in obs["frame_state"]["organs"]:
            if organ["sub_type"] == 4:
                end_pos_dis = RelativeDistance[organ["relative_pos"]["l2_distance"]]
                end_pos_dir = RelativeDirection[organ["relative_pos"]["direction"]]
                if organ["status"] != -1:
                    self.end_pos = (organ["pos"]["x"], organ["pos"]["z"])
                    self.is_end_pos_found = True
                # if end_pos is not found, use relative position to predict end_pos
                # 如果终点位置未找到，使用相对位置预测终点位置
                elif (not self.is_end_pos_found) and (
                    self.end_pos is None
                    or self.step_no % 100 == 0
                    or self.end_pos_dir != end_pos_dir
                    or self.end_pos_dis != end_pos_dis
                ):
                    distance = end_pos_dis * 20
                    theta = DirectionAngles[end_pos_dir]
                    delta_x = distance * math.cos(math.radians(theta))
                    delta_z = distance * math.sin(math.radians(theta))

                    self.end_pos = (
                        max(0, min(128, round(self.cur_pos[0] + delta_x))),
                        max(0, min(128, round(self.cur_pos[1] + delta_z))),
                    )

                    self.end_pos_dir = end_pos_dir
                    self.end_pos_dis = end_pos_dis

            # 观测到的宝箱位置
            if organ["sub_type"] == 1: #宝箱
                if organ["status"] == 1:
                    #观测到的宝箱位置特征
                    self.found_treasure = True
                    self.feature_treasure_pos = self._get_pos_feature(1, self.cur_pos, (organ["pos"]["x"], organ["pos"]["z"]))


        self.last_pos_norm = self.cur_pos_norm
        self.cur_pos_norm = norm(self.cur_pos, 128, -128)
        self.feature_end_pos = self._get_pos_feature(self.is_end_pos_found, self.cur_pos, self.end_pos)

        # History position feature
        # 历史位置特征
        self.feature_history_pos = self._get_pos_feature(1, self.cur_pos, self.history_pos[0])

        #Find Obstacle
        obstacle_relative_pos = get_obstacle(obs["map_info"])
        if (obstacle_relative_pos):
            self.found_obstacle = True
            obstacle_pos = (obstacle_relative_pos[0] + self.cur_pos[0] - 5,
                            obstacle_relative_pos[1] + self.cur_pos[1] - 5)
            self.feature_obstacle_pos = self._get_pos_feature(1, self.cur_pos, obstacle_pos)
        else:
            self.found_obstacle = False
            self.feature_obstacle_pos = self._get_pos_feature(0,self.cur_pos, (127,127))
        
        self.move_usable = True
        self.last_action = last_action
        
    def _get_exploration_features(self):
        """计算区块探索相关特征"""
        x, y = self.cur_pos
        
        # 1. 当前区块探索状态
        current_block_explored = self.exploration_tracker.get_bit(x, y)
        
        # 2. 已探索区块比例
        explored_count = self.exploration_tracker.get_explored_count()
        explored_ratio = explored_count / 1024.0
        
        # 3. 局部区域探索状态 (3x3区块)
        local_exploration = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                bx, by = x // 4 + dx, y // 4 + dy
                if 0 <= bx < 32 and 0 <= by < 32:
                    # 计算实际坐标（区块中心）
                    cx, cy = bx * 4 + 2, by * 4 + 2
                    local_exploration.append(self.exploration_tracker.get_bit(cx, cy))
                else:
                    local_exploration.append(0)  # 边界外视为未探索
        
        # 4. 终点方向区块探索状态
        if self.end_pos:
            end_x, end_y = self.end_pos
            direction_vector = (end_x - x, end_y - y)
            distance = np.linalg.norm(direction_vector)
            if distance > 0:
                # 终点方向上的区块
                target_dx, target_dy = direction_vector[0]/distance, direction_vector[1]/distance
                target_x, target_y = x + target_dx * 4, y + target_dy * 4
                target_explored = self.exploration_tracker.get_bit(target_x, target_y)
            else:
                target_explored = 1  # 已在终点
        else:
            target_explored = 0
        
        # 组合所有探索特征
        return np.array([
            current_block_explored,
            explored_ratio,
            np.mean(local_exploration),
            target_explored
        ])
        
    def process(self, frame_state, last_action):
        self.pb2struct(frame_state, last_action)

        # Legal action
        # 合法动作
        legal_action = self.get_legal_action()

        # Feature
        # 特征
        feature = np.concatenate([self.cur_pos_norm, 
                                  self.feature_end_pos, 
                                  self.feature_history_pos, 
                                  self.feature_obstacle_pos, #新增障碍特征
                                  self.feature_exploration,  # 新增探索特征
                                  self.feature_treasure_pos, #观测到的宝箱特征
                                  legal_action])

        return (
            feature,
            legal_action,
            reward_process(self.feature_end_pos[-1], 
                           self.feature_history_pos[-1],
                           self.found_obstacle,
                           self.feature_obstacle_pos[-1],
                           self.found_treasure,
                           self.feature_treasure_pos[-1],
                           self.current_exploration_status,
                           self.no_explore_steps,
                           self.step_no,
                           ),
        )

    def get_legal_action(self):
        # if last_action is move and current position is the same as last position, add this action to bad_move_ids
        # 如果上一步的动作是移动，且当前位置与上一步位置相同，则将该动作加入到bad_move_ids中
        if (
            abs(self.cur_pos_norm[0] - self.last_pos_norm[0]) < 0.001
            and abs(self.cur_pos_norm[1] - self.last_pos_norm[1]) < 0.001
            and self.last_action > -1
        ):
            self.bad_move_ids.add(self.last_action)
        else:
            self.bad_move_ids = set()

        legal_action = [self.move_usable] * self.move_action_num
        for move_id in self.bad_move_ids:
            legal_action[move_id] = 0

        if self.move_usable not in legal_action:
            self.bad_move_ids = set()
            return [self.move_usable] * self.move_action_num

        return legal_action

#下面是我的补充代码
class BlockExplorationTracker:
    def __init__(self):
        self.bitmap = [0] * 1024

    def get_bit(self, x, y):
        bx, by = int(x) // 4, int(y) // 4
        global_idx = bx + by * 32
        return self.bitmap[global_idx]

    def mark_explored(self, x, y):
        bx, by = int(x) // 4, int(y) // 4
        global_idx = bx + by * 32
        was_explored = self.bitmap[global_idx]
        self.bitmap[global_idx] = 1
        return not was_explored

    def get_explored_count(self):
        return np.sum(self.bitmap)
