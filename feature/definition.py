#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwu_agent.utils.common_func import attached, create_cls
from agent_target_dqn.conf.conf import Config

# The create_cls function is used to dynamically create a class. The first parameter of the function is the type name,
# and the remaining parameters are the attributes of the class, which should have a default value of None.
# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_act=None,
)


ActData = create_cls(
    "ActData",
    move_dir=None,
    use_talent=None,
)


SampleData = create_cls(
    "SampleData",
    obs=None,
    _obs=None,
    obs_legal=None,
    _obs_legal=None,
    act=None,
    rew=None,
    ret=None,
    done=None,
)

RelativeDistance = {
    "RELATIVE_DISTANCE_NONE": 0,
    "VerySmall": 1,
    "Small": 2,
    "Medium": 3,
    "Large": 4,
    "VeryLarge": 5,
}


RelativeDirection = {
    "East": 1,
    "NorthEast": 2,
    "North": 3,
    "NorthWest": 4,
    "West": 5,
    "SouthWest": 6,
    "South": 7,
    "SouthEast": 8,
}

DirectionAngles = {
    1: 0,
    2: 45,
    3: 90,
    4: 135,
    5: 180,
    6: 225,
    7: 270,
    8: 315,
}


"""def reward_process(end_dist, history_dist):
    # step reward
    # 步数奖励
    step_reward = -0.001

    # end reward
    # 终点奖励
    end_reward = -0.02 * end_dist

    # distance reward
    # 距离奖励
    dist_reward = 0.05 * history_dist

    return [step_reward + dist_reward + end_reward]"""

def reward_process(end_dist, 
                   history_dist, 
                   find_obstacle,
                   obstacle_dist,
                   find_treasure,
                   treasure_dist,
                   is_new_exploration, 
                   no_explore_steps,
                   step_count=0,
                   max_steps=1000,
                   is_collision=False):
    """
    优化后的迷宫探索奖励函数
    """

    # ===== 1. 探索效率奖励（增强 + 时间衰减） =====
    time_decay = 1 - min(step_count / max_steps, 1.0)
    explore_reward = 0.05 * time_decay if is_new_exploration else 0

    # ===== 2. 终点距离奖励（非线性 + 冲线鼓励） =====
    progress_ratio = 1 - end_dist
    end_reward = 0.02 * (progress_ratio ** 2)
    if end_dist < 0.03:
        end_reward += 1.0

    # ===== 3. 路径优化奖励（修正 history_dist 范围） =====
    MAX_HISTORY_DIST = 10 / 128  # ≈ 0.078125
    dist_reward = 0.015 * min(history_dist / MAX_HISTORY_DIST, 1.0)

    # ===== 4. 障碍物惩罚/奖励（缓冲带） =====
    if is_collision:
        obstacle_penalty = -2.0
    elif find_obstacle:
        if obstacle_dist < 0.01:
            obstacle_penalty = -1.0 * (0.01 - obstacle_dist)
        elif obstacle_dist < 0.05:
            obstacle_penalty = -0.05 * (0.05 - obstacle_dist)
        else:
            obstacle_penalty = 0.01 * obstacle_dist
    else:
        obstacle_penalty = 0.005  # 安全无障碍小奖励

    # ===== 5. 基础步数惩罚（略增） =====
    step_penalty = -0.003

    # ===== 6. 停滞惩罚（分段增长） =====
    if no_explore_steps < 100:
        stagnation_penalty = 0
    else:
        stagnation_penalty = -0.044 - 0.001 * (no_explore_steps - 100)

    # ===== 7. 宝箱相关奖励 =====
    if find_treasure:
        progress_ratio = 1 - treasure_dist
        treasure_end_reward = 0.015 * (progress_ratio ** 2)
        if treasure_dist < 0.03:
            treasure_end_reward += 1.5

    # ===== 8. 返回多个指标（便于训练监控） ====
    return [explore_reward, end_reward, dist_reward, obstacle_penalty, step_penalty, stagnation_penalty, treasure_end_reward]


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


@attached
def SampleData2NumpyData(g_data):
    return np.hstack(
        (
            np.array(g_data.obs, dtype=np.float32),
            np.array(g_data._obs, dtype=np.float32),
            np.array(g_data.obs_legal, dtype=np.float32),
            np.array(g_data._obs_legal, dtype=np.float32),
            np.array(g_data.act, dtype=np.float32),
            np.array(g_data.rew, dtype=np.float32),
            np.array(g_data.ret, dtype=np.float32),
            np.array(g_data.done, dtype=np.float32),
        )
    )


@attached
def NumpyData2SampleData(s_data):
    obs_data_size = Config.DIM_OF_OBSERVATION
    legal_data_size = Config.DIM_OF_ACTION_DIRECTION
    return SampleData(
        obs=s_data[:obs_data_size],
        _obs=s_data[obs_data_size : 2 * obs_data_size],
        obs_legal=s_data[2 * obs_data_size : 2 * obs_data_size + legal_data_size],
        _obs_legal=s_data[2 * obs_data_size + legal_data_size : 2 * obs_data_size + 2 * legal_data_size],
        act=s_data[-4],
        rew=s_data[-3],
        ret=s_data[-2],
        done=s_data[-1],
    )
