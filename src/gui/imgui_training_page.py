import os
import pickle
from argparse import Namespace
from enum import Enum

import imgui
import numpy as np
import pandas as pd
import traceback

from train_module import TrainManager
from graphic_module import GraphicManager
from gui import global_var as g
from gui import common
from gui import components as imgui_c
from geo import Road, Building, Region
from style_module import StyleManager
from utils import RoadLevel, RoadState
from utils import io_utils, graphic_uitls
from gui.animation_module import Animation
from DDPG import env as env
from DDPG.utils.reward_utils import *
print('training page loaded')

mSelectStartPointMode = False
mRoadInterpolateValue = 0.5
mRoadLevelStrs = [str(key) for key in RoadLevel]
mRoadLevels = [key for key in RoadLevel]
mCurrentRoadLevelIdx = 3

mNewRoads: dict[int, pd.Series] = {}

mRoadPtSeqData: list = []
mRoadAnimationData: dict = {}
mRoadGrowAnimation: Animation = Animation.blank()

mRoadNetAnimation: Animation = Animation.blank()
mRoadNetAnimationTimeGap = 0.1
mSyncMode = True
mNumAgents = 3


def show():
    global mRoadInterpolateValue, mNewRoads, mSelectStartPointMode, mCurrentRoadLevelIdx
    global mRoadPtSeqData, mRoadAnimationData, mRoadGrowAnimation
    global mRoadNetAnimation, mRoadNetAnimationTimeGap, mSyncMode, mNumAgents
    imgui.push_id('training page')
    if imgui.tree_node('训练工具', flags=imgui.TREE_NODE_DEFAULT_OPEN):
        imgui.text('最新版训练工具')
        _show_train_configs()
        if TrainManager.I.is_stopped:
            imgui.text('训练未开始')
            if imgui.button('TRAIN', width=imgui.get_content_region_available_width(), height=30 * g.GLOBAL_SCALE):
                _train()
        if TrainManager.I.is_paused:
            imgui.text('训练暂停')
            if imgui.button('CONTINUE', width=imgui.get_content_region_available_width(), height=30 * g.GLOBAL_SCALE):
                TrainManager.I.continue_train()
        if TrainManager.I.is_training:
            imgui.text('训练中')
            if imgui.button('PAUSE', width=imgui.get_content_region_available_width(), height=30 * g.GLOBAL_SCALE):
                TrainManager.I.pause_train()
            if imgui.button('STOP', width=imgui.get_content_region_available_width(), height=30 * g.GLOBAL_SCALE):
                TrainManager.I.stop_train()
        if 'save_buffer' in _shared_data:
            if not _shared_data['save_buffer']:
                if imgui.button('SAVE BUFFER', width=imgui.get_content_region_available_width(),
                                height=30 * g.GLOBAL_SCALE):
                    _shared_data['save_buffer'] = True
                if imgui.is_item_hovered():
                    imgui.set_tooltip('默认保存位置为src/logs文件夹\n'
                                      '如果您使用的是机械硬盘，该操作可能会消耗大量时间\n'
                                      '10_000条数据大约消耗680MB的空间（压缩前）， 压缩后大约8MB')
            else:
                imgui.button('SAVING...', width=imgui.get_content_region_available_width(), height=30 * g.GLOBAL_SCALE)
        _display_train_events()
        _display_timer_debug()
        _display_reward_debug()
        imgui.tree_pop()


    imgui.pop_id()


class ArgType(Enum):
    NONE = -1
    INTEGER = 0
    FLOAT = 1
    STRING = 2
    PATH = 3
    BOOLEAN = 4
    DICT = 5
    FLOAT2 = 6
    INT2 = 7


def _get_arg_types(args: dict):
    arg_types_dict = {}
    for key in args.keys():
        value = args[key]
        if isinstance(value, bool):
            arg_types_dict[key] = ArgType.BOOLEAN
            continue
        if isinstance(value, int):
            arg_types_dict[key] = ArgType.INTEGER
            continue
        if isinstance(value, float):
            arg_types_dict[key] = ArgType.FLOAT
            continue
        if isinstance(value, str):
            if 'path' in key or 'folder' in key or os.path.isfile(value) or os.path.isdir(value):
                arg_types_dict[key] = ArgType.PATH
                continue
            arg_types_dict[key] = ArgType.STRING
            continue
        if value is None:
            arg_types_dict[key] = ArgType.NONE
            continue
        if isinstance(value, dict):
            arg_types_dict[key] = ArgType.DICT
            continue
        if isinstance(value, tuple):
            if len(value) == 2:
                if isinstance(value[0], int):
                    arg_types_dict[key] = ArgType.INT2
                    continue
                if isinstance(value[0], float):
                    arg_types_dict[key] = ArgType.FLOAT2
                    continue
        raise Exception(f'遇到了不能识别的类型{value}, type={type(value)}，请检查arg的{key}数据')
    return arg_types_dict


def _arg_editor(args: dict, arg_type_dict: dict):
    width = g.LEFT_WINDOW_WIDTH / 2
    style: imgui.core.GuiStyle = imgui.get_style()
    for key in args.keys():
        tp: ArgType = arg_type_dict[key]
        if tp == ArgType.INTEGER:
            imgui.set_next_item_width(width)
            _, args[key] = imgui.input_int(key, args[key])
            continue
        if tp == ArgType.FLOAT:
            imgui.set_next_item_width(width)
            _, args[key] = imgui.input_float(key, args[key],format='%.5f')
            continue
        if tp == ArgType.BOOLEAN:
            imgui.set_next_item_width(width)
            _, args[key] = imgui.checkbox(key, args[key])
            continue
        if tp == ArgType.STRING:
            imgui.set_next_item_width(width)
            _, args[key] = imgui.input_text(key, args[key])
            continue
        if tp == ArgType.PATH:

            imgui.set_next_item_width(width - 45 * g.GLOBAL_SCALE - style.item_spacing[0])
            imgui.push_id(key)
            _, args[key] = imgui.input_text('', args[key])
            imgui.same_line()
            if imgui.button('...', width=45 * g.GLOBAL_SCALE):
                args[key] = io_utils.open_file_window()
            imgui.pop_id()
            imgui.same_line()
            imgui.text(key)
            continue
        if tp == ArgType.FLOAT2:
            imgui.set_next_item_width(width)
            _, args[key] = imgui.input_float2(key, *args[key])
            continue
        if tp == ArgType.INT2:
            imgui.set_next_item_width(width)
            _, args[key] = imgui.input_int2(key, *args[key])
            continue


_train_args = vars(TrainManager.I.default_train_args_copy)
_env_args = vars(TrainManager.I.default_env_args_copy)
_gui_args = vars(TrainManager.I.default_gui_args_copy)
_train_args_types = _get_arg_types(_train_args)
_env_args_types = _get_arg_types(_env_args)
_gui_args_types = _get_arg_types(_gui_args)
_shared_data = {'slow_ratio': 1}


def _show_train_configs():
    imgui.text('env_args')
    imgui.indent()
    _arg_editor(_env_args, _env_args_types)
    imgui.unindent()
    imgui.text('train_args')
    imgui.indent()
    _arg_editor(_train_args, _train_args_types)
    imgui.unindent()
    imgui.text('gui_args')
    imgui.indent()
    _arg_editor(_gui_args, _gui_args_types)
    _, _shared_data['slow_ratio'] = imgui.slider_int('slow_ratio', _shared_data['slow_ratio'], 1, 120)
    imgui.unindent()


def _train():
    TrainManager.I.start_train(
        env_args=Namespace(**_env_args),
        train_args=Namespace(**_train_args),
        gui_args=Namespace(**_gui_args),
        shared_data=_shared_data
    )


def _display_train_events():
    style: imgui.core.GuiStyle = imgui.get_style()
    stage = ''
    if 'stage' in _shared_data:
        stage = _shared_data["stage"]
        imgui.text(f'当前状态: {stage}')
    if 'epoch' in _shared_data:
        epoch = _shared_data["epoch"]
        if stage == 'warmup':
            imgui.progress_bar(
                epoch / _train_args["warmup_epochs"], (g.LEFT_WINDOW_WIDTH / 2, imgui.get_frame_height()))
            imgui.same_line()
            imgui.text(f'epoch: {epoch} / {_train_args["warmup_epochs"]}')
        elif stage == 'train':
            imgui.progress_bar(
                epoch / _train_args["train_epochs"], (g.LEFT_WINDOW_WIDTH / 2, imgui.get_frame_height()))
            imgui.same_line()
            imgui.text(f'epoch: {epoch} / {_train_args["train_epochs"]}')
    if 'episode_step' in _shared_data:
        imgui.progress_bar(
            _shared_data["episode_step"] / (_env_args["max_episode_step"] + 1),
            (g.LEFT_WINDOW_WIDTH / 2, imgui.get_frame_height()))
        imgui.same_line()
        imgui.text(f'step: ({_shared_data["episode_step"]})/ max({_env_args["max_episode_step"]})')
    imgui.separator()
    if 'sigma' in _shared_data:
        imgui.text(f'sigma: {_shared_data["sigma"]}')
    if 'action_0' in _shared_data:
        imgui.text(f'action_0: {_shared_data["action_0"]}')
    graph_size = (imgui.get_content_region_available_width(), 80 * g.GLOBAL_SCALE)
    if 'mean_return_list' in _shared_data and len(_shared_data["mean_return_list"] > 0):
        imgui.text(f'mean_return(last): {_shared_data["mean_return_list"][-1]}')
        imgui.plot_lines('mean_return_list', _shared_data["mean_return_list"], graph_size=graph_size)
    if 'mean_critic_loss' in _shared_data:
        imgui.text(f'mean_critic_loss(last): {_shared_data["mean_critic_loss"][-1]}')
        imgui.plot_lines('mean_critic_loss', _shared_data["mean_critic_loss"], graph_size=graph_size)
    if 'mean_actor_loss' in _shared_data:
        imgui.text(f'mean_actor_loss(last): {_shared_data["mean_actor_loss"][-1]}')
        imgui.plot_lines('mean_actor_loss', _shared_data["mean_actor_loss"], graph_size=graph_size)
    imgui.separator()


def _display_timer_debug():
    if 'time' not in _shared_data:
        return
    if imgui.tree_node('time debugger'):
        time_data = _shared_data['time']
        time_data = dict(sorted(time_data.items()))
        for level in time_data.keys():
            indent_value = level * 20 * g.GLOBAL_SCALE + 1
            imgui.indent(indent_value)
            imgui.text(f'level: {level}')
            time_dict = time_data[level]
            for _name, _time in time_dict.items():
                if _time is None:
                    _time_str = 'running...'
                else:
                    _time_str = f'{_time * 1000:.0f}ms'
                imgui.text(f'-{_name}: {_time_str}')
            imgui.unindent(indent_value)
        imgui.tree_pop()


_colors = [
    (0.992, 0.733, 0.176, 1.0),   # 黄色
    (0.678, 0.416, 0.71, 1.0),     # 紫色
    (0.859, 0.318, 0.42, 1.0),     # 红色
    (0.259, 0.561, 0.678, 1.0),    # 蓝色
    (0.184, 0.749, 0.686, 1.0),    # 青绿色
    (0.38, 0.58, 0.337, 1.0),      # 绿色
    (1.0, 0.522, 0.322, 1.0)       # 橙色
]

_reward_range = 500


def _display_reward_debug():
    if 'reward_info' not in _shared_data:
        return
    global _reward_range
    if imgui.tree_node('reward debugger'):
        _, _reward_range = imgui.slider_float('reward_range', _reward_range, 0, 1000)
        total_width = imgui.get_content_region_available_width()
        half_width = total_width / 2
        reward_info = _shared_data['reward_info']
        imgui.push_style_var(imgui.STYLE_ITEM_SPACING, (0, imgui.get_style().item_spacing[1]))
        for agent_idx, reward_dict in reward_info.items():
            imgui.text(f'agent-{agent_idx}')
            i = 0
            cursor_pos_y = imgui.get_cursor_pos_y()
            left_width = 0
            right_width = 0
            for reward_name, reward_value in reward_dict.items():
                if reward_name not in REWARD_KEYS:
                    continue
                if reward_value == 0:
                    continue
                item_width = float(abs(reward_value)) * total_width / _reward_range
                if item_width < 10:
                    item_width = 10
                if reward_value < 0:
                    imgui.set_cursor_pos_x(half_width - left_width - item_width)
                    imgui.set_cursor_pos_y(cursor_pos_y)
                    left_width += item_width
                else:
                    imgui.set_cursor_pos_x(half_width + right_width)
                    imgui.set_cursor_pos_y(cursor_pos_y)
                    right_width += item_width
                imgui.push_style_color(imgui.COLOR_BUTTON, *_colors[i])
                imgui.button('', item_width)
                imgui.pop_style_color()
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f'{REWARD_DISPLAY_NAMES[reward_name]}({reward_name}): {reward_value}')
                i += 1
        imgui.pop_style_var()
        imgui.tree_pop()


