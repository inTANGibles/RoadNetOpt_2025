import collections
import gzip
import math
import pickle
import traceback
from argparse import Namespace
from typing import Optional

import imgui
import numpy as np

from graphic_module import SimpleTexture, GraphicManager
from gui import global_var as g
from gui.icon_module import Spinner
from utils.io_utils import open_file_window
from DDPG.utils.reward_utils import *

mFilePath = ''
mReplayBuffer: Optional[collections.deque] = None
mMetaData: Optional[dict] = None

# region [caches]
mCachedReplayBufferSize: int = 0

mStateArray: Optional[np.ndarray] = None  # 记录了所有的state
mActionArray: Optional[np.ndarray] = None
mRewardArray: Optional[np.ndarray] = None
mNextStateArray: Optional[np.ndarray] = None
mDoneArray: Optional[np.ndarray] = None
mAllDoneArray: Optional[np.ndarray] = None

mActionArrayHistogram0: Optional[np.ndarray] = None
mActionArrayHistogram1: Optional[np.ndarray] = None

mAllDoneIndices: Optional[np.ndarray] = None  # All Done为True的idx序号
mEpochs: int = 0  # 总共有多少epoch
mNumAgents: int = 0
mNumActions: int = 0

mTrainArgs: Optional[Namespace] = None
mEnvArgs: Optional[Namespace] = None
mGuiArgs: Optional[Namespace] = None
mUserName = ''
mSaveTime = ''
mVersion = 000
mActionSpaceBound: Optional[np.ndarray] = None
mActionSpaceBoundMove: Optional[np.ndarray] = None

mRewardInfoBuffer: Optional[collections.deque] = None

# 对于某个epoch的切片
mStateSlice: Optional[np.ndarray] = None
mActionSlice: Optional[np.ndarray] = None
mRewardSlice: Optional[np.ndarray] = None
mNextStateSlice: Optional[np.ndarray] = None
mDoneSlice: Optional[np.ndarray] = None
mAllDoneSlice: Optional[np.ndarray] = None

mNumSteps: int = 0  # 当前epoch有多少步

# 对于某个step 和agent
mAction = None
mReward = None
mDone = None
mScaledAction = None
mActionDx = None
mActionDy = None

mRewardDetails = {}
mFinalRewardDetails = {}

# endregion

# region [parameters]
mCurrViewingEpoch = 0  # 正在查看的epoch
mCurrViewingStep = 0  # 正在查看的step

mCurrViewingAgentIdx = 0  # 正在查看的agent的index

mImgSize = 256  # 查看图片的大小

# endregion

# region [variables]
mCurrViewingBufferIdx = 0  # 正在查看第几个buffer， 由mCurrViewingEpoch与mCurrViewingStep决定
mIsLoadingBuffer = False


# endregion

def show():
    global mReplayBuffer
    if not g.mReplayBufferViewerWindowOpened:
        return
    flags = imgui.WINDOW_MENU_BAR
    expanded, g.mReplayBufferViewerWindowOpened = imgui.begin('replay buffer viewer', True, flags)
    _show_menu_bar()
    if mReplayBuffer is None:
        imgui.set_cursor_pos_x(imgui.get_content_region_available_width() / 2 - 100 * g.GLOBAL_SCALE)
        imgui.set_cursor_pos_y(imgui.get_window_height() / 2 - 12 * g.GLOBAL_SCALE)
        if not mIsLoadingBuffer:
            if imgui.button('Open File', 200 * g.GLOBAL_SCALE, 24 * g.GLOBAL_SCALE):
                _open_file_dialog()
        else:
            imgui.button('Loading...', 200 * g.GLOBAL_SCALE, 24 * g.GLOBAL_SCALE)
    Spinner.spinner('load_data')
    if mReplayBuffer is not None and not mIsLoadingBuffer:
        try:
            _show_buffer_viewer()
        except Exception as e:
            print(e)
    imgui.end()


def _show_menu_bar():
    if imgui.begin_menu_bar().opened:
        if imgui.begin_menu('File').opened:
            clicked, state = imgui.menu_item('Open File')
            if clicked:
                _open_file_dialog()
            clicked, state = imgui.menu_item('Close')
            if clicked:
                g.mReplayBufferViewerWindowOpened = False
            imgui.end_menu()
        imgui.end_menu_bar()


def _open_file_dialog():
    global mFilePath
    file_path = open_file_window(filetypes=[('Replay Buffer', '.pkl.gz')])
    if file_path is not None and file_path != '':
        mFilePath = file_path
        Spinner.start('load_data', target=_load_data, args=(file_path,))


def _rebuild_cache():
    """
    重建缓存数据
    buffer每一行数据：(epoch n)=> [state, action, reward, next_state, done, Done]
    :return:
    """
    # 等待replay buffer已经加载完毕后继续执行
    assert mReplayBuffer is not None
    assert mMetaData is not None
    # 计算长度
    global mCachedReplayBufferSize
    buffer_size = len(mReplayBuffer)
    mCachedReplayBufferSize = buffer_size

    # 计算数据维度
    global mNumAgents, mNumActions
    f_state, f_action, f_reward, f_next_state, f_done, f_all_done = mReplayBuffer[0]
    # (the word 'f' is short for first)
    # f_state :   (channel, height, width)
    # f_action:   (num_agents, num_actions)
    # f_reward:   (num_agents, 1)
    # f_done:     (num_agents, 1)
    # f_all_done: bool

    channel, height, width = f_state.shape
    num_agents, num_actions = f_action.shape
    mNumAgents, mNumActions = num_agents, num_actions

    # 转换为numpy
    global mStateArray, mActionArray, mRewardArray, mNextStateArray, mDoneArray, mAllDoneArray
    mStateArray = np.zeros((buffer_size, channel, height, width), dtype=np.uint8)
    mActionArray = np.zeros((buffer_size, num_agents, num_actions), dtype=np.float32)
    mRewardArray = np.zeros((buffer_size, num_agents, 1), dtype=np.float32)
    mNextStateArray = np.zeros((buffer_size, channel, height, width), dtype=np.uint8)
    mDoneArray = np.zeros((buffer_size, num_agents, 1), dtype=np.bool_)
    mAllDoneArray = np.zeros(buffer_size, dtype=np.bool_)
    for i in range(buffer_size):
        state, action, reward, next_state, done, all_done = mReplayBuffer[i]
        mStateArray[i] = state
        mActionArray[i] = action
        mRewardArray[i] = reward
        mNextStateArray[i] = next_state
        mDoneArray[i] = done
        mAllDoneArray[i] = all_done
    # 计算Done的位置和epoch数量
    global mAllDoneIndices, mEpochs
    mAllDoneIndices = np.where(mAllDoneArray)[0]
    mEpochs = len(mAllDoneIndices)
    # 计算action的分布
    action_array_0 = mActionArray[:, :, 0].flatten()
    action_array_1 = mActionArray[:, :, 1].flatten()

    # Calculate the histograms
    global mActionArrayHistogram0, mActionArrayHistogram1
    mActionArrayHistogram0, bins_0 = np.histogram(action_array_0, bins=20, range=(-1, 1))
    mActionArrayHistogram0 = mActionArrayHistogram0.astype(np.float32)
    mActionArrayHistogram1, bins_1 = np.histogram(action_array_1, bins=20, range=(-1, 1))
    mActionArrayHistogram1 = mActionArrayHistogram1.astype(np.float32)

    global mStateSlice, mNextStateSlice, mRewardSlice, mActionSlice, mDoneSlice, mAllDoneSlice
    global mNumSteps
    mStateSlice = None
    mNextStateSlice = None
    mRewardSlice = None
    mActionSlice = None
    mDoneSlice = None
    mAllDoneSlice = None
    mNumSteps = 0

    global mTrainArgs, mEnvArgs, mGuiArgs, mRewardInfoBuffer, mActionSpaceBound, mActionSpaceBoundMove, mUserName, mVersion, mSaveTime
    mTrainArgs = mMetaData['train_args']
    mEnvArgs = mMetaData['env_args']
    mGuiArgs = mMetaData['gui_args']
    mUserName = mMetaData['user_name']
    mVersion = mMetaData['train_version']
    mSaveTime = mMetaData['save_time']
    mRewardInfoBuffer = mMetaData['reward_info_buffer']
    _l = (mEnvArgs.action_step_range[1] - mEnvArgs.action_step_range[0]) / 2
    _b = mEnvArgs.action_step_range[1] - _l
    mActionSpaceBound = np.array([(math.pi, _l)])  # [((-π~π), (-l~l))]
    mActionSpaceBoundMove = np.array([(0, _b)])  # [((-π~π), (-l + b, l + b))]


def _reset_parameters():
    """重设imgui参数"""
    global mCurrViewingBufferIdx, mCurrViewingEpoch, mCurrViewingStep, mCurrViewingAgentIdx
    mCurrViewingBufferIdx = 0
    mCurrViewingEpoch = 0  # 正在查看的epoch
    mCurrViewingStep = 0  # 正在查看的step
    mCurrViewingAgentIdx = 0  # 正在查看的agent index


def _load_data(file_path):
    """
    此过程被包裹在thread上下文中
    从磁盘加载buffer数据， 并自动计算需要的数值
    :param file_path:
    :return:
    """
    global mReplayBuffer, mIsLoadingBuffer, mMetaData
    mIsLoadingBuffer = True
    # 首先清空mReplayBuffer
    mReplayBuffer = None

    # 从磁盘读取缓存文件（耗时操作）
    with gzip.open(file_path, "rb") as f:
        data = pickle.load(f)
        buffer: collections.deque = data['buffer']
        meta_data = data['meta_data']

    # 将mReplayBuffer赋值， 此时由于mIsLoadingBuffer依然为True，因此不会执行_show_buffer_viewer
    mReplayBuffer = buffer
    mMetaData = meta_data

    # 重建缓存信息（耗时操作）
    try:
        _rebuild_cache()
    except Exception as e:
        traceback.print_exc()
    # 重置参数信息
    _reset_parameters()

    mIsLoadingBuffer = False


def _show_buffer_viewer():

    global mCurrViewingBufferIdx, mCurrViewingEpoch, mCurrViewingStep, mCurrViewingAgentIdx
    global mStateSlice, mNextStateSlice, mRewardSlice, mActionSlice, mDoneSlice, mAllDoneSlice
    global mAction, mReward, mDone, mScaledAction, mActionDx, mActionDy
    global mNumSteps, mImgSize
    global mRewardDetails, mFinalRewardDetails
    imgui.text(f'buffer file path: {mFilePath}')
    imgui.text(f'buffer size = {mCachedReplayBufferSize}')
    imgui.same_line()
    imgui.text(f'epochs = {mEpochs}')
    imgui.same_line()
    imgui.text(f'version = {mVersion}')
    imgui.same_line()
    imgui.text(f'creator = {mUserName}')
    imgui.same_line()
    imgui.text(f'create time = {mSaveTime}')

    if imgui.button('view train_args'):
        imgui.open_popup('train_args')
    if imgui.begin_popup('train_args'):
        imgui.begin_table('train_args_table', 2, imgui.TABLE_ROW_BACKGROUND | imgui.TABLE_BORDERS)
        imgui.table_next_column()
        for key, value in vars(mTrainArgs).items():
            imgui.text(str(key))
            imgui.table_next_column()
            imgui.text(str(value))
            imgui.table_next_column()
        imgui.end_table()
        imgui.end_popup()
    imgui.same_line()
    if imgui.button('view env_args'):
        imgui.open_popup('env_args')
    if imgui.begin_popup('env_args'):
        imgui.begin_table('env_args_table', 2, imgui.TABLE_ROW_BACKGROUND | imgui.TABLE_BORDERS)
        imgui.table_next_column()
        for key, value in vars(mEnvArgs).items():
            imgui.text(str(key))
            imgui.table_next_column()
            imgui.text(str(value))
            imgui.table_next_column()
        imgui.end_table()
        imgui.end_popup()
    imgui.same_line()
    if imgui.button('view gui_args'):
        imgui.open_popup('gui_args')
    if imgui.begin_popup('gui_args'):
        imgui.begin_table('gui_args_table', 2, imgui.TABLE_ROW_BACKGROUND | imgui.TABLE_BORDERS)
        imgui.table_next_column()
        for key, value in vars(mGuiArgs).items():
            imgui.text(str(key))
            imgui.table_next_column()
            imgui.text(str(value))
            imgui.table_next_column()
        imgui.end_table()
        imgui.end_popup()
    # buttons

    epoch_updated = False  # 用于标记外部操作对epoch idx的改变
    step_updated = False  # 用于标记外部操作对step idx的改变

    changed, mCurrViewingEpoch = imgui.slider_int('curr epoch idx', mCurrViewingEpoch, 0, mEpochs - 1)
    epoch_updated |= changed
    imgui.same_line()
    if imgui.button('prev epoch') and mCurrViewingEpoch > 0:
        mCurrViewingEpoch = max(0, mCurrViewingEpoch - 1)
        epoch_updated |= True
    imgui.same_line()
    if imgui.button('next epoch') and mCurrViewingEpoch < mEpochs - 1:
        mCurrViewingEpoch = min(mCurrViewingEpoch + 1, mEpochs - 1)
        epoch_updated |= True

    if epoch_updated or mStateSlice is None:
        last_done_idx = -1 if mCurrViewingEpoch == 0 else mAllDoneIndices[mCurrViewingEpoch - 1]
        curr_done_idx = mAllDoneIndices[mCurrViewingEpoch]
        start = last_done_idx + 1
        end = curr_done_idx + 1
        mStateSlice = mStateArray[start:end]
        mNextStateSlice = mNextStateArray[start:end]
        mRewardSlice = mRewardArray[start:end]
        mActionSlice = mActionArray[start:end]
        mDoneSlice = mDoneArray[start:end]
        mAllDoneSlice = mAllDoneArray[start:end]

        mNumSteps = end - start
        mCurrViewingStep = 0  # 等epoch改变时，重置step
        step_updated |= True  # 标记step idx 已改变

    changed, mCurrViewingStep = imgui.slider_int('curr step idx', mCurrViewingStep, 0, mNumSteps - 1)
    step_updated |= changed
    imgui.same_line()
    if imgui.button('prev step') and mCurrViewingStep > 0:
        mCurrViewingStep = max(0, mCurrViewingStep - 1)
        step_updated |= True
    imgui.same_line()
    if imgui.button('next step') and mCurrViewingStep < mNumSteps - 1:
        mCurrViewingStep = min(mCurrViewingStep + 1, mNumSteps - 1)
        step_updated |= True

    if step_updated:
        print(f'update state')
        last_done_idx = -1 if mCurrViewingEpoch == 0 else mAllDoneIndices[mCurrViewingEpoch - 1]
        mCurrViewingBufferIdx = last_done_idx + 1 + mCurrViewingStep
        state_img = mStateSlice[mCurrViewingStep].transpose((1, 2, 0))
        GraphicManager.I.bilt_to('state_img', state_img, exposed=False)
        next_state_img = mNextStateSlice[mCurrViewingStep].transpose((1, 2, 0))
        GraphicManager.I.bilt_to('next_state_img', next_state_img, exposed=False)

        mReward = mRewardSlice[mCurrViewingStep][mCurrViewingAgentIdx][0]  # float
        mDone = mDoneSlice[mCurrViewingStep][mCurrViewingAgentIdx][0]  # float
        mAction = mActionSlice[mCurrViewingStep][mCurrViewingAgentIdx]  # (num_actions,)
        mScaledAction = mAction * mActionSpaceBound[0] + mActionSpaceBoundMove[0]
        mActionDx = np.cos(mScaledAction[0]) * mScaledAction[1]
        mActionDy = np.sin(mScaledAction[0]) * mScaledAction[1]

        reward_info = mRewardInfoBuffer[mCurrViewingBufferIdx][mCurrViewingAgentIdx]
        mRewardDetails = {REWARD_DISPLAY_NAMES[key]: value
                          for key, value in reward_info.items()
                          if key in REWARD_KEYS}
        mFinalRewardDetails = {REWARD_DISPLAY_NAMES[key]: value
                               for key, value in reward_info.items()
                               if key in FINAL_REWARD_KEYS}


    imgui.text(
        f'you are viewing epoch: {mCurrViewingEpoch}, step: {mCurrViewingStep} at buffer_idx: {mCurrViewingBufferIdx}')
    state_texture: SimpleTexture = GraphicManager.I.textures['state_img']
    next_state_texture: SimpleTexture = GraphicManager.I.textures['next_state_img']

    _, mImgSize = imgui.slider_int('img view size', mImgSize, 128, 1024)
    imgui.same_line()
    if imgui.button('use real size'):
        mImgSize = state_texture.width
    if mNumAgents > 1:
        _, mCurrViewingAgentIdx = imgui.slider_int('curr viewing agent idx', mCurrViewingAgentIdx, 0, mNumAgents - 1)

    # left image
    imgui.begin_group()
    imgui.text(f'state_img')
    imgui.text(f'w: {state_texture.width}, h: {state_texture.height}, c: {state_texture.channel}')
    imgui.image(state_texture.texture_id, mImgSize, mImgSize, (0, 0), (1, 1), (1, 1, 1, 1), (0.5, 0.5, 0.5, 1))

    imgui.end_group()

    # middle info
    imgui.same_line()
    imgui.begin_group()
    cursor_pos_x, cursor_pos_y = imgui.get_cursor_pos()

    imgui.text(f'action = {mAction[0]:.2f}, {mAction[1]:.2f}')
    imgui.text(f'dx = {mActionDx:.2f}, dy = {mActionDy:.2f}')
    imgui.text(f'----------------------->')
    text_width, _ = imgui.get_item_rect_size()

    imgui.begin_child('action gram', text_width, mEnvArgs.action_step_range[1] * 2, border=False)
    window_pos_x, window_pos_y = imgui.get_window_position()
    start_x = window_pos_x + text_width / 2
    start_y = window_pos_y + mEnvArgs.action_step_range[1]
    end_x = start_x + mActionDx
    end_y = start_y - mActionDy
    draw_list = imgui.get_window_draw_list()
    draw_list.add_line(start_x, start_y, end_x, end_y, imgui.get_color_u32_rgba(1, 0, 0, 1), 3)

    draw_list.add_circle_filled(start_x, start_y, mEnvArgs.action_step_range[1],
                                imgui.get_color_u32_rgba(0.5, 0.5, 0.5, 0.1))
    draw_list.add_circle(start_x, start_y, mEnvArgs.action_step_range[0], imgui.get_color_u32_rgba(0.5, 0.5, 0.5, 0.2),
                         thickness=1)

    imgui.end_child()
    # action 直方图
    imgui.text('action0_histogram')
    imgui.plot_lines("", mActionArrayHistogram0,
                     graph_size=(text_width, 25 * g.GLOBAL_SCALE))
    imgui.text('action1_histogram')
    imgui.plot_lines("", mActionArrayHistogram1,
                     graph_size=(text_width, 25 * g.GLOBAL_SCALE))
    imgui.end_group()

    # right image
    imgui.same_line()
    imgui.begin_group()
    imgui.text(f'next_state_img')
    imgui.text(f'w: {next_state_texture.width}, h: {next_state_texture.height}, c: {next_state_texture.channel}')
    imgui.image(next_state_texture.texture_id, mImgSize, mImgSize, (0, 0), (1, 1), (1, 1, 1, 1), (0.5, 0.5, 0.5, 1))

    imgui.end_group()

    # right info
    imgui.same_line()
    imgui.begin_group()
    imgui.text(f'done = {mDone}')
    imgui.text(f'reward = {mReward:.2f}')

    reward_info = mRewardInfoBuffer[mCurrViewingBufferIdx][mCurrViewingAgentIdx]

    #   - reward_info
    #       - reward_name : reward_value

    imgui.separator()
    imgui.text(f'reward_before_shift = {reward_info[REWARD_SUM]:.2f}')

    expanded, visible = imgui.collapsing_header('reward details', True, imgui.TREE_NODE_DEFAULT_OPEN)
    if expanded:
        imgui.begin_table('reward_details_table', 2, imgui.TABLE_ROW_BACKGROUND | imgui.TABLE_BORDERS)
        imgui.table_next_column()
        for key, value in mRewardDetails.items():
            imgui.text(str(key))
            imgui.table_next_column()
            imgui.text(f'{value:.2f}')
            imgui.table_next_column()
        imgui.end_table()

    imgui.separator()
    final_reward_shift = reward_info[FINAL_REWARD_SHIFT]
    final_reward_weight = reward_info[FINAL_REWARD_WEIGHT]
    imgui.text(f'final_reward_shift = {final_reward_shift:.2f} ({final_reward_weight * 100:.0f}%) ')
    expanded, visible = imgui.collapsing_header('final reward details', True, imgui.TREE_NODE_DEFAULT_OPEN)
    if expanded:
        imgui.begin_table('reward_details_table', 2, imgui.TABLE_ROW_BACKGROUND | imgui.TABLE_BORDERS)
        imgui.table_next_column()
        for key, value in mFinalRewardDetails.items():
            imgui.text(str(key))
            imgui.table_next_column()
            imgui.text(f'{value:.2f}')
            imgui.table_next_column()
        imgui.end_table()
    imgui.end_group()


