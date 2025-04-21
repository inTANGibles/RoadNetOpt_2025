import imgui
import gzip
import pickle
from gui import global_var as g
from gui import components as imgui_c
from gui.icon_module import Spinner
import numpy as np
from graphic_module import GraphicManager, SimpleTexture
from gui.imgui_replay_buffer_viewer_subwindow import _show_menu_bar, _open_file_dialog

print('logging subwindow loaded')

mFilePath = ''
mReplayBuffer = None
mMetaData = None

mEpochs = 0
mAllDoneIndices = None
mStateArray = None
mRewardArray = None
mRewardInfoBuffer = None

mCurrViewingEpoch = 0
mImgSize = 256


def show():
    if not g.mLoggingWindowOpened:
        return
    expanded, g.mLoggingWindowOpened = imgui.begin("Training Log Viewer", True)

    if mReplayBuffer is None:
        if not mFilePath:
            if imgui.button("Open Log File"):
                _open_file_dialog()
        Spinner.spinner('load_logger')
        imgui.end()
        return

    _show_log_viewer()
    imgui.end()


def _open_file_dialog():
    from utils.io_utils import open_file_window
    global mFilePath
    file_path = open_file_window(filetypes=[('Replay Buffer', '.pkl.gz')])
    if file_path:
        mFilePath = file_path
        Spinner.start('load_logger', target=_load_log_data, args=(file_path,))


def _load_log_data(file_path):
    global mReplayBuffer, mMetaData, mEpochs
    global mStateArray, mRewardArray, mAllDoneIndices, mRewardInfoBuffer

    with gzip.open(file_path, "rb") as f:
        data = pickle.load(f)
        mReplayBuffer = data['buffer']
        mMetaData = data['meta_data']

    # 解包缓存数组（训练中已经写入）
    mStateArray = np.array([t[0] for t in mReplayBuffer])
    mRewardArray = np.array([t[2] for t in mReplayBuffer])
    mAllDoneIndices = np.where([t[5] for t in mReplayBuffer])[0]
    mEpochs = len(mAllDoneIndices)
    mRewardInfoBuffer = mMetaData['reward_info_buffer']


def _show_log_viewer():
    global mCurrViewingEpoch

    changed, mCurrViewingEpoch = imgui.slider_int("Epoch", mCurrViewingEpoch, 0, mEpochs - 1)
    imgui.same_line()
    if imgui.button("Prev") and mCurrViewingEpoch > 0:
        mCurrViewingEpoch -= 1
    imgui.same_line()
    if imgui.button("Next") and mCurrViewingEpoch < mEpochs - 1:
        mCurrViewingEpoch += 1

    last_done_idx = -1 if mCurrViewingEpoch == 0 else mAllDoneIndices[mCurrViewingEpoch - 1]
    curr_done_idx = mAllDoneIndices[mCurrViewingEpoch]
    buffer_idx = curr_done_idx

    # 状态图像显示
    state_img = mStateArray[buffer_idx].transpose((1, 2, 0))
    GraphicManager.I.bilt_to("epoch_state", state_img, exposed=False)
    state_texture: SimpleTexture = GraphicManager.I.textures["epoch_state"]
    imgui.image(state_texture.texture_id, mImgSize, mImgSize)

    # reward 显示
    reward_vec = mRewardArray[buffer_idx]
    agent_rewards = [f"{i}: {reward_vec[i][0]:.2f}" for i in range(len(reward_vec))]
    imgui.text("Reward per agent:")
    for s in agent_rewards:
        imgui.bullet_text(s)

    reward_info = mRewardInfoBuffer[buffer_idx]
    imgui.separator()
    imgui.text("Reward Details:")
    for agent_idx, info in enumerate(reward_info):
        imgui.text(f"Agent {agent_idx}")
        for k, v in info.items():
            imgui.bullet_text(f"{k}: {v:.2f}")
        imgui.separator()


def _show_reward_table(epoch_idx: int, agent_idx: int):
    buffer_idx = mAllDoneIndices[epoch_idx]
    reward_info = mRewardInfoBuffer[buffer_idx][agent_idx]
    imgui.begin_table("RewardDetails", 2, imgui.TABLE_ROW_BACKGROUND | imgui.TABLE_BORDERS)
    imgui.table_setup_column("Key")
    imgui.table_setup_column("Value")
    imgui.table_headers_row()
    for key, value in reward_info.items():
        imgui.table_next_column()
        imgui.text(str(key))
        imgui.table_next_column()
        imgui.text(f"{value:.3f}")
    imgui.end_table()
