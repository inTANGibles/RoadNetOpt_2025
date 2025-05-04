import logging
from typing import Optional
import os
import moderngl
import moderngl_window.integrations.imgui
import configparser
import ast

if 'doc' in os.getcwd():
    os.chdir('../src')

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

# 大写字母的为常量，分为Mutable和Immutable，
# Mutable Consts可以被用户修改，保存在config.ini中
# Immutable Consts,
# 规定 _ 开头的变量不应该被其他代码读取

# ===================Mutable Consts======================
try:
    # 全局窗口
    GLOBAL_SCALE = config.getfloat('GUI', 'GLOBAL_SCALE')  # 全局缩放
    DARK_MODE = config.getboolean('GUI', 'DARK_MODE')  # 是否为暗色模式
    VSYNC = config.getboolean('GUI', 'VSYNC')  # 是否开启垂直同步

    _INIT_WINDOW_WIDTH = config.getint('GUI', 'INIT_WINDOW_WIDTH')
    _INIT_WINDOW_HEIGHT = config.getint('GUI', 'INIT_WINDOW_HEIGHT')
    _FONT_SIZE = config.getint('GUI', 'FONT_SIZE')
    _LEFT_WINDOW_WIDTH = config.getint('GUI', 'LEFT_WINDOW_WIDTH')
    _BOTTOM_WINDOW_HEIGHT = config.getint('GUI', 'BOTTOM_WINDOW_HEIGHT')

    DEFAULT_DATA_PATH = config.get('GUI', 'DEFAULT_DATA_PATH')

    # 主渲染窗口
    TEXTURE_SCALE = config.getfloat('GUI', 'TEXTURE_SCALE')  # Main Texture Buffer的渲染分辨率倍数
    LINE_WIDTH_SCALE = config.getfloat('GUI', 'LINE_WIDTH_SCALE')  # 主窗口道路线条宽度显示倍率
    MAIN_FRAME_BUFFER_TEXTURE_BACKGROUND_COLOR = ast.literal_eval(
        config.get('GUI', 'MAIN_FRAME_BUFFER_TEXTURE_BACKGROUND_COLOR'))
    ENABLE_RENDER_ROADS = config.getboolean('GUI', 'ENABLE_RENDER_ROADS')
    ENABLE_RENDER_BUILDINGS = config.getboolean('GUI', 'ENABLE_RENDER_BUILDINGS')
    ENABLE_RENDER_REGIONS = config.getboolean('GUI', 'ENABLE_RENDER_REGIONS')
    ENABLE_RENDER_NODES = config.getboolean('GUI', 'ENABLE_RENDER_NODES')
    ENABLE_RENDER_MOUSE_POINTER = config.getboolean('GUI', 'ENABLE_RENDER_MOUSE_POINTER')  # 是否在主窗口显示鼠标指针定位
    ENABLE_POST_PROCESSING = config.getboolean('GUI', 'ENABLE_POST_PROCESSING')  # 是否在主窗口中开启后处理
    ENABLE_WINDOW_BLUR = config.getboolean('GUI', 'ENABLE_WINDOW_BLUR')  # 是否开启窗口虚化
except Exception as e:
    print(e)
    print(f'ggui config not fount')
    # 全局窗口
    GLOBAL_SCALE = 1.0  # 全局缩放
    DARK_MODE = True  # 是否为暗色模式
    VSYNC = True  # 是否开启垂直同步

    _INIT_WINDOW_WIDTH = 1280
    _INIT_WINDOW_HEIGHT = 720
    _FONT_SIZE = 16
    _LEFT_WINDOW_WIDTH = 400
    _BOTTOM_WINDOW_HEIGHT = 32
    DEFAULT_DATA_PATH = '../data/VirtualEnv/BIGMAP.bin'

    # 主渲染窗口
    TEXTURE_SCALE = 0.5
    LINE_WIDTH_SCALE = 2.0
    MAIN_FRAME_BUFFER_TEXTURE_BACKGROUND_COLOR = (0.0, 0.0, 0.0, 1.0)
    ENABLE_RENDER_ROADS = True
    ENABLE_RENDER_BUILDINGS = True
    ENABLE_RENDER_REGIONS = True
    ENABLE_RENDER_NODES = True
    ENABLE_RENDER_MOUSE_POINTER = True
    ENABLE_POST_PROCESSING = False
    ENABLE_WINDOW_BLUR = True


def save_config():
    logging.info('saving config...')
    # 全局窗口
    config.set('GUI', 'GLOBAL_SCALE', str(GLOBAL_SCALE))
    config.set('GUI', 'DARK_MODE', str(DARK_MODE))
    config.set('GUI', 'VSYNC', str(VSYNC))
    config.set('GUI', 'INIT_WINDOW_WIDTH', str(_INIT_WINDOW_WIDTH))
    config.set('GUI', 'INIT_WINDOW_HEIGHT', str(_INIT_WINDOW_HEIGHT))
    config.set('GUI', 'FONT_SIZE', str(_FONT_SIZE))
    config.set('GUI', 'LEFT_WINDOW_WIDTH', str(int(LEFT_WINDOW_WIDTH / GLOBAL_SCALE)))
    config.set('GUI', 'BOTTOM_WINDOW_HEIGHT', str(_BOTTOM_WINDOW_HEIGHT))
    config.set('GUI', 'DEFAULT_DATA_PATH', str(DEFAULT_DATA_PATH))

    # 主渲染窗口
    config.set('GUI', 'TEXTURE_SCALE', str(TEXTURE_SCALE))
    config.set('GUI', 'LINE_WIDTH_SCALE', str(LINE_WIDTH_SCALE))
    config.set('GUI', 'MAIN_FRAME_BUFFER_TEXTURE_BACKGROUND_COLOR', str(MAIN_FRAME_BUFFER_TEXTURE_BACKGROUND_COLOR))
    config.set('GUI', 'ENABLE_RENDER_ROADS', str(ENABLE_RENDER_ROADS))
    config.set('GUI', 'ENABLE_RENDER_BUILDINGS', str(ENABLE_RENDER_BUILDINGS))
    config.set('GUI', 'ENABLE_RENDER_REGIONS', str(ENABLE_RENDER_REGIONS))
    config.set('GUI', 'ENABLE_RENDER_NODES', str(ENABLE_RENDER_NODES))
    config.set('GUI', 'ENABLE_RENDER_MOUSE_POINTER', str(ENABLE_RENDER_MOUSE_POINTER))
    config.set('GUI', 'ENABLE_POST_PROCESSING', str(ENABLE_POST_PROCESSING))
    config.set('GUI', 'ENABLE_WINDOW_BLUR', str(ENABLE_WINDOW_BLUR))

    with open('config.ini', 'w', encoding='utf-8') as f:
        config.write(f)


# ===================Immutable Consts====================
# （自动计算得到或固定值，代码运行过程中不应被修改）
INIT_WINDOW_WIDTH = _INIT_WINDOW_WIDTH * GLOBAL_SCALE  # 初始窗口大小
INIT_WINDOW_HEIGHT = _INIT_WINDOW_HEIGHT * GLOBAL_SCALE  # 初始窗口大小
FONT_SIZE = _FONT_SIZE * GLOBAL_SCALE  # 字体大小
FONT_SCALING_FACTOR = 1  # 字体超采样
LEFT_WINDOW_WIDTH = _LEFT_WINDOW_WIDTH * GLOBAL_SCALE  # 左边窗口的宽度
BOTTOM_WINDOW_HEIGHT = _BOTTOM_WINDOW_HEIGHT * GLOBAL_SCALE  # 底部窗口的宽度
IMAGE_WINDOW_INDENT_LEFT = 8
IMAGE_WINDOW_INDENT_RIGHT = 8
IMAGE_WINDOW_INDENT_TOP = 22 + FONT_SIZE
IMAGE_WINDOW_INDENT_BOTTOM = 8

DEFAULT_IMAGE_BUTTON_WIDTH = 20 * GLOBAL_SCALE
DEFAULT_IMAGE_BUTTON_HEIGHT = 20 * GLOBAL_SCALE
DEFAULT_ICON_WIDTH = 16 * GLOBAL_SCALE
DEFAULT_ICON_HEIGHT = 16 * GLOBAL_SCALE

RESOURCE_DIR = os.path.abspath('../resources/')  # 资源路径

# =======================Variables===========================
mWindowEvent: Optional[moderngl_window.WindowConfig] = None
mModernglWindowRenderer: Optional[moderngl_window.integrations.imgui.ModernglWindowRenderer] = None
mCtx: Optional[moderngl.Context] = None
mWindowSize = (-1, -1)

mChineseFont = None

mFirstLoop = True
mTime = 0
mFrameTime = 0
mSmoothedFrameTime = 0

mDxfWindowOpened = False
mInfoWindowOpened = True
mLoggingWindowOpened = False
mDebugWindowOpened = False
mReplayBufferViewerWindowOpened = False

mHoveringImageWindow = False
mHoveringInfoSubWindow = False
mHoveringMainTextureSubWindow = False
mHoveringDxfSubWindow = False
mHoveringLoggingSubWindow = False
mHoveringDebugSubWindow = False

mShowingMainTextureWindow = False
mImageSize = (0, 0)  # 纹理大小
mMousePosInImage = (0, 0)  # 鼠标位置（纹理空间
mImageWindowInnerSize = (0, 0)  # 图像控件的大小（屏幕空间
mImageWindowInnerPos = (0, 0)  # 图片控件在窗口的绝对位置（屏幕空间
mImageWindowDrawList = None

mSelectRoadsMode = True
mSelectedRoads = {}  # 被选中的道路 dict{uid:road}
mSelectedNodes = {}
mCurrentEditingRoad = None
mAddNodeMode = False

mShift = False
mCtrl = False
mAlt = False
