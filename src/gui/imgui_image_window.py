import imgui
import numpy as np
import pygame
from PIL import Image
from graphic_module import GraphicManager
from geo import Road
from utils import io_utils
from gui import global_var as g
from gui import components as imgui_c
from gui import common
from gui import imgui_toolbox_subwindow

mImageWindowSize = (0, 0)
mImageWindowPos = (0, 0)

mImageWindowMousePos = (0, 0)

mTextureInfo = {}

mObserverTableColumns = 4
mHighlightedObservers = set()
mHighlightObservers = False
mHighlightedRewardObservers = set()
mHighlightRewardObservers = False
print('image window loaded')


def show():
    global mImageWindowSize, mImageWindowPos, mImageWindowMousePos, mObserverTableColumns, mHighlightObservers, mHighlightRewardObservers
    screen_width, screen_height = g.mWindowSize
    flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS | imgui.WINDOW_NO_SCROLLBAR
    flags = flags | imgui.WINDOW_NO_SCROLL_WITH_MOUSE if g.mShowingMainTextureWindow else flags
    imgui.set_next_window_size(screen_width - g.LEFT_WINDOW_WIDTH, screen_height - g.BOTTOM_WINDOW_HEIGHT)
    imgui.set_next_window_position(g.LEFT_WINDOW_WIDTH, 0)
    # 开启窗口
    imgui.begin("image window", False, flags=flags)
    # 计算数值

    mImageWindowPos = (int(imgui.get_window_position()[0]), int(imgui.get_window_position()[1]))
    mImageWindowSize = imgui.get_window_size()
    g.mImageWindowInnerSize = (int(mImageWindowSize[0] - g.IMAGE_WINDOW_INDENT_LEFT - g.IMAGE_WINDOW_INDENT_RIGHT),
                               int(mImageWindowSize[1] - g.IMAGE_WINDOW_INDENT_TOP
                                   - g.IMAGE_WINDOW_INDENT_BOTTOM))
    g.mImageSize = (int(g.mImageWindowInnerSize[0] / g.TEXTURE_SCALE),
                    int(g.mImageWindowInnerSize[1] / g.TEXTURE_SCALE))
    g.mImageWindowInnerPos = (int(mImageWindowPos[0] + g.IMAGE_WINDOW_INDENT_LEFT),
                              int(mImageWindowPos[1] + g.IMAGE_WINDOW_INDENT_TOP))
    vec1 = (int(imgui.get_mouse_position()[0]), int(imgui.get_mouse_position()[1]))
    vec2 = g.mImageWindowInnerPos
    mImageWindowMousePos = (vec1[0] - vec2[0], vec1[1] - vec2[1])
    g.mMousePosInImage = (
        int(mImageWindowMousePos[0] / g.TEXTURE_SCALE), int(mImageWindowMousePos[1] / g.TEXTURE_SCALE))
    # g.mHoveringImageWindow = imgui_c.is_hovering_window()
    g.mHoveringImageWindow = imgui.is_window_hovered()
    g.mFocusingOnImageWindow = imgui.is_window_focused()
    g.mImageWindowDrawList = imgui.get_window_draw_list()

    textures_to_delete = set()  # 新建一个set用以记录哪些textures需要被删除
    flags = imgui.TAB_BAR_AUTO_SELECT_NEW_TABS | imgui.TAB_BAR_TAB_LIST_POPUP_BUTTON
    with imgui.begin_tab_bar('image_tab_bar', flags=flags):
        if GraphicManager.I.observers:
            # 首先显示observers
            selected, opened = imgui.begin_tab_item('observers')
            if selected:
                g.mShowingMainTextureWindow = False
                _, mObserverTableColumns = imgui.slider_int('scale', mObserverTableColumns, 1, 10)
                _, mHighlightObservers = imgui.checkbox('highlight observers', mHighlightObservers)
                imgui.same_line()
                imgui.text(f'({len(mHighlightedObservers)}) selected')
                with imgui.begin_table('observers_table', mObserverTableColumns):
                    imgui.table_next_column()
                    for observer in GraphicManager.I.observers.values():
                        if mHighlightObservers and observer not in mHighlightedObservers:
                            continue
                        imgui.image(observer.texture_id,
                                    imgui.get_content_region_available_width(),
                                    imgui.get_content_region_available_width(),
                                    border_color=(0, 0, 1, 1) if observer in mHighlightedObservers else (0, 0, 1, 0.1))
                        if imgui.is_item_clicked(0) and not mHighlightObservers:
                            if observer in mHighlightedObservers:
                                mHighlightedObservers.remove(observer)
                            else:
                                mHighlightedObservers.add(observer)
                        imgui.text(observer.name)
                        imgui.table_next_column()
                imgui.end_tab_item()
        if GraphicManager.I.reward_observers:
            # reward observers
            selected, opened = imgui.begin_tab_item('reward observers')
            if selected:
                g.mShowingMainTextureWindow = False
                _, mObserverTableColumns = imgui.slider_int('scale', mObserverTableColumns, 1, 10)
                _, mHighlightRewardObservers = imgui.checkbox('highlight reward observers', mHighlightRewardObservers)
                imgui.same_line()
                imgui.text(f'({len(mHighlightedRewardObservers)}) selected')
                with imgui.begin_table('reward_observers_table', mObserverTableColumns):
                    imgui.table_next_column()
                    for observer_name, observer in GraphicManager.I.reward_observers.items():
                        if mHighlightRewardObservers and observer not in mHighlightedRewardObservers:
                            continue
                        imgui.image(observer.texture_id,
                                    imgui.get_content_region_available_width(),
                                    imgui.get_content_region_available_width(),
                                    border_color=(0, 0, 1, 1) if observer in mHighlightedRewardObservers else (0, 0, 1, 0.1))
                        if imgui.is_item_clicked(0) and not mHighlightRewardObservers:
                            if observer in mHighlightedRewardObservers:
                                mHighlightedRewardObservers.remove(observer)
                            else:
                                mHighlightedRewardObservers.add(observer)
                        additional_text = ''
                        if observer_name in GraphicManager.I.reward_observer_values:
                            additional_value = GraphicManager.I.reward_observer_values[observer_name]
                            if additional_value is not None:
                                additional_text = str(additional_value)

                        imgui.text(f'{observer.name} {additional_text}')
                        imgui.table_next_column()
                imgui.end_tab_item()
        for graphic_texture in GraphicManager.I.textures.values():
            if not graphic_texture.exposed: continue  # 如果texture被设置为不暴露，则略过
            selected, opened = imgui.begin_tab_item(graphic_texture.name, imgui.TAB_ITEM_TRAILING)
            if not selected:
                continue  # 仅渲染被选中的texture
            if not opened:
                textures_to_delete.add(graphic_texture.name)
                imgui.end_tab_item()
                continue

            if graphic_texture.name == 'main':
                # 主视图
                g.mShowingMainTextureWindow = True
                texture_id = graphic_texture.get_final_texture_id()
                # 显示image
                imgui.image(texture_id, graphic_texture.width * g.TEXTURE_SCALE,
                            graphic_texture.height * g.TEXTURE_SCALE)
                # imgui.text(f"Displaying texture at: {graphic_texture.width}x{graphic_texture.height}")
                GraphicManager.I.MainTexture.render_draw_list()

            else:
                # 其他视图

                g.mShowingMainTextureWindow = False
                imgui.image(graphic_texture.texture_id, graphic_texture.width * g.TEXTURE_SCALE,
                            graphic_texture.height * g.TEXTURE_SCALE)


            # 显示侧边工具栏
            imgui_toolbox_subwindow.show(graphic_texture)

            imgui.end_tab_item()

    imgui.end()

    # 逻辑操作
    for name in textures_to_delete:
        GraphicManager.I.unregister_frame_buffer_texture(name)
