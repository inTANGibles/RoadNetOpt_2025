import imgui
import pygame

from gui import global_var as g
from gui import imgui_home_page
from gui import imgui_geo_page
from gui import imgui_training_page
from gui import imgui_tool_page
from gui import imgui_settings_page

print('main window loaded')
mDragging = False
mLastMousePosX = 0
def show():
    global mDragging, mLastMousePosX
    screen_width, screen_height = g.mWindowSize
    style: imgui.core.GuiStyle = imgui.get_style()
    imgui.set_next_window_size(g.LEFT_WINDOW_WIDTH, screen_height - g.BOTTOM_WINDOW_HEIGHT)
    imgui.set_next_window_position(0, 0)
    flags = imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
    expanded, opened = imgui.begin("main_window", False, flags=flags)
    imgui.push_id('main_window')

    curr_tab_item = '主页'
    with imgui.begin_tab_bar('main_tab_bar'):
        if imgui.begin_tab_item('主页').selected:
            curr_tab_item = '主页'
            # imgui_home_page.show()
            imgui.end_tab_item()
        if imgui.begin_tab_item('路网工具').selected:
            curr_tab_item = '路网工具'
            # imgui_geo_page.show()
            imgui.end_tab_item()
        if imgui.begin_tab_item('训练工具').selected:
            curr_tab_item = '训练工具'
            # imgui_training_page.show()
            imgui.end_tab_item()
        if imgui.begin_tab_item('实用工具').selected:
            curr_tab_item = '实用工具'
            # imgui_tool_page.show()
            imgui.end_tab_item()
        if imgui.begin_tab_item('设置').selected:
            curr_tab_item = '设置'
            # imgui_settings_page.show()
            imgui.end_tab_item()
    imgui.begin_child('main_window_content', g.LEFT_WINDOW_WIDTH - style.window_padding[0] * 2, 0, border=False)
    if curr_tab_item == '主页':
        imgui_home_page.show()
    elif curr_tab_item == '路网工具':
        imgui_geo_page.show()
    elif curr_tab_item == '训练工具':
        imgui_training_page.show()
    elif curr_tab_item == '实用工具':
        imgui_tool_page.show()
    elif curr_tab_item == '设置':
        imgui_settings_page.show()
    imgui.end_child()

    imgui.set_cursor_pos_y(0.0)
    imgui.set_cursor_pos_x(g.LEFT_WINDOW_WIDTH - style.window_padding[0])
    imgui.push_style_var(imgui.STYLE_FRAME_PADDING, (0, 0))
    imgui.push_style_color(imgui.COLOR_BUTTON, 0, 0, 0, 0)
    imgui.button('     split', width=style.window_padding[0], height=screen_height - g.BOTTOM_WINDOW_HEIGHT - style.window_padding[1])
    if mDragging:
        g.LEFT_WINDOW_WIDTH = g.LEFT_WINDOW_WIDTH + imgui.get_mouse_pos()[0] - mLastMousePosX
        mLastMousePosX = imgui.get_mouse_pos()[0]
    if imgui.is_mouse_released(0):
        mDragging = False
    if imgui.is_item_hovered() and not mDragging and imgui.is_mouse_down(0):
        mDragging = True
        mLastMousePosX = imgui.get_mouse_pos()[0]

    imgui.pop_style_color()
    imgui.pop_style_var()
    imgui.pop_id()
    imgui.end()
