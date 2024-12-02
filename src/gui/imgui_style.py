import os
import imgui
from gui import global_var as g

from gui.icon_module import Spinner, IconManager


def init_font():
    io = imgui.get_io()
    g.mChineseFont = io.fonts.add_font_from_file_ttf(
        os.path.join(g.RESOURCE_DIR, 'fonts/Unifont.ttf'),
        g.FONT_SIZE * g.FONT_SCALING_FACTOR,
        glyph_ranges=io.fonts.get_glyph_ranges_chinese_full()
    )
    io.font_global_scale /= g.FONT_SCALING_FACTOR
    g.mModernglWindowRenderer.refresh_font_texture()  # impl = g.mModernglWindowRenderer


def init_style_var():
    style: imgui.core.GuiStyle = imgui.get_style()
    style.window_rounding = 0
    style.frame_rounding = 0
    style.popup_rounding = 0
    style.item_spacing = (8, 8)


def push_style(dark_mode):
    if dark_mode:
        push_dark()
    else:
        push_light()


def push_dark():
    imgui.style_colors_dark()
    init_style_var()
    style: imgui.core.GuiStyle = imgui.get_style()
    style.colors[imgui.COLOR_WINDOW_BACKGROUND] = (0.1, 0.1, 0.1, 0.95)
    style.colors[imgui.COLOR_POPUP_BACKGROUND] = (0.08, 0.08, 0.08, 0.90)
    style.colors[imgui.COLOR_BORDER] = (0.32, 0.32, 0.32, 0.50)
    style.colors[imgui.COLOR_FRAME_BACKGROUND] = (0.25, 0.25, 0.25, 0.78)
    style.colors[imgui.COLOR_FRAME_BACKGROUND_HOVERED] = (0.26, 0.59, 0.98, 0.78)
    style.colors[imgui.COLOR_TITLE_BACKGROUND] = (0.21, 0.21, 0.21, 1.00)
    style.colors[imgui.COLOR_BUTTON] = (0.25, 0.25, 0.25, 0.78)
    style.colors[imgui.COLOR_BUTTON_HOVERED] = (0.19, 0.53, 0.92, 1.00)
    style.colors[imgui.COLOR_HEADER] = (0.55, 0.55, 0.55, 0.31)
    style.colors[imgui.COLOR_SEPARATOR] = (0.54, 0.54, 0.54, 0.50)
    style.colors[imgui.COLOR_TAB] = (0.32, 0.32, 0.32, 0.86)
    style.colors[imgui.COLOR_TAB_HOVERED] = (0.16, 0.47, 0.87, 1.00)
    style.colors[imgui.COLOR_TAB_ACTIVE] = (0.16, 0.47, 0.87, 1.00)
    style.colors[imgui.COLOR_PLOT_HISTOGRAM] = (0.14, 0.50, 0.90, 1.00)

    IconManager.set_mode(True)
    Spinner.set_mode(True)

    # icon = pygame.image.load(os.path.join(g.RESOURCE_DIR, 'textures/light/road-fill.png)'))
    # pygame.display.set_icon(icon)

    g.DARK_MODE = True


def push_light():
    imgui.style_colors_light()
    init_style_var()
    IconManager.set_mode(False)
    Spinner.set_mode(False)

    # icon = pygame.image.load(os.path.join(g.RESOURCE_DIR, 'textures/dark/road-fill.png)'))
    # pygame.display.set_icon(icon)

    g.DARK_MODE = False
