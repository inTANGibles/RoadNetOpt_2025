import functools
import sys
from abc import abstractmethod
import moderngl_window
from graphic_module import GraphicManager
from gui import global_var as g


def glcontext(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if "--window" not in sys.argv:
            sys.argv.append("--window")
            sys.argv.append("headless")

        class HeadlessWindow2(moderngl_window.WindowConfig):
            samples = 0  # Headless is not always happy with multisampling
            resource_dir = g.RESOURCE_DIR  # set resources dir
            vsync = False

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

                if self.wnd.name != 'headless':
                    raise RuntimeError('This example only works with --window headless option')

                g.mCtx = self.ctx
                g.mWindowEvent = self

                self.graphic_manager = GraphicManager(headless=True)

            def render(self, time, frame_time):
                func(*args, **kwargs)
                self.ctx.finish()
                self.wnd.close()

        HeadlessWindow2.run()
        return

    return wrapper


class GLContext(moderngl_window.WindowConfig):
    if "--window" not in sys.argv:
        sys.argv.append("--window")
        sys.argv.append("headless")

    samples = 0  # Headless is not always happy with multisampling
    resource_dir = g.RESOURCE_DIR  # set resources dir
    vsync = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.wnd.name != 'headless':
            raise RuntimeError('This example only works with --window headless option')

        g.mCtx = self.ctx
        g.mWindowEvent = self

        self.graphic_manager = GraphicManager(headless=True)

    def render(self, time, frame_time):
        self.main()
        self.ctx.finish()
        self.wnd.close()

    @abstractmethod
    def main(self):
        pass
