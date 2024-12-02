import sys
if "--window" not in sys.argv:
    sys.argv.append("--window")
    sys.argv.append("headless")

import moderngl_window
from graphic_module import GraphicManager
from train_module import TrainManager
from gui import global_var as g


class HeadlessTest(moderngl_window.WindowConfig):
    samples = 0  # Headless is not always happy with multisampling
    resource_dir = g.RESOURCE_DIR  # set resources dir

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.wnd.name != 'headless':
            raise RuntimeError('This example only works with --window headless option')

        g.mCtx = self.ctx
        g.mWindowEvent = self

        self.graphic_manager = GraphicManager(headless=True)
        self.train_manager = TrainManager()

    def render(self, time, frame_time):
        self.train_manager.train()
        self.ctx.finish()
        self.wnd.close()


if __name__ == '__main__':
    HeadlessTest.run()
