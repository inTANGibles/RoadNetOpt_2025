# region legacy
# import os.path
#
# import gui.global_var as g
# import moderngl
# import moderngl_window
#
#
# class HeadlessWindowEvent:
#     def __init__(self):
#         g.mWindowEvent = self
#         self.ctx = moderngl.create_standalone_context()
#         g.mCtx = self.ctx
#         g.mModernglWindowRenderer = HeadlessModernglWindowRenderer()
#         g.mWindowSize = (100, 100)
#         moderngl_window.activate_context(self.ctx)
#
#     def load_program(self, path):
#         # 读取顶点着色器文件内容
#         with open(os.path.join(g.RESOURCE_DIR, path), 'r') as f:
#             glsl_code = f.read()
#         # 分离顶点着色器和片段着色器代码
#         glsl_code = glsl_code.replace("#version 330", '')
#         vertex_shader_code = "#version 330\n#define VERTEX_SHADER\n" + glsl_code
#         fragment_shader_code = "#version 330\n#define FRAGMENT_SHADER\n" + glsl_code
#
#         prog = self.ctx.program(vertex_shader=vertex_shader_code, fragment_shader=fragment_shader_code)
#         return prog
#
#
# class HeadlessModernglWindowRenderer:
#     def __init__(self):
#         pass
#
#     def register_texture(self, texture):
#         pass
# endregion

