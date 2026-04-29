from layers import *
def init_geometry(self):
    self.backproject_depth = {}
    self.project_3d = {}
    for scale in self.opt.scales:
        h = self.opt.height // (2 ** scale)
        w = self.opt.width // (2 ** scale)

        self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w).to(self.device)
        self.project_3d[scale] = Project3D(self.opt.batch_size, h, w).to(self.device)
