def init_losses(self):
    """
    统一在此函数内完成损失函数的选择与实例化。
    要求：在调用本函数前，已完成 init_models(self)。
    注意：由于本函数可能新增可训练模块（例如特殊 loss 依赖的子网络），请在本函数之后再创建优化器。
    """
    method = self.opt.methods

    # —— Monodepth2 系列 Loss ——
    def _build_md2_loss():
        from methods.losses.monodepth2.monodepth2_loss import Monodepth2Loss
        print("[init_losses] Using loss: Monodepth2Loss")
        return Monodepth2Loss(self.opt, self.device)

    # —— 仅两个特殊分支：其余全部走 Monodepth2（默认） ——
    if method == "MRFEDepth":
        from methods.losses.MRFEDepth.MRFE_Loss import MRFELosses
        # 该 Loss 需要访问多个子模型
        self.loss = MRFELosses(self.opt, self.models, self.device)
        print("[init_losses] Using loss: MRFELosses (special branch: MRFEDepth)")

    elif method in {"Madhuanand", "madhuanand"}:
        from methods.losses.Mad.madhuanand_loss import MadhuanandLoss
        self.loss = MadhuanandLoss(self.opt, self.device)
        print("[init_losses] Using loss: MadhuanandLoss (special branch: Madhuanand)")

    else:
        # 统一回落：Monodepth2
        self.loss = _build_md2_loss()
