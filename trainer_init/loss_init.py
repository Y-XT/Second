def init_losses(self):
    """Initialize the photometric loss used by the supported paper methods."""
    from methods.losses.monodepth2.monodepth2_loss import Monodepth2Loss

    self.loss = Monodepth2Loss(self.opt, self.device)
    print("[init_losses] Using loss: Monodepth2Loss")
