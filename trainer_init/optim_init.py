import torch.optim as optim


def init_optimizers(self):
    """Initialize optimizer and scheduler for the supported paper methods."""
    self.model_optimizer = optim.Adam(
        self.parameters_to_train,
        self.opt.learning_rate,
    )
    self.model_lr_scheduler = optim.lr_scheduler.StepLR(
        self.model_optimizer,
        step_size=self.opt.scheduler_step_size,
        gamma=0.1,
    )
