# madhuanand_loss.py

import torch
from layers import get_smooth_loss
from methods.losses.monodepth2.monodepth2_loss import Monodepth2Loss
from methods.losses.Mad.contrastive_loss import PixelwiseContrastiveLoss  # 确保你有这个模块

class MadhuanandLoss(Monodepth2Loss):
    def __init__(self, opt, device):
        super().__init__(opt, device)
        # Paper-style fixed weights: L = 1*Lp + 0.001*Ls + 1*Lm + 0.5*Lc
        # (Lm is handled via automasking branch in this MD2-style implementation.)
        self.lambda_photo = 1.0
        self.lambda_smooth = 0.001
        self.lambda_contrast = 0.5
        self.contrastive = PixelwiseContrastiveLoss(
            margin=0.5,
            alpha=None,
            beta=None,
            n_neg=10,
        )

    def _source_frame_ids(self, inputs):
        source_ids = []
        for frame_id in self.opt.frame_ids[1:]:
            if frame_id == "s":
                continue
            if ("color", frame_id, 0) in inputs:
                source_ids.append(frame_id)
        return source_ids

    def compute_losses(self, inputs, outputs):
        losses = {}
        total_loss = 0
        source_frame_ids = self._source_frame_ids(inputs)
        if not source_frame_ids:
            raise ValueError("MadhuanandLoss requires at least one source frame")

        for scale in self.opt.scales:
            reprojection_losses = []

            source_scale = 0
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in source_frame_ids:
                pred = outputs.get(("color", frame_id, scale), None)
                if pred is None:
                    continue
                reproj_loss = self.compute_reprojection_loss(pred, target)

                if self.opt.distorted_mask:
                    valid_mask = outputs.get(("distorted_mask", frame_id, scale), None)
                    if valid_mask is not None:
                        reproj_loss = reproj_loss * valid_mask + (1.0 - valid_mask) * 1e5

                reprojection_losses.append(reproj_loss)

            if not reprojection_losses:
                raise KeyError(f"No reconstructed source images found at scale {scale}")
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_losses = []
                for frame_id in source_frame_ids:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_losses.append(self.compute_reprojection_loss(pred, target))
                identity_losses = torch.cat(identity_losses, 1)

                if self.opt.avg_reprojection:
                    identity_loss = identity_losses.mean(1, keepdim=True)
                else:
                    identity_loss = identity_losses

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                identity_loss += torch.randn(identity_loss.shape, device=self.device) * 1e-5
                combined = torch.cat((identity_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            to_optimise = combined if combined.shape[1] == 1 else torch.min(combined, dim=1)[0]

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                    torch.min(combined, dim=1)[1] > identity_loss.shape[1] - 1).float()

            photo_term = to_optimise.mean()

            contrastive_terms = []
            for frame_id in source_frame_ids:
                pred = outputs.get(("color", frame_id, scale), None)
                if pred is None:
                    continue
                contrastive_terms.append(self.contrastive(pred, target))
            if contrastive_terms:
                contrastive_term = torch.stack(contrastive_terms).mean()
            else:
                contrastive_term = photo_term.new_zeros(())

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_term = get_smooth_loss(norm_disp, color) / (2 ** scale)

            loss = (
                self.lambda_photo * photo_term
                + self.lambda_contrast * contrastive_term
                + self.lambda_smooth * smooth_term
            )

            total_loss += loss
            losses["loss/{}".format(scale)] = loss
            losses["loss_photo/{}".format(scale)] = photo_term
            losses["loss_contrast/{}".format(scale)] = contrastive_term
            losses["loss_smooth/{}".format(scale)] = smooth_term

        total_loss /= len(self.opt.scales)
        losses["loss"] = total_loss
        return losses
