# my_losses.py
import torch
import torch.nn.functional as F
from layers import SSIM

class MRFELosses:
    def __init__(self, opt, models, device):
        self.opt = opt
        self.models = models
        self.device = device
        self.no_ssim = bool(getattr(self.opt, "no_ssim", False))
        self.ssim = None if self.no_ssim else SSIM().to(self.device)
    def robust_l1(self, pred, target):
        eps = 1e-3
        return torch.sqrt(torch.pow(target - pred, 2) + eps ** 2)

    def computer_depth_loss(self, tgt_f, src_f):
        alafa = 0.05
        ru = 0.85
        gi = tgt_f - src_f
        term = torch.pow(gi, 2).mean(1, True) - ru * torch.pow(gi.mean(1, True), 2)
        term = torch.clamp(term, min=1e-12)
        return alafa * torch.sqrt(term)

    def compute_perceptional_loss(self, tgt_f, src_f):
        return 0.9 * self.robust_l1(tgt_f, src_f).mean(1, True) + self.computer_depth_loss(tgt_f, src_f)

    def compute_reprojection_loss(self, pred, target):
        photometric_loss = self.robust_l1(pred, target).mean(1, True)
        if self.no_ssim or self.ssim is None:
            return photometric_loss
        ssim_loss = self.ssim(pred, target).mean(1, True)
        return 0.85 * ssim_loss + 0.15 * photometric_loss

    def get_feature_regularization_loss(self, feature, img):
        b, _, h, w = feature.size()  ##[batch, channels, height, width]
        # print(rf'==> feature shape is {b} {_} {h} {w}')
        img = F.interpolate(img, (h, w), mode='area')

        feature_dx, feature_dy = self.gradient(feature)
        img_dx, img_dy = self.gradient(img)

        feature_dxx, feature_dxy = self.gradient(feature_dx)  # feature dxx, dyx
        feature_dyx, feature_dyy = self.gradient(feature_dy)  # feature dyx, dyy

        img_dxx, img_dxy = self.gradient(img_dx)  # img dxx, dyx
        img_dyx, img_dyy = self.gradient(img_dy)  # img dyx, dyy

        # dx+dy
        smooth1 = torch.mean(feature_dx.abs() * torch.exp(-img_dx.abs().mean(1, True))) + \
                  torch.mean(feature_dy.abs() * torch.exp(-img_dy.abs().mean(1, True)))

        # dxx + 2dxy +dyy
        smooth2 = torch.mean(feature_dxx.abs() * torch.exp(-img_dxx.abs().mean(1, True))) + \
                  torch.mean(feature_dxy.abs() * torch.exp(-img_dxy.abs().mean(1, True))) + \
                  torch.mean(feature_dyx.abs() * torch.exp(-img_dyx.abs().mean(1, True))) + \
                  torch.mean(feature_dyy.abs() * torch.exp(-img_dyy.abs().mean(1, True)))

        # print(f"smooth1:{smooth1}  smooth2:{smooth2}")
        total_feature_smooth_loss = -1e-3 * smooth1 + 1e-3 * smooth2
        # print(f"total_smooth:{total_feature_smooth_loss}")


        return total_feature_smooth_loss

    def get_smooth_loss(self, disp, img):
        b, _, h, w = disp.size()  # [batch, channels, height, width]
        img = F.interpolate(img, (h, w), mode='area')

        disp_dx, disp_dy = self.gradient(disp)  # disp
        img_dx, img_dy = self.gradient(img)  # img

        disp_dxx, disp_dxy = self.gradient(disp_dx)  # disp dxx, dyx
        disp_dyx, disp_dyy = self.gradient(disp_dy)  # disp dyx, dyy

        img_dxx, img_dxy = self.gradient(img_dx)  # img  dxx, dyx
        img_dyx, img_dyy = self.gradient(img_dy)  # img  dyx, dyy

        # dx+dy
        smooth1 = torch.mean(disp_dx.abs() * torch.exp(-0.5 * img_dx.abs().mean(1, True))) + \
                  torch.mean(disp_dy.abs() * torch.exp(-0.5 * img_dy.abs().mean(1, True)))

        # dxx + 2dxy +dyy
        smooth2 = torch.mean(disp_dxx.abs() * torch.exp(-0.5 * img_dxx.abs().mean(1, True))) + \
                  torch.mean(disp_dxy.abs() * torch.exp(-0.5 * img_dxy.abs().mean(1, True))) + \
                  torch.mean(disp_dyx.abs() * torch.exp(-0.5 * img_dyx.abs().mean(1, True))) + \
                  torch.mean(disp_dyy.abs() * torch.exp(-0.5 * img_dyy.abs().mean(1, True)))

        return smooth1 + smooth2
    def gradient(self, D):
        D_dy = D[:, :, 1:] - D[:, :, :-1]
        D_dx = D[:, :, :, 1:] - D[:, :, :, :-1]
        return D_dx, D_dy
    def compute_losses(self, inputs, outputs, features):
        losses = {}
        total_loss = 0
        target = inputs[("color", 0, 0)]
        tgt_f_base = features[0] if isinstance(features, (list, tuple)) and len(features) > 0 \
            else self.models["FeatureEncoder"](inputs[("color", 0, 0)])[0]

        for i in range(5):
            f = features[i]
            reg_loss = self.get_feature_regularization_loss(f, target)
            total_loss += (reg_loss / (2 ** i)) / 5

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            perceptional_losses = []

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]

            res_img = outputs[("res_img", 0, scale)]
            _, _, h, w = res_img.size()
            target_resize = F.interpolate(target, [h, w], mode="bilinear", align_corners=False)
            img_reconstruct_loss = self.compute_reprojection_loss(res_img, target_resize)
            loss += img_reconstruct_loss.mean() / len(self.opt.scales)

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.opt.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, 0)]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                identity_reprojection_loss = (identity_reprojection_losses.mean(1, True)
                                              if self.opt.avg_reprojection
                                              else identity_reprojection_losses)
            reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                if torch.cuda.is_available():
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(self.device) * 1e-5
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), 1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean() / len(self.opt.scales)

            for frame_id in self.opt.frame_ids[1:]:
                src_f = outputs[("feature", frame_id, 0)]
                perceptional_losses.append(self.compute_perceptional_loss(tgt_f_base, src_f))
            perceptional_loss = torch.cat(perceptional_losses, 1)
            min_perceptional_loss, outputs[("min_index", scale)] = torch.min(perceptional_loss, dim=1)
            fm_loss = 1e-3 * min_perceptional_loss.mean() / len(self.opt.scales)
            loss += fm_loss

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = self.get_smooth_loss(norm_disp, color)
            loss += 1e-3 * smooth_loss / (2 ** scale) / len(self.opt.scales)

            total_loss += loss

        losses["loss"] = total_loss
        return losses
