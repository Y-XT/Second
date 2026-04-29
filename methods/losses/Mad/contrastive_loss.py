import torch
import torch.nn as nn
from methods.losses.Mad import ops

Default = {'margin': 0.5, 'alpha': None, 'beta': None, 'n_neg': 10}


class PixelwiseContrastiveLoss(nn.Module):
    """
    Implementation of "pixel-wise" contrastive loss. Contrastive loss typically compares two whole images.
            L = (Y) * (1/2 * d**2) + (1 - Y) * (1/2 * max(0, margin - d)**2)
    In this instance, we instead compare pairs of features within those images.
    Positive matches are given by ground truth correspondences between images.
    Negative matches are generated on-the-fly based on provided parameters.
    Attributes:
        margin (float): Target margin distance between positives and negatives
        alpha (int): Minimum distance from original positive KeyPoint
        beta (int): Maximum distance from original positive KeyPoint
        n-neg (int): Number of negative samples to generate
    Methods:
        forward: Compute pixel-wise contrastive loss
        forward_eval: Detailed forward pass for logging
    """
    def __init__(self, margin=0.5, alpha=None, beta=None, n_neg=10):
        super(PixelwiseContrastiveLoss,self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.n_neg = n_neg
        self.lambda_neg = 1.0

        self._dist = nn.PairwiseDistance()

    def __repr__(self):
        return f'{self.__class__.__qualname__}({self.margin}, {self.alpha}, {self.beta}, {self.n_neg})'

    def __str__(self):
        return f'Min{self.alpha or 0}_Max{self.beta or "Inf"}'

    @staticmethod
    def create_parser(parser):
        parser.add_argument('--margin', default=0.5, help='Target distance between negative feature embeddings.')
        parser.add_argument('--alpha', default=None, type=float, help='Minimum distance from positive KeyPoint')
        parser.add_argument('--beta', default=None, type=float, help='Maximum distance from positive KeyPoint')
        parser.add_argument('--n_neg', default=10, help='Number of negative samples to generate')

    def forward(self, predicted, targetim):
        """ Compute pixel-wise contrastive loss.
        :param features: Vertically stacked feature maps (b, n-dim, h*2, w)
        :param labels: Horizontally stacked correspondence KeyPoints (b, n-kpts, 4) -> (x1, y1, x2, y2)
        :return: Loss
        """
        source = predicted
        target = targetim
        batch_size, _, height, width = source.shape

        num_points = self._adaptive_num_points(height, width)
        source_kpts = self._sample_keypoints(batch_size, height, width, source.device, num_points)
        target_kpts = source_kpts.clone()

        loss_pos = self._positive_loss(source, target, source_kpts, target_kpts)[0]
        if self.n_neg > 0 and self.lambda_neg > 0:
            loss_neg = self._negative_loss(source, target, source_kpts, target_kpts)[0]
        else:
            loss_neg = source.new_zeros(())
        return loss_pos + self.lambda_neg * loss_neg

    @staticmethod
    def _adaptive_num_points(height, width):
        total = max(1, int(height) * int(width))
        num_points = total // 2048
        num_points = max(16, min(512, num_points))
        return min(num_points, total)

    @staticmethod
    def _sample_keypoints(batch_size, height, width, device, num_points):
        x = torch.randint(0, int(width), (batch_size, num_points), device=device)
        y = torch.randint(0, int(height), (batch_size, num_points), device=device)
        return torch.stack([x, y], dim=-1).to(dtype=torch.float32)

    def _calc_distance(self, source, target, source_kpts, target_kpts):
        
        source_descriptors = ops.extract_kpt_vectors(source, source_kpts).permute([0, 2, 1])
        target_descriptors = ops.extract_kpt_vectors(target, target_kpts).permute([0, 2, 1])
        return self._dist(source_descriptors, target_descriptors)

    def _positive_loss(self, source, target, source_kpts, target_kpts):
        dist = self._calc_distance(source, target, source_kpts, target_kpts)
        loss = (dist**2).mean() / 2
        return loss, dist

    def _negative_loss(self, source, target, source_kpts, target_kpts):
        dsource_kpts, dtarget_kpts = self._generate_negative_like(source, source_kpts, target_kpts)

        dist = self._calc_distance(source, target, dsource_kpts, dtarget_kpts)
        margin_dist = (self.margin - dist).clamp(min=0.0)
        loss = (margin_dist ** 2).mean() / 2
        return loss, dist

    def _generate_negative_like(self, other, source_kpts, target_kpts):
        if self.n_neg <= 0:
            return source_kpts, target_kpts

        # Source points remain the same
        source_kpts = source_kpts.repeat([1, self.n_neg, 1])

        # Target points + offset according to method
        target_kpts = target_kpts.repeat([1, self.n_neg, 1])
        target_kpts = self._permute_negatives(target_kpts, other.shape)
        return source_kpts, target_kpts

    def _permute_negatives(self, kpts, shape):
        h, w = shape[-2:]
        max_hw = max(int(h), int(w))
        low = int(self.alpha) if self.alpha is not None else 1
        low = max(1, low)
        default_high = max(low, max_hw - 1)
        high = int(self.beta) if self.beta is not None else default_high
        high = max(low, high)

        shift_mag = torch.randint(
            low=low,
            high=high + 1,
            size=kpts.shape,
            device=kpts.device,
        )
        sign = torch.randint(0, 2, size=kpts.shape, device=kpts.device) * 2 - 1
        shift = (shift_mag * sign).to(dtype=kpts.dtype)

        new_kpts = kpts + shift
        x = torch.remainder(new_kpts[..., 0], float(w))
        y = torch.remainder(new_kpts[..., 1], float(h))
        return torch.stack([x, y], dim=-1)
