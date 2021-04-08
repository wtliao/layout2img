import torch
import torch.nn as nn
import torch.nn.functional as F


# Adaptive instance normalization
# modified from https://github.com/NVlabs/MUNIT/blob/d79d62d99b588ae341f9826799980ae7298da553/networks.py#L453-L482
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        # projection layer
        self.weight_proj = nn.Linear(num_w, num_features)
        self.bias_proj = nn.Linear(num_w, num_features)

    def forward(self, x, w):
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        weight, bias = self.weight_proj(w).contiguous().view(-1) + 1, self.bias_proj(w).contiguous().view(-1)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, weight, bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class SpatialAdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1):
        super(SpatialAdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

        # projection layer
        self.weight_proj = nn.Linear(num_w, num_features)
        self.bias_proj = nn.Linear(num_w, num_features)

    def forward(self, x, w, bbox):
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        return x


class AdaptiveBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True):
        super(AdaptiveBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        # projection layer
        self.weight_proj = nn.Linear(num_w, num_features)
        self.bias_proj = nn.Linear(num_w, num_features)

    def forward(self, x, w):
        self._check_input_dim(x)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(x, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)

        size = output.size()
        weight, bias = self.weight_proj(w) + 1, self.bias_proj(w)
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class SpatialAdaptiveBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, num_w=512, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(SpatialAdaptiveBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        # projection layer
        self.weight_proj = nn.Linear(num_w, num_features)
        self.bias_proj = nn.Linear(num_w, num_features)

    def forward(self, x, vector, bbox):
        """
        :param x: input feature map (b, c, h, w)
        :param vector: latent vector (b*o, dim_w)
        :param bbox: bbox map (b, o, h, w)
        :return:
        """
        self._check_input_dim(x)
        exponential_average_factor = 0.0
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        output = F.batch_norm(x, self.running_mean, self.running_var,
                              self.weight, self.bias,
                              self.training or not self.track_running_stats,
                              exponential_average_factor, self.eps)

        b, o, _, _ = bbox.size()
        _, _, h, w = x.size()
        bbox = F.interpolate(bbox, size=(h, w), mode='bilinear')
        # calculate weight and bias
        weight, bias = self.weight_proj(vector), self.bias_proj(vector)

        weight, bias = weight.view(b, o, -1), bias.view(b, o, -1)

        weight = torch.sum(bbox.unsqueeze(2) * weight.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                 (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6) + 1
        bias = torch.sum(bbox.unsqueeze(2) * bias.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
               (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6)
        return weight * output + bias

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


from .sync_batchnorm import SynchronizedBatchNorm2d


class SpatialAdaptiveSynBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_w=512, batchnorm_func=SynchronizedBatchNorm2d, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(SpatialAdaptiveSynBatchNorm2d, self).__init__()
        # projection layer
        self.num_features = num_features
        self.weight_proj = nn.utils.spectral_norm(nn.Linear(num_w, num_features))
        self.bias_proj = nn.utils.spectral_norm(nn.Linear(num_w, num_features))
        self.batch_norm2d = batchnorm_func(num_features, eps=eps, momentum=momentum,
                                           affine=affine)

    def forward(self, x, vector, bbox):
        """
        :param x: input feature map (b, c, h, w)
        :param vector: latent vector (b*o, dim_w)
        :param bbox: bbox map (b, o, h, w)
        :return:
        """
        # self._check_input_dim(x)
        output = self.batch_norm2d(x)

        b, o, bh, bw = bbox.size()
        _, _, h, w = x.size()
        if bh != h or bw != w:
            bbox = F.interpolate(bbox, size=(h, w), mode='bilinear')
        # calculate weight and bias
        weight, bias = self.weight_proj(vector), self.bias_proj(vector)

        weight, bias = weight.view(b, o, -1), bias.view(b, o, -1)

        weight = torch.sum(bbox.unsqueeze(2) * weight.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                 (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6) + 1
        bias = torch.sum(bbox.unsqueeze(2) * bias.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
               (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6)
        return weight * output + bias

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
class SpatialAdaptiveSynBatchNorm2d_part(nn.Module):
    def __init__(self, num_features, num_w=512, batchnorm_func=SynchronizedBatchNorm2d, eps=1e-5, momentum=0.1, affine=False,
                 track_running_stats=True):
        super(SpatialAdaptiveSynBatchNorm2d_part, self).__init__()
        # projection layer
        self.num_features = num_features
        self.weight_proj = nn.utils.spectral_norm(nn.Linear(num_w, num_features*3))
        self.bias_proj = nn.utils.spectral_norm(nn.Linear(num_w, num_features*3))
        self.batch_norm2d = batchnorm_func(num_features, eps=eps, momentum=momentum,
                                           affine=affine)

    def forward(self, x, vector, bbox, bbox1, bbox2):
        """
        :param x: input feature map (b, c, h, w)
        :param vector: latent vector (b*o, dim_w)
        :param bbox: bbox map (b, o, h, w)
        :return:
        """
        # self._check_input_dim(x)
        #print(vector.shape)
        output = self.batch_norm2d(x)
        #print(output.shape)

        b, o, bh, bw = bbox.size()
        _, _, h, w = x.size()
        if bh != h or bw != w:
            bbox = F.interpolate(bbox, size=(h, w), mode='bilinear')
            bbox1 = F.interpolate(bbox1, size=(h, w), mode='bilinear')
            bbox2 = F.interpolate(bbox2, size=(h, w), mode='bilinear')
        # calculate weight and bias
        weight_all, bias_all = self.weight_proj(vector), self.bias_proj(vector)
        #print(weight_all.shape)
        #print(bias_all.shape)
        weight, bias = weight_all[:,:self.num_features], bias_all[:,:self.num_features]
        weight1, bias1 = weight_all[:,self.num_features:2*self.num_features], bias_all[:,self.num_features:2*self.num_features]
        weight2, bias2 = weight_all[:,2*self.num_features:], bias_all[:,2*self.num_features:]

        weight, bias = weight.view(b, o, -1), bias.view(b, o, -1)
        weight1, bias1 = weight1.view(b, o, -1), bias1.view(b, o, -1)
        weight2, bias2 = weight2.view(b, o, -1), bias2.view(b, o, -1)

        weight = torch.sum(bbox.unsqueeze(2) * weight.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                 (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6) + 1
        weight1 = torch.sum(bbox1.unsqueeze(2) * weight1.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                 (torch.sum(bbox1.unsqueeze(2), dim=1, keepdim=False) + 1e-6) + 1
        weight2 = torch.sum(bbox2.unsqueeze(2) * weight2.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
                 (torch.sum(bbox2.unsqueeze(2), dim=1, keepdim=False) + 1e-6) + 1
        #print(weight.shape)
        #print(weight1.shape)
        bias = torch.sum(bbox.unsqueeze(2) * bias.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
               (torch.sum(bbox.unsqueeze(2), dim=1, keepdim=False) + 1e-6)
        bias1 = torch.sum(bbox1.unsqueeze(2) * bias1.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
               (torch.sum(bbox1.unsqueeze(2), dim=1, keepdim=False) + 1e-6)
        bias2 = torch.sum(bbox2.unsqueeze(2) * bias2.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=False) / \
               (torch.sum(bbox2.unsqueeze(2), dim=1, keepdim=False) + 1e-6)
        #print(bias.shape)
        return (weight * output + bias) + (weight1 * output + bias1) + (weight2 * output + bias2)

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
