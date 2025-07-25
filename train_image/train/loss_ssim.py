# Copied from SubtleMR 2.3 repo by Zechen Zhou

import warnings

import torch
import torch.nn.functional as F


def _fspecial_gauss_1d(size, sigma):
    r"""Create 1-D gauss kernel
    Args:
        size (int): the size of gauss kernel
        sigma (float): sigma of normal distribution

    Returns:
        torch.Tensor: 1D kernel (1 x 1 x size)
    """
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2

    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()

    return g.unsqueeze(0).unsqueeze(0)


def gaussian_filter(input, win):
    r""" Blur input with 1-D kernel
    Args:
        input (torch.Tensor): a batch of tensors to be blurred
        window (torch.Tensor): 1-D gauss kernel

    Returns:
        torch.Tensor: blurred tensors
    """
    assert all([ws == 1 for ws in win.shape[1:-1]]), win.shape
    if len(input.shape) == 4:
        conv = F.conv2d
    elif len(input.shape) == 5:
        conv = F.conv3d
    else:
        raise NotImplementedError(input.shape)

    C = input.shape[1]
    out = input
    for i, s in enumerate(input.shape[2:]):
        if s >= win.shape[-1]:
            out = conv(out, weight=win.transpose(2 + i, -1), stride=1, padding=0, groups=C)
        else:
            warnings.warn(
                f"Skipping Gaussian Smoothing at dimension 2+{i} for input: {input.shape} and win size: {win.shape[-1]}"
            )

    return out


def _ssim(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)  # set alpha=beta=gamma=1
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    ssim_per_channel = torch.flatten(ssim_map, 2).mean(-1)
    cs = torch.flatten(cs_map, 2).mean(-1)
    return ssim_per_channel, cs


def _ssim_lcs(X, Y, data_range, win, size_average=True, K=(0.01, 0.03)):

    r""" Calculate ssim index for X and Y

    Args:
        X (torch.Tensor): images
        Y (torch.Tensor): images
        win (torch.Tensor): 1-D gauss kernel
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar

    Returns:
        torch.Tensor: ssim results.
    """
    K1, K2 = K
    # batch, channel, [depth,] height, width = X.shape
    compensation = 1.0

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    win = win.to(X.device, dtype=X.dtype)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = compensation * (gaussian_filter(X * X, win) - mu1_sq)
    sigma2_sq = compensation * (gaussian_filter(Y * Y, win) - mu2_sq)
    sigma12 = compensation * (gaussian_filter(X * Y, win) - mu1_mu2)

    # set alpha=beta=gamma=1
    l_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    return torch.flatten(l_map, 2).mean(-1), torch.flatten(cs_map, 2).mean(-1)


def ssim(
    X,
    Y,
    data_range=255,
    size_average=True,
    win_size=11,
    win_sigma=1.5,
    win=None,
    K=(0.01, 0.03),
    nonnegative_ssim=False,
):
    r""" interface of ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,H,W)
        Y (torch.Tensor): a batch of images, (N,C,H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu

    Returns:
        torch.Tensor: ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if len(X.shape) not in (4, 5):
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    ssim_per_channel, cs = _ssim(X, Y, data_range=data_range, win=win, size_average=False, K=K)
    if nonnegative_ssim:
        ssim_per_channel = torch.relu(ssim_per_channel)

    if size_average:
        return ssim_per_channel.mean()
    else:
        return ssim_per_channel.mean(1)


def ms_ssim(
    X, Y, data_range=255, size_average=True, win_size=11, win_sigma=1.5, win=None, weights=None, K=(0.01, 0.03)
):

    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        win_size: (int, optional): the size of gauss kernel
        win_sigma: (float, optional): sigma of normal distribution
        win (torch.Tensor, optional): 1-D gauss kernel. if None, a new kernel will be created according to win_size and win_sigma
        weights (list, optional): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if len(X.shape) == 4:
        avg_pool = F.avg_pool2d
    elif len(X.shape) == 5:
        avg_pool = F.avg_pool3d
    else:
        raise ValueError(f"Input images should be 4-d or 5-d tensors, but got {X.shape}")

    if win is not None:  # set win_size
        win_size = win.shape[-1]

    if not (win_size % 2 == 1):
        raise ValueError("Window size should be odd.")

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    #weights = torch.FloatTensor(weights, device=X.device, dtype=X.dtype)
    weights = torch.tensor(weights, device=X.device, dtype=X.dtype)
    levels = weights.shape[0]

    smaller_side = min(X.shape[-2:])
    assert smaller_side > (win_size - 1) * (2 ** (levels-1)), \
    "Image size should be larger than %d due to the %d downsamplings in ms-ssim" % ((win_size - 1) * (2 ** (levels-1)), \
                                                                                    levels-1)

    if win is None:
        win = _fspecial_gauss_1d(win_size, win_sigma)
        win = win.repeat([X.shape[1]] + [1] * (len(X.shape) - 1))

    mcs = []
    for i in range(levels):
        ssim_per_channel, cs = _ssim(X, Y, win=win, data_range=data_range, size_average=False, K=K)

        if i < levels - 1:
            mcs.append(torch.relu(cs))
            padding = [s % 2 for s in X.shape[2:]]
            X = avg_pool(X, kernel_size=2, padding=padding)
            Y = avg_pool(Y, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


def define_filter(win_sz_min, win_sz_max, scale_num, channel, spatial_dim):
    if win_sz_max < win_sz_min:
        raise ValueError(f"Input image size should be larger than 3 in all spatial dimensions, but got {X.shape}")

    scale_num_max = round((win_sz_max - win_sz_min) / 2) + 1
    if scale_num > scale_num_max:
        warnings.warn(
            f"Too many # of scales, change it from {scale_num} to {scale_num_max}"
        )
        scale_num = scale_num_max

    gr = (torch.sqrt(torch.tensor(5)) - 1) / 2 # golden ratio ~ 0.618
    if scale_num == 1:
        if win_sz_max >= 11:
            # default setting for SSIM
            win_sz = torch.tensor([11.])
            sigma = torch.tensor([1.5])
        else:
            win_sz = torch.tensor([win_sz_max])
            sigma = win_sz / 6 # window size -> sigma using the +/-3 sigma criteria
    elif scale_num == 2:
        win_sz = torch.tensor([max(11 * gr, win_sz_min), min(11 / gr, win_sz_max)]) // 2 * 2 + 1
        sigma = win_sz / 6
    elif scale_num > 2 and scale_num <= scale_num_max // 2:
        # uniform window size increasing
        #win_sz = torch.linspace(win_sz_min, win_sz_max, steps=scale_num)

        # non-uniform window size increasing
        # 0 ~ scale_num_max - 1
        # 0 ~ (scale_num_max - 1) * 2 = (win_sz_max - win_sz_min)
        # win_sz_min ~ win_sz_max
        win_sz = win_sz_min + ((gr * scale_num_max * torch.arange(scale_num)) % scale_num_max) * 2
        win_sz, _ = torch.sort(win_sz)

        win_sz = win_sz // 2 * 2 + 1
        sigma = win_sz / 6
    elif scale_num > scale_num_max // 2 and scale_num < scale_num_max:
        # uniform window size increasing
        #win_sz = torch.linspace(win_sz_min, win_sz_max, steps=scale_num)
        #win_sz = win_sz // 2 * 2 + 1

        # non-uniform window size increasing
        # randomly drop from 1 to scale_num_max-1
        # 0 ~ scale_num_max - 2
        # 1 ~ scale_num_max - 1
        # 2 ~ (scale_num_max - 1) * 2 = (win_sz_max - win_sz_min)
        # win_sz_min + 2 ~ win_sz_max
        win_sz_drop = win_sz_min + ((gr * (scale_num_max - 1) * torch.arange(scale_num_max - scale_num) +
                                     scale_num_max - 2) % (scale_num_max - 1) + 1) * 2
        win_sz_drop = win_sz_drop // 2 * 2 + 1

        win_sz = []
        for ii in range(1, scale_num_max+1):
            if 2*ii+1 not in win_sz_drop:
                win_sz.append(2*ii+1)
        win_sz = torch.tensor(win_sz)

        sigma = win_sz / 6
    elif scale_num == scale_num_max:
        win_sz = torch.linspace(win_sz_min, win_sz_max, steps=scale_num) // 2 * 2 + 1
        sigma = win_sz / 6
    else:
        raise ValueError(f"# of scales has to be a positive number, but got {scale_num}")
    #print(win_sz)

    g_filter = []
    for ind in range(scale_num):
        g = _fspecial_gauss_1d(win_sz[ind], sigma[ind])
        g = g.repeat([channel, 1] + [1] * spatial_dim)
        g_filter.append(g)

    return g_filter


def ms_ssim_(X, Y, data_range=255, size_average=True,
             weights=[0.2]*5, g_filter=None, K=(0.01, 0.03)):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        weights (list): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if g_filter is None:
        smaller_side = min(X.shape[2:])
        #win_sz_min = 1
        win_sz_min = 3
        win_sz_max = smaller_side - 1 + smaller_side % 2
        g_filter = define_filter(win_sz_min, win_sz_max, len(weights), X.shape[1], len(X.shape)-2)
    scale_num = len(g_filter)
    weights = torch.tensor(weights[:scale_num], device=X.device, dtype=X.dtype)

    mcs = []
    for i in range(scale_num):
        ssim_per_channel, cs = _ssim(X, Y, win=g_filter[i], data_range=data_range, size_average=False, K=K)

        if i < scale_num - 1:
            mcs.append(torch.relu(cs))
            #mcs.append(cs)

    ssim_per_channel = torch.relu(ssim_per_channel)  # (batch, channel)
    mcs_and_ssim = torch.stack(mcs + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)

def ms_ssim_opt(X, Y, data_range=255, size_average=True,
                weights=[7.78397865e-1, 6.28404664e-6, 6.07334001e-1, 3.59218858e-1,
                         5.29029266e-1, 8.83466833e-1, 5.03892419e-1, 1.14383229e0,
                         4.65256320e-1, 1.67463995e0], g_filter=None, K=(0.01, 0.03)):
    r""" interface of ms-ssim
    Args:
        X (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        Y (torch.Tensor): a batch of images, (N,C,[T,]H,W)
        data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
        size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
        weights (list): weights for different levels
        K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
    Returns:
        torch.Tensor: ms-ssim results
    """
    if not X.shape == Y.shape:
        raise ValueError("Input images should have the same dimensions.")

    for d in range(len(X.shape) - 1, 1, -1):
        X = X.squeeze(dim=d)
        Y = Y.squeeze(dim=d)

    if not X.type() == Y.type():
        raise ValueError("Input images should have the same dtype.")

    if g_filter is None:
        smaller_side = min(X.shape[2:])
        #win_sz_min = 1
        win_sz_min = 3
        win_sz_max = smaller_side - 1 + smaller_side % 2
        g_filter = define_filter(win_sz_min, win_sz_max, len(weights), X.shape[1], len(X.shape)-2)
    scale_num = len(g_filter)
    weights = torch.tensor(weights[:scale_num*2], device=X.device, dtype=X.dtype)

    mlcs = []
    for i in range(scale_num):
        lu, cs = _ssim_lcs(X, Y, win=g_filter[i], data_range=data_range, size_average=False, K=K)
        mlcs.append(torch.relu(lu))
        mlcs.append(torch.relu(cs))

    mlcs = torch.stack(mlcs, dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mlcs ** weights.view(-1, 1, 1), dim=0)

    if size_average:
        return ms_ssim_val.mean()
    else:
        return ms_ssim_val.mean(1)


class SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        K=(0.01, 0.03),
        nonnegative_ssim=False,
    ):
        r""" class for ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
            nonnegative_ssim (bool, optional): force the ssim response to be nonnegative with relu.
        """

        super(SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.K = K
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, X, Y):
        return ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            K=self.K,
            nonnegative_ssim=self.nonnegative_ssim,
        )


class MS_SSIM(torch.nn.Module):
    def __init__(
        self,
        data_range=255,
        size_average=True,
        win_size=11,
        win_sigma=1.5,
        channel=3,
        spatial_dims=2,
        weights=None,
        K=(0.01, 0.03),
    ):
        r""" class for ms-ssim
        Args:
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            win_size: (int, optional): the size of gauss kernel
            win_sigma: (float, optional): sigma of normal distribution
            channel (int, optional): input channels (default: 3)
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MS_SSIM, self).__init__()
        self.win_size = win_size
        self.win = _fspecial_gauss_1d(win_size, win_sigma).repeat([channel, 1] + [1] * spatial_dims)
        self.size_average = size_average
        self.data_range = data_range
        self.weights = weights
        self.K = K

    def forward(self, X, Y):
        return ms_ssim(
            X,
            Y,
            data_range=self.data_range,
            size_average=self.size_average,
            win=self.win,
            weights=self.weights,
            K=self.K,
        )


class MSF_SSIM(torch.nn.Module):
    def __init__(
        self,
        img_shape,
        data_range=255,
        size_average=True,
        weights=[0.2]*5,
        K=(0.01, 0.03),
        win_sz_min=3,
        kadid10k_opt=True
    ):
        r""" class for ms-ssim
        Args:
            img_shape (list): C [x D] x H x W
            data_range (float or int, optional): value range of input images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            weights (list, optional): weights for different levels
            K (list or tuple, optional): scalar constants (K1, K2). Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.
        """

        super(MSF_SSIM, self).__init__()
        self.kadid10k_opt = kadid10k_opt
        smaller_side = min(img_shape[1:])
        if self.kadid10k_opt:
            self.g_filter = define_filter(win_sz_min=3, win_sz_max=47, scale_num=5,
                                          channel=img_shape[0], spatial_dim=len(img_shape)-1)
            self.weights = [7.78397865e-1, 6.28404664e-6,
                            6.07334001e-1, 3.59218858e-1,
                            5.29029266e-1, 8.83466833e-1,
                            5.03892419e-1, 1.14383229e0,
                            4.65256320e-1, 1.67463995e0]
        else:
            self.g_filter = define_filter(win_sz_min=win_sz_min, win_sz_max=smaller_side - 1 + smaller_side % 2,
                                          scale_num=len(weights), channel=img_shape[0], spatial_dim=len(img_shape)-1)
            self.weights = weights
        self.size_average = size_average
        self.data_range = data_range
        self.K = K

    def forward(self, X, Y):
        if self.kadid10k_opt:
            return ms_ssim_opt(
                X,
                Y,
                data_range=self.data_range,
                size_average=self.size_average,
                g_filter=self.g_filter,
                weights=self.weights,
                K=self.K,
            )
        else:
            return ms_ssim_(
                X,
                Y,
                data_range=self.data_range,
                size_average=self.size_average,
                g_filter=self.g_filter,
                weights=self.weights,
                K=self.K,
            )
