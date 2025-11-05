import math
import numbers
import random
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class WhiteNoise(nn.Module):
    def __init__(self, std: float = 0.1):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * self.std
        return x + noise


class MeanDriftNoise(nn.Module):
    def __init__(self, std: float = 0.1):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError("MeanDriftNoise expects (T, C) or (B, T, C) tensors")

        drift_shape = list(x.shape)
        drift_shape[-2] = 1  # single frame drift per channel
        noise = torch.randn(drift_shape, device=x.device, dtype=x.dtype) * self.std
        return x + noise

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= (
                1
                / (std * math.sqrt(2 * math.pi))
                * torch.exp(-(((mgrid - mean) / std) ** 2) / 2)
            )

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer("weight", kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                "Only 1, 2 and 3 dimensions are supported. Received {}.".format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding="same")


def _bandpass_filter_fft(x: torch.Tensor, sample_rate: float, band: Sequence[float]) -> torch.Tensor:
    time_dim = x.shape[-1]
    freqs = torch.fft.fftfreq(time_dim, d=1.0 / sample_rate).to(x.device)
    low, high = float(band[0]), float(band[1])
    mask = (freqs.abs() >= low) & (freqs.abs() <= high)
    mask = mask.to(x.device, x.dtype)

    spectrum = torch.fft.fft(x, dim=-1)
    spectrum = spectrum * mask
    return torch.fft.ifft(spectrum, dim=-1)


def _hilbert_transform_fft(x: torch.Tensor) -> torch.Tensor:
    time_dim = x.shape[-1]
    spectrum = torch.fft.fft(x, dim=-1)
    if x.is_complex():
        base_dtype = torch.float32 if x.dtype == torch.complex64 else torch.float64
    else:
        base_dtype = torch.float32 if x.dtype in (torch.float16, torch.float32) else torch.float64
    dtype = torch.complex64 if base_dtype == torch.float32 else torch.complex128
    h = torch.zeros(time_dim, device=x.device, dtype=dtype)
    if time_dim % 2 == 0:
        h[0] = 1
        h[time_dim // 2] = 1
        h[1 : time_dim // 2] = 2
    else:
        h[0] = 1
        h[1 : (time_dim + 1) // 2] = 2
    return torch.fft.ifft(spectrum * h, dim=-1)


class AdditiveNoise(nn.Module):
    """Add Gaussian noise with configurable probability."""

    def __init__(self, std: float = 0.05, p: float = 0.5, clamp: Optional[float] = None):
        super().__init__()
        self.std = std
        self.p = p
        self.clamp = clamp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or random.random() > self.p:
            return x

        noise = torch.randn_like(x) * self.std
        out = x + noise
        if self.clamp is not None:
            out = torch.clamp(out, min=-self.clamp, max=self.clamp)
        return out


class ChannelDropout(nn.Module):
    """Randomly drop entire channels across the temporal dimension."""

    def __init__(
        self,
        dropout_rate: float = 0.1,
        p: float = 0.5,
        fill_value: float = 0.0,
        min_channels: int = 1,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.p = p
        self.fill_value = fill_value
        self.min_channels = min_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or self.dropout_rate <= 0 or random.random() > self.p:
            return x

        if x.dim() not in (2, 3):
            raise ValueError("ChannelDropout expects tensors shaped (T, C) or (B, T, C)")

        channels = x.shape[-1]
        if channels <= self.min_channels:
            return x

        mask = torch.rand(channels, device=x.device) < self.dropout_rate
        if mask.sum().item() >= channels:
            # Ensure at least one channel survives
            keep_idx = random.randrange(channels)
            mask[keep_idx] = False

        out = x.clone()
        if x.dim() == 2:
            out[:, mask] = self.fill_value
        else:
            out[:, :, mask] = self.fill_value
        return out


class TimeWarp(nn.Module):
    """Apply random time warping by stretching/compressing sequences."""

    def __init__(
        self,
        max_warp: float = 0.2,
        p: float = 0.5,
        mode: str = "linear",
    ) -> None:
        super().__init__()
        if max_warp < 0:
            raise ValueError("max_warp must be non-negative")
        self.max_warp = max_warp
        self.p = p
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0 or self.max_warp <= 0 or random.random() > self.p:
            return x

        if x.dim() not in (2, 3):
            raise ValueError("TimeWarp expects tensors shaped (T, C) or (B, T, C)")

        if x.dim() == 2:
            x_work = x.unsqueeze(0).transpose(1, 2)  # (1, C, T)
        else:
            x_work = x.transpose(1, 2)  # (B, C, T)

        original_len = x_work.shape[-1]
        scale = 1.0 + random.uniform(-self.max_warp, self.max_warp)
        new_len = max(2, int(round(original_len * scale)))

        interpolate_kwargs = {
            "size": new_len,
            "mode": self.mode,
        }
        if self.mode in {"linear", "bilinear"}:
            interpolate_kwargs["align_corners"] = False

        warped = F.interpolate(x_work, **interpolate_kwargs)

        if new_len > original_len:
            start = random.randint(0, new_len - original_len)
            warped = warped[..., start : start + original_len]
        elif new_len < original_len:
            pad_total = original_len - new_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            warped = F.pad(warped, (pad_left, pad_right))

        if x.dim() == 2:
            return warped.transpose(1, 2).squeeze(0)
        return warped.transpose(1, 2)


class HighGammaPower(nn.Module):
    """Approximate high-gamma power via band-pass filtering and Hilbert envelope."""

    def __init__(
        self,
        sample_rate: float,
        band: Iterable[float],
        log_power: bool = True,
        zscore: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.sample_rate = float(sample_rate)
        if len(list(band)) != 2:
            raise ValueError("band must contain two elements: [low, high]")
        self.band = (float(band[0]), float(band[1]))
        self.log_power = log_power
        self.zscore = zscore
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in (2, 3):
            raise ValueError("HighGammaPower expects tensors shaped (T, C) or (B, T, C)")

        if x.dim() == 2:
            x_work = x.transpose(0, 1).unsqueeze(0)  # (1, C, T)
        else:
            x_work = x.transpose(1, 2)  # (B, C, T)

        filtered = _bandpass_filter_fft(x_work, self.sample_rate, self.band)
        analytic = _hilbert_transform_fft(filtered)
        power = analytic.abs() ** 2

        if self.log_power:
            power = torch.log(power + self.eps)

        if self.zscore:
            mean = power.mean(dim=-1, keepdim=True)
            std = power.std(dim=-1, keepdim=True) + self.eps
            power = (power - mean) / std

        if x.dim() == 2:
            return power.squeeze(0).transpose(0, 1)
        return power.transpose(1, 2)


class MultiBandHilbertEnvelope(nn.Module):
    """Compute Hilbert envelopes for multiple frequency bands."""

    def __init__(
        self,
        sample_rate: float,
        bands: Sequence[Sequence[float]],
        log_amplitude: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if not bands:
            raise ValueError("bands must contain at least one frequency range")
        self.sample_rate = float(sample_rate)
        self.bands = [tuple(map(float, band)) for band in bands]
        self.log_amplitude = log_amplitude
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in (2, 3):
            raise ValueError("MultiBandHilbertEnvelope expects (T, C) or (B, T, C) tensors")

        if x.dim() == 2:
            x_work = x.transpose(0, 1).unsqueeze(0)
        else:
            x_work = x.transpose(1, 2)

        envelopes = []
        for band in self.bands:
            filtered = _bandpass_filter_fft(x_work, self.sample_rate, band)
            analytic = _hilbert_transform_fft(filtered)
            amplitude = analytic.abs()
            if self.log_amplitude:
                amplitude = torch.log(amplitude + self.eps)
            envelopes.append(amplitude)

        combined = torch.cat(envelopes, dim=1)

        if x.dim() == 2:
            return combined.squeeze(0).transpose(0, 1)
        return combined.transpose(1, 2)


class Compose(nn.Module):
    def __init__(self, transforms: Iterable[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(list(transforms))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


def build_augmentation(name: str, params: Optional[Dict] = None, *, input_channels: Optional[int] = None) -> nn.Module:
    params = params.copy() if params else {}
    registry = {
        "GaussianSmoothing": GaussianSmoothing,
        "WhiteNoise": WhiteNoise,
        "MeanDriftNoise": MeanDriftNoise,
        "AdditiveNoise": AdditiveNoise,
        "ChannelDropout": ChannelDropout,
        "TimeWarp": TimeWarp,
        "HighGammaPower": HighGammaPower,
        "MultiBandHilbertEnvelope": MultiBandHilbertEnvelope,
    }

    if name not in registry:
        raise KeyError(f"Unknown augmentation: {name}")

    if name == "GaussianSmoothing" and "channels" not in params:
        if input_channels is None:
            raise ValueError(
                "GaussianSmoothing requires 'channels'; provide input_channels to build_augmentation"
            )
        params = params.copy()
        params["channels"] = input_channels

    return registry[name](**params)


def build_augmentations(
    transform_configs: Optional[Iterable],
    *,
    input_channels: Optional[int] = None,
) -> Optional[nn.Module]:
    if not transform_configs:
        return None

    modules: List[nn.Module] = []
    for spec in transform_configs:
        if isinstance(spec, str):
            name = spec
            params = {}
        elif isinstance(spec, dict):
            name = spec.get("name")
            params = spec.get("params", {})
            if name is None:
                raise ValueError(f"Transform specification missing 'name': {spec}")
        else:
            raise TypeError(f"Unsupported transform specification type: {type(spec)!r}")

        modules.append(build_augmentation(name, params, input_channels=input_channels))

    if not modules:
        return None

    if len(modules) == 1:
        return modules[0]

    return Compose(modules)


__all__ = [
    "AdditiveNoise",
    "ChannelDropout",
    "Compose",
    "GaussianSmoothing",
    "HighGammaPower",
    "MultiBandHilbertEnvelope",
    "MeanDriftNoise",
    "TimeWarp",
    "WhiteNoise",
    "build_augmentation",
    "build_augmentations",
]
