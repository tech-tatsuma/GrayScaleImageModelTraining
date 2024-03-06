import torch
import math
import warnings


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    An internal function to initialize a tensor with a truncated normal distribution
    while avoiding unnecessary gradient computations.
    """
    def norm_cdf(x):
        # Computes the cumulative distribution function (CDF) for a standard normal distribution
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    # Warns if the mean is more than 2 standard deviations away from the [a, b] interval
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Processing within a context where gradients are not computed

        # Fills the tensor with values from a truncated uniform distribution
        # and transforms them using the inverse CDF of the normal distribution
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Fills the tensor with values using a uniform distribution
        # and transforms them using the inverse function of the error function
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        # Applies standard deviation and mean to transform
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        # Clamps the values to ensure they are within the [a, b] range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    """
    Fills the input tensor with values drawn from a truncated normal distribution.
    Values are drawn from a normal distribution \mathcal{N}(mean, std^2) and
    values outside the [a, b] range are redrawn until they fall within this interval.
    This method of generating random values works best when a <= mean <= b.

    Arguments:
        tensor: A n-dimensional `torch.Tensor`
        mean: The mean of the normal distribution
        std: The standard deviation of the normal distribution
        a: The minimum truncation threshold
        b: The maximum truncation threshold

    Example:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    # Calls the internal function to initialize the tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
