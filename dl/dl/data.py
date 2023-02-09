import torch
import torch.utils.data


def parabola_1d(n_samples):
    """
    1d parabola on uniform grid points.

    """
    x = torch.linspace(-1, 1, 100).reshape((-1,1))
    y = x**2

    dataset = torch.utils.data.TensorDataset(x, y)
    dataset.data = x
    dataset.targets = y

    return dataset
