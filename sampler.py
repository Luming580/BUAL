"""Custom samplers used by active-learning acquisition."""

from typing import Iterator, Sequence

import torch


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """Sample a fixed subset in deterministic order."""

    def __init__(self, indices: Sequence[int]):
        self.indices = list(indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
