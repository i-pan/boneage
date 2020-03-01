from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    MultiStepLR,
    OneCycleLR,
    CyclicLR
)

from .onecycle import CustomOneCycleLR
from .multistepcosine import MultiStepCosineAnneal