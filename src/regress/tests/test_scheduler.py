from factory.train import MultiStepCosineAnneal
from factory.optim import AdamW 

import torch
from torch.autograd import Variable

INIT_LR = 1e-3
FINAL_LR = 1e-8
GAMMA = 0.1
STEPS_PER_EPOCH = 1000
NUM_EPOCHS = 100
MILESTONES = [10, 30, 95]

optimizer = AdamW(iter(Variable(torch.Tensor(1))), INIT_LR)
scheduler = MultiStepCosineAnneal(
                optimizer, 
                INIT_LR,
                MILESTONES, 
                NUM_EPOCHS, 
                STEPS_PER_EPOCH,
                gamma=GAMMA,
                final_lr=FINAL_LR
            )

lrs = []
for e in range(NUM_EPOCHS):
    for s in range(STEPS_PER_EPOCH):
        #lrs.append(optimizer.param_groups[0]['lr'])
        lrs.append(scheduler.get_lr())
        optimizer.step()
        scheduler.step()

import matplotlib.pyplot as plt

plt.plot(range(NUM_EPOCHS*STEPS_PER_EPOCH), lrs)
plt.yscale('log')
#plt.ylim([INIT_LR/1000., INIT_LR+INIT_LR*0.05])
plt.show()