import numpy as np

class Sample(Callback):

    def __init__(self, num_samples = 6):
        super().__init__()
        self.num_samples = num_samples

    def on_epoch_end(self, trainer, pl_module):
        # Randomly pick a sample from validation dataset to encoder
        data, *_ = pl_module.valid_dataset[np.random.choice(len(pl_module.valid_dataset))]
        # Add batch dimension and move it to device
        data = data.unsqueeze(1).to(pl_module.device)

        # Sample
        pl_module.sampler.sample(data, pl_module.temperature, self.num_samples)
