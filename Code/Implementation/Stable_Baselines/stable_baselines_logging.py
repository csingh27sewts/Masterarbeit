import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.logger import Image
# from PIL import Image
import matplotlib.pyplot as plt
import os

class FigureRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(FigureRecorderCallback, self).__init__(verbose)

    def _on_step(self):
        # Plot values (here a random variable)
        figure = plt.figure()
        figure.add_subplot().plot(np.random.random(3))
        # Close the figure after logging it
        self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
        plt.close()
        return True

class ImageRecorderCallback(BaseCallback):
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(ImageRecorderCallback, self).__init__(verbose)

    def _on_step(self):
        image = self.training_env.render(mode="rgb_array")
        # "HWC" specify the dataformat of the image, here channel last
        # (H for height, W for width, C for channel)
        # See https://pytorch.org/docs/stable/tensorboard.html
        # for supported formats
        # plt.imshow(image)
        # plt.show(block=False)
        # plt.pause(0.2)
        # plt.close()
        #plt.savefig('ok.png')
        StopTrainingOnRewardThreshold(reward_threshold=490, verbose=1)
        self.logger.record("trajectory/image", Image(image, 'HWC'), exclude=("stdout", "log", "json", "csv"))
        return True



