from torch import optim
import os

class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.cluster = '0'
        self.base_dir = "./Data"
        self.img_dir = os.path.join(self.base_dir,f'{self.cluster}_train')
        self.lab_dir = os.path.join(self.base_dir,'coords')
        self.modelfile = "./Data/yolos_weight.pt" #yolo path
        self.stopind = 8 # stop sign index
        self.goind = 9 # sOriginal sign index
        self.screen = os.path.join(self.base_dir, 'screen_model.pt')


        self.start_learning_rate = 0.1

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', factor=0.7,patience=100)#patience=50
        self.max_tv = 0.165

        self.batch_size = 16



class ReproducePaperObj(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size =32
        self.patch_size_H = 300
        self.patch_size_W = 512
        self.patch_expand = 1.0
        self.patch_path = os.path.join(self.base_dir,f'{self.cluster}.pt')
        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj


patch_configs = {
    "base": BaseConfig,
    "paper_obj": ReproducePaperObj
}
