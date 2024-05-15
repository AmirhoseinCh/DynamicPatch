"""
Training code for Adversarial patch training


"""
# Reference for the used methodology:
# Thys, S., Van Ranst, W., & Goedemé, T. (2019). Fooling automated surveillance cameras: adversarial patches to attack person detection. In CVPRW: Workshop on The Bright and Dark Sides of Computer Vision: Challenges and Opportunities for Privacy and Security.

import PIL

import load_data
from tqdm import tqdm
import torch
from load_data import *
import gc
from torch import autograd
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess
from torchvision.transforms import ToPILImage
#import torch.autograd.profiler as profiler
import patch_config
import sys
import time
import sys
sys.path.insert(0,'/path/to/yolov5 dir')

def custom_collate(batch):
    images = []
    labels = []
    for img, lab in batch:
        img_array = np.array(img)
        images.append(torch.tensor(img_array, dtype=torch.float32))
        labels.append(lab)
    return images, labels

class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config.patch_configs[mode]()
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.config.modelfile, force_reload=True, _verbose=False, autoshape = False)
        is_grad_enabled = any(param.requires_grad for param in self.model.parameters())
        if is_grad_enabled:
            print("Gradient mode is enabled.")
        else:
            print("Gradient mode is disabled.")
            for param in self.model.parameters():
                param.requires_grad_(True)
            print("Gradient mode is enabled Now!")

        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = MaxProbExtractor(cls_id=1, num_cls=3).cuda()
        self.total_variation = TotalVariation().cuda()

        self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'new_runs/{time_str}_{name}')
        else:
            return SummaryWriter()

    def train(self):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """

        img_size = 640
        batch_size = self.config.batch_size
        print("batch Size: ",batch_size)
        n_epochs = 5000

        expand = 1.0
        max_lab = 14
        gradp = torch.zeros(1) 

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch = self.generate_patch('random')

        adv_patch.requires_grad_(True)

        # Create the custom dataset
        dataset = CustomDataset(self.config.img_dir, self.config.lab_dir)
        # Create the DataLoader
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)


        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        optimizer = optim.Adam([adv_patch], lr=self.config.start_learning_rate, amsgrad=True)
        scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        
        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            bt0 = time.time()
            for i_batch, (img_batch, side_batch, front_batch, car_batch) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
                                                        total=self.epoch_length):

                
                adv_batch_t = self.patch_transformer(adv_patch, batch_size)
                p_img_batch = self.patch_applier(img_batch, adv_batch_t, side_batch,car_batch, expand)
                output = self.model(p_img_batch)
                max_prob,objconf, clsconf_stop, iou = self.prob_extractor(output[0],front_batch,car_batch,img_batch.shape)
                tv = self.total_variation(adv_patch)
                tv_loss = tv
                det_loss = torch.mean(max_prob)
                loss = torch.max(det_loss, torch.tensor(0.05).cuda()) + 0.2*torch.max(tv_loss, torch.tensor(0.07).cuda())
                ep_loss += loss
            

                loss.backward()
                
                gradp = adv_patch.grad.mean()
                
                optimizer.step()
                optimizer.zero_grad()
                adv_patch.data.clamp_(0,0.9999)       #keep patch in image range

                bt1 = time.time()
            
                if i_batch%5 == 0:
                    iteration = self.epoch_length * epoch + i_batch

                    self.writer.add_scalar('loss/total_loss', loss, iteration)
                    self.writer.add_scalar('loss/det_loss', det_loss, iteration)
                    self.writer.add_scalar('loss/iou_loss', iou, iteration)
                    self.writer.add_scalar('loss/tv_loss', tv_loss, iteration)
                    self.writer.add_scalar('loss/obj_conf', objconf, iteration)
                    self.writer.add_scalar('loss/cls_conf_stop', clsconf_stop, iteration)
                    self.writer.add_scalar('Patch gradiant', gradp, iteration)
                    self.writer.add_scalar('misc/epoch', epoch, iteration)
                    self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)

                    self.writer.add_image('patch', adv_patch, iteration)
            
                if i_batch + 1 >= len(train_loader):
                    print('\n')
                else:

                    del adv_batch_t, output, max_prob, det_loss, p_img_batch, loss,tv_loss 
                    gc.collect()
                    torch.cuda.empty_cache()

                

            et1 = time.time()
            ep_loss = ep_loss/len(train_loader)

          

            scheduler.step(ep_loss)
            if True:
                print('  EPOCH NR: ', epoch),
                print('EPOCH LOSS: ', ep_loss.item())
                print('EPOCH DET-LOSS: ', det_loss.item())
                print('EPOCH TV-LOSS: ', tv_loss.item())
                print('EPOCH TIME: ', et1-et0)

                del adv_batch_t, output, max_prob, det_loss, p_img_batch,  loss ,tv_loss
                gc.collect()
                torch.cuda.empty_cache()
            et0 = time.time()
            
        torch.save(adv_patch.detach().cpu(), self.config.patch_path)
        print("Done")


    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch = torch.rand((3, self.config.patch_size_H, self.config.patch_size_W), device='cuda')
        return adv_patch

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """

        adv_patch_cpu = torch.load(path).to('cuda')
        return adv_patch_cpu


def main():

    

    trainer = PatchTrainer('paper_obj')
    trainer.train()

if __name__ == '__main__':
    main()


