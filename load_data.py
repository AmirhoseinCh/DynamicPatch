import fnmatch
import math
import random
import os
import sys
import time
from operator import itemgetter
import cv2
import gc
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from torchvision.io import read_image
import torchvision.ops as ops
from torchvision.transforms import functional as TF
import patch_config as patch_config


# Define Screen Network class
class Network(nn.Module):
    def __init__(self, n_neurons_l1):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, n_neurons_l1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(n_neurons_l1, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.tanh(x)
        return x

class ComplexNetwork(nn.Module):
    def __init__(self, n_neurons_l1, n_neurons_l2):
        super(ComplexNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, n_neurons_l1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(n_neurons_l1, n_neurons_l2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(n_neurons_l2, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.tanh(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None, device='cuda'):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.device = device
        self.transform = transform

        # Get the list of image and label file names
        self.img_files = os.listdir(img_dir)
        self.label_files = os.listdir(label_dir)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Get the file names for the current index
        img_filename = self.img_files[idx]
        label_filename = self.img_files[idx][:-3]+"txt"

        # Load the image
        img_path = os.path.join(self.img_dir, img_filename)
        image = read_image(img_path).to(self.device).float() / 255   # Load the image using torchvision.io.read_image()
        # Apply the transform if specified
        if self.transform:
            image = self.transform(image)

        # Load the labels
        label_path = os.path.join(self.label_dir, label_filename)
        side = [(0, 0), (0, 0), (0, 0), (0, 0)]
        front = [(0, 0), (0, 0), (0, 0), (0, 0)]
        car = [(0, 0), (0, 0), (0, 0), (0, 0)]


        if os.path.exists(label_path):
            with open(label_path, "r") as file:
                for line in file:
                    points = line.strip().split()
                    if len(points) != 5:
                        continue  # Skip lines that don't have the expected number of points

                    label = int(points[0])
                    coordinates = [tuple(map(int, point.split(","))) for point in points[1:]]

                    if label == 0:
                        side = self.order_four_points(coordinates)
                        side = torch.tensor(side, dtype=torch.float32, device=self.device)
                    elif label == 1:
                        front = self.order_four_points(coordinates)
                        front = torch.tensor(front, dtype=torch.float32, device=self.device)
                    elif label == 2:
                        car = self.order_four_points(coordinates)
                        car = torch.tensor(car, dtype=torch.float32, device=self.device)
                



        return image, side, front, car

    def order_four_points(self,points):
        points.sort(key=lambda p: p[1])

        Tops = points[:2]
        Tops.sort(key=lambda p: p[0])
        Bottoms = points[2:]
        Bottoms.sort(key=lambda p: -p[0])


        return Tops+Bottoms
    

    def collate_fn(self, batch):
        images, sides, fronts, cars = zip(*batch)

        # Pad sequences to the maximum size
        padded_images = pad_sequence(images, batch_first=True)  # Pad images
        padded_sides = pad_sequence([torch.tensor(side, dtype=torch.float32) for side in sides], batch_first=True)  # Pad sides
        padded_fronts = pad_sequence([torch.tensor(front, dtype=torch.float32) for front in fronts], batch_first=True)  # Pad fronts
        padded_cars = pad_sequence([torch.tensor(car, dtype=torch.float32) for car in cars], batch_first=True)

        return padded_images, padded_sides, padded_fronts, padded_cars

class MaxProbExtractor(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls):
        super(MaxProbExtractor, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.indices = None
       

    def forward(self, predictions, coordinate_batch, car_batch, shape):
   
        goind = patch_config.BaseConfig().goind
        stpind = patch_config.BaseConfig().stopind  
        max_conf_values = []
        cls_conf_stop = []
        #cls_conf_car = []
        cls_conf_go = []
        obj_conf = []
        indices = []

        for xi, x in enumerate(predictions):
            
        
            x1, Go_box, _ =  self.filtered2(coordinate_batch[xi],x,shape,c=0.15)

            placeholder = x1.clone()
            
            placeholder[:, 5:] = x1[:, 5:] * x1[:, 4:5] 

            max_conf_obj , _ = torch.max(placeholder[:, 4], dim=0)


            if placeholder[:,goind].size(0) >= 3:
                top_values2, _ = torch.topk(placeholder[:, goind], 3)
            else:
                top_values2 = torch.tensor([0], dtype=torch.float32, device='cuda')

            max_conf_go = torch.mean(top_values2)

    
            if placeholder[:,stpind].size(0) >= 3:
                top_values3, indices3 = torch.topk(placeholder[:, stpind], 3)
            elif placeholder[:,stpind].size(0) >= 1 and placeholder[:,stpind].size(0) < 3:
                top_value3, index3  = torch.max(placeholder[:, stpind], dim=0)
                top_values3 = top_value3.unsqueeze(0)
                indices3 = index3.unsqueeze(0)
            else:
                print("No S-sign")


            top_values3 = placeholder[indices3, stpind]
        
            max_conf_stop = torch.mean(top_values3)
            

            Stop_box = ops.box_convert(placeholder[indices3,0:4],in_fmt='cxcywh',out_fmt='xyxy')
            #print("STOP BOX:",Stop_box)
            iout = ops.box_iou(Go_box.unsqueeze(0), Stop_box)*top_values3
            iou = iout.mean()
            

            max_conf_values.append(1-iou)
            obj_conf.append(max_conf_obj.mean())
            cls_conf_stop.append(max_conf_stop) 
            indices.append(indices3)
            

        max_conf_tensor = torch.stack(max_conf_values)  # Convert list to tensor
        obj_conf_tensor = torch.stack(obj_conf).mean()
        cls_conf_stop_tensor = torch.stack(cls_conf_stop).mean()


        
        return max_conf_tensor , obj_conf_tensor, cls_conf_stop_tensor, iou 
    

    def index_finder(self,predictions, coordinate_batch, shape):
        indices = []
        for xi, x in enumerate(predictions):
            _, Go_box, xc =  self.filtered2(coordinate_batch[xi],x,shape,c=0.15)
            placeholder = x.clone()
            placeholder[:, 5:] = x[:, 5:] * x[:, 4:5] 

            boxes = ops.box_convert(placeholder[xc,0:4],in_fmt='cxcywh',out_fmt='xyxy')

            iout = ops.box_iou(Go_box.unsqueeze(0), boxes)
            _, indices3 = torch.topk(iout, 2)
            true_indices = torch.nonzero(xc).squeeze()
            indices.append(true_indices[indices3])
        self.indices = torch.stack(indices)


    def filtered(self, side, x, shape):
        coordinates = side.tolist()
        cxL = (coordinates[0][0] + coordinates[1][0]) / 2 *640/shape[3]
        cxR = (coordinates[2][0] + coordinates[3][0]) / 2 *640/shape[3]
        cyU = (coordinates[0][1] + coordinates[3][1]) / 2 *640/shape[2]
        cyD = (coordinates[1][1] + coordinates[2][1]) / 2 *640/shape[2]
        w = cxR-cxL
        h = cyD-cyU
     
        filtered1 = (x[..., 0]-x[..., 2]/2 > cxL-1.5*w) & (x[..., 0]+x[..., 2]/2 < cxL-0.5*w) & (x[..., 1]-x[..., 3]/2 > cyU-1.5*h) & (x[..., 1]+x[..., 3]/2 < cyD+1.5*h)
        filtered2 = (x[..., 0]-x[..., 2]/2 > cxR+0.5*w) & (x[..., 0]+x[..., 2]/2 < cxR+1.5*w) & (x[..., 1]-x[..., 3]/2 > cyU-1.5*h) & (x[..., 1]+x[..., 3]/2 < cyD+1.5*h)
        filtered3 = (x[..., 0]-x[..., 2]/2 > cxL-0.5*w) & (x[..., 0]+x[..., 2]/2 < cxR+0.5*w) & (x[..., 1]-x[..., 3]/2 > cyU-1.5*h) & (x[..., 1]+x[..., 3]/2 < cyU-0.5*h) 
        filtered4 = (x[..., 0]-x[..., 2]/2 > cxL-0.5*w) & (x[..., 0]+x[..., 2]/2 < cxR+0.5*w) & (x[..., 1]-x[..., 3]/2 > cyD+0.5*h) & (x[..., 1]+x[..., 3]/2 < cyD+1.5*h)
        xc = filtered1 | filtered2 | filtered3 | filtered4 
        if not torch.any(xc):
            xc= torch.zeros((x.shape[0]),dtype=bool)
            xc[16040] = True

        x = x[xc]
        return x
    
    def filtered2(self,front, x, shape,c):
        coordinates = front.tolist()
        cxL = coordinates[0][0] *640/shape[3]
        cxR = coordinates[2][0] *640/shape[3] 
        cyU = coordinates[0][1] *640/shape[2]
        cyD = coordinates[2][1] *640/shape[2]
        Go_box = torch.tensor([cxL,cyU,cxR,cyD],device='cuda')
        w = cxR-cxL
        h = cyD-cyU
        cxL = cxL-c*w
        cxR = cxR+c*w
        cyU = cyU-c*h
        cyD = cyD+c*h
        
        xc = (x[..., 0]-x[..., 2]/2 > cxL) & (x[..., 0]+x[..., 2]/2 < cxR) & (x[..., 1]-x[..., 3]/2 > cyU) & (x[..., 1]+x[..., 3]/2 < cyD) 

        if not torch.any(xc):
            xc= torch.zeros((x.shape[0]),dtype=bool)
            xc[16040] = True

        x = x[xc]
        
        return x, Go_box, xc
    
    def filtered3(self,car, x, shape,c=0.1):
        coordinates = car.tolist()
        cxL = coordinates[0][0] *640/shape[3]
        cxR = coordinates[2][0] *640/shape[3] 
        cyU = coordinates[0][1] *640/shape[2]
        cyD = coordinates[2][1] *640/shape[2]
        box1 = torch.tensor([cxL,cyU,cxR,cyD],device='cuda')
        w = cxR-cxL
        h = cyD-cyU
        cxL = cxL-c*w
        cxR = cxR+c*w
        cyU = cyU-c*h
        cyD = cyD+c*h
        
        xc = (x[..., 0]-x[..., 2]/2 > cxL) & (x[..., 0]+x[..., 2]/2 < cxR) & (x[..., 1]-x[..., 3]/2 > cyU) & (x[..., 1]+x[..., 3]/2 < cyD) 
        if not torch.any(xc):
            xc= torch.zeros((x.shape[0]),dtype=bool)
            xc[16040] = True

        x = x[xc]
        holder = x.clone()
        holder[:, 5:] = x[:, 5:] * x[:, 4:5]
        max_conf , max_index = torch.max(holder[:, 5], dim=0)
        box2 = ops.box_convert(x[max_index,0:4],in_fmt='cxcywh',out_fmt='xyxy')
        iou = ops.box_iou(box1.unsqueeze(0), box2.unsqueeze(0))

        return iou[0]-torch.abs(max_conf-0.9)
    
    def filtered4(self,car, x, shape,c=0.1):
        coordinates = car.tolist()
        cxL = coordinates[0][0] *640/shape[3]
        cxR = coordinates[2][0] *640/shape[3] 
        cyU = coordinates[0][1] *640/shape[2]
        cyD = coordinates[2][1] *640/shape[2]
        box1 = torch.tensor([cxL,cyU,cxR,cyD],device='cuda')
        w = cxR-cxL
        h = cyD-cyU
        cxL = cxL-c*w
        cxR = cxR+c*w
        cyU = cyU-c*h
        cyD = cyD+c*h
        
        xc = (x[..., 0] > cxL) & (x[..., 0] < cxR) & (x[..., 1] > cyU) & (x[..., 1] < cyD) 
        if not torch.any(xc):
            xc= torch.zeros((x.shape[0]),dtype=bool)
            xc[16040] = True
            print("All False")
            print(w,h)
        x = x[xc]
        holder = x.clone()
        holder[:, 5:] = x[:, 5:] * x[:, 4:5]
        max_conf , max_index = torch.max(holder[:, 5], dim=0)
        box2 = ops.box_convert(x[max_index,0:4],in_fmt='cxcywh',out_fmt='xyxy')
        iou = ops.box_iou(box1.unsqueeze(0), box2.unsqueeze(0))

        return iou[0]-torch.abs(max_conf-0.9)
  

class TotalVariation(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TotalVariation, self).__init__()

    def forward(self, adv_patch):
        # bereken de total variation van de adv_patch
        tvcomp1 = torch.sum(torch.abs(adv_patch[:, :, 1:] - adv_patch[:, :, :-1]+0.000001),0)
        tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
        tvcomp2 = torch.sum(torch.abs(adv_patch[:, 1:, :] - adv_patch[:, :-1, :]+0.000001),0)
        tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
        tv = tvcomp1 + tvcomp2
        return tv/torch.numel(adv_patch)

class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformer, self).__init__()
        self.noise_factor = 0.10
        self.screen_model = torch.load(patch_config.BaseConfig().screen)
        self.screen_model.eval()  # Set the model to evaluation mode
        


    
    def forward(self, adv_patch, btch_size):
        
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        adv_batch = adv_patch.expand(btch_size, -1, -1, -1)

        # Apply Screen model
        screen_batch = self.screen_model(adv_batch) 
        advsc_batch = torch.clamp(screen_batch, 0.000001, 0.9999)


        return advsc_batch

class PatchApplier(nn.Module):
    def __init__(self):
        super(PatchApplier, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.min_contrast = 0.85
        self.max_contrast = 1.15
        self.min_brightness = -0.05
        self.max_brightness = 0.05
        self.noise_factor = 0.10

    def forward(self, img_batch, adv_batch, side_batch, front_batch, expand=1, test=False):
        if img_batch.max() > 5:
            imgs = img_batch[:, :3, :, :]/255
        else:
            imgs = img_batch[:, :3, :, :]

        ptchd_image_batch = self.patch_applier(imgs, adv_batch, side_batch, front_batch,expand,imgs.shape, test)

        batch_size = ptchd_image_batch.shape[0]
        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_(self.min_contrast, self.max_contrast)

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_(self.min_brightness, self.max_brightness)
     
        
        ptchd_image_batch_n = ptchd_image_batch * contrast + brightness 
        ptchd_image_batch_c = torch.clamp(ptchd_image_batch_n, 0.000001, 0.9999)


        return ptchd_image_batch_c
    
    def crop_images(self, batch_of_tensors, top, left, height, width):
        cropped_images = []
        for image_tensor in batch_of_tensors:
            cropped_image = TF.crop(image_tensor, top, left, height, width)
            resized_image = TF.resize(cropped_image, (640,640))
            cropped_images.append(resized_image)
        return torch.stack(cropped_images)

    def patch_applier(self, images, patches, coordinates, go_coorbatch,expand,org_shape, test):
        
        RRot=False #To randomly rotate patches
        images = F.interpolate(images, (640, 640))
        batch_size = images.shape[0]
        h = patches.shape[2]
        w = patches.shape[3]
        resized_patch = F.pad(patches[0], (0, images.shape[2]-w, 0, images.shape[3]-h))
        start = [[0, 0],[w , 0], [w , h ],[0, h ]]
        
        transformed_patches = []
        masks = []

        
        for i in range(batch_size):
            if int(torch.mean(coordinates[i])) == 0:
                coordinates_i = [[0, 0], [1, 0], [1, 1], [0, 1]]
                perspective_end = coordinates_i
                perspective_start = start

            else:
                
                # Randomly apply perspective transformation with some randomness
                perspective_prob = 0.1  # Adjust the probability as desired
                if random.random() < perspective_prob and RRot:
                    coordinates_i = self.expand_polygon(coordinates[i].tolist(),expand+random.uniform(-0.05, 0.05),org_shape)
                    perspective_randomness = 5  # Adjust the randomness as desired
                    perspective_start = [[random.uniform(-perspective_randomness, perspective_randomness) + x, random.uniform(-perspective_randomness, perspective_randomness) + y] for x, y in start]
                    perspective_end = [[random.uniform(-perspective_randomness, perspective_randomness) + x, random.uniform(-perspective_randomness, perspective_randomness) + y] for x, y in coordinates_i]
                else:
                    coordinates_i = self.expand_polygon(coordinates[i].tolist(),expand,org_shape)
                    perspective_start = start
                    perspective_end = coordinates_i

            # Calculate the middle point of perspective_end
            middle_point = [(perspective_end[0][0] + perspective_end[1][0]) / 2, (perspective_end[1][1] + perspective_end[2][1]) / 2]
            
            # Perform perspective transform on the patch image
            perspective_transform = transforms.functional.perspective
            transformed_patch = perspective_transform(resized_patch, startpoints=perspective_start, endpoints=perspective_end)

            
            # Create a mask for the transformed patch
            mask = torch.ones_like(transformed_patch, device=self.device)
            mask = perspective_transform(mask, startpoints=[[0, 0], [mask.shape[2] , 0], [mask.shape[2] , mask.shape[1] ], [0, mask.shape[1] ]], endpoints=perspective_end)
            
            # Randomly apply rotation to the patch and mask
            rotation_prob = 0.1  # Adjust the probability as desired
            if random.random() < rotation_prob and RRot:
                angle = random.uniform(-10, 10)  # Adjust the angle range as desired
                transformed_patch = TF.rotate(transformed_patch, angle, center=middle_point)
                mask = TF.rotate(mask, angle, center=middle_point)

            transformed_patches.append(transformed_patch)
            masks.append(mask)

        transformed_patches = torch.stack(transformed_patches)
        masks = torch.stack(masks)

        
        # Overlay the transformed patches on the source images using the masks

        if test:
            print('mask:',masks.shape)
            print('images:',images.shape)
            pp = images * masks
            pp[:,0,:,:] = pp[:,0,:,:]*1.5 
            images = images * (1 - masks) + pp
            outputs = torch.clamp(images, min=0, max=1)
        else:
            outputs = images * (1 - masks) +  transformed_patches 

        
        return outputs
    
    def expand_polygon(self,coordinates, expansion_distance,shape):
        # Calculate centroid
        cx = (coordinates[0][0] + coordinates[1][0] + coordinates[2][0] + coordinates[3][0]) / 4
        cy = (coordinates[0][1] + coordinates[1][1] + coordinates[2][1] + coordinates[3][1]) / 4

        expanded_coordinates = []
        for point in coordinates:
            dx = point[0] - cx
            dy = point[1] - cy
            new_x = (cx + dx * expansion_distance)*640/shape[3]
            new_y = (cy + dy * expansion_distance)*640/shape[2]
            expanded_coordinates.append([new_x, new_y])

        expanded_coordinates = self.order_four_points(expanded_coordinates)


        return expanded_coordinates
    
    def order_four_points(self,points):
        points.sort(key=lambda p: p[1])

        Tops = points[:2]
        Tops.sort(key=lambda p: p[0])
        Bottoms = points[2:]
        Bottoms.sort(key=lambda p: -p[0])

        return Tops+Bottoms




if __name__ == '__main__':
    if len(sys.argv) == 3:
        img_dir = sys.argv[1]
        lab_dir = sys.argv[2]

    else:
        print('Usage: ')
        print('  python load_data.py img_dir lab_dir')
        sys.exit()
