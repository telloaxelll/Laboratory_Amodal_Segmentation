import torch
from torch import nn
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.utils import make_grid
from torchvision.io import write_video

from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from IPython.display import Video

tensor_to_image = ToPILImage()
image_to_tensor = ToTensor()

def get_img_dict(img_dir):
    """Organizes image files in a directory into a dictionary by type (e.g., 'rgba', 'segmentation').
    
    Args:
        img_dir (Path): Path to the directory containing image files.
    
    Returns:
        dict: Dictionary with keys as image types and values as lists of file paths.
    """
    img_files = [x for x in img_dir.iterdir() if x.name.endswith('.png') or x.name.endswith('.tiff')]
    img_files.sort()

    img_dict = {}
    for img_file in img_files:
        img_type = img_file.name.split('_')[0]
        if img_type not in img_dict:
            img_dict[img_type] = []
        img_dict[img_type].append(img_file)
    return img_dict

def get_sample_dict(sample_dir):
    """Creates a dictionary of camera directories and their organized image files for a sample.
    
    Args:
        sample_dir (str or Path): Path to the sample directory containing camera folders.
    
    Returns:
        dict: Dictionary with camera names as keys and image dictionaries as values.
    """
    sample_dir = Path(sample_dir)
    cam_dirs = [x for x in sample_dir.iterdir() if x.is_dir() and 'camera' in x.name]
    cam_dirs.sort()

    sample_dict = {}
    for cam_dir in cam_dirs:
        cam_name = cam_dir.name
        sample_dict[cam_name] = get_img_dict(cam_dir)
    return sample_dict

def make_obj_viz(cam_dict, cam_num=0):
    """Creates visualization grids for object modal and amodal data from a camera dictionary.
    
    Args:
        cam_dict (dict): Dictionary containing image data for a camera.
        cam_num (int): Camera number (default 0).
    
    Returns:
        list: List of grid tensors for visualization.
    """
    n_frames = 24
    n_cols = 6

    all_obj_ids = [x for x in sample_dict['camera_0000'].keys() if 'obj_' in x]
    obj_id_str = random.sample(all_obj_ids, k=1)[0]
    obj_id_int = int(obj_id_str.split('_')[1])

    grid_tensors = []
    for i in range(n_frames):
        grid = []

        # Modal RGB (square 1)
        scene_rgb_tensor = image_to_tensor(Image.open(cam_dict['scene']['rgba'][i]).convert('RGB'))
        grid.append(scene_rgb_tensor)


        # Modal segmentation (square 2)
        scene_masks_tensor = image_to_tensor(Image.open(cam_dict['scene']['segmentation'][i]).convert('RGB'))
        grid.append(scene_masks_tensor)


        # Modal mask (square 3)
        scene_masks_p = Image.open(cam_dict['scene']['segmentation'][i])
        scene_masks_p_tensor = torch.tensor(np.array(scene_masks_p))

        obj_modal_tensor = (scene_masks_p_tensor==obj_id_int)
        blended_obj_modal_tensor = scene_masks_tensor*obj_modal_tensor
        grid.append(blended_obj_modal_tensor)


        # Amodal mask (square 4)
        obj_amodal_tensor = image_to_tensor(Image.open(cam_dict[obj_id_str]['segmentation'][i]).convert('RGB'))
        blended_obj_amodal_tensor = blended_obj_modal_tensor + (obj_amodal_tensor != obj_modal_tensor)
        grid.append(blended_obj_amodal_tensor)


        # Amodal RGB (square 5)
        obj_rgb_tensor = image_to_tensor(Image.open(cam_dict[obj_id_str]['rgba'][i]).convert('RGB'))
        grid.append(obj_rgb_tensor)


        # Scene + amodal (square 6)
        blended_scene_obj_tensor = (scene_rgb_tensor/3 + 2*blended_obj_amodal_tensor/3)
        grid.append(blended_scene_obj_tensor)


        grid_tensors.append(make_grid(grid, nrow=n_cols, padding=2, pad_value=127))

    return grid_tensors

def make_vid(grid_tensors, save_path):
    """Creates a video from a list of grid tensors and saves it to a file.
    
    Args:
        grid_tensors (list): List of tensors representing frames.
        save_path (str): Path to save the video file.
    """
    vid_tensor = torch.stack(grid_tensors, dim=1).permute(1, 2, 3, 0)
    vid_tensor = (vid_tensor*255).long()
    write_video(save_path, vid_tensor, fps=5, options={'crf':'20'})

# Model class
class SimpleConv2DModel(nn.Module):
    """A simple convolutional neural network for binary segmentation tasks.
    
    This model consists of convolutional layers with ReLU activations and a final convolution
    for outputting binary masks.
    """
    def __init__(self):
        """Initializes the model layers."""
        super(SimpleConv2DModel, self).__init__()
        # Define multiple Conv2D layers with 'same' padding
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.final_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        """Defines the forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, height, width).
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, 1, height, width).
        """
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.final_conv(x)
        return x
