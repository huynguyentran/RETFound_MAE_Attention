import torch
import torch
import models_vit
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import torchvision.transforms as transforms
import skimage.io 
import skimage.segmentation
import numpy as np
from PIL import Image

# import PIL.Image
from matplotlib import pylab as P
import os
import saliency.core as saliency 
import torch.nn.functional as F
import keras
import tensorflow as tf
from keras.applications.imagenet_utils import decode_predictions
import skimage.io 
import skimage.segmentation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
import copy
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode

# call the model
model = models_vit.__dict__['vit_large_patch16'](
    num_classes=2,
    drop_path_rate=0.2,
    global_pool=True,
)


# Boilerplate methods.
def ShowImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')
  P.imshow(im)
  P.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')

  P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
  P.title(title)

def ShowHeatMap(im, title, ax=None):
  if ax is None:
    P.figure()
  P.axis('off')
  P.imshow(im, cmap='inferno')
  P.title(title)

def LoadImage(file_path):
  im = Image.open(file_path)
  im = im.resize((224,224))
  im = np.asarray(im)
  return im

def PreprocessImage(im):
  im = tf.keras.applications.vgg16.preprocess_input(im)
  return im


weight_path = '/content/drive/MyDrive/huyn/LACDHS_task_unfreeze/Taskcheckpoint-best.pth'
# load RETFound weights
checkpoint = torch.load(weight_path, map_location='cpu')
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# interpolate position embedding
interpolate_pos_embed(model, checkpoint_model)

# load pre-trained model
msg = model.load_state_dict(checkpoint_model, strict=False)

# assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

# manually initialize fc layer
trunc_normal_(model.head.weight, std=2e-5)

# print("Model = %s" % str(model))


transform_test = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BILINEAR),  # Resize the shorter side to the input size
    transforms.CenterCrop(224),  # Center crop to the input size
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with the ImageNet statistics
])

np.random.seed(42)
# image_dir = '/content/drive/MyDrive/huyn/retfound/Images'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()  # Set the model to evaluation mode

save_dir = '/content/drive/MyDrive/huyn/LACDHS_task_unfreeze_XRAI/task/mae'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
images_dir = '/content/drive/MyDrive/huyn/LACDHS_task_unfreeze_XRAI/images'

class_idx_str = 'class_idx_str'
def call_model_function(images, call_model_args=None, expected_keys=None):
    target_class_idx = call_model_args[class_idx_str]
    
    # Check if images is a NumPy array and convert to tensor
    if isinstance(images, np.ndarray):
        images = torch.tensor(images).float()  # Convert to tensor and ensure it's float type

    # Move images to the appropriate device
    images = images.squeeze(1).to(device)  # Ensure images are on the right device
    images.requires_grad_()  # Enable gradient tracking
    print("Input tensor shape:", images.shape)  # Check the shape of the input tensor
    with torch.autograd.set_detect_anomaly(True):
        output_layer = model(images)  # Forward pass
        output_layer = output_layer[:, target_class_idx]  # Select the target class

        # Get gradients
        gradients = torch.autograd.grad(outputs=output_layer, inputs=images, retain_graph=True)[0]
        gradients = gradients.unsqueeze(1)
    # Ensure gradients are on CPU and convert to NumPy
    return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients.cpu().numpy()}

   

  #  input_tensor.requires_grad_()  # Enable gradient tracking
  #   with torch.autograd.set_detect_anomaly(True):
  #       output_layer = model(input_tensor)  # Forward pass
  #       output_layer = output_layer[:, target_class_idx]  # Select the target class output

  #       # Compute gradients
  #       gradients = torch.autograd.grad(outputs=output_layer, inputs=input_tensor)[0]

  #   return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}

image_path = '/content/drive/MyDrive/huyn/LACDHS_task_unfreeze_XRAI/images/Copy of sq1000038_1098492_11801509_Right Field 2.jpg'

image = Image.open(image_path).convert('RGB')
input_tensor = transform_test(image)
input_tensor = input_tensor.unsqueeze(0).to(device) 

with torch.no_grad():
    output = model(input_tensor)
    probabilities = F.softmax(output, dim=1)
    _, predicted_class = torch.max(probabilities, 1)
    predicted_class = predicted_class.cpu().item()

call_model_args = {class_idx_str: predicted_class}

# Construct the saliency object
gradient_saliency = saliency.GradientSaliency()

# Compute vanilla and smoothed masks (move tensors to CPU for saliency calculations)
vanilla_mask_3d = gradient_saliency.GetMask(input_tensor.cpu(), call_model_function, call_model_args)
# smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(input_tensor.cpu(), call_model_function, call_model_args)


vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
# smoothgrad_mask_grayscale = gradient_saliency.GetSmoothedMask(input_tensor.cpu().numpy(), call_model_function, call_model_args)


print("Shape of vanilla_mask_grayscale:", vanilla_mask_grayscale.shape)

# Ensure it's 2D before saving
if len(vanilla_mask_grayscale.shape) == 3:  # If it has a color dimension
    vanilla_mask_grayscale = vanilla_mask_grayscale[:, :, 0]  
    
plt.imsave(os.path.join(save_dir, 'vanilla_gradient.png'), vanilla_mask_grayscale, cmap='gray')
# plt.imsave(os.path.join(save_dir, 'smoothgrad.png'), smoothgrad_mask_grayscale, cmap='gray')

print("Saliency masks saved successfully.")