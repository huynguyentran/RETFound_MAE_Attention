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
import os
import torch.nn.functional as F
import keras
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


weight_path = '/content/drive/MyDrive/huyn/retfound/glaucoma_finetune_unfreeze/glaucomacheckpoint-best.pth'
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
image_dir = '/content/drive/MyDrive/huyn/retfound/Images'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()  # Set the model to evaluation mode


save_dir = '/content/drive/MyDrive/huyn/retfound/Images/Untitled Folder'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def predict(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)

    # Get predicted class
    _, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), probabilities.squeeze().cpu().numpy(), input_tensor.squeeze().cpu().numpy()

# Function to perturb the image
def perturb_image(img, perturbation, segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1 
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image * mask[:, :, np.newaxis]
    return perturbed_image

results = []
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)
    if img_path.lower().endswith('.jpg'):
        predicted_class, probabilities, original_image = predict(img_path, model, transform_test, device)
        print(f'Class: {predicted_class}')
        print(f'Probabilities: {probabilities.tolist()}')

        # Convert the image to double-precision float
        original_image = original_image.transpose(1, 2, 0).astype(np.float64)

        # Generate superpixels
        superpixels = skimage.segmentation.quickshift(original_image, kernel_size=4, max_dist=200, ratio=0.2)
        num_superpixels = np.unique(superpixels).shape[0]

        # Generate perturbations
        num_perturb = 150
        perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
        
        # Get predictions for perturbations
        predictions = []
        for pert in perturbations:
            perturbed_img = perturb_image(original_image, pert, superpixels)
            perturbed_img = torch.tensor(perturbed_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(perturbed_img)
            predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Calculate distances and weights
        original_superpixels = np.ones(num_superpixels)[np.newaxis, :]  # Perturbation with all superpixels enabled
        distances = pairwise_distances(perturbations, original_superpixels, metric='cosine').ravel()
        kernel_width = 0.25
        weights = np.sqrt(np.exp(-(distances**2) / kernel_width**2))  # Kernel function
        
        # Fit a simpler model
        simpler_model = LinearRegression()
        simpler_model.fit(X=perturbations, y=predictions[:, :, predicted_class], sample_weight=weights)
        coeff = simpler_model.coef_[0]
        
        # Identify top features
        num_top_features = 4
        top_features = np.argsort(coeff)[-num_top_features:]
        
        # Visualize the perturbed image
        mask = np.zeros(num_superpixels)
        mask[top_features] = True  # Activate top superpixels
        highlighted_image = perturb_image(original_image, mask, superpixels)
        
        # Ensure image is in the range [0, 1] for floats
        if highlighted_image.max() > 1:
            highlighted_image = highlighted_image / 255.0
        
        # Save the images
        skimage.io.imsave(os.path.join(save_dir, f'{img_name}_superpixels.png'),
                          skimage.segmentation.mark_boundaries(original_image / 2 + 0.5, superpixels))

        skimage.io.imsave(os.path.join(save_dir, f'{img_name}_highlighted.png'), highlighted_image)

        results.append({
            'image': img_name,
            'predicted_class': predicted_class,
            'probabilities': probabilities.tolist(),
            'highlighted_image': highlighted_image
        })
        print(f'Image: {img_name}')

# Print the results
for result in results:
    print(f"Image: {result['image']}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Probabilities: {result['probabilities']}\n")
    
#RESNET

# import torch

# import torchvision.transforms as transforms
# import skimage.io 
# import skimage.segmentation
# import numpy as np
# from PIL import Image
# import os
# import torch.nn.functional as F
# import keras
# from keras.applications.imagenet_utils import decode_predictions
# import skimage.io 
# import skimage.segmentation
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import pairwise_distances
# import copy
# import matplotlib.pyplot as plt
# from torchvision.transforms import InterpolationMode


# transform_test = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# def denormalize(image_tensor, mean, std):
#     mean = np.array(mean).reshape(1, 1, 3)
#     std = np.array(std).reshape(1, 1, 3)
#     denormalized_image = image_tensor * std + mean
#     return denormalized_image

# np.random.seed(42)
# image_dir = '/content/drive/MyDrive/Images'


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# model.eval()  # Set the model to evaluation mode


# save_dir = '/content/drive/MyDrive/huyn/resnet50/glaucoma/Images/freeze'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# def predict(image_path, model, transform, device):
#     image = Image.open(image_path).convert('RGB')
#     input_tensor = transform(image)
#     input_tensor = input_tensor.unsqueeze(0)  
#     input_tensor = input_tensor.to(device)
#     with torch.no_grad():
#         output = model(input_tensor)
#         probabilities = F.softmax(output, dim=1)
#     _, predicted_class = torch.max(probabilities, 1)
#     return predicted_class.item(), probabilities.squeeze().cpu().numpy(), input_tensor.squeeze().cpu().numpy()

# def perturb_image(img, perturbation, segments):
#     active_pixels = np.where(perturbation == 1)[0]
#     mask = np.zeros(segments.shape)
#     for active in active_pixels:
#         mask[segments == active] = 1 
#     perturbed_image = copy.deepcopy(img)
#     perturbed_image = perturbed_image * mask[:, :, np.newaxis]
#     return perturbed_image

# # Main processing loop
# results = []

# for img_name in os.listdir(image_dir):
#     img_path = os.path.join(image_dir, img_name)
#     if img_path.lower().endswith('.jpg'):
#         predicted_class, probabilities, original_image = predict(img_path, model, transform_test, device)

#         # Convert the image to float64 and transpose
#         original_image = original_image.transpose(1, 2, 0).astype(np.float64)

#         # Denormalize the image
#         mean = [0.485, 0.456, 0.406]
#         std = [0.229, 0.224, 0.225]
#         original_image = denormalize(original_image, mean, std)

#         # Clip to [0, 1] range
#         original_image = np.clip(original_image, 0, 1)

#         # Generate superpixels
#         superpixels = skimage.segmentation.quickshift(original_image, kernel_size=4, max_dist=200, ratio=0.2)
#         num_superpixels = np.unique(superpixels).shape[0]

#         # Generate perturbations
#         num_perturb = 150
#         perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
        
#         # Get predictions for perturbations
#         predictions = []
#         for pert in perturbations:
#             perturbed_img = perturb_image(original_image, pert, superpixels)
#             perturbed_img = torch.tensor(perturbed_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 pred = model(perturbed_img)
#             predictions.append(pred.cpu().numpy())
        
#         predictions = np.array(predictions)
        
#         # Calculate distances and weights
#         original_superpixels = np.ones(num_superpixels)[np.newaxis, :]  # Perturbation with all superpixels enabled
#         distances = pairwise_distances(perturbations, original_superpixels, metric='cosine').ravel()
#         kernel_width = 0.25
#         weights = np.sqrt(np.exp(-(distances**2) / kernel_width**2))  # Kernel function
        
#         # Fit a simpler model
#         simpler_model = LinearRegression()
#         simpler_model.fit(X=perturbations, y=predictions[:, :, predicted_class], sample_weight=weights)
#         coeff = simpler_model.coef_[0]
        
#         # Identify top features
#         num_top_features = 4
#         top_features = np.argsort(coeff)[-num_top_features:]
        
#         # Visualize the perturbed image
#         mask = np.zeros(num_superpixels, dtype=bool)
#         mask[top_features] = True  # Activate top superpixels
#         highlighted_image = perturb_image(original_image, mask, superpixels)
        
#         # Ensure highlighted image is in the correct range and type
#         highlighted_image = np.clip(highlighted_image, 0, 1)
#         highlighted_image = (highlighted_image * 255).astype(np.uint8)

#         # Convert original image to uint8 and ensure correct shape
#         original_image = (original_image * 255).astype(np.uint8)

#         # Save the images
#         skimage.io.imsave(os.path.join(save_dir, f'{img_name}_superpixels.png'),
#                   (skimage.segmentation.mark_boundaries(original_image, superpixels) * 255).astype(np.uint8))

#         skimage.io.imsave(os.path.join(save_dir, f'{img_name}_highlighted.png'), highlighted_image)

#         results.append({
#             'image': img_name,
#             'predicted_class': predicted_class,
#             'probabilities': probabilities.tolist(),
#             'highlighted_image': highlighted_image
#         })

# # Print the results
# for result in results:
#     print(f"Image: {result['image']}")
#     print(f"Predicted class: {result['predicted_class']}")
#     print(f"Probabilities: {result['probabilities']}\n")
    


# VGG19

# import torchvision.transforms as transforms
# import skimage.io
# import skimage.segmentation
# import numpy as np
# from PIL import Image
# import os
# import torch.nn.functional as F
# import keras
# from keras.applications.imagenet_utils import decode_predictions
# import skimage.io
# import skimage.segmentation
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import pairwise_distances
# import copy
# import matplotlib.pyplot as plt
# from torchvision.transforms import InterpolationMode

# transform_test = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# np.random.seed(42)
# image_dir = '/content/drive/MyDrive/Images'


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.eval()  


# save_dir = '/content/drive/MyDrive/huyn/vgg19/glaucoma/Images/unfreeze'
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# def predict(image_path, model, transform, device):
#     image = Image.open(image_path).convert('RGB')
#     input_tensor = transform(image)
#     input_tensor = input_tensor.unsqueeze(0)
#     input_tensor = input_tensor.to(device)

#     print(f"Image Path: {image_path}")
#     print(f"Transformed Image Tensor: {input_tensor.shape}, dtype: {input_tensor.dtype}, device: {input_tensor.device}")

#     with torch.no_grad():
#         output = model(input_tensor)
#         probabilities = F.softmax(output, dim=1)

#     _, predicted_class = torch.max(probabilities, 1)

#     return predicted_class.item(), probabilities.squeeze().cpu().numpy(), input_tensor.squeeze().cpu().numpy()

# def perturb_image(img, perturbation, segments):
#     active_pixels = np.where(perturbation == 1)[0]
#     mask = np.zeros(segments.shape)
#     for active in active_pixels:
#         mask[segments == active] = 1
#     perturbed_image = copy.deepcopy(img)
#     perturbed_image = perturbed_image * mask[:, :, np.newaxis]
#     return perturbed_image

# def denormalize(image_tensor, mean, std):
#     mean = np.array(mean).reshape(1, 1, 3)
#     std = np.array(std).reshape(1, 1, 3)
#     denormalized_image = image_tensor * std + mean
#     return denormalized_image

# # Main processing loop
# results = []

# for img_name in os.listdir(image_dir):
#     img_path = os.path.join(image_dir, img_name)
#     print(img_path)
#     if img_path.lower().endswith('.jpg'):
#         predicted_class, probabilities, original_image = predict(img_path, model, transform_test, device)
#         print(predicted_class)
#         print(probabilities.tolist())
#         # Convert the image to float64 and transpose
#         original_image = original_image.transpose(1, 2, 0).astype(np.float64)

#         # Denormalize the image
#         mean = [0.485, 0.456, 0.406]
#         std = [0.229, 0.224, 0.225]
#         original_image = denormalize(original_image, mean, std)

#         # Clip to [0, 1] range
#         original_image = np.clip(original_image, 0, 1)

#         # Generate superpixels
#         superpixels = skimage.segmentation.quickshift(original_image, kernel_size=4, max_dist=200, ratio=0.2)
#         num_superpixels = np.unique(superpixels).shape[0]

#         # Generate perturbations
#         num_perturb = 150
#         perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))

#         # Get predictions for perturbations
#         predictions = []
#         for pert in perturbations:
#             perturbed_img = perturb_image(original_image, pert, superpixels)
#             perturbed_img = torch.tensor(perturbed_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 pred = model(perturbed_img)
#             predictions.append(pred.cpu().numpy())

#         predictions = np.array(predictions)

#         # Calculate distances and weights
#         original_superpixels = np.ones(num_superpixels)[np.newaxis, :]  # Perturbation with all superpixels enabled
#         distances = pairwise_distances(perturbations, original_superpixels, metric='cosine').ravel()
#         kernel_width = 0.25
#         weights = np.sqrt(np.exp(-(distances**2) / kernel_width**2))  # Kernel function

#         # Fit a simpler model
#         simpler_model = LinearRegression()
#         simpler_model.fit(X=perturbations, y=predictions[:, :, predicted_class], sample_weight=weights)
#         coeff = simpler_model.coef_[0]

#         # Identify top features
#         num_top_features = 4
#         top_features = np.argsort(coeff)[-num_top_features:]

#         # Visualize the perturbed image
#         mask = np.zeros(num_superpixels, dtype=bool)
#         mask[top_features] = True  # Activate top superpixels
#         highlighted_image = perturb_image(original_image, mask, superpixels)

#         # Ensure highlighted image is in the correct range and type
#         highlighted_image = np.clip(highlighted_image, 0, 1)
#         highlighted_image = (highlighted_image * 255).astype(np.uint8)

#         # Convert original image to uint8 and ensure correct shape
#         original_image = (original_image * 255).astype(np.uint8)

#         # Save the images
#         skimage.io.imsave(os.path.join(save_dir, f'{img_name}_superpixels.png'),
#                   (skimage.segmentation.mark_boundaries(original_image, superpixels) * 255).astype(np.uint8))

#         skimage.io.imsave(os.path.join(save_dir, f'{img_name}_highlighted.png'), highlighted_image)

#         results.append({
#             'image': img_name,
#             'predicted_class': predicted_class,
#             'probabilities': probabilities.tolist(),
#             'highlighted_image': highlighted_image
#         })
#         print(img_name)

# # Print the results
# for result in results:
#     print(f"Image: {result['image']}")
#     print(f"Predicted class: {result['predicted_class']}")
#     print(f"Probabilities: {result['probabilities']}\n")