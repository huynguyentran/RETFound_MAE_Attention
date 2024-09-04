import sys
import os
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import models_mae


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

# def run_one_image(img, model, save_path):
#     x = torch.tensor(img)

#     # make it a batch-like
#     x = x.unsqueeze(dim=0)
#     x = torch.einsum('nhwc->nchw', x)

#     # run MAE
#     loss, y, mask = model(x.float(), mask_ratio=0.75)
#     y = model.unpatchify(y)
#     y = torch.einsum('nchw->nhwc', y).detach().cpu()

#     # visualize the mask
#     mask = mask.detach()
#     mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)  # (N, H*W, p*p*3)
#     mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
#     mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

#     x = torch.einsum('nchw->nhwc', x)

#     # masked image
#     im_masked = x * (1 - mask)

#     # MAE reconstruction pasted with visible patches
#     im_paste = x * (1 - mask) + y * mask

#     # make the plt figure larger
#     plt.rcParams['figure.figsize'] = [24, 24]

#     plt.subplot(1, 4, 1)
#     plt.imshow(torch.clip((x[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int())
#     plt.title("original")
#     plt.axis('off')

#     plt.subplot(1, 4, 2)
#     plt.imshow(torch.clip((im_masked[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int())
#     plt.title("masked")
#     plt.axis('off')

#     plt.subplot(1, 4, 3)
#     plt.imshow(torch.clip((y[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int())
#     plt.title("reconstruction")
#     plt.axis('off')

#     plt.subplot(1, 4, 4)
#     plt.imshow(torch.clip((im_paste[0] * imagenet_std + imagenet_mean) * 255, 0, 255).int())
#     plt.title("reconstruction + visible")
#     plt.axis('off')

#     # Save the figure
#     plt.savefig(save_path, bbox_inches='tight')
#     plt.close()

def save_image(tensor_image, save_path):
    # Convert tensor image to PIL image and save
    tensor_image = torch.clip((tensor_image * imagenet_std + imagenet_mean) * 255, 0, 255).int()
    tensor_image = tensor_image.permute(1, 2, 0)  # Change to [H, W, C]
    pil_image = Image.fromarray(tensor_image.numpy().astype(np.uint8))
    pil_image.save(save_path)


def run_one_image(img, model, save_dir):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.75)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # Save images separately
    save_image(x[0], os.path.join(save_dir, "original.png"))
    save_image(im_masked[0], os.path.join(save_dir, "masked.png"))
    save_image(y[0], os.path.join(save_dir, "reconstruction.png"))
    save_image(im_paste[0], os.path.join(save_dir, "reconstruction_visible.png"))


path = '/content/drive/MyDrive/huyn/Lime-uncropped/08 30 2024/Lime'
save_dir = os.path.join(path, "visualize")
os.makedirs(save_dir, exist_ok= True)

# This is an MAE model trained with pixels as targets for visualization (ViT-Large, training mask ratio=0.75)

# download checkpoint if not exist
chkpt_dir = '/content/drive/MyDrive/huyn/LACDHS_task_unfreeze/Taskcheckpoint-best.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
print('Model loaded.')

torch.manual_seed(2)

for entry in os.listdir(path):
        entry_path = os.path.join(path, entry) 

        # Check if the entry is a file
        if os.path.isfile(entry_path):
            print(f"Processing file: {entry_path}")
            image_name = entry
            image_path = os.path.join(path, image_name)

            img_url = image_path
            img = Image.open(img_url)
            img = img.resize((224, 224))
            img = np.array(img) / 255.

            assert img.shape == (224, 224, 3)

            # normalize by ImageNet mean and std
            img = img - imagenet_mean
            img = img / imagenet_std

            plt.rcParams['figure.figsize'] = [5, 5]
            show_image(torch.tensor(img))
            print('MAE with pixel reconstruction:')
            temp_save_dir = os.path.join(save_dir, entry