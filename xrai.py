import torch
import torchvision.transforms as transforms
import numpy as np
import models_vit
import saliency.core as saliency
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
import PIL.Image
import matplotlib.pyplot as plt
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os
import random


def pre_input_vgg19(x):
    return tf.keras.applications.vgg19.preprocess_input(x)

def preprocess_image_vgg19(file_path):
  im = PIL.Image.open(file_path)
  im = im.resize((224,224))
  im = np.asarray(im)
  im = pre_input_vgg19(im)  # Apply preprocessing for VGG19
  im = np.expand_dims(im, axis=0)
  return im

def build_transform():
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    t = []
    input_size = 224
    # if input_size <= 224:
    #     crop_pct = 224 / 256
    # else:
    #     crop_pct = 1.0
    # size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def retfound_preprocess_image(image_path):
    transform = build_transform()
    image = Image.open(image_path).convert('RGB')
    transformed_image = transform(image)
    return transformed_image


def call_model_function_vgg19(images, call_model_args=None, expected_keys=None):
    target_class_idx = call_model_args['class_idx_str']
    images = call_model_args['pre_image']
    model = call_model_args['model']
    images = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            output_layer = model(images)
            output_layer = output_layer[:,target_class_idx]
            gradients = np.array(tape.gradient(output_layer, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            conv_layer, output_layer = model(images)
            gradients = np.array(tape.gradient(output_layer, conv_layer))
            return {saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                    saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}

def call_model_function_retfound(images, call_model_args=None, expected_keys=None):
    target_class_idx = call_model_args['class_idx_str']
    images = call_model_args['pre_image']
    model = call_model_args['model']
    images.requires_grad = True
    output = model(images)
    probabilities = torch.nn.Softmax(dim=1)(output)
    if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
        outputs = probabilities[:, target_class_idx]
        grads = torch.autograd.grad(outputs, images, grad_outputs=torch.ones_like(outputs), create_graph=True)
        grads = torch.movedim(grads[0], 1, 3)
        gradients = grads.cpu().detach().numpy()
        # gradients = grads[0].detach().cpu().numpy()
        # gradients = np.transpose(gradients, (0, 2, 3, 1))
        return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
    else:
        return None
def main():
    retfound_model = models_vit.__dict__['vit_large_patch16'](
        img_size=224,
        num_classes=2,
        drop_path_rate=0.2,
        global_pool=True,
    )
    weight_path = '/content/drive/MyDrive/huyn/Sreenihi_models/checkpoint-best.pth'
    # weight_path = '/content/drive/MyDrive/huyn/LACDHS_task_unfreeze/Taskcheckpoint-best.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(weight_path, map_location='cpu')
    retfound_model.load_state_dict(checkpoint['model'])
    weights_path = "/content/drive/MyDrive/huyn/Sreenihi_models/VGG19_224_Hyper_Parameter_Tuned_SG_1_NG_0_Dec@-05___Full_Fundus_Entire_Dataset_class_each.h5"
    vgg_19 = load_model(weights_path)
    vgg_19.summary()

    image_directory_glaucoma = '/content/drive/MyDrive/huyn/LACDHS_splitted_data/test/Glaucoma'


    image_files = [f for f in os.listdir(image_directory_glaucoma) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # First random sample
    sampled_images = set(random.sample(image_files, min(50, len(image_files))))

    # Exclude previously sampled images and sample another 50
    remaining_images = [f for f in image_files if f not in sampled_images]
    another_sample = random.sample(remaining_images, min(50, len(remaining_images)))
    save_dir = '/content/drive/MyDrive/huyn/LACDHS_XRAI_VGG19_RETFOUND/25/'
    os.makedirs(save_dir, exist_ok=True)

    for image in sampled_images:
        image_path = os.path.join(image_directory_glaucoma, image)
        gradient_saliency = saliency.GradientSaliency()
        image_name = image_path.split('/')[-1]

        original_image = Image.open(image_path).convert('RGB').resize((224, 224))
        original_image_np = np.array(original_image)

        xrai_object = saliency.XRAI()


        # RETFound

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        retfound_input_tensor  = retfound_preprocess_image(image_path).unsqueeze(0).to(device)
        retfound_model.to(device)
        predictions = retfound_model(retfound_input_tensor)
        retfound_prediction_class = torch.argmax(predictions[0]).item()

        retfound_image_np = retfound_input_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        retfound_image_np = (retfound_image_np - retfound_image_np.min()) / (retfound_image_np.max() - retfound_image_np.min())

        call_model_args = {'class_idx_str': retfound_prediction_class,
                            'model': retfound_model,
                        'vit': True,
                        'pre_image': retfound_input_tensor}

        retfound_xrai_attributions = xrai_object.GetMask(retfound_image_np, call_model_function_retfound, call_model_args, batch_size=1)
        #VGG19

        vgg_input_tensor = preprocess_image_vgg19(image_path)
        predictions = vgg_19(vgg_input_tensor)
        vgg19_prediction_class = np.argmax(predictions[0])
        call_model_args = {'class_idx_str': vgg19_prediction_class,
                        'model': vgg_19,
                        'vit': False,
                        'pre_image': vgg_input_tensor}

        vgg_image_np = vgg_input_tensor[0]
        vgg_image_np= (vgg_image_np - vgg_image_np.min()) / (vgg_image_np.max() - vgg_image_np.min())

        vgg_19_xrai_attributions = xrai_object.GetMask(vgg_image_np, call_model_function_vgg19, call_model_args, batch_size=1)

        first_number = random.randint(1, 2)
        second_number = 3 - first_number

        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        fig.suptitle("Saliency Map Visualizations", fontsize=16)


        axes[0].imshow(retfound_image_np)
        axes[0].axis('off')
        axes[0].set_title("Original Glaucoma Image.")

        # Apply mask for retfound model attributions (adjust `percent` as needed)
        percent = 75
        mask = retfound_xrai_attributions >= np.percentile(retfound_xrai_attributions, percent)
        highlighted_image = np.array(original_image)
        highlighted_image[~mask] = 0
        axes[first_number].imshow(highlighted_image)  # Show original with applied mask in gray
        axes[first_number].axis('off')
        axes[first_number].set_title(f"RETFound Predicted {retfound_prediction_class}")

        # Apply mask for VGG19 model attributions
        mask = vgg_19_xrai_attributions >= np.percentile(vgg_19_xrai_attributions, percent)
        highlighted_image = np.array(original_image)
        highlighted_image[~mask] = 0
        # Show the saliency map with the 'inferno' colormap and alpha blending
        axes[second_number].imshow(highlighted_image)  # Add alpha for transparency
        axes[second_number].axis('off')
        axes[second_number].set_title(f"VGG19 Predicted {vgg19_prediction_class}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        extracted_name = image_name.split('.')[0]
        gt = f"{extracted_name}_groundtruth.jpg"
        save_path = os.path.join(save_dir, gt)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        fig.suptitle("Saliency Map Visualizations", fontsize=16)


        axes[0].imshow(retfound_image_np)
        axes[0].axis('off')
        axes[0].set_title("Original Glaucoma Image.")

        # Apply mask for retfound model attributions (adjust `percent` as needed)
        percent = 75
        mask = retfound_xrai_attributions >= np.percentile(retfound_xrai_attributions, percent)
        highlighted_image = np.array(original_image)
        highlighted_image[~mask] = 0
        axes[first_number].imshow(highlighted_image)  # Show original with applied mask in gray
        axes[first_number].axis('off')

        # Apply mask for VGG19 model attributions
        mask = vgg_19_xrai_attributions >= np.percentile(vgg_19_xrai_attributions, percent)
        highlighted_image = np.array(original_image)
        highlighted_image[~mask] = 0
        # Show the saliency map with the 'inferno' colormap and alpha blending
        axes[second_number].imshow(highlighted_image)  # Add alpha for transparency
        axes[second_number].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()
        save_path = os.path.join(save_dir, image_name)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        another_sample = random.sample(remaining_images, min(50, len(remaining_images)))
        save_dir = '/content/drive/MyDrive/huyn/LACDHS_XRAI_VGG19_RETFOUND/10/'
        os.makedirs(save_dir, exist_ok=True)
        
    for image in another_sample:
        image_path = os.path.join(image_directory_glaucoma, image)
        gradient_saliency = saliency.GradientSaliency()
        image_name = image_path.split('/')[-1]

        original_image = Image.open(image_path).convert('RGB').resize((224, 224))
        original_image_np = np.array(original_image)

        xrai_object = saliency.XRAI()


        # RETFound

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        retfound_input_tensor  = retfound_preprocess_image(image_path).unsqueeze(0).to(device)
        retfound_model.to(device)
        predictions = retfound_model(retfound_input_tensor)
        retfound_prediction_class = torch.argmax(predictions[0]).item()

        retfound_image_np = retfound_input_tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        retfound_image_np = (retfound_image_np - retfound_image_np.min()) / (retfound_image_np.max() - retfound_image_np.min())

        call_model_args = {'class_idx_str': retfound_prediction_class,
                            'model': retfound_model,
                        'vit': True,
                        'pre_image': retfound_input_tensor}

        retfound_xrai_attributions = xrai_object.GetMask(retfound_image_np, call_model_function_retfound, call_model_args, batch_size=1)
        #VGG19

        vgg_input_tensor = preprocess_image_vgg19(image_path)
        predictions = vgg_19(vgg_input_tensor)
        vgg19_prediction_class = np.argmax(predictions[0])
        call_model_args = {'class_idx_str': vgg19_prediction_class,
                        'model': vgg_19,
                        'vit': False,
                        'pre_image': vgg_input_tensor}

        vgg_image_np = vgg_input_tensor[0]
        vgg_image_np= (vgg_image_np - vgg_image_np.min()) / (vgg_image_np.max() - vgg_image_np.min())

        vgg_19_xrai_attributions = xrai_object.GetMask(vgg_image_np, call_model_function_vgg19, call_model_args, batch_size=1)

        first_number = random.randint(1, 2)
        second_number = 3 - first_number

        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        fig.suptitle("Saliency Map Visualizations", fontsize=16)


        axes[0].imshow(retfound_image_np)
        axes[0].axis('off')
        axes[0].set_title("Original Glaucoma Image.")

        # Apply mask for retfound model attributions (adjust `percent` as needed)
        percent = 90
        mask = retfound_xrai_attributions >= np.percentile(retfound_xrai_attributions, percent)
        highlighted_image = np.array(original_image)
        highlighted_image[~mask] = 0
        axes[first_number].imshow(highlighted_image)  # Show original with applied mask in gray
        axes[first_number].axis('off')
        axes[first_number].set_title(f"RETFound Predicted {retfound_prediction_class}")

        # Apply mask for VGG19 model attributions
        mask = vgg_19_xrai_attributions >= np.percentile(vgg_19_xrai_attributions, percent)
        highlighted_image = np.array(original_image)
        highlighted_image[~mask] = 0
        # Show the saliency map with the 'inferno' colormap and alpha blending
        axes[second_number].imshow(highlighted_image)  # Add alpha for transparency
        axes[second_number].axis('off')
        axes[second_number].set_title(f"VGG19 Predicted {vgg19_prediction_class}")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        extracted_name = image_name.split('.')[0]
        gt = f"{extracted_name}_groundtruth.jpg"
        save_path = os.path.join(save_dir, gt)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        fig.suptitle("Saliency Map Visualizations", fontsize=16)


        axes[0].imshow(retfound_image_np)
        axes[0].axis('off')
        axes[0].set_title("Original Glaucoma Image.")

        # Apply mask for retfound model attributions (adjust `percent` as needed)
        percent = 90
        mask = retfound_xrai_attributions >= np.percentile(retfound_xrai_attributions, percent)
        highlighted_image = np.array(original_image)
        highlighted_image[~mask] = 0
        axes[first_number].imshow(highlighted_image)  # Show original with applied mask in gray
        axes[first_number].axis('off')

        # Apply mask for VGG19 model attributions
        mask = vgg_19_xrai_attributions >= np.percentile(vgg_19_xrai_attributions, percent)
        highlighted_image = np.array(original_image)
        highlighted_image[~mask] = 0
        # Show the saliency map with the 'inferno' colormap and alpha blending
        axes[second_number].imshow(highlighted_image)  # Add alpha for transparency
        axes[second_number].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()
        save_path = os.path.join(save_dir, image_name)
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)






if __name__ == '__main__':
    main()
