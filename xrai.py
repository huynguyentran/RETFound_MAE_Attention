import torch
import torchvision.transforms as transforms
import numpy as np
import models_vit
import saliency.core as saliency
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os
import random

def build_transform():
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    t = []
    input_size = 224
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
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


    image_directory_glaucoma = '/content/drive/MyDrive/huyn/LACDHS_splitted_data/test/Glaucoma'

    # predictions = []
    # ground_truths = [] 

    
    # for image_path in image_directory_glaucoma:
    #     image_name = os.path.basename(image_path)

    #     # Preprocess image
    #     retfound_input_tensor  = retfound_preprocess_image(image_path).unsqueeze(0).to(device)

    #     # Make prediction
    #     predictions_tensor = retfound_model(retfound_input_tensor)
    #     predicted_class = torch.argmax(predictions_tensor[0]).item()

    #     # Store results
    #     predictions.append(predicted_class)
    #     ground_truths.append(0)  # Replace `0` with the actual label if available

    # # Save predictions to CSV
    # results_df = pd.DataFrame({
    #     "Image_Name": image_files,
    #     "Ground_Truth": ground_truths,
    #     "Prediction": predictions
    # })
    # save_dir = '/content/drive/MyDrive/huyn/XRAI_VGG19_RETFOUND'
    # os.makedirs(save_dir, exist_ok=True)
    # output_file = os.path.join(save_dir, "retfound.csv")
    # results_df.to_csv(output_file, index=False)
    



    image_files = [f for f in os.listdir(image_directory_glaucoma) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    file_path_25 = '/content/drive/MyDrive/huyn/LACDHS_XRAI_VGG19_RETFOUND/sampled_25.xlsx'
    file_path_10 = '/content/drive/MyDrive/huyn/LACDHS_XRAI_VGG19_RETFOUND/sampled_10.xlsx'


    save_dir = '/content/drive/MyDrive/huyn/LACDHS_XRAI_VGG19_RETFOUND'
    os.makedirs(save_dir, exist_ok=True)

    save_dir_retfound_25 = f'{save_dir}/RETFound/25_2'
    save_dir_retfound_10 = f'{save_dir}/RETFound/10_2'
    os.makedirs(save_dir_retfound_25, exist_ok=True)
    os.makedirs(save_dir_retfound_10, exist_ok=True)
    
    preds_10 = []
    preds_25 = []

    for index, row in df_25.iterrows():
        image_name = row[0]
        image_path = os.path.join(image_directory_glaucoma, image)
        # image_name = image_path.split('/')[-1]

        original_image = Image.open(image_path).convert('RGB').resize((224, 224))
        original_image_np = np.array(original_image)
        xrai_object = saliency.XRAI()


        # RETFound
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        retfound_input_tensor  = retfound_preprocess_image(image_path).unsqueeze(0).to(device)
        retfound_model.to(device)
        predictions = retfound_model(retfound_input_tensor)
        retfound_prediction_class = torch.argmax(predictions[0]).item()
        preds_25.append(retfound_prediction_class)
        call_model_args = {'class_idx_str': retfound_prediction_class,
                            'model': retfound_model,
                        'vit': True,
                        'pre_image': retfound_input_tensor}

        retfound_xrai_attributions = xrai_object.GetMask(original_image_np, call_model_function_retfound, call_model_args, batch_size=1)

        percent = 75
        mask = retfound_xrai_attributions >= np.percentile(retfound_xrai_attributions, percent)
        highlighted_image = np.array(original_image)
        highlighted_image[~mask] = 0

        highlighted_image_path = os.path.join(save_dir_retfound_25, image_name)
        Image.fromarray(highlighted_image).save(highlighted_image_path)
        
    for index, row in df_10.iterrows():
        image_name = row[0]
        image_path = os.path.join(image_directory_glaucoma, image_name)

        original_image = Image.open(image_path).convert('RGB').resize((224, 224))
        original_image_np = np.array(original_image)
        xrai_object = saliency.XRAI()


        # RETFound
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        retfound_input_tensor  = retfound_preprocess_image(image_path).unsqueeze(0).to(device)
        retfound_model.to(device)
        predictions = retfound_model(retfound_input_tensor)
        retfound_prediction_class = torch.argmax(predictions[0]).item()
        preds_10.append(retfound_prediction_class)
        call_model_args = {'class_idx_str': retfound_prediction_class,
                            'model': retfound_model,
                        'vit': True,
                        'pre_image': retfound_input_tensor}

        retfound_xrai_attributions = xrai_object.GetMask(original_image_np, call_model_function_retfound, call_model_args, batch_size=1)

        percent = 90
        mask = retfound_xrai_attributions >= np.percentile(retfound_xrai_attributions, percent)
        highlighted_image = np.array(original_image)
        highlighted_image[~mask] = 0

        highlighted_image_path = os.path.join(save_dir_retfound_10, image_name)
        Image.fromarray(highlighted_image).save(highlighted_image_path)



    # output_file_25 = f"{save_dir}/sampled_25.xlsx"
    # df_25['RETFound_preds'] = preds_25
    # df_25.to_csv(output_file_25, index=False)

    # output_file_10 = f"{save_dir}/sampled_10.xlsx"
    # df_10['RETFound_preds'] = preds_10
    # df_10.to_csv(output_file_10, index=False)



if __name__ == '__main__':
    main()
