# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import math
import sys
import csv
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy
from typing import Iterable, Optional
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score,multilabel_confusion_matrix, confusion_matrix
from pycm import *
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
import skimage.io 
import skimage.segmentation
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
import cv2
import copy
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances


def misc_measures(confusion_matrix):
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    
    for i in range(1, confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
        specificity.append(specificity_)
        precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_*specificity_))
        F1_score_2.append(2*precision_*sensitivity_/(precision_+sensitivity_))
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    result = result.transpose(2, 3).transpose(1, 2)
    return result


# def denormalize(tensor, mean, std):
#     mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)  # Shape (3, 1, 1)
#     std = torch.tensor(std).view(3, 1, 1).to(tensor.device)    # Shape (3, 1, 1)
#     return tensor * std + mean


def denormalize(image, mean, std):
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    return image * std + mean

def visualize_cam_for_image(model, input_image, target_layer, save_dir, device, prediction, batch_idx):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    denormalized_image = denormalize(input_image, mean, std)

    input_tensor = input_image.unsqueeze(0).to(device)  
    original_image = denormalized_image.cpu().numpy().transpose(1, 2, 0)  
    original_image = np.clip(original_image * 255, 0, 255).astype(np.uint8)  
    os.makedirs(save_dir, exist_ok=True)

    original_image_path = os.path.join(save_dir, f"layer_21_batch_{batch_idx}_original.jpg")
    Image.fromarray(original_image).save(original_image_path)

    cam = GradCAM(model=model, target_layers=target_layer, reshape_transform=reshape_transform, use_cuda=device.type == 'cuda')
    cam_output = cam(input_tensor=input_tensor)
    if cam_output[0].max() != cam_output[0].min():
        cam_output = cam_output[0]
        
        cam_output = (cam_output - cam_output.min()) / (cam_output.max() - cam_output.min() + 1e-8)  
        cam_output = np.uint8(255 * cam_output)  
        cam_output = cv2.equalizeHist(cam_output)
        cam_colored = cv2.applyColorMap(cam_output, cv2.COLORMAP_JET)
        save_path = os.path.join(save_dir, f"layer_21_batch_{batch_idx}_prediction_{prediction}_grad_cam.jpg")
        save_path_2 = os.path.join(save_dir, f"layer_21_batch_{batch_idx}_prediction_{prediction}_grad_cam_colored.jpg")
        Image.fromarray(cam_output).save(save_path)
        Image.fromarray(cam_colored).save(save_path_2)



def perturb_image(img, perturbation, segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1 
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image * mask[:, :, np.newaxis]
    return perturbed_image



@torch.no_grad()
def evaluate(data_loader, model, device, task, epoch, mode, num_class):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    if not os.path.exists(task):
        os.makedirs(task)

    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []
    

    # features = {}  # Dictionary to store features**

    # def get_features(name):
    #     def hook(model, input, output):
    #         features[name] = output.detach()
    #     return hook
    # model.blocks[-1].norm1.register_forward_hook(get_features('vit_last_block'))
    dataset = data_loader.dataset
    class_names = dataset.classes  # List of class names



    # switch to evaluation mode
    model.eval()

    

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        true_label=F.one_hot(target.to(torch.int64), num_classes=num_class)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            prediction_softmax = nn.Softmax(dim=1)(output)
            _,prediction_decode = torch.max(prediction_softmax, 1)
            _,true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())

        acc1,_ = accuracy(output, target, topk=(1,2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

        # if mode == 'test':
        #     with torch.enable_grad():
        #         save_dir = os.path.join(task, 'GradCam')
        #         for i in range(batch_size):
        #             if true_label_decode[i].item() == 0:  # Check if true label is 0
        #                 input_image = images[i]  # Add batch dimension
        #                 prediction = prediction_decode[i].item()

        #                 true_label_name = class_names[true_label_decode[i].item()]
        #                 prediction_name = class_names[prediction]
        #                 # Run Grad-CAM for each target layer
        #                 if true_label_name == '2SKA_Suspected_Glaucoma':
        #                     mean = [0.485, 0.456, 0.406]
        #                     std = [0.229, 0.224, 0.225]
        #                     denormalized_image = denormalize(input_image, mean, std)

        #                     input_tensor = input_image.unsqueeze(0).to(device)  
        #                     original_image = denormalized_image.cpu().numpy().transpose(1, 2, 0)  
        #                     original_image = np.clip(original_image * 255, 0, 255).astype(np.uint8)  
        #                     os.makedirs(save_dir, exist_ok=True)

        #                     original_image_path = os.path.join(save_dir, f"batch_{i}_orginal_label_{true_label_name}.jpg")
        #                     Image.fromarray(original_image).save(original_image_path)



        #                     for j in [23]:
        #                         target_layer = [model.module.blocks[j].attn]
        #                         cam = GradCAM(model=model, target_layers=target_layer, reshape_transform=reshape_transform, use_cuda=device.type == 'cuda')
        #                         cam_output = cam(input_tensor=input_tensor)
        #                         if cam_output[0].max() != cam_output[0].min():
        #                             cam_output = cam_output[0]
                                    
        #                             cam_output = (cam_output - cam_output.min()) / (cam_output.max() - cam_output.min() + 1e-8)  
        #                             cam_output = np.uint8(255 * cam_output)  
        #                             cam_output = cv2.equalizeHist(cam_output)
        #                             cam_colored = cv2.applyColorMap(cam_output, cv2.COLORMAP_JET)
                                    
        #                             save_path = os.path.join(save_dir, f"batch_{i}_layer_{j}_prediction_{prediction_name}_grad_cam.jpg")
        #                             save_path_colored = os.path.join(save_dir, f"batch_{i}_layer_{j}_prediction_{prediction_name}_grad_cam_colored.jpg")
        #                             Image.fromarray(cam_output).save(save_path)
        #                             Image.fromarray(cam_colored).save(save_path_colored)
        #                     # Print the true label and prediction
        #                     true_label_name = class_names[true_label_decode[i].item()]
        #                     prediction_name = class_names[prediction]
        #                     print(f"Image {i}: True Label = {true_label_name} (ID: {true_label_decode[i].item()}) | Prediction = {prediction_name} (ID: {prediction})")

                    
        if mode == 'test':
            results = []
            save_dir = os.path.join(task, 'Lime')
            for i in range(batch_size):
                if true_label[i, 0].item() == 0:
                    input_image = images[i]
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225]
                    denormalized_image = denormalize(input_image.permute(1, 2, 0).cpu().numpy(), mean, std)
                    denormalized_image = np.clip(denormalized_image, 0, 1)  # Clip values to [0, 1]
                    
                    superpixels = skimage.segmentation.quickshift(denormalized_image, kernel_size=4, max_dist=200, ratio=0.2)
                    num_superpixels = np.unique(superpixels).shape[0]
                    
                    predicted_class = prediction_decode[i].item()
                    num_perturb = 300
                    perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
                    predictions = []
                    for pert in perturbations:
                        perturbed_img = perturb_image(denormalized_image, pert, superpixels)
                        perturbed_img = torch.tensor(perturbed_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                        with torch.no_grad():
                            pred = model(perturbed_img)
                        predictions.append(pred.cpu().numpy())

                    predictions = np.array(predictions)
                    original_superpixels = np.ones(num_superpixels)[np.newaxis, :]
                    distances = pairwise_distances(perturbations, original_superpixels, metric='cosine').ravel()
                    kernel_width = 0.25
                    weights = np.sqrt(np.exp(-(distances**2) / kernel_width**2))

                    simpler_model = LinearRegression()
                    simpler_model.fit(X=perturbations, y=predictions[:, :, predicted_class], sample_weight=weights)
                    coeff = simpler_model.coef_[0]

                    num_top_features = 10
                    top_features = np.argsort(coeff)[-num_top_features:]

                    mask = np.zeros(num_superpixels)
                    mask[top_features] = True
                    highlighted_image = perturb_image(denormalized_image, mask, superpixels)
                    if highlighted_image.max() > 1:
                        highlighted_image = highlighted_image / 255.0

                    # Save the images
                    skimage.io.imsave(os.path.join(save_dir, f'batch_{i}_prediction_{predicted_class}_superpixels.png'),
                                    skimage.segmentation.mark_boundaries(denormalized_image, superpixels))

                    skimage.io.imsave(os.path.join(save_dir, f'batch_{i}_prediction_{predicted_class}_highlighted.png'), highlighted_image)

                    results.append({
                        'image': i,
                        'predicted_class': predicted_class,
                        'true_label': true_label[i, 0].item(),
                        'highlighted_image': highlighted_image,
                    })
            
        # if mode == 'test':
        #     for i in range(batch_size):
        #         if true_label[i, 0].item() == 0:  # Change this condition as needed
        #             mean = [0.485, 0.456, 0.406]
        #             std = [0.229, 0.224, 0.225]
        #             input_image = images[i]
        #             save_dir = os.path.join(task, 'features')
        #             os.makedirs(save_dir, exist_ok=True)

        #             # Save original image
        #             denormalized_image = denormalize(input_image, mean, std)
        #             original_image = denormalized_image.cpu().numpy().transpose(1, 2, 0)
        #             original_image = np.clip(original_image * 255, 0, 255).astype(np.uint8)
        #             original_image_path = os.path.join(save_dir, f"image_batch_{i}_original.jpg")
        #             Image.fromarray(original_image).save(original_image_path)

        #             # Save feature maps as individual images
        #             feature_maps = features['vit_last_block'][i].cpu().numpy()
        #             for j, feature_map in enumerate(feature_maps):
        #                 # Normalize each feature map to [0, 1] range
        #                 feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map))
        #                 feature_map = (feature_map * 255).astype(np.uint8)  # Scale to [0, 255]

        #                 # Save the feature map as an image
        #                 feature_map_image = Image.fromarray(feature_map)
        #                 feature_map_path = os.path.join(save_dir, f"image_{i}_feature_{j}.jpg")
        #                 feature_map_image.save(feature_map_path)

        #             predicted_class = prediction_decode[i].item()
        #             print(f"Saved features and original image for image {i} with predicted class {predicted_class}")
    
        

    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)
    confusion_matrix = multilabel_confusion_matrix(true_label_decode_list, prediction_decode_list,labels=[i for i in range(num_class)])
    acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
    
    auc_roc = roc_auc_score(true_label_onehot_list, prediction_list,multi_class='ovr',average='macro')
    auc_pr = average_precision_score(true_label_onehot_list, prediction_list,average='macro')          
            
    metric_logger.synchronize_between_processes()
    
    print('Sklearn Metrics - Acc: {:.4f} AUC-roc: {:.4f} AUC-pr: {:.4f} F1-score: {:.4f} MCC: {:.4f}'.format(acc, auc_roc, auc_pr, F1, mcc)) 
    results_path = task+'_metrics_{}.csv'.format(mode)
    with open(results_path,mode='a',newline='',encoding='utf8') as cfa:
        wf = csv.writer(cfa)
        data2=[[acc,sensitivity,specificity,precision,auc_roc,auc_pr,F1,mcc,metric_logger.loss]]
        for i in data2:
            wf.writerow(i)
            
    
    if mode=='test':
        cm = ConfusionMatrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        # cm = confusion_matrix(actual_vector=true_label_decode_list, predict_vector=prediction_decode_list)
        cm.plot(cmap=plt.cm.Blues,number_label=True,normalized=True,plot_lib="matplotlib")
        plt.savefig(task+'confusion_matrix_test.jpg',dpi=600,bbox_inches ='tight')
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},auc_roc

