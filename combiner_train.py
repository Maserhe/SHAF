from comet_ml import Experiment
import json
import multiprocessing
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean, harmonic_mean, geometric_mean
from typing import List
import clip
import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import base_path, squarepad_transform, FashionIQDataset, targetpad_transform, ShoesDataset, Fashion200kDataset
from combiner import Combiner
from utils import collate_fn, update_train_running_results, set_train_bar_description, save_model, \
    extract_index_features, generate_randomized_fiq_caption, device
from validate import compute_fiq_val_metrics, compute_shoes_val_metrics

import random
# fashionCLIP
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import global_val



def smooth_loss(pred, gold):
    eps = 0.05
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1).mean()
    return loss


class CustomSchedule(object):
    def __init__(self, d_model, warmup_steps=4000, optimizer=None):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.steps = 1.
        self.optimizer = optimizer

    def step(self):
        arg1 = self.steps ** -0.5
        arg2 = self.steps * (self.warmup_steps ** -1.5)
        self.steps += 1.
        lr = (self.d_model ** -0.5) * min(arg1, arg2)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        return lr



def hook_text_function(module, input, output):
    # 在这里存储或处理中间输出
    # global text_intermediate_outputs
    global_val.set_text_value(output[0][:, 0:1, :])

def hook_image_function(module, input, output):
    # 在这里存储或处理中间输出
    # global image_intermediate_outputs
    global_val.set_image_value(output[0][:, 0:1, :])


def combiner_training_fiq(train_dress_types: List[str], val_dress_types: List[str],
                          projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                          combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int,
                          transform: str, save_training: bool, save_best: bool, **kwargs):
    """
    Train the Combiner on FashionIQ dataset keeping frozed the CLIP model
    :param train_dress_types: FashionIQ categories to train on
    :param val_dress_types: FashionIQ categories to validate on
    :param projection_dim: Combiner projection dimension
    :param hidden_dim: Combiner hidden dimension
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param combiner_lr: Combiner learning rate
    :param batch_size: batch size of the Combiner training
    :param clip_bs: batch size of the CLIP feature extraction
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param save_best: when True save only the weights of the best Combiner wrt the average_recall metric
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg. If you want to load a
                fine-tuned version of clip you should provide `clip_model_path` as kwarg.
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    training_path: Path = Path(
        base_path / f"models/fashioniq_model_preAtten/fashionclip_pre_Atten_2个_加晚期融合_种子20_combiner_trained_on_fiq_{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    # clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    clip_preprocess = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    clip_model.to(device=device)
    clip_model.eval()
    input_dim = 224
    feature_dim = 512

    for i in range(1, 13):  # 从第 1 层到第 12 层
        layer = clip_model.vision_model.encoder.layers[i - 1]  # 注意：层的索引从 0 开始
        layer.register_forward_hook(hook_image_function)

    for i in range(1, 13):  # 从第 1 层到第 12 层
        layer = clip_model.text_model.encoder.layers[i - 1]  # 注意：层的索引从 0 开始
        layer.register_forward_hook(hook_text_function)


    if kwargs.get("clip_model_path"):
        print('Trying to load the CLIP model')
        clip_model_path = kwargs["clip_model_path"]
        saved_state_dict = torch.load(clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    # clip_model = clip_model.float()

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    index_features_list = []
    index_names_list = []

    global_image_feature_list = []

    # Define the validation datasets and extract the validation index features for each dress_type
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', clip_preprocess)
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', clip_preprocess)
        index_features_and_names = extract_index_features(classic_val_dataset, clip_model)
        index_features_list.append(index_features_and_names[0])
        index_names_list.append(index_features_and_names[1])

        global_image_feature_list.append(index_features_and_names[2])
        global_val.clear_image()

    # Define the combiner and the train dataset
    combiner = Combiner(feature_dim, projection_dim, hidden_dim).to(device, non_blocking=True)
    relative_train_dataset = FashionIQDataset('train', train_dress_types, 'relative', clip_preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=1, pin_memory=True, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.Adam(combiner.parameters(), lr=combiner_lr)


    # lr_scheduler = CustomSchedule(64, optimizer=optimizer)

    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')



    for epoch in range(num_epochs):
        # if torch.cuda.is_available():  # RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
        #     clip.model.convert_weights(clip_model)  # Convert CLIP model in fp16 to reduce computation and memory
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            combiner.train()
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):  # Load a batch of triplets
                step = len(train_bar) * epoch + idx
                images_in_batch = reference_images['pixel_values'].size(0)

                optimizer.zero_grad()

                reference_images = reference_images.to(device)
                target_images = target_images.to(device)

                # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
                flattened_captions: list = np.array(captions).T.flatten().tolist()
                input_captions = generate_randomized_fiq_caption(flattened_captions)
                # text_inputs = clip.tokenize(input_captions, truncate=True).to(device, non_blocking=True)

                text_embedding = clip_preprocess(text=input_captions, return_tensors="pt", max_length=77,
                                                 padding="max_length", truncation=True)
                text_embedding.to(device)

                # reference_global_image_feature = torch.empty((0, 50, 768)).to(device, non_blocking=True)
                # global_text_features = torch.empty((0, 50, 768)).to(device, non_blocking=True)

                # Extract the features with CLIP
                with torch.no_grad():
                    target_image_features = clip_model.get_image_features(**target_images)
                    global_val.clear_image()

                    reference_image_features = clip_model.get_image_features(**reference_images)
                    # reference_global_image_feature = image_output1

                    text_features = clip_model.get_text_features(**text_embedding)

                    # global_text_features = text_output1

                # Compute the logits and the loss
                with torch.cuda.amp.autocast():

                    image_intermediate_outputs = global_val.get_image_value()
                    text_intermediate_outputs = global_val.get_text_value()

                    global_val.clear_image()
                    global_val.clear_text()
                    # print(len(image_intermediate_outputs), " === ", len(text_intermediate_outputs))
                    # print(" ============111===================")

                    logits = combiner(reference_image_features, text_features, target_image_features, image_intermediate_outputs, text_intermediate_outputs)

                    # image_intermediate_outputs = []
                    # text_intermediate_outputs = []

                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = crossentropy_criterion(logits, ground_truth)
                    # loss = smooth_loss(logits, ground_truth)

                # Backprogate and update the weights
                scaler.scale(loss).backward()
                # lr_scheduler.step()

                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            # clip_model = clip_model.float()  # In validation we use fp32 CLIP model
            with experiment.validate():
                combiner.eval()
                recalls_at10 = []
                recalls_at50 = []

                # Compute and log validation metrics for each validation dataset (which corresponds to a different
                # FashionIQ category)
                for relative_val_dataset, index_features, global_image_features, index_names, idx in zip(relative_val_datasets,
                                                                                  index_features_list,
                                                                                  global_image_feature_list,
                                                                                  index_names_list,
                                                                                  idx_to_dress_mapping):
                    recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, clip_model, index_features, global_image_features,
                                                                       index_names, combiner.combine_features, clip_preprocess)
                    recalls_at10.append(recall_at10)
                    recalls_at50.append(recall_at50)

                results_dict = {}
                for i in range(len(recalls_at10)):
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                results_dict.update({
                    f'average_recall_at10': mean(recalls_at10),
                    f'average_recall_at50': mean(recalls_at50),
                    f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
                })

                print(json.dumps(results_dict, indent=4))
                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Validation CSV logging
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            # Save model
            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    save_model('combiner', epoch, combiner, training_path)
                elif not save_best:
                    save_model(f'combiner_{epoch}', epoch, combiner, training_path)


def combiner_training_shoes(train_dress_types: List[str], val_dress_types: List[str],
                          projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                          combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int,
                          transform: str, save_training: bool, save_best: bool, **kwargs):
    """
    Train the Combiner on FashionIQ dataset keeping frozed the CLIP model
    :param train_dress_types: FashionIQ categories to train on
    :param val_dress_types: FashionIQ categories to validate on
    :param projection_dim: Combiner projection dimension
    :param hidden_dim: Combiner hidden dimension
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param combiner_lr: Combiner learning rate
    :param batch_size: batch size of the Combiner training
    :param clip_bs: batch size of the CLIP feature extraction
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param save_best: when True save only the weights of the best Combiner wrt the average_recall metric
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg. If you want to load a
                fine-tuned version of clip you should provide `clip_model_path` as kwarg.
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    training_path: Path = Path(
        base_path / f"models/shoes_model_preAtten/fashionclip_combiner_种子20_trained_on_shoes_{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    # clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    clip_preprocess = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    clip_model.to(device=device)
    clip_model.eval()
    input_dim = 224
    feature_dim = 512

    for i in range(1, 13):  # 从第 1 层到第 12 层
        layer = clip_model.vision_model.encoder.layers[i - 1]  # 注意：层的索引从 0 开始
        layer.register_forward_hook(hook_image_function)

    for i in range(1, 13):  # 从第 1 层到第 12 层
        layer = clip_model.text_model.encoder.layers[i - 1]  # 注意：层的索引从 0 开始
        layer.register_forward_hook(hook_text_function)


    if kwargs.get("clip_model_path"):
        print('Trying to load the CLIP model')
        clip_model_path = kwargs["clip_model_path"]
        saved_state_dict = torch.load(clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    # clip_model = clip_model.float()

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    index_features_list = []
    index_names_list = []

    global_image_feature_list = []

    # Define the validation datasets and extract the validation index features for each dress_type
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = ShoesDataset('val', 'relative', clip_preprocess)
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = ShoesDataset('val', 'classic', clip_preprocess)
        index_features_and_names = extract_index_features(classic_val_dataset, clip_model)
        index_features_list.append(index_features_and_names[0])
        index_names_list.append(index_features_and_names[1])

        global_image_feature_list.append(index_features_and_names[2])
        global_val.clear_image()



    # Define the combiner and the train dataset
    combiner = Combiner(feature_dim, projection_dim, hidden_dim).to(device, non_blocking=True)
    relative_train_dataset = ShoesDataset('train', 'relative', clip_preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=1, pin_memory=True, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.Adam(combiner.parameters(), lr=combiner_lr)
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')


    for epoch in range(num_epochs):
        # if torch.cuda.is_available():  # RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
        #     clip.model.convert_weights(clip_model)  # Convert CLIP model in fp16 to reduce computation and memory
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            combiner.train()
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):  # Load a batch of triplets
                step = len(train_bar) * epoch + idx
                images_in_batch = reference_images['pixel_values'].size(0)

                optimizer.zero_grad()

                reference_images = reference_images.to(device)
                target_images = target_images.to(device)

                # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
                # flattened_captions: list = np.array(captions).T.flatten().tolist()
                # input_captions = generate_randomized_fiq_caption(flattened_captions)

                input_captions = captions
                # text_inputs = clip.tokenize(input_captions, truncate=True).to(device, non_blocking=True)

                text_embedding = clip_preprocess(text=input_captions, return_tensors="pt", max_length=77,
                                                 padding="max_length", truncation=True)
                text_embedding.to(device)

                # Extract the features with CLIP
                with torch.no_grad():
                    target_image_features = clip_model.get_image_features(**target_images)
                    global_val.clear_image()

                    reference_image_features = clip_model.get_image_features(**reference_images)
                    # reference_global_image_feature = image_output1

                    text_features = clip_model.get_text_features(**text_embedding)

                    # global_text_features = text_output1

                # Compute the logits and the loss
                with torch.cuda.amp.autocast():
                    image_intermediate_outputs = global_val.get_image_value()
                    text_intermediate_outputs = global_val.get_text_value()

                    global_val.clear_image()
                    global_val.clear_text()
                    # print(len(image_intermediate_outputs), " === ", len(text_intermediate_outputs))
                    # print(" ============111===================")

                    logits = combiner(reference_image_features, text_features, target_image_features,
                                      image_intermediate_outputs, text_intermediate_outputs)

                    # image_intermediate_outputs = []
                    # text_intermediate_outputs = []

                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = crossentropy_criterion(logits, ground_truth)
                    # loss = smooth_loss(logits, ground_truth)

                # Backprogate and update the weights
                scaler.scale(loss).backward()
                # lr_scheduler.step()

                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            # clip_model = clip_model.float()  # In validation we use fp32 CLIP model
            with experiment.validate():
                combiner.eval()
                recalls_at1 = []
                recalls_at10 = []
                recalls_at50 = []

                # Compute and log validation metrics for each validation dataset (which corresponds to a different
                # FashionIQ category)
                for relative_val_dataset, index_features, global_image_features, index_names, idx in zip(relative_val_datasets,
                                                                                  index_features_list,
                                                                                  global_image_feature_list,
                                                                                  index_names_list,
                                                                                  idx_to_dress_mapping):

                    recall_at1, recall_at10, recall_at50 = compute_shoes_val_metrics(relative_val_dataset, clip_model, index_features, global_image_features,
                                                                       index_names, combiner.combine_features, clip_preprocess)


                    recalls_at1.append(recall_at1)
                    recalls_at10.append(recall_at10)
                    recalls_at50.append(recall_at50)

                results_dict = {}
                for i in range(len(recalls_at10)):
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at1'] = recalls_at1[i]
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                results_dict.update({
                    # f'average_recall_at1': mean(recalls_at1),
                    # f'average_recall_at10': mean(recalls_at10),
                    # f'average_recall_at50': mean(recalls_at50),
                    f'average_recall': (mean(recalls_at1) + mean(recalls_at50) + mean(recalls_at10)) / 3
                })

                print(json.dumps(results_dict, indent=4))
                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Validation CSV logging
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            # Save model
            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    save_model('combiner', epoch, combiner, training_path)
                elif not save_best:
                    save_model(f'combiner_{epoch}', epoch, combiner, training_path)


def combiner_training_fashion200k(train_dress_types: List[str], val_dress_types: List[str],
                          projection_dim: int, hidden_dim: int, num_epochs: int, clip_model_name: str,
                          combiner_lr: float, batch_size: int, clip_bs: int, validation_frequency: int,
                          transform: str, save_training: bool, save_best: bool, **kwargs):
    """
    Train the Combiner on FashionIQ dataset keeping frozed the CLIP model
    :param train_dress_types: FashionIQ categories to train on
    :param val_dress_types: FashionIQ categories to validate on
    :param projection_dim: Combiner projection dimension
    :param hidden_dim: Combiner hidden dimension
    :param num_epochs: number of epochs
    :param clip_model_name: CLIP model you want to use: "RN50", "RN101", "RN50x4"...
    :param combiner_lr: Combiner learning rate
    :param batch_size: batch size of the Combiner training
    :param clip_bs: batch size of the CLIP feature extraction
    :param validation_frequency: validation frequency expressed in epoch
    :param transform: preprocess transform you want to use. Should be in ['clip', 'squarepad', 'targetpad']. When
                targetpad is also required to provide `target_ratio` kwarg.
    :param save_training: when True save the weights of the Combiner network
    :param save_best: when True save only the weights of the best Combiner wrt the average_recall metric
    :param kwargs: if you use the `targetpad` transform you should prove `target_ratio` as kwarg. If you want to load a
                fine-tuned version of clip you should provide `clip_model_path` as kwarg.
    """

    training_start = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    training_path: Path = Path(
        base_path / f"models/fashion200k_model_pre/fashionclip_combiner_种子20_trained_on_Fashion200k_{clip_model_name}_{training_start}")
    training_path.mkdir(exist_ok=False, parents=True)

    # Save all the hyperparameters on a file
    with open(training_path / "training_hyperparameters.json", 'w+') as file:
        json.dump(training_hyper_params, file, sort_keys=True, indent=4)

    # clip_model, clip_preprocess = clip.load(clip_model_name, device=device, jit=False)
    clip_model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
    clip_preprocess = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

    clip_model.to(device=device)
    clip_model.eval()
    input_dim = 224
    feature_dim = 512


    if kwargs.get("clip_model_path"):
        print('Trying to load the CLIP model')
        clip_model_path = kwargs["clip_model_path"]
        saved_state_dict = torch.load(clip_model_path, map_location=device)
        clip_model.load_state_dict(saved_state_dict["CLIP"])
        print('CLIP model loaded successfully')

    # clip_model = clip_model.float()

    idx_to_dress_mapping = {}
    relative_val_datasets = []
    index_features_list = []
    index_names_list = []

    global_image_feature_list = []

    # Define the validation datasets and extract the validation index features for each dress_type
    for idx, dress_type in enumerate(val_dress_types):
        idx_to_dress_mapping[idx] = dress_type
        relative_val_dataset = Fashion200kDataset('val', 'relative', clip_preprocess)
        relative_val_datasets.append(relative_val_dataset)
        classic_val_dataset = Fashion200kDataset('val', 'classic', clip_preprocess)
        index_features_and_names = extract_index_features(classic_val_dataset, clip_model)
        index_features_list.append(index_features_and_names[0])
        index_names_list.append(index_features_and_names[1])
        global_image_feature_list.append(index_features_and_names[2])


    # Define the combiner and the train dataset
    combiner = Combiner(feature_dim, projection_dim, hidden_dim).to(device, non_blocking=True)
    relative_train_dataset = Fashion200kDataset('train', 'relative', clip_preprocess)
    relative_train_loader = DataLoader(dataset=relative_train_dataset, batch_size=batch_size,
                                       num_workers=1, pin_memory=True, collate_fn=collate_fn,
                                       drop_last=True, shuffle=True)

    # Define the optimizer, the loss and the grad scaler
    optimizer = optim.Adam(combiner.parameters(), lr=combiner_lr)
    crossentropy_criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()

    # When save_best == True initialize the best result to zero
    if save_best:
        best_avg_recall = 0

    # Define dataframes for CSV logging
    training_log_frame = pd.DataFrame()
    validation_log_frame = pd.DataFrame()

    # Start with the training loop
    print('Training loop started')

    # 绑定钩子函数
    clip_model.vision_model.encoder.layers[6].register_forward_hook(hook1)
    clip_model.text_model.encoder.layers[6].register_forward_hook(hook2)


    for epoch in range(num_epochs):
        # if torch.cuda.is_available():  # RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
        #     clip.model.convert_weights(clip_model)  # Convert CLIP model in fp16 to reduce computation and memory
        with experiment.train():
            train_running_results = {'images_in_epoch': 0, 'accumulated_train_loss': 0}
            combiner.train()
            train_bar = tqdm(relative_train_loader, ncols=150)
            for idx, (reference_images, target_images, captions) in enumerate(train_bar):  # Load a batch of triplets
                step = len(train_bar) * epoch + idx
                images_in_batch = reference_images['pixel_values'].size(0)

                optimizer.zero_grad()

                reference_images = reference_images.to(device)
                target_images = target_images.to(device)

                # Randomize the training caption in four way: (a) cap1 and cap2 (b) cap2 and cap1 (c) cap1 (d) cap2
                # flattened_captions: list = np.array(captions).T.flatten().tolist()
                # input_captions = generate_randomized_fiq_caption(flattened_captions)

                input_captions = captions
                # text_inputs = clip.tokenize(input_captions, truncate=True).to(device, non_blocking=True)

                text_embedding = clip_preprocess(text=input_captions, return_tensors="pt", max_length=77,
                                                 padding="max_length", truncation=True)
                text_embedding.to(device)


                # Extract the features with CLIP
                with torch.no_grad():


                    reference_image_features = clip_model.get_image_features(**reference_images)
                    reference_global_image_feature = image_output1

                    target_image_features = clip_model.get_image_features(**target_images)
                    text_features = clip_model.get_text_features(**text_embedding)

                    global_text_features = text_output1


                    # reference_image_features = clip_model.get_image_features(**reference_images)
                    # target_image_features = clip_model.get_image_features(**target_images)
                    # text_features = clip_model.get_text_features(**text_embedding)


                # Compute the logits and the loss
                with torch.cuda.amp.autocast():
                    logits = combiner(reference_image_features, text_features, target_image_features, reference_global_image_feature, global_text_features)
                    ground_truth = torch.arange(images_in_batch, dtype=torch.long, device=device)
                    loss = crossentropy_criterion(logits, ground_truth)

                # Backprogate and update the weights
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                experiment.log_metric('step_loss', loss.detach().cpu().item(), step=step)
                update_train_running_results(train_running_results, loss, images_in_batch)
                set_train_bar_description(train_bar, epoch, num_epochs, train_running_results)

            train_epoch_loss = float(
                train_running_results['accumulated_train_loss'] / train_running_results['images_in_epoch'])
            experiment.log_metric('epoch_loss', train_epoch_loss, epoch=epoch)

            # Training CSV logging
            training_log_frame = pd.concat(
                [training_log_frame,
                 pd.DataFrame(data={'epoch': epoch, 'train_epoch_loss': train_epoch_loss}, index=[0])])
            training_log_frame.to_csv(str(training_path / 'train_metrics.csv'), index=False)

        if epoch % validation_frequency == 0:
            # clip_model = clip_model.float()  # In validation we use fp32 CLIP model
            with experiment.validate():
                combiner.eval()
                recalls_at1 = []
                recalls_at10 = []
                recalls_at50 = []

                # Compute and log validation metrics for each validation dataset (which corresponds to a different
                # FashionIQ category)
                for relative_val_dataset, index_features, global_image_features, index_names, idx in zip(
                        relative_val_datasets,
                        index_features_list,
                        global_image_feature_list,
                        index_names_list,
                        idx_to_dress_mapping):
                    recall_at1, recall_at10, recall_at50 = compute_shoes_val_metrics(relative_val_dataset, clip_model, index_features,
                                                                       global_image_features,
                                                                       index_names, combiner.combine_features,
                                                                       clip_preprocess)
                    recalls_at1.append(recall_at1)
                    recalls_at10.append(recall_at10)
                    recalls_at50.append(recall_at50)

                results_dict = {}
                for i in range(len(recalls_at10)):
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at1'] = recalls_at1[i]
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at10'] = recalls_at10[i]
                    results_dict[f'{idx_to_dress_mapping[i]}_recall_at50'] = recalls_at50[i]
                results_dict.update({
                    # f'average_recall_at1': mean(recalls_at1),
                    # f'average_recall_at10': mean(recalls_at10),
                    # f'average_recall_at50': mean(recalls_at50),
                    f'average_recall': (mean(recalls_at1) + mean(recalls_at50) + mean(recalls_at10)) / 3
                })

                print(json.dumps(results_dict, indent=4))
                experiment.log_metrics(
                    results_dict,
                    epoch=epoch
                )

                # Validation CSV logging
                log_dict = {'epoch': epoch}
                log_dict.update(results_dict)
                validation_log_frame = pd.concat([validation_log_frame, pd.DataFrame(data=log_dict, index=[0])])
                validation_log_frame.to_csv(str(training_path / 'validation_metrics.csv'), index=False)

            # Save model
            if save_training:
                if save_best and results_dict['average_recall'] > best_avg_recall:
                    best_avg_recall = results_dict['average_recall']
                    save_model('combiner', epoch, combiner, training_path)
                elif not save_best:
                    save_model(f'combiner_{epoch}', epoch, combiner, training_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    # 设置随机数种子
    setup_seed(3407)
    global_val._init()

    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="should be either 'fashion200k' or 'fashionIQ' or 'shoes' ")
    parser.add_argument("--api-key", type=str, help="api for Comet logging")
    parser.add_argument("--workspace", type=str, help="workspace of Comet logging")
    parser.add_argument("--experiment-name", type=str, help="name of the experiment on Comet")
    parser.add_argument("--projection-dim", default=640 * 4, type=int, help='Combiner projection dim')
    parser.add_argument("--hidden-dim", default=640 * 8, type=int, help="Combiner hidden dim")
    parser.add_argument("--num-epochs", default=300, type=int, help="number training epochs")
    parser.add_argument("--clip-model-name", default="RN50x4", type=str, help="CLIP model to use, e.g 'RN50', 'RN50x4'")
    parser.add_argument("--clip-model-path", type=str, help="Path to the fine-tuned CLIP model")
    parser.add_argument("--combiner-lr", default=2e-5, type=float, help="Combiner learning rate")
    parser.add_argument("--batch-size", default=1024, type=int, help="Batch size of the Combiner training")
    parser.add_argument("--clip-bs", default=32, type=int, help="Batch size during CLIP feature extraction")
    parser.add_argument("--validation-frequency", default=3, type=int, help="Validation frequency expressed in epochs")
    parser.add_argument("--target-ratio", default=1.25, type=float, help="TargetPad target ratio")
    parser.add_argument("--transform", default="targetpad", type=str,
                        help="Preprocess pipeline, should be in ['clip', 'squarepad', 'targetpad'] ")
    parser.add_argument("--save-training", dest="save_training", action='store_true',
                        help="Whether save the training model")
    parser.add_argument("--save-best", dest="save_best", action='store_true',
                        help="Save only the best model during training")

    args = parser.parse_args()
    if args.dataset.lower() not in ['fashioniq', 'fashion200k', "shoes"]:
        raise ValueError("Dataset should be either 'fashion200k' or 'FashionIQ")

    training_hyper_params = {
        "projection_dim": args.projection_dim,
        "hidden_dim": args.hidden_dim,
        "num_epochs": args.num_epochs,
        "clip_model_name": args.clip_model_name,
        "clip_model_path": args.clip_model_path,
        "combiner_lr": args.combiner_lr,
        "batch_size": args.batch_size,
        "clip_bs": args.clip_bs,
        "validation_frequency": args.validation_frequency,
        "transform": args.transform,
        "target_ratio": args.target_ratio,
        "save_training": args.save_training,
        "save_best": args.save_best,
    }

    if args.api_key and args.workspace:
        print("Comet logging ENABLED")
        experiment = Experiment(
            api_key=args.api_key,
            project_name=f"{args.dataset} combiner training",
            workspace=args.workspace,
            disabled=False
        )
        if args.experiment_name:
            experiment.set_name(args.experiment_name)
    else:
        print("Comet loging DISABLED, in order to enable it you need to provide an api key and a workspace")
        experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
            disabled=True
        )

    experiment.log_code(folder=str(base_path / 'src'))
    experiment.log_parameters(training_hyper_params)

    if args.dataset.lower() == 'fashioniq':
        training_hyper_params.update(
            {'train_dress_types': ['dress', 'toptee', 'shirt'], 'val_dress_types': ['dress', 'toptee', 'shirt']})
        combiner_training_fiq(**training_hyper_params)

    elif args.dataset.lower() == 'shoes':
        training_hyper_params.update(
            {'train_dress_types': ['shoes'], 'val_dress_types': ['shoes']})
        combiner_training_shoes(**training_hyper_params)

    elif args.dataset.lower() == 'fashion200k':
        training_hyper_params.update(
            {'train_dress_types': ['fashion200k'], 'val_dress_types': ['shoes']})
        combiner_training_fashion200k(**training_hyper_params)