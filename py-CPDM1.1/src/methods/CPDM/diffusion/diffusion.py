import os
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from copy import deepcopy
from collections import Counter
import itertools

from safetensors.torch import load_file

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import distributed as dist
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from diffusers import UNet2DConditionModel, DDPMPipeline, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel

from accelerate import Accelerator

import src.utilities.utils as utils
from . import dist_util, logger
from .my_diffusers.schedulers.scheduling_ddpm import MyDDPMScheduler
from .discriminator import Discriminator
from .unet_discriminator import get_encoder_unet_condition_model


from torch.autograd import Variable
import torch_dct
import kornia as ki

def get_mean(dataset):
    if dataset == 'cifar100CI':
        return torch.FloatTensor([0.5070751592371323, 0.48654887331495095, 0.4409178433670343])
    else:
        return torch.FloatTensor([0.485, 0.456, 0.406])

def get_std(dataset):
    if dataset == 'cifar100CI':
        return torch.FloatTensor([0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    else:
        return torch.FloatTensor([0.229, 0.224, 0.225])


def get_unet_condition_model(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        'DownBlock2D',
        'DownBlock2D',
        'DownBlock2D',
        'DownBlock2D',
        'CrossAttnDownBlock2D',
        'DownBlock2D'
    ),
    up_block_types=(
        'UpBlock2D',
        'CrossAttnUpBlock2D',
        'UpBlock2D',
        'UpBlock2D',
        'UpBlock2D',
        'UpBlock2D'
    ),
    cross_attention_dim=768,
    encoder_hid_dim=768,
    encoder_hid_dim_type='text_proj',
    attention_head_dim=8,
    encoder_hid_proj_as_identity=False,
    encoder_hid_proj_no_bias=False
):
    assert not (encoder_hid_proj_as_identity and encoder_hid_proj_no_bias)
    model = UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        cross_attention_dim=cross_attention_dim,
        encoder_hid_dim=encoder_hid_dim,
        encoder_hid_dim_type=encoder_hid_dim_type,
        attention_head_dim=attention_head_dim,
    )
    if encoder_hid_proj_as_identity:
        model.encoder_hid_proj = nn.Identity()
    elif encoder_hid_proj_no_bias:
        encoder_hid_proj = nn.Linear(encoder_hid_dim, cross_attention_dim, bias=False)
        encoder_hid_proj.weight.data = model.encoder_hid_proj.weight.data
        model.encoder_hid_proj = encoder_hid_proj
    return model


def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.item())


def output_img(filepath,img):
    Image.fromarray(img).save(filepath)


def save_to_JPEG(num_samples,all_images,all_labels,all_class_name,output_path):
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_samples]
    try:
        label_arr = np.concatenate(all_labels, axis=0)
    except:
        label_arr = np.array(all_labels)
    label_arr = label_arr[: num_samples]
    class_to_idx = set()
    classes = set()
    img_path_label_list = []
    if dist.get_rank() == 0:
        num_fig = arr.shape[0]
        for fig_count in range(num_fig):
            class_name = all_class_name[label_arr[fig_count]]
            class_to_idx.add(label_arr[fig_count])
            classes.add(class_name)
            filepath = os.path.join(output_path,
                                    "{}_generator{}.JPEG".format(class_name, fig_count))
            img_path_label_list.append((filepath, label_arr[fig_count]))
            output_img(filepath, np.array(arr[fig_count]))
    dist.barrier()
    class_to_idx = list(class_to_idx)
    class_to_idx.sort()
    classes = list(classes)
    classes.sort()
    print("sampling complete")
    return img_path_label_list, classes, class_to_idx

def get_in_channels(image_condition):
    if image_condition == 'none':
        return 3
    elif image_condition == 'canny':
        return 4
    else:
        return 6


class Diffusion():
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_schedule='linear',
        num_inference_timesteps=20,
        **kwargs,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.num_inference_timesteps = num_inference_timesteps
        self.args = kwargs.get('args')
        self.task_counter = kwargs.get('task_counter', 0)
        self.labels_embedding = kwargs.get('labels_embedding')
        self.labels_metric = kwargs.get('labels_metric')
        self.curtask_labels = kwargs.get('curtask_labels')
        self.all_labels = kwargs.get('all_labels')
        self.prevtask_path = kwargs.get('prevtask_path')
        self.curtask_path = kwargs.get('curtask_path')
        self.curtask_model_path = os.path.join(self.curtask_path, 'diffusion')

        self.train_scheduler = MyDDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule
        )

        if self.args.ddpm:
            self.inference_scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule
            )
        else:
            self.inference_scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule
            )
        self.inference_scheduler.set_timesteps(num_inference_timesteps)

        self.model = get_unet_condition_model(
            sample_size=self.args.image_size,
            in_channels=get_in_channels(self.args.image_condition),
            encoder_hid_proj_as_identity=(self.args.embed_condition == 'identity'),
            encoder_hid_proj_no_bias=(self.args.embed_condition == 'no_bias')
        )
        # if self.task_counter == 0:
        #     print("LOAD MODELLLLLLLLLL")
        #     self.model.load_state_dict(
        #         load_file(
        #             os.path.join(
        #                 self.curtask_path,
        #                 'diffusion',
        #                 'unet',
        #                 'diffusion_pytorch_model.safetensors'
        #             )
        #         )
        #     )
        if (self.task_counter > 0) and (self.prevtask_path is not None):
            self.model.load_state_dict(
                load_file(
                    os.path.join(
                        self.prevtask_path,
                        'diffusion',
                        'unet',
                        'diffusion_pytorch_model.safetensors'
                    )
                )
            )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.diffusion_lr, weight_decay=self.args.diffusion_weight_decay)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.args.diffusion_lr_warmup_steps,
            num_training_steps=self.args.lr_anneal_steps,
        )

        self.discriminator = None
        self.discriminator_lr_scheduler = None
        if self.args.use_discriminator:
            if self.args.unet_discriminator:
                self.discriminator = get_encoder_unet_condition_model()
                if (self.task_counter > 0) and (self.prevtask_path is not None):
                    self.discriminator.load_state_dict(
                        load_file(
                            os.path.join(
                                self.prevtask_path,
                                'diffusion',
                                'discriminator',
                                'diffusion_pytorch_model.safetensors'
                            )
                        )
                    )
            else:
                self.discriminator = Discriminator()
                if (self.task_counter > 0) and (self.prevtask_path is not None):
                    self.discriminator.load_state_dict(torch.load(os.path.join(self.prevtask_path, 'diffusion/discriminator.pt')))
            self.discriminator_optimizer = torch.optim.AdamW(self.discriminator.parameters(), lr=self.args.discriminator_lr)
            if self.args.unet_discriminator:
                self.discriminator_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=self.discriminator_optimizer,
                    num_warmup_steps=self.args.diffusion_lr_warmup_steps,
                    num_training_steps=self.args.lr_anneal_steps,
                )
            else:
                self.discriminator_lr_scheduler = None

        self.step = 0
        self.accelerator = Accelerator(
            mixed_precision='fp16' if self.args.use_fp16 else 'no',   
        )

        self.ema_model = None
        if self.args.ema:
            self.ema_model = EMAModel(self.model.parameters(), decay=self.args.ema_rate, use_ema_warmup=True, inv_gamma=1.0, power=3 / 4)
            self.accelerator.register_for_checkpointing(self.ema_model)
            self.ema_model.to(self.accelerator.device)
            
        self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler
        )
        if self.labels_embedding is not None:
            self.labels_embedding = self.labels_embedding.to(self.accelerator.device)
        if (self.labels_metric is not None) and (self.task_counter > 0):
            self.nearest = self.labels_metric[(len(self.all_labels) - len(self.curtask_labels)):len(self.all_labels), :(len(self.all_labels) - len(self.curtask_labels))].argmin(dim=1)
            self.nearest = self.nearest.to(self.accelerator.device)

            self.furthest = self.labels_metric[(len(self.all_labels) - len(self.curtask_labels)):len(self.all_labels), :(len(self.all_labels) - len(self.curtask_labels))].argmax(dim=1)
            self.furthest = self.furthest.to(self.accelerator.device)

        if self.discriminator is not None:
            self.discriminator, self.discriminator_optimizer = self.accelerator.prepare(self.discriminator, self.discriminator_optimizer)
        if self.discriminator_lr_scheduler is not None:
            self.discriminator_lr_scheduler = self.accelerator.prepare(self.discriminator_lr_scheduler)

        self.most_confidence = None
        if self.args.image_condition != 'none':
            self.load_most_confidence()

    def load_most_confidence(self):
        self.most_confidence = dict()
        if (self.task_counter > 0) and (self.prevtask_path is not None):
            # for label in self.all_labels:
            #     if label not in self.curtask_labels:
            #         self.most_confidence[label] = torch.load(
            #             os.path.join(self.prevtask_path, 'image-condition', 'end', 'tensors', f'{label}.pt')
            #         )
            #         self.most_confidence[label].requires_grad = False
            for file in os.listdir(os.path.join(self.prevtask_path, 'image-condition', 'end', 'tensors')):
                label = int(file.split('.', 1)[0])
                self.most_confidence[label] = torch.load(
                    os.path.join(self.prevtask_path, 'image-condition', 'end', 'tensors', file)
                )
                self.most_confidence[label].requires_grad = False
        os.makedirs(os.path.join(self.curtask_path, 'image-condition', 'start', 'tensors'), exist_ok=True)
        for key, value in self.most_confidence.items():
            torch.save(
                value.cpu(),
                os.path.join(self.curtask_path, 'image-condition', 'start', 'tensors', f'{key}.pt')
            )

    def update_most_confidence(self):
        for label in self.curtask_labels:
            self.most_confidence[label] = torch.load(
                os.path.join(self.curtask_path, 'image-condition', 'start', 'tensors', f'{label}.pt')
            )
            if self.args.image_condition == 'buffer':
                self.most_confidence[label].requires_grad = False
            elif self.args.image_condition in ['learn', 'learn_from_noise']:
                self.most_confidence[label].requires_grad = True
        os.makedirs(os.path.join(self.curtask_path, 'image-condition', 'start', 'images'), exist_ok=True)
        mean = get_mean(self.args.ds_name)
        std = get_std(self.args.ds_name)
        for key, value in self.most_confidence.items():
            output_img(
                os.path.join(self.curtask_path, 'image-condition', 'start', 'images', f'{key}.jpg'),
                (value.permute(1, 2, 0).mul(std).add(mean).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            )
        for key, value in self.most_confidence.items():
            self.most_confidence[key] = value.to(self.accelerator.device)
        if self.args.image_condition in ['learn', 'learn_from_noise']:
            for label in self.curtask_labels:
                self.most_confidence[label].retain_grad()

    def save_most_confidence(self):
        os.makedirs(os.path.join(self.curtask_path, 'image-condition', 'end', 'tensors'), exist_ok=True)
        os.makedirs(os.path.join(self.curtask_path, 'image-condition', 'end', 'images'), exist_ok=True)
        mean = get_mean(self.args.ds_name)
        std = get_std(self.args.ds_name)
        for key, value in self.most_confidence.items():
            torch.save(
                value.cpu(),
                os.path.join(self.curtask_path, 'image-condition', 'end', 'tensors', f'{key}.pt')
            )
            output_img(
                os.path.join(self.curtask_path, 'image-condition', 'end', 'images', f'{key}.jpg'),
                (value.cpu().permute(1, 2, 0).mul(std).add(mean).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
            )

    def save(self):
        if self.args.ema:
            self.ema_model.copy_to(self.model.parameters())
        pipeline = DDPMPipeline(
            unet=self.accelerator.unwrap_model(self.model),
            scheduler=self.train_scheduler
        )
        pipeline.save_pretrained(self.curtask_model_path)
        if self.discriminator is not None:
            if self.args.unet_discriminator:
                self.accelerator.unwrap_model(self.discriminator).save_pretrained(os.path.join(self.curtask_model_path, 'discriminator'))
            else:
                torch.save(self.discriminator.state_dict(), os.path.join(self.curtask_model_path, 'discriminator.pt'))

    def sample_correct(self, args, model_ft, samples_path, label_order_list, dataset_path, combine_label_list, use_cuda = False):
        print('Sample imgs ...')
        all_class_name = os.path.join(os.path.dirname(os.path.dirname(dataset_path)),"order_seed={}.pkl".format(self.args.CI_order_rndseed))
        all_class_name_list = utils.unpickle(all_class_name)
        print("sampling...")
        all_images = []
        all_labels = []
        num_samples = self.args.num_samples * len(label_order_list)
        label_counter = 0
        generated = [0] * 100
        print(f"created 0 / {num_samples} samples", end="\r")
        model_ft.train(False)
        with torch.no_grad():
            while len(all_images) < num_samples:
                
                done = False
                resample_counter = 0
                missing_idx = []
                current_idx = []
                force_sample = False

                while not done:
                    
                    if resample_counter > 10:
                        print("Resample counter exceeded 10")
                        force_sample = True
                    
                    rnd_label = []

                    if len(missing_idx) == 0:
                        while (len(rnd_label) < 100) and (label_counter < len(label_order_list)):
                            rnd_label.extend([label_order_list[label_counter] for _ in range(self.args.num_samples)])
                            current_idx.append(label_order_list[label_counter])
                            label_counter += 1
                    else:
                        for idx in missing_idx:
                            rnd_label.extend([idx for _ in range((1 + resample_counter) * self.args.num_samples)])
                        missing_idx = []

                        
                    rnd_label = np.array(rnd_label)
                    rnd_label = torch.from_numpy(rnd_label).to(dist_util.dev())
                    encoder_hidden_states = self.labels_embedding[rnd_label]
                    zeros = torch.zeros_like(encoder_hidden_states)
                    noise = torch.randn((rnd_label.shape[0], 3, self.args.image_size, self.args.image_size)).to(dist_util.dev())
                    if self.most_confidence is not None:
                        image_conditions = torch.stack(
                            [self.most_confidence[label.item()] for label in rnd_label],
                            dim=0
                        ).to(dist_util.dev())
                    sample = noise
                    for t in tqdm(
                        self.inference_scheduler.timesteps,
                        desc=f"created {len(all_images)} / {num_samples} samples",
                        leave=False
                    ):
                        if self.most_confidence is not None:
                            model_input = torch.cat((image_conditions, sample), dim=1)
                        else:
                            model_input = sample
                        with torch.no_grad():
                            if self.args.w > 0.0:
                                noisy_residual = (self.args.w + 1) * self.model(model_input, t, encoder_hidden_states).sample \
                                                - self.args.w * self.model(model_input, t, zeros).sample
                            else:
                                noisy_residual = self.model(model_input, t, encoder_hidden_states).sample
                        previous_noisy_sample = self.inference_scheduler.step(noisy_residual, t, sample).prev_sample
                        sample = previous_noisy_sample

                    # TODO: ensure all samples created can be correctly classified
                    inputs = sample
                    labels = rnd_label

                    if use_cuda:
                        inputs, labels = Variable(inputs.cuda(non_blocking=False)), \
                                            Variable(labels.cuda(non_blocking=False))
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)


                    if args.class_incremental or args.class_incremental_repetition:
                        logits = model_ft(inputs,combine_label_list)
                    else:
                        logits = model_ft(inputs)
                    _, preds = torch.max(logits.data, 1)

                    # check if correctly classified
                    # if not, discard and resample
                    
                    for count in range(len(labels)):
                        if generated[labels[count]] < args.num_samples:
                            if not force_sample:
                                if preds[count] == labels[count]:
                                        # unnormalize the image
                                        
                                        img = inputs[count] * get_std(args.ds_name).view(3, 1, 1) + get_mean(args.ds_name).view(3, 1, 1)
                                        img = img.unsqueeze(0)

                                        ###
                                        img = (img / 2 + 0.5).clamp(0, 1)
                                        img = (img.permute(0, 2, 3, 1) * 255).round().to(torch.uint8)
                                        img = img.contiguous()

                                        all_images.append(img.cpu().numpy())
                                        all_labels.append(labels[count].cpu().numpy())
                                        generated[labels[count]] += 1
                            else:
                                    img = inputs[count].unsqueeze(0)
                                    img = (img / 2 + 0.5).clamp(0, 1)
                                    img = (img.permute(0, 2, 3, 1) * 255).round().to(torch.uint8)
                                    img = img.contiguous()

                                    all_images.append(img.cpu().numpy())
                                    all_labels.append(labels[count].cpu().numpy())
                                    generated[labels[count]] += 1
                                
                    #get unique labels in labels

                    for idx in current_idx:
                        if generated[idx] < args.num_samples:
                            missing_idx.append(idx)

                    if len(missing_idx) == 0:
                        done = True

                    if done or force_sample:
                        
                        resample_counter = 0
                    else:
                        resample_counter += 1

                    
                                

        print(f"created {num_samples} / {num_samples} samples")
        utils.create_dir(samples_path)
        generator_img_path_label_list, generator_classes, generator_class_to_idx = save_to_JPEG(num_samples,
                                                                                            all_images,
                                                                                            all_labels,
                                                                                            all_class_name_list,
                                                                                            samples_path)
        # exit(0)
        return generator_img_path_label_list, generator_classes, generator_class_to_idx

    def sample(self, samples_path, label_order_list, dataset_path):
        print('Sample imgs ...')
        all_class_name = os.path.join(os.path.dirname(os.path.dirname(dataset_path)),"order_seed={}.pkl".format(self.args.CI_order_rndseed))
        all_class_name_list = utils.unpickle(all_class_name)
        print("sampling...")
        all_images = []
        all_labels = []
        num_samples = self.args.num_samples * len(label_order_list)
        label_counter = 0
        print(f"created 0 / {num_samples} samples", end="\r")
        while sum([s.shape[0] for s in all_images]) < num_samples:
            rnd_label = []
            while (len(rnd_label) < 100) and (label_counter < len(label_order_list)):
                rnd_label.extend([label_order_list[label_counter] for _ in range(self.args.num_samples)])
                label_counter += 1
            rnd_label = np.array(rnd_label)
            rnd_label = torch.from_numpy(rnd_label).to(dist_util.dev())
            encoder_hidden_states = self.labels_embedding[rnd_label]
            zeros = torch.zeros_like(encoder_hidden_states)
            noise = torch.randn((rnd_label.shape[0], 3, self.args.image_size, self.args.image_size)).to(dist_util.dev())
            if self.most_confidence is not None:
                image_conditions = torch.stack(
                    [self.most_confidence[label.item()] for label in rnd_label],
                    dim=0
                ).to(dist_util.dev())
            sample = noise
            for t in tqdm(
                self.inference_scheduler.timesteps,
                desc=f"created {sum([s.shape[0] for s in all_images])} / {num_samples} samples",
                leave=False
            ):
                if self.most_confidence is not None:
                    model_input = torch.cat((image_conditions, sample), dim=1)
                else:
                    model_input = sample
                with torch.no_grad():
                    if self.args.w > 0.0:
                        noisy_residual = (self.args.w + 1) * self.model(model_input, t, encoder_hidden_states).sample \
                                        - self.args.w * self.model(model_input, t, zeros).sample
                    else:
                        noisy_residual = self.model(model_input, t, encoder_hidden_states).sample
                previous_noisy_sample = self.inference_scheduler.step(noisy_residual, t, sample).prev_sample
                sample = previous_noisy_sample
            sample = (sample / 2 + 0.5).clamp(0, 1)
            sample = (sample.permute(0, 2, 3, 1) * 255).round().to(torch.uint8)
            sample = sample.contiguous()
            gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            rnd_label = rnd_label.to(dist_util.dev())
            gathered_labels = [torch.zeros_like(rnd_label) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, rnd_label)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        print(f"created {num_samples} / {num_samples} samples")
        utils.create_dir(samples_path)
        generator_img_path_label_list, generator_classes, generator_class_to_idx = save_to_JPEG(num_samples,
                                                                                            all_images,
                                                                                            all_labels,
                                                                                            all_class_name_list,
                                                                                            samples_path)
        # exit(0)
        return generator_img_path_label_list, generator_classes, generator_class_to_idx

    def train(self, dataset, batch_size, num_steps=0, classifier=None):
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last=True, pin_memory=True, persistent_workers=True)
        data_loader = self.accelerator.prepare(data_loader)
        if classifier is not None:
            classifier = self.accelerator.prepare(classifier)

        keep_training = True
        while keep_training:
            for batch, cond in tqdm(data_loader):
                self.step += 1
                batch = batch.to(self.accelerator.device, non_blocking=True)
                labels = cond['y'].to(self.accelerator.device, non_blocking=True)

                noise = torch.randn(batch.shape).to(self.accelerator.device, non_blocking=True)
                bs = batch.shape[0]
                timesteps = torch.randint(
                    0, self.train_scheduler.config.num_train_timesteps, (bs,), device=self.accelerator.device
                ).long()
                noisy_images = self.train_scheduler.add_noise(batch, noise, timesteps)
                model_input = noisy_images
                if self.most_confidence is not None:

                    if self.args.image_condition == 'dct':
                        im_cond = []
                        for label in labels:
                            fft2d = torch.zeros(noisy_images.shape).to(self.accelerator.device, non_blocking=True)
                            # self.most_confidence[label.item()] = torch.cat([fft2d[0, 0:args.dct_luma, 0:args.dct_luma].reshape(1,1,-1), fft2d[1, 0:args.dct_chroma, 0:args.dct_chroma].reshape(1,1,-1), fft2d[2, 0:args.dct_chroma, 0:args.dct_chroma].reshape(1,1,-1)], dim=2)
                            # reconstruct fft2d from it
                            fft2d[0, 0:self.args.dct_luma, 0:self.args.dct_luma] = self.most_confidence[label.item()][:,:, 0:self.args.dct_luma ** 2].reshape(1,self.args.dct_luma,self.args.dct_luma)
                            fft2d[1, 0:self.args.dct_chroma, 0:self.args.dct_chroma] = self.most_confidence[label.item()][:,:, self.args.dct_luma ** 2 :  self.args.dct_luma ** 2 + self.args.dct_chroma **2].reshape(1,self.args.dct_chroma,self.args.dct_chroma)
                            fft2d[2, 0:self.args.dct_chroma, 0:self.args.dct_chroma] = self.most_confidence[label.item()][:,:,self.args.dct_luma ** 2 + self.args.dct_chroma **2:].reshape(1,self.args.dct_chroma,self.args.dct_chroma)
                            im_cond.append(ki.color.ycbcr.ycbcr_to_rgb(torch_dct.idct_2d(fft2d)))

                        image_conditions = torch.stack(im_cond, dim=0)


                    else:
                        image_conditions = torch.stack(
                            [self.most_confidence[label.item()] for label in labels],
                            dim=0
                        )


                    model_input = torch.cat((image_conditions, noisy_images), dim=1)

                mask_adaptive_shape = [bs] + [1] * (len(self.labels_embedding.shape) - 1)
                mask = torch.where(
                    torch.rand(size=(bs,), device=self.accelerator.device) >= self.args.drop_labels_prob,
                    True,
                    False
                )
                encoder_hidden_states = torch.where(
                    mask.view(mask_adaptive_shape),
                    self.labels_embedding[labels],
                    0.0
                )

                losses = dict()
                loss = torch.scalar_tensor(0.0, dtype=torch.float32, device=self.accelerator.device)
                mse_loss = torch.scalar_tensor(0.0, dtype=torch.float32, device=self.accelerator.device)

                accumulated_list = [self.model]
                if classifier is not None:
                    accumulated_list.append(classifier)
                if self.discriminator is not None:
                    accumulated_list.append(self.discriminator)
                with self.accelerator.accumulate(accumulated_list):
                    noise_pred = self.model(model_input, timesteps, encoder_hidden_states, return_dict=False)[0]
                    mse_loss += F.mse_loss(noise_pred, noise)
                    loss += mse_loss
                    if (self.labels_metric is not None) and (self.task_counter > 0):
                        
                        if self.args.contrastive_loss:
                            c_labels = labels[labels >= min(self.curtask_labels)]
                            c_model_input = model_input[labels >= min(self.curtask_labels)]
                            c_timesteps = timesteps[labels >= min(self.curtask_labels)]
                            nearest_labels = self.nearest[c_labels - min(self.curtask_labels)]
                            c_noise_pred = self.model(
                                    c_model_input, c_timesteps, self.labels_embedding[c_labels], return_dict=False
                                )[0]

                            furthest_labels = self.furthest[c_labels - min(self.curtask_labels)]

                            c_model_input = torch.cat((c_model_input, c_model_input), dim=0)
                            c_timesteps = torch.cat((c_timesteps, c_timesteps), dim=0)
                            label_emb = torch.cat((self.labels_embedding[nearest_labels], self.labels_embedding[furthest_labels]), dim=0)


                            pred = self.model(
                                c_model_input, c_timesteps, label_emb, return_dict=False
                            )[0].detach().clone()
                            nearest_noise_pred = pred[:len(c_noise_pred)]
                            furthest_noise_pred = pred[len(c_noise_pred):]
                            # TODO normalize them !!!!
                            
                            cosine_triplet_loss = torch.mean(
                                F.triplet_margin_with_distance_loss(c_noise_pred.reshape(len(c_noise_pred), -1), nearest_noise_pred.reshape(len(c_noise_pred), -1), furthest_noise_pred.reshape(len(c_noise_pred), -1), distance_function = nn.CosineSimilarity() ,reduction='none') \
                                * c_timesteps[:len(c_noise_pred)].type(torch.float32)

                            )
                            # TODO try cosine embedding loss as well

                            loss+= self.args.rho * cosine_triplet_loss
                            losses['cosine_triplet_loss'] = cosine_triplet_loss

                        
                        elif self.args.diversity_loss:
                            c_labels = labels[labels >= min(self.curtask_labels)]
                            c_model_input = model_input[labels >= min(self.curtask_labels)]
                            c_timesteps = timesteps[labels >= min(self.curtask_labels)]
                            nearest_labels = self.nearest[c_labels - min(self.curtask_labels)]
                            c_noise_pred = self.model(
                                    c_model_input, c_timesteps, self.labels_embedding[c_labels], return_dict=False
                                )[0]
                            nearest_noise_pred = self.model(
                                    c_model_input, c_timesteps, self.labels_embedding[nearest_labels], return_dict=False
                                )[0].detach().clone()
                            mse_loss_rc = torch.mean(
                                F.mse_loss(c_noise_pred, nearest_noise_pred, reduction='none').mean(dim=tuple(range(1, len(model_input.shape)))) \
                                * c_timesteps.type(torch.float32)
                            )
                            loss += self.args.tau * mse_loss_rc
                            losses['mse_loss_rc'] = mse_loss_rc
                    
                    if self.args.ce_enhanced_weight > 0.0 or (self.args.ce_uncertainty_weight > 0.0 and self.task_counter > 0) or self.discriminator is not None:
                        previous_noisy_images = self.train_scheduler.my_step(noise_pred, timesteps, noisy_images).prev_sample
                        weight = (self.train_scheduler.config.num_train_timesteps - timesteps.type(torch.float32)) \
                                / self.train_scheduler.config.num_train_timesteps
                        weight = weight.to(self.accelerator.device, non_blocking=True)
                        
                        if self.args.ce_enhanced_weight > 0.0:
                            assert classifier is not None
                            ce_enhanced_loss = torch.scalar_tensor(0.0, dtype=torch.float32, device=self.accelerator.device)
                            classifier_outputs = classifier(previous_noisy_images, self.all_labels)
                            ce_enhanced_losses = F.cross_entropy(classifier_outputs, labels, reduction='none')
                            _weight = torch.where(mask, weight, 0.0).to(self.accelerator.device, non_blocking=True)
                            # _weight = weight.to(self.accelerator.device, non_blocking=True)
                            ce_enhanced_loss += torch.mean(ce_enhanced_losses * _weight)
                            loss += self.args.ce_enhanced_weight * ce_enhanced_loss
                            losses['ce_enhanced_loss'] = ce_enhanced_loss

                        if self.args.ce_uncertainty_weight > 0.0 and self.task_counter > 0:
                            assert classifier is not None
                            ce_uncertainty_loss = torch.scalar_tensor(0.0, dtype=torch.float32, device=self.accelerator.device)
                            classifier_outputs = classifier(previous_noisy_images, self.curtask_labels)
                            ce_uncertainty_losses = F.cross_entropy(
                                classifier_outputs,
                                torch.ones(size=(bs, len(self.curtask_labels)), device=self.accelerator.device) / len(self.curtask_labels),
                                reduction='none'
                            )
                            _weight = torch.where(mask & (labels < min(self.curtask_labels)), weight, 0.0).to(self.accelerator.device, non_blocking=True)
                            ce_uncertainty_loss += torch.mean(ce_uncertainty_losses * _weight)
                            loss += self.args.ce_uncertainty_weight * ce_uncertainty_loss
                            losses['ce_uncertainty_loss'] = ce_uncertainty_loss

                        if self.discriminator is not None:
                            g_loss = torch.scalar_tensor(0.0, dtype=torch.float32, device=self.accelerator.device)
                            d_loss = torch.scalar_tensor(0.0, dtype=torch.float32, device=self.accelerator.device)
                            if self.args.wgan:
                                self.discriminator.freeze()
                                if self.args.wgan_set_weight:
                                    g_loss += - torch.mean(
                                        self.discriminator(previous_noisy_images, self.labels_embedding[labels]).squeeze() \
                                        * weight
                                    )
                                else:
                                    g_loss += - torch.mean(
                                        self.discriminator(previous_noisy_images, self.labels_embedding[labels]).squeeze() \
                                        # * weight
                                    )
                                self.discriminator.unfreeze()

                                if self.step % self.args.wgan_n_critic == 0:
                                    d_real = torch.mean(
                                        self.discriminator(batch, self.labels_embedding[labels]).squeeze()
                                    )
                                    if self.args.wgan_set_weight:
                                        d_fake = torch.mean(
                                            self.discriminator(previous_noisy_images.detach().clone(), self.labels_embedding[labels]).squeeze() \
                                            * (1.0 - weight)
                                        )
                                        gradient_penalty = compute_gradient_penalty(
                                            self.discriminator,
                                            batch,
                                            previous_noisy_images.detach().clone(),
                                            self.labels_embedding[labels],
                                            weight
                                        )
                                    else:
                                        d_fake = torch.mean(
                                            self.discriminator(previous_noisy_images.detach().clone(), self.labels_embedding[labels]) \
                                            # * (1.0 - weight)
                                        )
                                        gradient_penalty = compute_gradient_penalty(
                                            self.discriminator,
                                            batch,
                                            previous_noisy_images.detach().clone(),
                                            self.labels_embedding[labels],
                                            # 1.0 - weight
                                        )
                                    d_pure_loss = - d_real + d_fake
                                    d_loss += d_pure_loss + self.args.wgan_gradient_penalty_weight * gradient_penalty
                                    losses['d_real'] = d_real
                                    losses['d_fake'] = d_fake
                                    losses['d_pure_loss'] = d_pure_loss
                                    losses['gradient_penalty'] = gradient_penalty
                            elif self.args.unet_discriminator:
                                real_labels = torch.ones_like(labels, dtype=torch.float32)
                                fake_labels = torch.zeros_like(labels, dtype=torch.float32)
                                self.discriminator.freeze()
                                g_loss += F.binary_cross_entropy_with_logits(
                                    self.discriminator(previous_noisy_images, timesteps, self.labels_embedding[labels], return_dict=False)[0].squeeze(),
                                    real_labels
                                )
                                self.discriminator.unfreeze()

                                real_loss = F.binary_cross_entropy_with_logits(
                                    self.discriminator(
                                        self.train_scheduler.add_noise(
                                            batch,
                                            noise,
                                            torch.where(timesteps >= 1, timesteps - 1, 0)
                                        ),
                                        timesteps,
                                        self.labels_embedding[labels],
                                        return_dict=False
                                    )[0].squeeze(),
                                    real_labels
                                )
                                fake_loss = F.binary_cross_entropy_with_logits(
                                    self.discriminator(previous_noisy_images.detach().clone(), timesteps, self.labels_embedding[labels], return_dict=False)[0].squeeze(),
                                    fake_labels
                                )
                                d_loss += (real_loss + fake_loss) / 2
                            else:
                                if self.args.random_label:
                                    real_labels = torch.rand_like(labels, dtype=torch.float32) * (weight / 2)
                                    fake_labels = torch.rand_like(labels, dtype=torch.float32) * (1.0 - weight) / 2 + (weight / 2 + 0.5)
                                else:
                                    real_labels = weight / 2
                                    fake_labels = weight / 2 + 0.5
                                self.discriminator.freeze()
                                g_loss += F.binary_cross_entropy_with_logits(
                                    self.discriminator(previous_noisy_images, self.labels_embedding[labels]).squeeze(),
                                    fake_labels
                                )
                                self.discriminator.unfreeze()

                                real_loss = F.binary_cross_entropy_with_logits(
                                    self.discriminator(batch, self.labels_embedding[labels]).squeeze(),
                                    torch.ones_like(labels, dtype=torch.float32)
                                )
                                fake_loss = F.binary_cross_entropy_with_logits(
                                    self.discriminator(previous_noisy_images.detach().clone(), self.labels_embedding[labels]).squeeze(),
                                    real_labels
                                )
                                d_loss += (real_loss + fake_loss) / 2
                            if self.step % self.args.n_critic == 0:
                                losses['g_loss'] = g_loss
                                loss += g_loss * self.args.g_loss_weight
                            losses['d_loss'] = d_loss

                    self.accelerator.backward(loss, retain_graph=True)
                    if self.accelerator.sync_gradients and self.args.max_grad_norm:
                        if self.args.use_discriminator and self.args.unet_discriminator:
                            self.accelerator.clip_grad_norm_(itertools.chain(*[self.model.parameters(), self.discriminator.parameters()]), 2.0)
                        else:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    if (self.discriminator is not None) and (self.step % self.args.wgan_n_critic == 0):
                        self.accelerator.backward(d_loss)
                        self.discriminator_optimizer.step()
                        self.discriminator_optimizer.zero_grad()
                        if self.discriminator_lr_scheduler is not None:
                            self.discriminator_lr_scheduler.step()
                    if self.args.image_condition in ['learn', 'learn_from_noise']:
                        counter = Counter(labels.cpu().numpy().tolist())
                        for label in self.curtask_labels:
                            if self.most_confidence[label].grad is not None:
                                self.most_confidence[label] = self.most_confidence[label] * (1.0 - self.args.image_condition_learn_L2) \
                                                            - self.most_confidence[label].grad * self.args.image_condition_learn_lr / counter[label]
                                self.most_confidence[label].grad = None
                if self.accelerator.sync_gradients and self.args.ema:
                    self.ema_model.step(self.model.parameters())

                losses['mse_loss'] = mse_loss
                losses['loss'] = loss
                if self.accelerator.is_main_process:
                    if self.step % self.args.log_interval == 0:
                        logger.dumpkvs()
                        log_loss_dict(losses=losses)
                    if self.step % self.args.save_interval == 0:
                        self.save()
                # if (self.step % 5000 == 0) and (self.step >= 15000) and self.args.toy:
                if (self.step % 500 == 0) and self.args.toy:
                    self.toy_sample(
                        os.path.join(self.curtask_path, 'toy_sample_step_{}'.format(self.step)),
                        self.curtask_labels
                    )
                if self.step >= num_steps:
                    keep_training = False
                    break
        if self.accelerator.is_main_process:
            self.save()

    def toy_sample(self, samples_path, label_order_list):
        print('Sample imgs ...')
        print("sampling...")
        all_images = []
        all_labels = []
        num_samples = 5 * len(label_order_list)
        label_counter = 0
        print(f"created 0 / {num_samples} samples", end="\r")
        while sum([s.shape[0] for s in all_images]) < num_samples:
            rnd_label = []
            while (len(rnd_label) < 500) and (label_counter < len(label_order_list)):
                rnd_label.extend([label_order_list[label_counter] for _ in range(5)])
                label_counter += 1
            rnd_label = np.array(rnd_label)
            rnd_label = torch.from_numpy(rnd_label).to(dist_util.dev())
            encoder_hidden_states = self.labels_embedding[rnd_label]
            zeros = torch.zeros_like(encoder_hidden_states)
            noise = torch.randn((rnd_label.shape[0], 3, self.args.image_size, self.args.image_size)).to(dist_util.dev())
            if self.most_confidence is not None:
                image_conditions = torch.stack(
                    [self.most_confidence[label.item()] for label in rnd_label],
                    dim=0
                ).to(dist_util.dev())
            sample = noise
            for t in tqdm(
                self.inference_scheduler.timesteps,
                desc=f"created {sum([s.shape[0] for s in all_images])} / {num_samples} samples",
                leave=False
            ):
                if self.most_confidence is not None:
                    model_input = torch.cat((image_conditions, sample), dim=1)
                else:
                    model_input = sample
                with torch.no_grad():
                    if self.args.w > 0.0:
                        noisy_residual = (self.args.w + 1) * self.model(model_input, t, encoder_hidden_states).sample \
                                        - self.args.w * self.model(model_input, t, zeros).sample
                    else:
                        noisy_residual = self.model(model_input, t, encoder_hidden_states).sample
                previous_noisy_sample = self.inference_scheduler.step(noisy_residual, t, sample).prev_sample
                sample = previous_noisy_sample
            sample = (sample / 2 + 0.5).clamp(0, 1)
            sample = (sample.permute(0, 2, 3, 1) * 255).round().to(torch.uint8)
            sample = sample.contiguous()
            gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            rnd_label = rnd_label.to(dist_util.dev())
            gathered_labels = [torch.zeros_like(rnd_label) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, rnd_label)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        print(f"created {num_samples} / {num_samples} samples")
        utils.create_dir(samples_path)
        toy_save_to_JPEG(num_samples, all_images, all_labels, samples_path)


def toy_save_to_JPEG(num_samples,all_images,all_labels,output_path):
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: num_samples]
    class_to_idx = set()
    classes = set()
    img_path_label_list = []
    if dist.get_rank() == 0:
        num_fig = arr.shape[0]
        for fig_count in range(num_fig):
            class_name = label_arr[fig_count]
            class_to_idx.add(label_arr[fig_count])
            classes.add(class_name)
            filepath = os.path.join(output_path,
                                    "{}_generator{}.JPEG".format(class_name, fig_count))
            img_path_label_list.append((filepath, label_arr[fig_count]))
            output_img(filepath, np.array(arr[fig_count]))
    dist.barrier()
    class_to_idx = list(class_to_idx)
    class_to_idx.sort()
    classes = list(classes)
    classes.sort()
    print("sampling complete")
    return img_path_label_list, classes, class_to_idx


def compute_gradient_penalty(discriminator, real_samples, fake_samples, labels_embedding, weight=1.0):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha_shape = [real_samples.shape[0]] + [1] * (len(real_samples.shape) - 1)
    alpha = torch.rand(size=tuple(alpha_shape), device=real_samples.device)
    # # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates, labels_embedding)
    fake = torch.autograd.Variable(torch.ones(size=(real_samples.shape[0],), device=real_samples.device), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = torch.mean(((gradients.norm(2, dim=1) - 1) ** 2) * weight)
    return gradient_penalty
    