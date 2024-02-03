import torch
import torch.nn as nn
import os
import time

import torch.optim as optim
from torch.autograd import Variable
import src.utilities.utils as utils
import numpy as np
import json
from copy import deepcopy

import src.methods.CPDM.train_CPDM as CPDM_train
from src.data.imgfolder import ImageFolderTrainVal

import torch.distributed as dist
import torch.nn.functional as F

from src.methods.CPDM.diffusion import dist_util, logger
from src.methods.CPDM.diffusion.image_datasets import load_data
from src.methods.CPDM.diffusion.diffusion import Diffusion

import src.framework.new_eval as test

from src.score.fid import get_statistics
from src.score.both import get_inception_and_fid_score
from src.score.cmmd import compute_cmmd

TRAINING_DONE_TOKEN = 'DIFFUSION.TRAINING.DONE'

def fine_tune_train_CPDM(dataset_path, args,previous_task_model_path, exp_dir, task_counter, labels_embedding, labels_metric, batch_size=200,
                               num_epochs=100, lr=0.0008, weight_decay=0, head_shared=False,
                               saving_freq=5,classifier_type="pretrained",
                               manager=None, test_ds_path=[], test_model_paths=[]):
    start_mem = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    if not os.path.exists(exp_dir):
        print("Going to exp_dir=", exp_dir)
        os.makedirs(exp_dir)

    dist_util.setup_dist()
    utils.create_dir(os.path.join(exp_dir,"logger_file"))
    logger.configure(dir=os.path.join(exp_dir,"logger_file"))
    since = time.time()
    use_cuda = torch.cuda.is_available()
    dsets = torch.load(dataset_path)

    ## TODO compute ground truth metrics
    # m1, s1 = get_statistics(
    #     images=dsets['train'].samples[0], num_images=None, batch_size=50,
    #     use_torch=False, verbose=False, parallel=False)


    
    print(dataset_path)

    dset_sizes = {x: len(dsets[x]) for x in ['train', 'val']}
    dset_classes = dsets['train'].classes

    if task_counter == 0:
        print("Loading prev model from path: ", previous_task_model_path)
        model_ft = torch.load(previous_task_model_path)
        if not head_shared:
            last_layer_index = str(len(model_ft.classifier._modules) - 1)
            num_ftrs = model_ft.classifier._modules[last_layer_index].in_features
            model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))
            print("NEW FC CLASSIFIER HEAD with {} units".format(len(dset_classes)))
        criterion = nn.CrossEntropyLoss()
        temp_tensor = torch.zeros((2,2)).cpu()
        if use_cuda:
            criterion.cuda()
            model_ft.cuda()
            temp_tensor = Variable(temp_tensor.cuda())

        device = temp_tensor.device
        tg_params = model_ft.parameters()
        optimizer_ft = optim.SGD(tg_params, lr=lr, momentum=0.9, weight_decay=weight_decay)

        print("creating model and diffusion... args.num_classes={}".format(args.num_classes))
        pl_list = dsets['train'].get_allfigs_filepath()
        dset_path_list = [item[0] for item in pl_list]
        labels_list = [item[1] for item in pl_list]
        temp_ll = deepcopy(labels_list)
        combine_label_list = list(sorted(set(temp_ll)))
        resume = os.path.join(exp_dir, 'epoch.pth.tar')
        test_ds_path.append(dataset_path)
        model_ft, best_acc = CPDM_train.train_model(args=args,model=model_ft, criterion=criterion,
                                                    optimizer=optimizer_ft,lr = lr,
                                                    dsets = dsets, batch_size = batch_size, dset_sizes = dset_sizes,
                                                    use_cuda = use_cuda, num_epochs =num_epochs,task_counter = task_counter,
                                                    exp_dir=exp_dir,resume=resume, saving_freq=saving_freq,device=device,
                                                    combine_label_list=combine_label_list,gen_dset=None,
                                                    test_ds_path=test_ds_path)
        test_model_paths.append(os.path.join(exp_dir, 'best_model.pth.tar'))
        combine_order_name = os.path.join(exp_dir,
                                        "generator_label_list_order_seed={}.pkl".format(args.CI_order_rndseed))
        print("Order name:{}".format(combine_order_name))
        print("Current label_list: {}".format(combine_label_list))
        utils.savepickle(combine_label_list, combine_order_name)
        test.main(args, manager, test_ds_path, test_model_paths)
        
        if (args.max_task_count > 1) and (not os.path.exists(os.path.join(exp_dir, TRAINING_DONE_TOKEN))):
            print("CPDM training preparing")
            diffusion = Diffusion(
                num_train_timesteps=args.diffusion_steps,
                beta_schedule='linear',
                num_inference_timesteps=args.num_inference_timesteps,
                args=args,
                task_counter=task_counter,
                labels_embedding=labels_embedding,
                labels_metric=labels_metric[args.nearest_label] if args.nearest_label is not None else None,
                curtask_labels=combine_label_list,
                all_labels=combine_label_list,
                curtask_path=exp_dir
            )
            model_ft = torch.load(os.path.join(exp_dir, 'best_model.pth.tar'))
            start_preprocess_time = time.time()
            generator_data = load_data(data_dir=dset_path_list,
                                batch_size=args.diffusion_batch_size,
                                image_size=args.image_size,
                                classes_list=labels_list,
                                class_cond=args.class_cond,)
            print("CPDM training starts")
            if (args.ce_enhanced_weight > 0.0) or (args.ce_uncertainty_weight > 0.0):
                classifier = deepcopy(model_ft)
                classifier.freeze()
            else:
                classifier = None
            if args.image_condition != 'none':
                if (not os.path.exists(os.path.join(exp_dir, 'image-condition', 'start', 'tensors'))) \
                    or (len(os.listdir(os.path.join(exp_dir, 'image-condition', 'start', 'tensors'))) < len(combine_label_list)):
                    if args.image_condition == 'learn_from_noise':
                        most_confidence = CPDM_train.do_random(args, combine_label_list)
                    elif args.learning_start == 'less':
                        most_confidence = CPDM_train.do_find_less_confidence(args, model_ft, dsets['train'], batch_size, use_cuda, combine_label_list, combine_label_list)
                    elif args.learning_start == 'avg':
                        most_confidence = CPDM_train.do_find_avg(args, model_ft, dsets['train'], batch_size, use_cuda, combine_label_list, combine_label_list)
                    else:
                        most_confidence = CPDM_train.do_find_most_confidence(args, model_ft, dsets['train'], batch_size, use_cuda, combine_label_list, combine_label_list)
                    os.makedirs(os.path.join(exp_dir, 'image-condition', 'start', 'tensors'), exist_ok=True)
                    for key, value in most_confidence.items():
                        torch.save(value, os.path.join(exp_dir, 'image-condition', 'start', 'tensors', f'{key}.pt'))
                diffusion.update_most_confidence()
            diffusion.train(dataset=generator_data, batch_size=args.diffusion_batch_size, num_steps=args.lr_anneal_steps, classifier=classifier)
            if args.image_condition != 'none':
                diffusion.save_most_confidence()
            update_stage_times(0, time.time() - start_preprocess_time, filename=exp_dir + '/' + 'diffusion_training_times.json')
            torch.save('', os.path.join(exp_dir, TRAINING_DONE_TOKEN))

    else:
        print('load model')
        model_ft = torch.load(previous_task_model_path)
        if not head_shared:
            last_layer_index = str(len(model_ft.classifier._modules) - 1)
            num_ftrs = model_ft.classifier._modules[last_layer_index].in_features
            model_ft.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, len(dset_classes))
            print("NEW FC CLASSIFIER HEAD with {} units".format(len(dset_classes)))
        criterion = nn.CrossEntropyLoss()
        temp_tensor = torch.zeros((2, 2)).cpu()
        if use_cuda:
            criterion.cuda()
            model_ft.cuda()
            temp_tensor = Variable(temp_tensor.cuda())
        device = temp_tensor.device
        tg_params = model_ft.parameters()
        optimizer_ft = optim.SGD(tg_params, lr=lr, momentum=0.9, weight_decay=weight_decay)

        print("Loard model and diffusion... args.num_classes={}".format(args.num_classes))

        order_root_path = os.path.dirname(previous_task_model_path)
        order_name = os.path.join(order_root_path,
                                  "generator_label_list_order_seed={}.pkl".format(args.CI_order_rndseed))
        print("Order name:{}".format(order_name))
        label_order_list = utils.unpickle(order_name)
        pl_list = dsets['train'].get_allfigs_filepath()
        dset_path_list = [item[0] for item in pl_list]
        labels_list = [item[1] for item in pl_list]
        temp_ll = deepcopy(labels_list)
        temp_ll.extend(label_order_list)
        combine_label_list = list(sorted(set(temp_ll)))
        samples_path = os.path.join(exp_dir, 'samples')
        generated = os.path.exists(samples_path) or (
            os.path.exists(os.path.join(exp_dir,"generator_img_path_label_list_stage.pkl")) and \
            os.path.exists(os.path.join(exp_dir,"generator_classes_stage.pkl")) and \
            os.path.exists(os.path.join(exp_dir,"generator_class_to_idx_stage.pkl"))
        )
        if (not generated):
            diffusion = Diffusion(
                num_train_timesteps=args.diffusion_steps,
                beta_schedule='linear',
                num_inference_timesteps=args.num_inference_timesteps,
                args=args,
                task_counter=task_counter,
                labels_embedding=labels_embedding,
                labels_metric=labels_metric[args.nearest_label] if args.nearest_label is not None else None,
                curtask_labels=list(sorted(set(labels_list))),
                all_labels=combine_label_list,
                prevtask_path=order_root_path,
                curtask_path=exp_dir
            )
            sample_start_time = time.time()
            generator_img_path_label_list, generator_classes, generator_class_to_idx = diffusion.sample(
                samples_path=samples_path,
                label_order_list=label_order_list,
                dataset_path=dataset_path
            )
            utils.savepickle(data=generator_img_path_label_list,file_path=os.path.join(exp_dir,"generator_img_path_label_list_stage.pkl"))
            utils.savepickle(data=generator_classes,file_path=os.path.join(exp_dir, "generator_classes_stage.pkl"))
            utils.savepickle(data=generator_class_to_idx, file_path=os.path.join(exp_dir, "generator_class_to_idx_stage.pkl"))
            sample_end_time = time.time()
            update_stage_times(0, sample_end_time - sample_start_time, filename=exp_dir + '/' + 'sample_times.json')
        else:
            print(print('Samples are in {}'.format(samples_path)))
            generator_img_path_label_list = utils.unpickle(os.path.join(exp_dir,"generator_img_path_label_list_stage.pkl"))
            generator_classes = utils.unpickle(os.path.join(exp_dir,"generator_classes_stage.pkl"))
            generator_class_to_idx = utils.unpickle(os.path.join(exp_dir,"generator_class_to_idx_stage.pkl"))
        gen_dset = combine_data_loader(args,
                                        generator_img_path_label_list,
                                        generator_classes,
                                        generator_class_to_idx,
                                        dsets)

        # TODO compute new metrics
        gen_dset_loaders_temp = {x: torch.utils.data.DataLoader(gen_dset[x], batch_size=batch_size, num_workers=4,
                                            shuffle=True, pin_memory=True, persistent_workers=True)
                for x in ['train', 'val']}
        gen_ziploaders_temp =  enumerate(gen_dset_loaders_temp['train'])
        images = []
        labels = []
        for _, (inputs, labels_temp) in gen_ziploaders_temp:
            for (img, label) in zip(inputs, labels_temp):
                images.append(img)
                labels.append(label)

        print(f'number of replay images: {len(images)}')
        parts = dataset_path.split('/')
        extracted_path = '/'.join(parts[:-2]) if len(parts) > 2 else dataset_path
        fid_cache = os.path.join(extracted_path, 'stats', 'task_{}_stats.npz'.format(task_counter))
        print(f"Statistics Task {task_counter}\n")
        try:
            is_score, fid_score = get_inception_and_fid_score(images, labels,  fid_cache, num_images=None, splits=10, batch_size=50) # 6, 31
            cmmd = compute_cmmd(dataset_path, samples_path, batch_size=50, max_count=-1)
        except ValueError:
            print(f'Error in computing IS and FID for task {task_counter}')
        print()

        # parts = dataset_path.split('/')
        # extracted_path = '/'.join(parts[:-2]) if len(parts) > 2 else dataset_path
        
        # is_list = []
        # fid_list = []

        # print("Statistics\n")
        # for class_id in range(100):

        #     fid_cache = os.path.join(extracted_path, 'stats', f'class_{class_id}_stats.npz')
        #     is_score, fid_score = get_inception_and_fid_score(images[class_id], labels[class_id],  fid_cache, num_images=None, splits=10, batch_size=50)

        #     is_list.append(is_score)
        #     fid_list.append(fid_score)

        #     print(f'Class {class_id}: IS: {is_score}, FID: {fid_score}')
        
        # print(f'Average IS: {np.mean(is_list)}, Average FID: {np.mean(fid_list)}')

        _labels_list = deepcopy(labels_list)
        if not args.diffusion_without_replay:
            gen_pl_list = gen_dset['train'].get_allfigs_filepath()
            gen_dset_path_list = [item[0] for item in gen_pl_list]
            gen_labels_list = [item[1] for item in gen_pl_list]
            dset_path_list.extend(gen_dset_path_list)
            labels_list.extend(gen_labels_list)
        resume = os.path.join(exp_dir, 'epoch.pth.tar')
        test_ds_path.append(dataset_path)




        
        model_ft, best_acc = CPDM_train.train_model(args=args,model=model_ft, criterion=criterion,
                                                    optimizer=optimizer_ft,lr = lr,
                                                    dsets = dsets, batch_size = batch_size, dset_sizes = dset_sizes,
                                                    use_cuda = use_cuda, num_epochs = num_epochs,task_counter = task_counter,
                                                    exp_dir=exp_dir,resume=resume, saving_freq=saving_freq,device=device,
                                                    combine_label_list=combine_label_list,gen_dset=gen_dset,
                                                    test_ds_path=test_ds_path)
        test_model_paths.append(os.path.join(exp_dir, 'best_model.pth.tar'))
        combine_order_name = os.path.join(exp_dir,
                                        "generator_label_list_order_seed={}.pkl".format(args.CI_order_rndseed))
        print("Order name:{}".format(combine_order_name))
        print("Current label_list: {}".format(combine_label_list))
        utils.savepickle(combine_label_list, combine_order_name)
        test.main(args, manager, test_ds_path, test_model_paths)

        dont_train_diffusion = (task_counter == args.max_task_count - 1)
        if (not dont_train_diffusion) and (not os.path.exists(os.path.join(exp_dir, TRAINING_DONE_TOKEN))):
            print("CPDM training preparing")
            diffusion = Diffusion(
                num_train_timesteps=args.diffusion_steps,
                beta_schedule='linear',
                num_inference_timesteps=args.num_inference_timesteps,
                args=args,
                task_counter=task_counter,
                labels_embedding=labels_embedding,
                labels_metric=labels_metric[args.nearest_label] if args.nearest_label is not None else None,
                curtask_labels=list(sorted(set(_labels_list))),
                all_labels=combine_label_list,
                prevtask_path=order_root_path,
                curtask_path=exp_dir
            )
            model_ft = torch.load(os.path.join(exp_dir, 'best_model.pth.tar'))
            start_preprocess_time = time.time()
            generator_data = load_data(data_dir=dset_path_list,
                                    batch_size=args.diffusion_batch_size,
                                    image_size=args.image_size,
                                    classes_list=labels_list,
                                    class_cond=args.class_cond,)
            print("CPDM training starts")
            if (args.ce_enhanced_weight > 0.0) or (args.ce_uncertainty_weight > 0.0):
                classifier = deepcopy(model_ft)
                classifier.freeze()
            else:
                classifier = None
            if args.image_condition != 'none':
                if (not os.path.exists(os.path.join(exp_dir, 'image-condition', 'start', 'tensors'))) \
                    or (len(os.listdir(os.path.join(exp_dir, 'image-condition', 'start', 'tensors'))) < len(combine_label_list)):
                    if args.image_condition == 'learn_from_noise':
                        most_confidence = CPDM_train.do_random(args, list(sorted(set(labels_list))))
                    else:
                        most_confidence = CPDM_train.do_find_most_confidence(args, model_ft, dsets['train'], batch_size, use_cuda, combine_label_list, list(sorted(set(labels_list))))
                    os.makedirs(os.path.join(exp_dir, 'image-condition', 'start', 'tensors'), exist_ok=True)
                    for key, value in most_confidence.items():
                        torch.save(value, os.path.join(exp_dir, 'image-condition', 'start', 'tensors', f'{key}.pt'))
                diffusion.update_most_confidence()
            diffusion.train(dataset=generator_data, batch_size=args.diffusion_batch_size, num_steps=args.lr_anneal_steps, classifier=classifier)
            if args.image_condition != 'none':
                diffusion.save_most_confidence()
            update_stage_times(0, time.time() - start_preprocess_time, filename=exp_dir + '/' + 'diffusion_training_times.json')
            torch.save('', os.path.join(exp_dir, TRAINING_DONE_TOKEN))

    time_cost = time.time() - since
    save_runningtime(str(time_cost), filename=exp_dir + '/' + 'runningtime.txt')
    end_mem = torch.cuda.max_memory_allocated()
    save_runningtime(str(start_mem)+","+str(end_mem),filename=exp_dir + '/' + 'mem.txt')
    combine_order_name = os.path.join(exp_dir,
                                      "generator_label_list_order_seed={}.pkl".format(args.CI_order_rndseed))
    print("Order name:{}".format(combine_order_name))
    print("Current label_list: {}".format(combine_label_list))
    utils.savepickle(combine_label_list, combine_order_name)
    return model_ft, best_acc

def combine_data_loader(args, generator_img_path_label_list,generator_classes,generator_class_to_idx,dset):
    gimg = deepcopy(generator_img_path_label_list)
    gimg_split = {
        'train': [],
        'val': []
    }
    for _class in generator_class_to_idx:
        cls_gimg = [img for img in gimg if img[1] == _class]
        gimg_split['train'].extend(cls_gimg[:round(0.8*len(cls_gimg))])
        gimg_split['val'].extend(cls_gimg[round(0.8*len(cls_gimg)):])
    combine_dset = {}
    for x in ['train', 'val']:
        imgs = deepcopy(gimg_split[x])
        root = deepcopy(dset[x].get_root())
        classes = deepcopy(generator_classes)
        classes.sort()
        class_to_idx = deepcopy(generator_class_to_idx)
        class_to_idx.sort()
        transform = deepcopy(dset[x].get_trans())
        combine_dset[x] = ImageFolderTrainVal(root,None,transform=transform,classes=classes,class_to_idx=class_to_idx,
                                              imgs=imgs)
    return combine_dset


def save_runningtime(time_elapsed,filename='runningtime.txt'):
    File = open(filename,'w')
    File.write(time_elapsed)
    File.close()

def update_stage_times(stage, sample_time, filename):
    if os.path.exists(filename):
        OldFile = open(filename, 'r')
        sample_times = json.load(OldFile)
        OldFile.close()
    else:
        sample_times = dict()
    File = open(filename, 'w')
    sample_times[stage] = sample_time
    json.dump(sample_times, File)
    File.close()
