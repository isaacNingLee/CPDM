import os
import configparser
import sys
import random
import time

import numpy
import warnings
import shutil
import datetime
import copy
from torchvision import models

import torch.nn as nn
import torch

import subprocess
try:
    import cPickle as pickle
except:
    import pickle

def savepickle(data, file_path):
    mkdir_p(os.path.dirname(file_path), delete=False)
    print('pickle into', file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def mkdir_p(path, delete=False, print_info=True):
    if path == '': return

    if delete:
        subprocess.call(('rm -r ' + path).split())
    if not os.path.exists(path):
        if print_info:
            print('mkdir -p  ' + path)
        subprocess.call(('mkdir -p ' + path).split())

def init():
    set_random()

def get_root_src_path():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))


def get_parsed_config():

    config = configparser.ConfigParser()
    src_path = get_root_src_path()
    config.read(os.path.join(src_path, 'config.init'))

    for key, path in config['DEFAULT'].items():
        if '.' == read_from_config(config, key).split(os.path.sep)[0]:
            config['DEFAULT'][key] = os.path.join(src_path, read_from_config(config, key)[2:])
            create_dir(config['DEFAULT'][key], key)

    return config


def read_from_config(config, key_value):
    return os.path.normpath(config['DEFAULT'][key_value]).replace("'", "").replace('"', "")


def parse_str_to_floatlist(str_in):
    return list(map(float, str_in.replace(' ', '').split(',')))


def set_random(seed=7):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_now():
    return str(datetime.datetime.now().date()) + "_" + ':'.join(str(datetime.datetime.now().time()).split(':')[:-1])


def replace_last_classifier_layer(model, out_dim):
    last_layer_index = str(len(model.classifier._modules) - 1)
    num_ftrs = model.classifier._modules[last_layer_index].in_features
    model.classifier._modules[last_layer_index] = nn.Linear(num_ftrs, out_dim).cuda()
    return model


def get_first_FC_layer(seq_module):

    for module in seq_module.modules():
        if isinstance(module, nn.Linear):
            return module
    raise Exception("No LINEAR module in sequential found...")


def save_cuda_mem_req(out_dir, out_filename='cuda_mem_req.pth.tar'):

    out_dir = os.path.dirname(out_dir)
    out_path = os.path.join(out_dir, out_filename)

    mem_req = {}
    mem_req['cuda_memory_allocated'] = torch.cuda.memory_allocated(device=None)
    mem_req['cuda_memory_cached'] = torch.cuda.memory_cached(device=None)

    torch.save(mem_req, out_path)
    print("SAVED CUDA MEM REQ {} to path: {}".format(mem_req, out_path))


def save_preprocessing_time(out_dir, time, out_filename='preprocess_time.pth.tar'):
    if os.path.isfile(out_dir):
        out_dir = os.path.dirname(out_dir)
    out_path = os.path.join(out_dir, out_filename)
    torch.save(time, out_path)
    print_timing(time, "PREPROCESSING")


def print_timing(timing, title=''):
    title = title.strip() + ' ' if title != '' else title
    print("{}TIMING >>> {} <<<".format(title, str(timing)))


def reset_stats():
    try:
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
        print("RESETTED STATS")
    except:
        print("PYTORCH VERSION NOT ENABLING RESET STATS")


def print_stats():
    print("CUDA MAX MEM ALLOC >>> {} <<<".format(torch.cuda.max_memory_allocated()))
    print("CUDA MAX MEM CACHE >>> {} <<<".format(torch.cuda.max_memory_cached()))


def get_exp_name(args, method):
    exp_name = ["dm={}".format(args.drop_margin),
                "df={}".format(args.decaying_factor),
                "e={}".format(args.num_epochs),
                "bs={}".format(args.batch_size)]
    if args.method_name in ["DDGR", "CPDM"]:
        exp_name.append("diffusion_steps={}".format(args.diffusion_steps))
        exp_name.append("num_samples={}".format(args.num_samples))
        if args.method_name in ["CPDM"]:
            if args.ddpm:
                exp_name.append("ddpm={}".format(args.num_inference_timesteps))
            else:
                exp_name.append("ddim={}".format(args.num_inference_timesteps))
            if args.diffusion_without_replay:
                exp_name.append("diffusion_without_replay")
            if args.embed_condition != 'default':
                exp_name.append("embed_condition={}".format(args.embed_condition))
            if args.ce_enhanced_weight > 0.0:
                exp_name.append("ce_enhanced_weight={}".format(args.ce_enhanced_weight))
            if args.ce_uncertainty_weight > 0.0:
                exp_name.append("ce_uncertainty_weight={}".format(args.ce_uncertainty_weight))
            if args.use_discriminator:
                exp_name.append("discriminator")
                exp_name.append("g_loss_weight={}".format(args.g_loss_weight))
                if args.wgan:
                    exp_name.append("wgan")
                    exp_name.append("wgan_gradient_penalty_weight={}".format(args.wgan_gradient_penalty_weight))
                    exp_name.append("wgan_n_critic={}".format(args.wgan_n_critic))
                    if args.wgan_set_weight:
                        exp_name.append("wgan_set_weight")
                elif args.random_label:
                    exp_name.append("random_label")
                elif args.unet_discriminator:
                    exp_name.append("unet")
                exp_name.append("{}".format(args.n_critic))
            if args.nearest_label is not None:
                exp_name.append("nearest_label={}".format(args.nearest_label))
                exp_name.append("tau={}".format(args.tau))
            if args.image_condition != 'none':
                exp_name.append("image_condition={}".format(args.image_condition))
                if args.image_condition in ['learn', 'learn_from_noise']:
                    if args.learning_start != 'most':
                        exp_name.append(args.learning_start)
                    exp_name.append("image_condition_learn_lr={}".format(args.image_condition_learn_lr))
                    exp_name.append("image_condition_learn_L2={}".format(args.image_condition_learn_L2))
            if args.max_grad_norm:
                exp_name.append("max_grad_norm")
            if args.ema:
                exp_name.append("ema")
                exp_name.append("ema_rate={}".format(args.ema_rate))
            if args.toy:
                exp_name.append("toy")
    if args.weight_decay != 0:
        exp_name.append("L2={}".format(args.weight_decay))
    for h_key, h_val in method.hyperparams.items():
        exp_name.append("{}={}".format(h_key, h_val))
    if hasattr(method, 'static_hyperparams'):
        for h_key, h_val in method.static_hyperparams.items():
            exp_name.append("{}={}".format(h_key, h_val))
    exp_name = '_'.join(exp_name)
    return exp_name


def get_starting_model_path(root_path, dataset_obj, model_name, exp_name, method_name, append_filename=True):


    path = os.path.join(root_path, dataset_obj.train_exp_results_dir, method_name, model_name)
    path = os.path.join(path, 'gridsearch', 'first_task_basemodel', exp_name, 'task_1', 'TASK_TRAINING')

    if append_filename:
        path = os.path.join(path, 'best_model.pth.tar')
    return path


def get_test_results_path(root_path, dataset_obj, method_name, model_name,
                          gridsearch_name=None, exp_name=None, subset='test', create=False):

    path = os.path.join(root_path, 'results', dataset_obj.test_results_dir, method_name, model_name)

    if gridsearch_name is not None:
        path = os.path.join(path, gridsearch_name)
    if exp_name is not None:
        if subset != 'test':
            exp_name = '{}_{}'.format(exp_name, subset)
        path = os.path.join(path, exp_name)
    if create:
        create_dir(path)
    return path

def get_records_path(root_path, dataset_obj, method_name, model_name,
                          gridsearch_name=None, exp_name=None, subset='test', create=False):

    rp = root_path.replace('test', 'records')
    path = os.path.join(rp, 'results', dataset_obj.test_results_dir, method_name, model_name)

    if gridsearch_name is not None:
        path = os.path.join(path, gridsearch_name)
    if exp_name is not None:
        if subset != 'test':
            exp_name = '{}_{}'.format(exp_name, subset)
        path = os.path.join(path, exp_name)
    if create:
        create_dir(path)
    return path


def get_test_results_filename(method_name, task_number):
    return "test_method_performances" + method_name + str(int(task_number) - 1) + ".pth"


def get_train_results_path(tr_results_root_path, dataset_obj, method_name=None, model_name=None, gridsearch=True,
                           gridsearch_name=None, exp_name=None, filename=None, create=False,args=None):

    if create and filename is not None:
        print("WARNING: filename is not being created, but superdirs are if not existing.")

    path = os.path.join(tr_results_root_path, dataset_obj.train_exp_results_dir)
    if method_name is not None:
        if 'IMM' in method_name:
            path = os.path.join(path, args.method_name)
        else:
            path = os.path.join(path, method_name)
    if model_name is not None:
        path = os.path.join(path, model_name)
    if gridsearch:
        path = os.path.join(path, 'gridsearch')
    if gridsearch_name is not None:
        path = os.path.join(path, gridsearch_name)
    if exp_name is not None:
        path = os.path.join(path, exp_name)
    if create:
        create_dir(path)
    if filename is not None:
        path = os.path.join(path, filename)
    if args.contrastive_loss:#####
        path = os.path.join(path, 'contrastive_loss')
    elif args.diversity_loss:#########
        path = os.path.join(path, 'diversity_loss')
    else:
        path = os.path.join(path, 'no_add_loss')
    return path


def get_perf_output_filename(method_name, dataset_index, joint_full_batch=False):

    if joint_full_batch:
        return 'test_method_performancesJOINT_FULL_BATCH.pth'
    else:
        return 'test_method_performances' + method_name + str(dataset_index) + ".pth"


def get_hyperparams_output_filename():
    return 'hyperparams.pth.tar'


def get_prev_heads(prev_head_model_paths, head_layer_idx):

    if not isinstance(prev_head_model_paths, list):
        prev_head_model_paths = [prev_head_model_paths]

    if len(prev_head_model_paths) == 0:
        return []

    heads = []

    for head_model_path in prev_head_model_paths:
        previous_model_ft = torch.load(head_model_path)
        if isinstance(previous_model_ft, dict):
            previous_model_ft = previous_model_ft['model']

        head = previous_model_ft.classifier._modules[head_layer_idx]
        assert isinstance(head, torch.nn.Linear), type(head)
        heads.append(copy.deepcopy(head.cuda()))
        del previous_model_ft

    return heads


def get_immediate_subdirectories(parent_dir_path, path_mode=False, sort=False):

    if not path_mode:
        dirs = [name for name in os.listdir(parent_dir_path)
                if os.path.isdir(os.path.join(parent_dir_path, name))]
    else:
        dirs = [os.path.join(parent_dir_path, name) for name in os.listdir(parent_dir_path)
                if os.path.isdir(os.path.join(parent_dir_path, name))]
    if sort:
        dirs.sort()
    return dirs


def get_relative_path(absolute_path, segments=1):

    return os.path.sep.join(list(filter(None, absolute_path.split(os.path.sep)))[-segments:])


def attempt_move(src_path, dest_path):
    try:
        shutil.move(src_path, dest_path)
    except Exception:
        if not os.path.exists(dest_path):
            print("Dest path not existing: ", dest_path)
        if not os.path.exists(src_path):
            print("SRC path not existing: ", src_path)


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def append_to_file(filepath, msg):
    write_mode = 'w' if not os.path.exists(filepath) else 'a'
    with open(filepath, write_mode) as f:
        f.write(msg + "\n")


def rm_dir(path_to_dir, delete_subdirs=True, content_only=True):
    if path_to_dir is not None and os.path.exists(path_to_dir):
        for the_file in os.listdir(path_to_dir):
            file_path = os.path.join(path_to_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path) and delete_subdirs:
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        print("REMOVED CONTENTS FROM DIR: ", path_to_dir)

        if not content_only:
            try:
                shutil.rmtree(path_to_dir)
            except Exception as e:
                print(e)
            print("REMOVED DIR AND ALL ITS CONTENTS: ", path_to_dir)


def create_dir(dirpath, print_description=""):
    try:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, mode=0o750)
    except Exception as e:
        print(e)
        print("ERROR IN CREATING ", print_description, " PATH:", dirpath)


def create_symlink(src, ln):
    if not os.path.exists(ln):
        create_dir(os.path.dirname(ln))
        os.symlink(src, ln)


def float_to_scientific_str(value, sig_count=1):

    from decimal import Decimal
    format_str = '%.' + str(sig_count) + 'E'
    return format_str % Decimal(value)


def debug_add_sys_args(string_cmd, set_debug_option=True):

    warnings.warn("=" * 20 + "SEVERE WARNING: DEBUG CMD ARG USED, TURN OF FOR REAL RUNS" + "=" * 20)
    args = string_cmd.split(' ')
    if set_debug_option:
        args.insert(0, "--debug")
    for arg in args:
        sys.argv.append(str(arg))
