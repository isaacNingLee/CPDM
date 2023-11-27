import sys
import torch
import os
import traceback
import time

import src.data.dataset as datasets
import src.methods.method as methods
import src.utilities.utils as utils
import warnings
warnings.filterwarnings('ignore')


def main(args, manager, ds_paths, model_paths):

    args.test_max_task_count = manager.dataset.task_count if args.test_max_task_count is None else args.test_max_task_count
    ds_paths = ds_paths[0] if len(ds_paths) == 1 and isinstance(ds_paths[0], list) else ds_paths  # Joint

    args.out_path = utils.get_test_results_path(args.test_results_root_path, manager.dataset,
                                                method_name=manager.method.eval_name,
                                                model_name=manager.base_model.name,
                                                gridsearch_name=args.gridsearch_name,
                                                exp_name=args.exp_name,
                                                subset=args.test_set,
                                                create=True)

    args.records_path = utils.get_records_path(args.test_results_root_path, manager.dataset,
                                                method_name=manager.method.eval_name,
                                                model_name=manager.base_model.name,
                                                gridsearch_name=args.gridsearch_name,
                                                exp_name=args.exp_name,
                                                subset=args.test_set,
                                                create=True)

    max_out_filepath = os.path.join(args.out_path, utils.get_perf_output_filename(manager.method.eval_name,
                                                                                  manager.dataset.task_count - 1))

    if not args.debug and not args.test_overwrite_mode and os.path.exists(max_out_filepath):
        print("[OVERWRITE=False] SKIPPING EVAL, Already exists: ", max_out_filepath)
        exit(0)

    args.task_lengths = datasets.get_nc_per_task(manager.dataset)

    if hasattr(manager.method, 'eval_model_preprocessing'):
        model_paths = manager.method.eval_model_preprocessing(args)

    print("\nTESTING ON DATASETS:")
    print('\n'.join(ds_paths))

    print("\nTESTING ON MODELS:")
    print('\n'.join(model_paths))

    print("Testing on ", len(ds_paths), " task datasets")

    eval_latest_model_existed_tasks(args, manager, ds_paths, model_paths)

    utils.print_stats()
    print("FINISHED testing of: ", args.exp_name)

def eval_latest_model_existed_tasks(args, manager, ds_paths, model_paths):
    print("TEST AFTER TRAINING TASK {}:".format(args.task_counter))
    model_idx = args.task_counter - 1
    print("MODEL PATH: {}".format(model_paths[model_idx]))

    acc_all = {}
    start_time = time.time()

    for dataset_index in range(args.test_starting_task_count - 1, args.task_counter):
        args.eval_dset_idx = dataset_index
        try:
            string_name = 'Testing_task: ' + str(dataset_index) + '\t'
            accuracy = eval_task_steps_accuracy(args, manager, ds_paths, model_paths, string_name, model_idx)
            acc_all[dataset_index] = accuracy
            if args.class_incremental_repetition:
                break

        except Exception as e:
            print("TESTING ERROR: ", e)
            print("No results saved...")
            traceback.print_exc()
            break
        if args.class_incremental_repetition:
            break

    total_string = ""
    if args.class_incremental or args.class_incremental_repetition:
        for dataset_index in range(args.test_starting_task_count - 1, args.task_counter):
            s_name = 'Testing_dataset: ' + str(dataset_index) + '\tAccuracy: {:.8f}\n'.format(acc_all[dataset_index])
            print(s_name)
            total_string += s_name
            if args.class_incremental_repetition:
                break
    elapsed_time = time.time() - start_time
    if args.test_overwrite_mode:
        total_string += 'TOTAL_EVAL_time: {}'.format(elapsed_time)
        file = open(os.path.join(args.records_path,'eval_records_at_task_{}.txt'.format(args.task_counter)),'w')
        file.write(total_string)
        file.close()
    utils.print_timing(elapsed_time, title="TOTAL EVAL")


class EvalMetrics(object):
    def __init__(self):
        self.seq_acc = []
        self.seq_forgetting = []
        self.seq_head_acc = []


def eval_task_steps_accuracy(args, manager, ds_paths, model_paths, string_name, trained_model_idx):

    if not args.class_incremental:
        print("TESTING ON TASK ", args.eval_dset_idx + 1)
    accuracy = None

    args.dset_path = ds_paths[args.eval_dset_idx]
    args.head_paths = model_paths[args.eval_dset_idx]

    string_name += 'Testing_model:\t'
    if args.debug:
        print("Testing Dataset = ", args.dset_path)

    args.trained_model_idx = trained_model_idx
    args.eval_model_path = model_paths[trained_model_idx]
    if args.debug:
        print("Testing on model = ", args.eval_model_path)

    try:
        ssss = string_name + str(trained_model_idx)
        accuracy, _ = manager.method.inference_eval(args, manager,ssss)

    except Exception:
        print("ERROR in Testing model, trained until TASK ", str(trained_model_idx + 1))
        print("Aborting testing on further models")
        traceback.print_exc(5)
            # break
    return accuracy

