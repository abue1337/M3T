import logging
import os

from model import input_fn, model_fn
from model.maml import MAML
from model import test_script
from utils import utils_params, utils_misc, utils_devices, utils_plots, utils_read_write
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import gin


def run_test(ds, path_model_id='', run_paths=''):

    ds_test, ds_test_info = input_fn.gen_pipeline_test_time(ds_name=ds, split='test')

    # Define model
    target_model = model_fn.gen_model()
    update_model = model_fn.gen_model()

    accuracies, losses = test_script.test(ds_test, target_model, update_model, run_paths)

    for i in range(len(accuracies)):
        logging.info(
            f"Test acc after {i} gradient steps: {accuracies[i]} Test loss after "
            f"{i} gradient steps:{losses[i]}")
    # utils_plots.plot_test_time_behaviour(maml.losses, maml.accuracies, run_paths)
    # utils_read_write.write_loss_acc_to_file(run_paths, maml.losses, maml.accuracies)
    test_result = {'accuracy': accuracies, 'losses': losses}

    return test_result


if __name__ == '__main__':
    # Define model path and test dataset
    path_model_id = 'C:\\Users\\andre\\Desktop\\experiments\\models\\after_fix\\batch_aug\\run_2020-10-22T17-38-25'
    #path_model_id = '/misc/usrhomes/s1353/MetaBYOL/experiments/models/run_2020-10-22T17-38-25/'  # define model for test
    LEVEL = 5
    USE_ALL = False
    if USE_ALL:
        test_datasets = [f'cifar10_corrupted/brightness_{LEVEL}',
                         f'cifar10_corrupted/contrast_{LEVEL}',
                         f'cifar10_corrupted/defocus_blur_{LEVEL}',
                         f'cifar10_corrupted/elastic_{LEVEL}',
                         f'cifar10_corrupted/fog_{LEVEL}',
                         f'cifar10_corrupted/frost_{LEVEL}',
                         f'cifar10_corrupted/frosted_glass_blur_{LEVEL}',
                         f'cifar10_corrupted/gaussian_blur_{LEVEL}',
                         f'cifar10_corrupted/impulse_noise_{LEVEL}',
                         f'cifar10_corrupted/jpeg_compression_{LEVEL}',
                         f'cifar10_corrupted/motion_blur_{LEVEL}',
                         f'cifar10_corrupted/pixelate_{LEVEL}',
                         f'cifar10_corrupted/saturate_{LEVEL}',
                         f'cifar10_corrupted/shot_noise_{LEVEL}',
                         f'cifar10_corrupted/snow_{LEVEL}',
                         f'cifar10_corrupted/spatter_{LEVEL}',
                         f'cifar10_corrupted/speckle_noise_{LEVEL}',
                         f'cifar10_corrupted/zoom_blur_{LEVEL}']
    else:
        test_datasets = ['cifar10_corrupted/fog_5', 'cifar10_corrupted/snow_5']

    # generate folder structures
    run_paths = utils_params.gen_run_folder(path_model_id=path_model_id)

    # gin config files
    config_names = ['config.gin']
    # config
    utils_params.inject_gin(config_names, path_model_id=path_model_id)
    # utils_params.inject_gin(config_names, path_model_id=path_model_id, bindings=bindings, from_operative=True, run_paths=run_paths)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_test'], logging.INFO)

    results = {}
    for dataset in test_datasets:
        # start testing
        result = run_test(ds=dataset, path_model_id=path_model_id, run_paths=run_paths)
        results[dataset] = result

    # Convert all results to json and print all results
    utils_read_write.save_result_json(os.path.join(run_paths['path_model_id'], 'test_results.json'), results)

    # Print results
    for key, value in results.items():
        logging.info(f"{key:>40}: {value}")