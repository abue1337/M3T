import logging
from model import input_fn, model_fn
from model.maml import MAML
from model import test_script
from utils import utils_params, utils_misc, utils_devices
from model import classic_training
import tensorflow_datasets as tfds
import gin
import tensorflow as tf
import matplotlib.pyplot as plt

@gin.configurable()
def set_up_train(path_model_id='', device='0', config_names=['config.gin']):  # 'sinusoid'
    # generate folder structures
    run_paths = utils_params.gen_run_folder(path_model_id=path_model_id)

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # config
    utils_params.inject_gin(config_names, path_model_id=path_model_id)

    ds_train, ds_train_info = input_fn.gen_pipeline_train(split='train[:1%]')
    ds_val, ds_val_info = input_fn.gen_pipeline_train(split='test[:1%]', validation_set=True)

    # set device params
    #utils_devices.set_devices(device)

    # Define model
    target_model = model_fn.gen_model

    #### Classic Training ####
    #trained_classic_model = classic_training.train_classic(model, ds_train, run_paths)

    #### MAML ####
    # Initialize MAML training
    maml = MAML(target_model, ds_train._flat_shapes[0][1:])

    # Meta Train
    trained_maml_model = maml.train(
        ds_train,ds_val,
        run_paths)


    gin_string = gin.operative_config_str()
    logging.info(f'Fetched config parameters: {gin_string}.')
    utils_params.save_gin(run_paths['path_gin'], gin_string)


if __name__ == '__main__':
    device = '0'
    path_model_id = ''  # only to use if starting from existing model

    # gin config files
    config_names = ['config.gin']

    # start training
    set_up_train(path_model_id=path_model_id, device=device, config_names=config_names)
