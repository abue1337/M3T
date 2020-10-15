import logging
from model import input_fn, model_fn
from model.maml import MAML
from utils import utils_params, utils_misc, utils_devices, utils_plots
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import gin


def set_up_eval( path_model_id='', run_paths=''):

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_eval'], logging.INFO)

    ds_test, ds_test_info = input_fn.gen_pipeline_test_time(split='test')
    #ds_test, ds_val_info = input_fn.gen_pipeline_train(split='test', validation_set=True)


    # Define model
    target_model = model_fn.gen_model

    maml = MAML(target_model, ds_test._flat_shapes[0][0:], test_time=True)

    for a in range(maml.num_steps_ml+1):
        maml.num_test_time_steps = a
        maml.test(ds_test, run_paths)
    for i in range(maml.num_steps_ml+1):
        logging.info(
            f"Test acc after {i} gradient steps: {maml.accuracies[i]} Test loss after "
            f"{i} gradient steps:{maml.losses[i]}")
    utils_plots.plot_test_time_behaviour(maml.losses, maml.accuracies, run_paths)



def eval_main(path_model_id='', bindings=[], inject_gin=True):

    # generate folder structures
    run_paths = utils_params.gen_run_folder(path_model_id=path_model_id)

    # gin config files
    if inject_gin:
        config_names = ['config.gin']
        # config
        utils_params.inject_gin(config_names, path_model_id=path_model_id)
        #utils_params.inject_gin(config_names, path_model_id=path_model_id, bindings=bindings, from_operative=True, run_paths=run_paths)

    # start testing
    set_up_eval(path_model_id=path_model_id, run_paths=run_paths)


if __name__ == '__main__':
    path_model_id = 'C:\\Users\\andre\\Desktop\\experiments\\models\\batch_spec_aug\\run_2020-10-13T18-56-10'
    eval_main(path_model_id=path_model_id)
