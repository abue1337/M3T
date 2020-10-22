import logging
from model import input_fn, model_fn
from model.maml import MAML
from model import test_script
from utils import utils_params, utils_misc, utils_devices, utils_plots, utils_read_write
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
    target_model = model_fn.gen_model()
    update_model = model_fn.gen_model()

    accuracies,losses = test_script.test(ds_test, target_model, update_model, run_paths)

    for i in range(len(accuracies)):
        logging.info(
            f"Test acc after {i} gradient steps: {accuracies[i]} Test loss after "
            f"{i} gradient steps:{losses[i]}")
    # utils_plots.plot_test_time_behaviour(maml.losses, maml.accuracies, run_paths)
    # utils_read_write.write_loss_acc_to_file(run_paths, maml.losses, maml.accuracies)



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
    path_model_id = 'C:\\Users\\andre\\Desktop\\experiments\\models\\after_fix\\run_2020-10-21T22-03-09'
    eval_main(path_model_id=path_model_id)
