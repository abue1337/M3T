from MetaBYOL.utils import utils_plots, utils_read_write

path = 'C:\\Users\\andre\\Desktop\\experiments\\models\\batch_spec_aug\\blur20flip50\\run_2020-10-16T10-12-27\\graphs\\8_eval'
acc, loss = utils_read_write.read_loss_acc_from_file(path)
utils_plots.plot_test_time_behaviour_2(loss, acc, path)