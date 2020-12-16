import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
#sns.set_context(context='paper')

if __name__ == '__main__':
    # Define json file(s)
    #'C:\\Users\\andre\\Desktop\\experiments\\models\\base_models\\run_2020-10-22T13-47-42-872076_single\\test_results.json',
    #'C:\\Users\\andre\\Desktop\\experiments\\models\\base_models\\run_2020-10-22T13-49-11-115948_batch\\test_results.json',
    result_path = [
        'C:\\Users\\andre\\Desktop\\experiments\\models\\base_models\\run_2020-10-22T09-43-58-651902_simple\\test_results.json',
        'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\no_inner_steps\\olr0_01_nobatchaug\\run_2020-12-01T08-58-24\\logs\\test\\grad5_lr0_1_bs32\\test_results.json',
        'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step1_nobatchaug\\run_2020-11-26T22-05-17\\logs\\grad5_lr0_1_bs8\\test_results.json',
        'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step1_nobatchaug\\run_2020-11-26T22-05-17\\logs\\grad5_lr0_1_bs8\\test_results.json']
    # 'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step2\\run_2020-11-23T11-47-27\\logs\\test\\ckpt200_bs32_grad5_lr0_1\\test_results.json',
    # 'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step2\\run_2020-11-23T11-47-27\\logs\\test\\ckpt200_bs32_grad5_lr0_1\\test_results.json']
    """
    ###version two- Rotation
    result_path = ['C:\\Users\\andre\\Desktop\\experiments\\models\\base_models\\run_2020-10-22T09-43-58-651902_simple\\test_results.json',
                   #'C:\\Users\\andre\\Desktop\\experiments\\models\\base_models\\run_2020-10-22T13-47-42-872076_single\\test_results.json',
                   'C:\\Users\\andre\\Desktop\\experiments\\models\\base_models\\run_2020-10-22T13-49-11-115948_batch\\test_results.json',
                   'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\no_inner_steps\\olr0_01\\run_2020-12-01T09-05-20\\logs\\test\\grad5_bs32_lr0_1\\test_results.json',
                   'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step1\\run_2020-11-26T09-35-03\\logs\\test\\bs8duplicate_grad5_lr_0_1\\test_results.json',
                   'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step1\\run_2020-11-26T09-35-03\\logs\\test\\bs8duplicate_grad5_lr_0_1\\test_results.json']
        #'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step2\\run_2020-11-23T11-47-27\\logs\\test\\ckpt200_bs32_grad5_lr0_1\\test_results.json',
        #'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step2\\run_2020-11-23T11-47-27\\logs\\test\\ckpt200_bs32_grad5_lr0_1\\test_results.json']
    """

    ###version one - BYOL
    result_path = [
        'C:\\Users\\andre\\Desktop\\experiments\\models\\base_models\\run_2020-10-22T09-43-58-651902_simple\\test_results.json',
        'C:\\Users\\andre\\Desktop\\experiments\\models\\base_models\\run_2020-10-22T13-49-11-115948_batch\\test_results.json',
        'C:\\Users\\andre\\Desktop\\experiments\\models\\inner_byol\\withpredictor\\no_inner_steps\\olr0_01_ilr0_0_gradsteps_0\\run_2020-12-07T09-08-37\\logs\\test\\grad5_lr0_1_bs8\\test_results.json',
        #'C:\\Users\\andre\\Desktop\\experiments\\models\\inner_byol\\withpredictor\\olr0_01_ilr_0_1_step1\\run_2020-12-07T14-36-49\\logs\\test\\grad5_lr0_1_bs8\\test_results.json',
        #'C:\\Users\\andre\\Desktop\\experiments\\models\\inner_byol\\withpredictor\\olr0_01_ilr_0_1_step1\\run_2020-12-07T14-36-49\\logs\\test\\grad5_lr0_1_bs8\\test_results.json']
        'C:\\Users\\andre\\Desktop\\experiments\\models\\inner_byol\\withpredictor\\olr0_01_ilr_0_1_step2\\run_2020-12-07T14-49-58\\logs\\test\\grad5_lr0_1_bs8\\test_results.json',
        'C:\\Users\\andre\\Desktop\\experiments\\models\\inner_byol\\withpredictor\\olr0_01_ilr_0_1_step2\\run_2020-12-07T14-49-58\\logs\\test\\grad5_lr0_1_bs8\\test_results.json']

    labels = ['Baseline Simple','Baseline Batch', 'Baseline Joint','Meta Trained 0 ttt step','Meta Trained 2 ttt step']
    save_to = 'result_bar_byol_2step_vs_joint.pdf'
    NUM_RESULTS = len(result_path)
    METRIC = 'accuracy'
    WIDTH = 0.15

    step = [0, 0, 0, 0, 2]
    # Iterate over files to get names and data
    metric_vals = []
    for idx, file in enumerate(result_path):
        # Load json to dict
        with open(file) as f:
            result_dict = json.load(f)
        # get dataset names
        if idx == 0:
            datasets = [ x.split('/')[-1] for x in list(result_dict.keys())]
        # Get metrics

        metric_val = []
        for dataset, result in result_dict.items():
            # get metric
            if result[METRIC][0] == '[':
                metric_val.append(json.loads(result[METRIC])[step[idx]]*100)  #TODO: change list to dict in test script
            else:
                metric_val.append(float(result[METRIC]) * 100)
        metric_vals.append(metric_val)

    # Prepare plot
    plt.figure(figsize=(12, 4))
    y_pos = np.arange(len(datasets))
    # Plot bar
    for idx, metric_val in enumerate(metric_vals):
        print('')
        plt.bar(y_pos + idx*WIDTH, metric_val, width=WIDTH, label=labels[idx])

    plt.xticks(rotation=45)
    plt.legend(prop={'size': 12})
    plt.ylim([20, 100])
    plt.ylabel('Accuracy (%)')
    plt.xticks(y_pos + WIDTH / NUM_RESULTS, datasets)
    plt.legend(loc='lower right')
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_to)
    plt.close()
    print('')