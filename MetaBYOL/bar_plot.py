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

    result_path = ['C:\\Users\\andre\\Desktop\\experiments\\models\\base_models\\run_2020-10-22T09-43-58-651902_simple\\test_results.json',
                   #'C:\\Users\\andre\\Desktop\\experiments\\models\\base_models\\run_2020-10-22T13-47-42-872076_single\\test_results.json',
                   'C:\\Users\\andre\\Desktop\\experiments\\models\\base_models\\run_2020-10-22T13-49-11-115948_batch\\test_results.json',
                   'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\no inner steps\\olr0_01\\run_2020-12-01T09-05-20\\logs\\test\\grad5_bs32_lr0_1\\test_results.json',
                   'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step1\\run_2020-11-26T09-35-03\\logs\\test\\bs32duplicate_grad10_lr_0_01\\test_results.json',
        'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step1\\run_2020-11-26T09-35-03\\logs\\test\\bs32duplicate_grad10_lr_0_01\\test_results.json']
    labels = ['Baseline Simple','Baseline Batch', 'Baseline Joint','Meta Trained 0 ttt step','Meta Trained 1 ttt step']
    save_to = 'C:\\Users\\andre\\Desktop\\result_bar_rotation_vs_joint.pdf'
    NUM_RESULTS = len(result_path)
    METRIC = 'accuracy'
    WIDTH = 0.2

    step = [0, 0, 0, 0, 1]
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
    plt.figure(figsize=(20, 5))
    y_pos = np.arange(len(datasets))
    # Plot bar
    for idx, metric_val in enumerate(metric_vals):
        print('')
        plt.bar(y_pos + idx*WIDTH, metric_val, width=WIDTH, label=labels[idx])

    plt.xticks(rotation=45)
    plt.legend(prop={'size': 12})
    plt.ylim([0, 110])
    plt.ylabel('Accuracy (%)')
    plt.xticks(y_pos + WIDTH / NUM_RESULTS, datasets)
    plt.legend(loc='upper right')
    plt.tight_layout()
    #plt.show()
    plt.savefig(save_to)
    plt.close()
    print('')