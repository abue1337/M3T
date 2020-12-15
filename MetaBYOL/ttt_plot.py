import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
import os

# sns.set_context(context='paper')

if __name__ == '__main__':
    # Define json file(s)
    single_plot = False
    CORRIDOR = False
    corridor = 0.1
    METRIC = 'accuracy'
    metric = {'accuracy': 0,
              'losses': 1
              }
    if single_plot:
        path = 'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\no inner steps\\run_2020-11-19T13-38-35\\logs\\test\\bs32duplicate_grad10_lr_0_01\\'
    if not single_plot:
        paths = ['C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\no_inner_steps\\olr0_01\\run_2020-12-01T09-05-20\\logs\\test\\grad5_bs32_lr0_1\\',
        'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step1\\run_2020-11-26T09-35-03\\logs\\test\\bs32duplicate_grad5_lr_0_1\\',
                 'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step1\\run_2020-11-26T09-35-03\\logs\\test\\bs32duplicate_grad5_lr_0_05\\']

    if single_plot:
        result_path = os.path.join(path, 'test_results.json')
        labels = ['Meta Trained']
        save_to = os.path.join(path, f'result_ttt_plot_{METRIC}_{corridor}.pdf')

        NUM_COLS = 6

        with open(result_path) as f:
            result_dict = json.load(f)
        # get dataset names
        datasets = [x.split('/')[-1] for x in list(result_dict.keys())]

        metric_vals = []
        for a, values in result_dict.items():
            metric_vals.append(json.loads(values[METRIC]))

        NUM_ROWS = int(np.ceil(len(metric_vals) / NUM_COLS))
        fig, ax = plt.subplots(NUM_ROWS, NUM_COLS,figsize=(30, 20), tight_layout=True)

        for i, dataset in enumerate(datasets):
            if CORRIDOR is True:
                ax[int(i / NUM_COLS)][i % NUM_COLS].set_ylim((np.mean(metric_vals[i]) - corridor/2, np.mean(metric_vals[i]) + corridor/2))
            ax[int(i / NUM_COLS)][i % NUM_COLS].plot(range(len(metric_vals[i])), metric_vals[i], '--*')
            ax[int(i / NUM_COLS)][i % NUM_COLS].set_ylabel(f'Test {METRIC}')
            ax[int(i / NUM_COLS)][i % NUM_COLS].set_xlabel('Number of gradient steps')
            ax[int(i / NUM_COLS)][i % NUM_COLS].set_title(dataset)

        #fig.suptitle(f'Test Time Training - {METRIC}')
        plt.savefig(save_to)
        plt.show()
        print('')

    if not single_plot:

        # labels = ['Meta Trained']
        save_to = os.path.join(paths[0], f'result_compare_ttt_plot_{METRIC}_{corridor}.pdf')



        for count, path in enumerate(paths):
            result_path = os.path.join(path, 'test_results.json')
            with open(result_path) as f:
                result_dict = json.load(f)
            # get dataset names
            datasets = [x.split('/')[-1] for x in list(result_dict.keys())]

            metric_vals = []
            for a, values in result_dict.items():
                metric_vals.append(json.loads(values[METRIC]))

            if count==0:
                NUM_COLS = 6
                NUM_ROWS = int(np.ceil(len(metric_vals) / NUM_COLS))
                fig, ax = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(30, 20), tight_layout=True)

            for i, dataset in enumerate(datasets):
                if CORRIDOR is True:
                    ax[int(i / NUM_COLS)][i % NUM_COLS].set_ylim(
                        (np.mean(metric_vals[i]) - corridor / 2, np.mean(metric_vals[i]) + corridor / 2))
                ax[int(i / NUM_COLS)][i % NUM_COLS].plot(range(len(metric_vals[i])), metric_vals[i], '--*')
                ax[int(i / NUM_COLS)][i % NUM_COLS].set_ylabel(f'Test {METRIC}')
                ax[int(i / NUM_COLS)][i % NUM_COLS].set_xlabel('Number of gradient steps')
                ax[int(i / NUM_COLS)][i % NUM_COLS].set_title(dataset)

            # fig.suptitle(f'Test Time Training - {METRIC}')
        plt.savefig(save_to)
        plt.show()
        print('')
