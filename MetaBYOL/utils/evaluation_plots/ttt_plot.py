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
    all_plots = False
    CORRIDOR = False
    corridor = 0.1
    METRIC = 'accuracy'
    metric = {'accuracy': 0,
              'losses': 1
              }
    legend = ['classic training','ilr 0.1 steps 1', 'ilr 0.1 steps 2']
    if  not all_plots:
        selected = ['fog_5','frosted_glass_blur_5','impulse_noise_5','jpeg_compression_5']
    if single_plot:
        path = 'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\no inner steps\\run_2020-11-19T13-38-35\\logs\\test\\bs32duplicate_grad10_lr_0_01\\'
    if not single_plot:
        paths = ['C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\no_inner_steps\\olr0_01\\run_2020-12-01T09-05-20\\logs\\test\\grad5_bs32_lr0_1\\',
        'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step1\\run_2020-11-26T09-35-03\\logs\\test\\bs8duplicate_grad5_lr_0_1\\',
                 'C:\\Users\\andre\\Desktop\\experiments\\models\\rotation_try\\olr0_01_ilr0_1_step2\\run_2020-11-23T11-47-27\\logs\\test\\ckpt200_bs8_grad5_lr0_1\\']
        ###byol with aug
        """
        paths = [
            #'C:\\Users\\andre\\Desktop\\experiments\\models\\inner_byol\\withpredictor\\olr0_01_ilr_0_1_step1\\run_2020-12-03T09-23-03\\logs\\test\\grad5_lr0_1_bs8rep\\',
            #'C:\\Users\\andre\\Desktop\\experiments\\models\\inner_byol\\withpredictor\\olr0_01_ilr_0_1_step2\\run_2020-12-07T14-49-58\\logs\\test\\grad5_lr0_1_bs8\\',
            'C:\\Users\\andre\\Desktop\\experiments\\models\\inner_byol\\withpredictor\\no_inner_steps\\olr0_01_ilr0_0_gradsteps_0\\run_2020-12-07T09-08-37\\logs\\test\\grad5_lr0_1_bs8\\',
            'C:\\Users\\andre\\Desktop\\experiments\\models\\inner_byol\\withpredictor\\olr0_01_ilr_0_1_step1\\run_2020-12-07T14-36-49\\logs\\test\\grad5_lr0_1_bs8\\',
            'C:\\Users\\andre\\Desktop\\experiments\\models\\inner_byol\\withpredictor\\olr0_01_ilr_0_1_step2\\run_2020-12-07T14-49-58\\logs\\test\\grad5_lr0_1_bs8\\'
        ]
        """

        """
        paths = [
            'C:\\Users\\andre\\Desktop\\experiments\\models\\inner_byol\\withpredictor\\olr0_01_ilr_0_1_step1_nobatchaug\\run_2020-12-07T09-05-54\\logs\\test\\grad5_lr0_1_bs8\\',
            'C:\\Users\\andre\\Desktop\\experiments\\models\\inner_byol\\withpredictor\\no_inner_steps\\olr0_01_ilr0_0_gradsteps_0_no_batchaug\\run_2020-12-07T09-10-52\\logs\\test\\grad5_lr0_1_bs8\\']
        """
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
        if all_plots:
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
        else:
            # labels = ['Meta Trained']
            #save_to = os.path.join(paths[0], f'result_compare_ttt_plot_selected.pdf')
            save_to = 'result_compare_ttt_plot_selected.pdf'
            plots = []
            for count, path in enumerate(paths):
                result_path = os.path.join(path, 'test_results.json')
                with open(result_path) as f:
                    result_dict = json.load(f)
                # get dataset names
                datasets = [x.split('/')[-1] for x in list(result_dict.keys())]

                metric_vals = []
                metric_vals2 = []
                for a, values in result_dict.items():
                    metric_vals.append(json.loads(values['accuracy']))
                    metric_vals2.append(json.loads(values['losses']))


                if count == 0:
                    NUM_COLS = len(selected)
                    NUM_ROWS = 1
                    fig, ax = plt.subplots(NUM_ROWS, ncols=NUM_COLS, figsize=(len(selected)*3,4),tight_layout=True)



                for i, dataset in enumerate(selected):
                    dataset_index = datasets.index(dataset)
                    if CORRIDOR is True:
                        ax[int(i / NUM_COLS)][i % NUM_COLS].set_ylim(
                            (np.mean(metric_vals[dataset_index]) - corridor / 2, np.mean(metric_vals[dataset_index]) + corridor / 2))
                    cur, = ax[i % NUM_COLS].plot(range(len(metric_vals[dataset_index])), metric_vals[dataset_index], '--*')
                    ax[i % NUM_COLS].set_ylabel(f'Test {METRIC}')
                    ax[i % NUM_COLS].set_xlabel('Number of gradient steps')
                    ax[i % NUM_COLS].set_title(dataset)
                plots.append(cur)

            ax[-1].legend(plots, legend, loc=4)
            #fig.legend( loc='center right',borderaxespad=0.1)
            #fig.subplots_adjust(right=0.85)
            # Shrink current axis's height by 10% on the bottom
            #box = ax[-1].get_position()
            #ax[-1].set_position([box.x0, box.y0 + box.height * 0.1,
            #                 box.width, box.height * 0.9])

            #ax[1][i % NUM_COLS].plot(range(len(metric_vals2[dataset_index])),
                    #                                         metric_vals2[dataset_index], '--*')
                    #ax[1][i % NUM_COLS].set_ylabel(f'Test loss')
                    #ax[1][i % NUM_COLS].set_xlabel('Number of gradient steps')
                    # fig.suptitle(f'Test Time Training - {METRIC}')

        plt.savefig(save_to)
        plt.show()
        print('')
