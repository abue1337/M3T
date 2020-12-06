import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
import os
import pandas as pd
# Read data from file 'filename.csv'
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later)

paths = ["C:\\Users\\andre\\Desktop\\experiments\\models\\inner_byol\\run_2020-11-23T12-57-39\\logs\\"]
files = ['run-.-tag-Average meta test tasks accuracy.csv','run-.-tag-Average meta val accuracy.csv',
         'run-.-tag-Average pre update test tasks accuracy.csv']

for i, path in enumerate(paths):

    data = []
    for j, file in enumerate(files):
        data.append(pd.read_csv(path+files[j]))

    f=open(path.split('logs\\')[0] + 'config_operative.gin','r')
    contents = f.read()
    fig, ax = plt.subplots()
    for metric in data:
        ax.plot(metric['Step'], metric['Value'])
        #plt.plot(metric['Value'])
    ax.set(xlabel='Epoch', ylabel='Accuracy',
           title='Meta training behaviour')
    #ax.grid()

    # fig.savefig("meta_training_behaviour.png")
    plt.show()
    print(i)