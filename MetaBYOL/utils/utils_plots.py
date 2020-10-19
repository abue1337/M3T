import matplotlib.pyplot as plt
import os


def plot_omniglot_img(img):
    plt.gray()
    if img.ndim == 4:
        for i in range(img.shape[0]):
            plt.imshow(img[i, :, :, 0])
            plt.show()
    else:
        plt.imshow(img[:, :, 0])
        plt.show()

def plot_img(img):
    #plt.gray()
    if img.ndim == 4:
        for i in range(img.shape[0]):
            plt.imshow(img[i, :, :, :])
            plt.show()
    else:
        plt.imshow(img[:, :, :])
        plt.show()


def plot_test_time_behaviour(losses, accuracies, run_paths):
    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
    ax1.plot(range(len(losses)), losses, '--*')
    fig.suptitle('Test Time Training Behaviour')
    ax1.set_ylabel('Test Loss')
    ax1.set_xlabel('Number of gradient steps')
    ax1.set_title('  ')
    ax2.plot(range(len(accuracies)), accuracies, '--*')

    ax2.set_ylabel('Test accuracy')
    ax2.set_xlabel('Number of gradient steps')
    my_path = os.path.dirname(run_paths['path_graphs_eval'])
    my_file = 'Test_time_behaviour.png'
    plt.savefig(os.path.join(my_path, my_file))
    plt.show()


def plot_test_time_behaviour_2(losses, accuracies, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, tight_layout=True)
    ax1.plot(range(len(losses)), losses, '--*')
    fig.suptitle('Test Time Training Behaviour')
    ax1.set_ylabel('Test Loss')
    ax1.set_xlabel('Number of gradient steps')
    ax1.set_title('  ')
    ax2.plot(range(len(accuracies)), accuracies, '--*')

    ax2.set_ylabel('Test accuracy')
    ax2.set_xlabel('Number of gradient steps')
    my_path = path
    my_file = 'Test_time_behaviour.png'
    plt.savefig(os.path.join(my_path, my_file))
    plt.show()