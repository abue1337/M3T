import matplotlib.pyplot as plt


def plot_omniglot_img(img):
    plt.gray()
    if img.ndim == 4:
        for i in range(img.shape[0]):
            plt.imshow(img[i, :, :, 0])
            plt.show()
    else:
        plt.imshow(img[:, :, 0])
        plt.show()
