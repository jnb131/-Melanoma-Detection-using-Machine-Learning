from sklearn.decomposition import PCA
import img_proc
import matplotlib.pyplot as plt
def main():
    step_size = 0.01
    TRAIN_DATA_BATCH_SIZE = 3000
    BATCH_SIZE = 200
    data_dir = 'data'

    data_gen_train = img_proc.Data_Generator(data_dir + '/train_sep', TRAIN_DATA_BATCH_SIZE, shuffle=True,
                                             flatten=True)
    x_train, y_train = data_gen_train.__getitem__(0)

    pca = PCA(n_components=5)
    pca.fit_transform(x_train)
    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


if __name__ == '__main__':
    main()