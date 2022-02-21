from PIL import Image
from vis.visualization import visualize_saliency, overlay
from vis.utils import utils
from keras import activations

# printing images 9 and 10
n_H = np.shape(X_train)[1]
n_W = np.shape(X_train)[2]
n_C = np.shape(X_train)[3]
ex1 = np.reshape(X_train[9, :, :, :], (n_H, n_W, n_C)).astype(dtype=np.uint8)
ex2 = np.reshape(X_train[10, :, :, :], (n_H, n_W, n_C)).astype(dtype=np.uint8)
img1 = Image.fromarray(ex1, 'RGB')
img2 = Image.fromarray(ex2, 'RGB')
img1.save('img1.jpeg')
img2.save('img2.jpeg')
img1.show()
img2.show()

ex1_n = np.reshape(normalize(X_train)[9, :, :, :], (n_H, n_W, n_C)).astype(dtype=np.uint8)
ex2_n = np.reshape(normalize(X_train)[10, :, :, :], (n_H, n_W, n_C)).astype(dtype=np.uint8)

model = CNN(normalize(X_train), y_train, batch_size=10, epochs=1)
y_train_pred = model.predict(normalize(X_train))

# printing saliency maps for two example images (see second link in email)
layer_idx = utils.find_layer_idx(model, 'conv3')

f, ax = plt.subplots(1, 2)
for i, img in enumerate([ex1_n, ex2_n]):
    # 20 is the index of CONV2D last layer
    # print(f"len(img1.shape) = {len(img1.shape)}")
    grads = visualize_saliency(model, layer_idx, filter_indices=20, seed_input=img)

    # visualize grads as heatmap
    ax[i].imshow(grads, cmap='jet')