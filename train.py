from nets.segnet import convnet_segnet
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import keras
from keras import backend as K
import numpy as np
from keras.utils.data_utils import get_file
NCLASSES = 2
HEIGHT = 416
WIDTH = 416

def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = Image.open(r".\dataset2\jpg" + '/' + name)
            img = img.resize((WIDTH,HEIGHT))
            img = np.array(img)
            img = img/255
            X_train.append(img)

            name = (lines[i].split(';')[1]).replace("\n", "")
            # 从文件中读取图像
            img = Image.open(r".\dataset2\png" + '/' + name)
            img = img.resize((int(WIDTH/2),int(HEIGHT/2)))
            img = np.array(img)
            seg_labels = np.zeros((int(HEIGHT/2),int(WIDTH/2),NCLASSES))
            for c in range(NCLASSES):
                seg_labels[: , : , c ] = (img[:,:,0] == c ).astype(int)
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i+1) % n
        yield (np.array(X_train),np.array(Y_train))

def loss(y_true, y_pred):
    crossloss = K.binary_crossentropy(y_true,y_pred)
    # 这个乘除是为了求平均……相当于每个像素点的交叉熵。
    loss = 4 * K.sum(crossloss)/HEIGHT/WIDTH
    return loss

if __name__ == "__main__":
    log_dir = "logs/"
    # 获取model
    model = convnet_segnet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
    # model.summary()
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='/SegNet_Conv2/models')
                                    
    model.load_weights(weights_path,by_name=True)
   
