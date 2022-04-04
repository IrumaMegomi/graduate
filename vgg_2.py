import scipy.io
import numpy as np
import os
import scipy.misc
import cv2
from PIL import Image
from PIL import ImageEnhance
import matplotlib
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
matplotlib.rcParams['font.family'] = 'simHei'
matplotlib.rcParams['axes.unicode_minus'] = False
#变量列表
C_ki_i=[0]*36#红外提取特征后的范数
sum_i=0#红外范数和
ave_i=0#红外范数求均值
C_ki_v=[0]*36#可见光提取特征后的范数
sum_v=0#可见光范数和
ave_v=0#可见光范数求均值
img_high=cv2.imread('visible12.jpg')
cnt_i=0
cnt_v=0


img_infrared = cv2.imread('infrared12.jpg')
img_gray_infrared = cv2.cvtColor(img_infrared,cv2.COLOR_BGR2GRAY)
f = np.fft.fft2(img_gray_infrared)
#shift 将低频移到中间
fshift = np.fft.fftshift(f)
rows,cols = fshift.shape
mid_x,mid_y = int((rows)/2),(int((cols)/2))
#高通
mask1 = np.ones((rows,cols),dtype=np.uint8)
mask1[mid_x-5:mid_x+5,mid_y-5:mid_y+5] = 0
fshift1 = mask1*fshift
isshift1 = np.fft.ifftshift(fshift1)

#低通蒙板
mask2 = np.zeros((rows,cols),dtype=np.uint8)
mask2[mid_x-10:mid_x+10,mid_y-10:mid_y+10] = 1
fshift2 = mask2*fshift
isshift2 = np.fft.ifftshift(fshift2)

high = np.fft.ifft2(isshift1)
low = np.fft.ifft2(isshift2)

img_high_infrared = np.abs(high) #高通图像
img_high_infrared=img_high_infrared.astype('uint8')
img_high_infrared=cv2.cvtColor(img_high_infrared,cv2.COLOR_GRAY2BGR)
img_low_infrared = np.abs(low)  #低通图像
shape_i=img_low_infrared.shape
#红外部分高低通结束

#可见光部分高低通
img_visible = cv2.imread('visible12.jpg')
img_gray_visible = cv2.cvtColor(img_visible,cv2.COLOR_BGR2GRAY)
f_v = np.fft.fft2(img_gray_visible)
#shift 将低频移到中间
fshift_v = np.fft.fftshift(f_v)
rows_v,cols_v = fshift_v.shape
mid_x_v,mid_y_v = int((rows_v)/2),(int((cols_v)/2))
#高通
mask1_v = np.ones((rows_v,cols_v),dtype=np.uint8)
mask1_v[mid_x_v-5:mid_x_v+5,mid_y_v-5:mid_y_v+5] = 0
fshift1_v = mask1_v*fshift_v
isshift1_v = np.fft.ifftshift(fshift1_v)

#低通蒙板
mask2_v = np.zeros((rows_v,cols_v),dtype=np.uint8)
mask2_v[mid_x_v-10:mid_x_v+10,mid_y_v-10:mid_y_v+10] = 1
fshift2_v = mask2_v*fshift_v
isshift2_v = np.fft.ifftshift(fshift2_v)

high_v = np.fft.ifft2(isshift1_v)
low_v = np.fft.ifft2(isshift2_v)

img_high_visible = np.abs(high_v) #高通图像
img_high_visible=img_high_visible.astype('uint8')
img_high_visible=cv2.cvtColor(img_high_visible,cv2.COLOR_GRAY2BGR)
img_low_visible = np.abs(low_v)  #低通图像
shape_v=img_low_visible.shape
#低通最大化融合
img_low=img_low_visible#自定义变量img_low，作为最终低通的部分

for i in range (0,232):
    for j in range (0,328):
        if(img_low_infrared[i,j]>img_low_visible[i,j]):
            img_low[i,j]=img_low_infrared[i,j]
        else:
            img_low[i,j]=img_low_visible[i,j]
#低通最大化融合结束
img_low=img_low.astype('uint8')
img_low=cv2.cvtColor(img_low,cv2.COLOR_GRAY2BGR)

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)
def _pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
            padding='SAME')
def preprocess(image, mean_pixel):
    return image - mean_pixel
def unprocess(image, mean_pixel):
    return image + mean_pixel
def imread(path):
    return scipy.misc.imread(path).astype(np.float)
def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(path, img)
print ("Functions for VGG ready")



def net(data_path, input_image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    net = {}
    current = input_image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current)
        net[name] = current
    assert len(net) == len(layers)
    return net, mean_pixel, layers
print ("Network for VGG ready")


#红外特征提取
cwd  = os.getcwd()
VGG_PATH_i =  "imagenet-vgg-verydeep-19.mat"
input_image_i=img_high_infrared
'''
input_image_i=cv2.cvtColor(input_image_i,cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16, 16))
input_image_i = clahe.apply(input_image_i)
input_image_i=cv2.cvtColor(input_image_i,cv2.COLOR_GRAY2BGR)
'''
shape_i = (1,input_image_i.shape[0],input_image_i.shape[1],input_image_i.shape[2])
with tf.compat.v1.Session() as sess:
    image_i = tf.placeholder('float', shape=shape_i)
    nets_i, mean_pixel_i, all_layers_i = net(VGG_PATH_i, image_i)
    input_image_pre_i = np.array([preprocess(input_image_i, mean_pixel_i)])
    layers_i = all_layers_i # For all layers
    # layers = ('relu2_1', 'relu3_1', 'relu4_1')
    for i_i, layer_i in enumerate(layers_i):
        print ("[%d/%d] %s" % (i_i+1, len(layers_i), layer_i))
        features_i = nets_i[layer_i].eval(feed_dict={image_i: input_image_pre_i})
        features_i=features_i[0,:,:,0]
        C_ki_i[i_i]=np.linalg.norm(features_i,ord=1)
        sum_i=sum_i+C_ki_i[i_i]
        cnt_i+=1
        if(cnt_i==9):
            break
        #上面是提取特征图
        # 特征图绘制
'''
        if 1:
            plt.figure(i_i+1, figsize=(10, 5))
            plt.matshow(features_i, cmap=plt.cm.gray, fignum=i_i+1)
            plt.title("" + layer_i)
            plt.colorbar()
            plt.show()
'''
#sum_i=C_ki_i[1]+C_ki_i[3]+C_ki_i[6]+C_ki_i[8]+C_ki_i[11]+C_ki_i[13]+C_ki_i[15]+C_ki_i[17]+C_ki_i[20]+C_ki_i[22]+C_ki_i[24]+C_ki_i[26]+C_ki_i[29]+C_ki_i[31]+C_ki_i[33]+C_ki_i[35]
sum_i=2.0*C_ki_i[0]+1.2*C_ki_i[1]+1.1*C_ki_i[2]+1.1*C_ki_i[3]+1.1*C_ki_i[5]+C_ki_i[7]
ave_i=sum_i/2
print(ave_i)
#红外特征提取结束

#可见光特征提取
VGG_PATH_v =  "imagenet-vgg-verydeep-19.mat"
IMG_PATH_v =  "visible12.jpg"
input_image_v=img_high_visible
'''
input_image_v=cv2.cvtColor(input_image_v,cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(2, 2))
input_image_v = clahe.apply(input_image_v)
input_image_v=cv2.cvtColor(input_image_v,cv2.COLOR_GRAY2BGR)
'''
shape_v = (1,input_image_v.shape[0],input_image_v.shape[1],input_image_v.shape[2])
with tf.compat.v1.Session() as sess:
    image_v = tf.placeholder('float', shape=shape_v)
    nets_v, mean_pixel_v, all_layers_v = net(VGG_PATH_v, image_v)
    input_image_pre_v = np.array([preprocess(input_image_v, mean_pixel_v)])
    layers_v = all_layers_v # For all layers
    # layers = ('relu2_1', 'relu3_1', 'relu4_1')
    for i_v, layer_v in enumerate(layers_v):
        print ("[%d/%d] %s" % (i_v+1, len(layers_v), layer_v))
        features_v = nets_v[layer_v].eval(feed_dict={image_v: input_image_pre_v})
        features_v=features_v[0,:,:,0]
        C_ki_v[i_v]=np.linalg.norm(features_v,ord=1)
        sum_v=sum_v+C_ki_v[i_v]
        cnt_v+=1
        if(cnt_v==9):
            break
'''
        if 1:
            plt.figure(i_v+1, figsize=(10, 5))
            plt.matshow(features_v, cmap=plt.cm.gray, fignum=i_v+1)
            plt.title("" + layer_v)
            plt.colorbar()
            plt.show()
'''
#sum_v=C_ki_v[1]+C_ki_v[3]+C_ki_v[6]+C_ki_v[8]+C_ki_v[11]+C_ki_v[13]+C_ki_v[15]+C_ki_v[17]+C_ki_v[20]+C_ki_v[22]+C_ki_v[24]+C_ki_v[26]+C_ki_v[29]+C_ki_v[31]+C_ki_v[33]+C_ki_v[35]
sum_v=2.0*C_ki_v[0]+1.2*C_ki_v[1]+1.1*C_ki_v[2]+1.1*C_ki_v[3]+1.1*C_ki_v[5]+C_ki_v[7]
ave_v=sum_v/2
print(ave_v)
#可见光特征提取结束
w_ki_i=ave_i/(ave_i+ave_v)
w_ki_v=ave_v/(ave_v+ave_i)
print(w_ki_v,'\n',w_ki_i)
#红外可见光权重确定完毕

#加权融合部分
img_i=cv2.imread('infrared12.jpg')
img_v=cv2.imread('visible12.jpg')
img_high_infrared=cv2.cvtColor(img_i,cv2.COLOR_BGR2GRAY)
img_high_visible=cv2.cvtColor(img_v,cv2.COLOR_BGR2GRAY)
img_high=img_high_visible
img_low=cv2.cvtColor(img_low,cv2.COLOR_BGR2GRAY)
img_last=img_high_visible
for i in range (0,232):
    for j in range (0,328):
        img_high[i,j]=w_ki_i*img_high_infrared[i,j]+w_ki_v*img_high_visible[i,j]
#加权融合结束
print(img_high.shape)
print(img_low.shape)
'''
plt.subplot(111)
plt.imshow(img_high,'gray')
plt.show()
'''
#准备和低频加和
for i in range(0,232):
    for j in range(0,328):
        img_last[i,j]=0.9*img_high[i,j]+0.1*img_low[i,j]-15
        if(img_last[i,j]>=255):
            img_last[i,j]=235
        elif(img_last[i,j]<=0):
            img_last[i,j]=15

plt.subplot(111)
plt.imshow(img_last,'gray')
plt.show()
'''
plt.subplot(321)
plt.imshow(img_high_infrared,'gray')
plt.title('红外高频')

plt.subplot(322)
plt.imshow(img_high_visible,'gray')
plt.title('可见光高频')

plt.subplot(323)
plt.imshow(img_high,'gray')
plt.title('高频融合')

plt.subplot(324)
plt.imshow(img_last)
plt.title('最终融合效果')
plt.show()
'''