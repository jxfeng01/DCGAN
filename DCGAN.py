
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Input,Dense,Conv2D, Conv2DTranspose, Reshape, Flatten,\
    LeakyReLU, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import load_model
from keras.datasets import cifar10
from keras.utils.vis_utils import plot_model
import keras.backend as K

#For FID calculation
from fid import scale_images, calculate_fid
from keras.applications.inception_v3 import InceptionV3

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

class DCGAN():
    def __init__(self):
        
        #Hyperparameters
        self.LEARNING_RATE = 0.0001
        self.BATCH_SIZE = 128
        self.KERNEL_INIT = RandomNormal(0.0,0.02)
        self.BN_MOMENTUM = 0.9
        self.BETA1 = 0.5 
        self.KERNEL_SIZE = int(512)         
        self.INPUT_SHAPE = (32,32,3)
    
        
        self.generator = self.create_generator() 
        self.discriminator = self.create_discriminator()
        self.test_noise = np.random.randn(25,100)
        self.fid_test_noise = np.random.randn(2048,100)
        self.gen_train_model = None
        self.dis_train_model = None
        self.fid_score = []
        
    
    def build(self):
        self.dis_train_model = self.create_dis_model()
        self.gen_train_model = self.create_gen_model()

    def create_generator(self):
        model = Sequential()
        model.add(Dense(4*4*self.KERNEL_SIZE,kernel_initializer = self.KERNEL_INIT,input_dim = 100))
        model.add(Reshape((4,4,self.KERNEL_SIZE)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization(momentum = self.BN_MOMENTUM))
        model.add(Conv2DTranspose(int(self.KERNEL_SIZE/2), kernel_size = 4, strides = 2, padding = 'same', kernel_initializer = self.KERNEL_INIT, activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization(momentum = self.BN_MOMENTUM))
        model.add(Conv2DTranspose(int(self.KERNEL_SIZE/4), kernel_size = 4, strides = 2, padding = 'same', kernel_initializer = self.KERNEL_INIT, activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization(momentum = self.BN_MOMENTUM))
        model.add(Conv2DTranspose(int(self.KERNEL_SIZE/8), kernel_size = 4, strides = 2, padding = 'same', kernel_initializer = self.KERNEL_INIT, activation = 'relu'))
        model.add(BatchNormalization(momentum = self.BN_MOMENTUM))
        model.add(Conv2DTranspose(3, kernel_size = 3, strides = 1, padding = 'same', kernel_initializer = self.KERNEL_INIT, activation = 'tanh'))
        print(model.summary())
        return model

    def create_discriminator(self):
        model = Sequential()
        model.add(Conv2D(int(self.KERNEL_SIZE/8),kernel_size = 3, input_shape = self.INPUT_SHAPE,padding = 'same',kernel_initializer=self.KERNEL_INIT))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(BatchNormalization(momentum = self.BN_MOMENTUM))
        model.add(Conv2D(int(self.KERNEL_SIZE/4),kernel_size = 4, strides = 2, padding = 'same',kernel_initializer=self.KERNEL_INIT))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(BatchNormalization(momentum = self.BN_MOMENTUM))
        model.add(Conv2D(int(self.KERNEL_SIZE/2),kernel_size = 4, strides = 2, padding = 'same',kernel_initializer=self.KERNEL_INIT))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(BatchNormalization(momentum = self.BN_MOMENTUM))
        model.add(Conv2D(self.KERNEL_SIZE,kernel_size = 4, strides = 2, padding = 'same',kernel_initializer=self.KERNEL_INIT))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1,kernel_initializer = self.KERNEL_INIT,activation = 'sigmoid'))
        print(model.summary())
        return model
    
    def create_gen_model(self):
        z = Input(shape=(100,))
        gen_fake = self.generator(z)
        validate = self.discriminator(gen_fake)
        gen_model = Model(z,validate)
        self.discriminator.trainable = False
        self.generator.trainable = True
        gen_model.compile(optimizer = Adam(self.LEARNING_RATE,self.BETA1),loss = 'binary_crossentropy')
        return gen_model

    def create_dis_model(self):
        z = Input(shape = (100,))
        fake_img = self.generator(z)
        real_img= Input(shape = self.INPUT_SHAPE)
        valid_fake = self.discriminator(fake_img)
        valid_real = self.discriminator(real_img)
        dis_model = Model([real_img,z],[valid_real,valid_fake])
        self.generator.trainable = False
        self.discriminator.trainable = True
        dis_model.compile(optimizer = Adam(self.LEARNING_RATE*3,self.BETA1),loss = ['binary_crossentropy','binary_crossentropy'])
        return dis_model

    def fit(self,epochs,X):
        real = np.ones((self.BATCH_SIZE,1))
        fake = np.zeros((self.BATCH_SIZE,1))
        for epoch in range(epochs+1):
            np.random.shuffle(X)
            num_of_batch = int(X.shape[0]//self.BATCH_SIZE)
            progress = tf.keras.utils.Progbar(target = num_of_batch)
            for i in range(num_of_batch):
                progress.update(i)
                real_img = X[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]
                noise = np.random.randn(self.BATCH_SIZE,100)
                self.discriminator.trainable = True
                self.generator.trainable = False
                dis_loss = self.dis_train_model.train_on_batch([real_img,noise],[real,fake])
                if (i%2 == 0):
                    self.discriminator.trainable = False
                    self.generator.trainable = True
                    gen_loss = self.gen_train_model.train_on_batch(noise,real)
                    
            if((epoch)%2 == 0):
                self.generate_img(epoch+1)
#                self.save(epoch+1)
                fid_s = self.fid(X)
                print(fid_s)
                self.fid_score.append(fid_s)
                
        

                
    def fid(self,X):
        model_fid =InceptionV3(include_top = False, pooling = 'avg', input_shape = (299,299,3))
        random_index = np.random.randint(0, X.shape[0], 2048)
        real_img = X[random_index]
        fake_img = self.generator.predict(self.fid_test_noise)
        real_img = scale_images(real_img,(299,299,3))
        fake_img = scale_images(fake_img,(299,299,3))
        fid_score = calculate_fid(model_fid,real_img,fake_img)
        return fid_score

    def save(self,epoch):
        self.generator.save(('Generator Saved Models/celeba_TTS'+str(epoch)+'.h5'))
        self.discriminator.save(('Discriminator Saved Models/celeba_TTS'+str(epoch)+'.h5'))
        self.save('Model/TTS_'+str(epoch))

    def load(self,gen,dis):
        self.generator = load_model(gen)
        self.discriminator = load_model(dis)
    
    def generate_img(self,epoch):
        img = self.generator.predict(self.test_noise)
        img = 0.5 * img +0.5
        n = 0
        fig,loc = plt.subplots(5,5)
        for i in range(5):
            for j in range(5):
                loc[i,j].imshow(img[n])
                loc[i,j].axis('off')
                n+=1
        fig.savefig('Cifarimages/CELEB_'+str(epoch)+'.png')
        plt.close()



if __name__ =="__main__":
    dataset = np.load('celebaData_x32.npy')
#    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#    dataset = np.concatenate((x_train,x_test),axis = 0)    
    dataset = dataset / 127.5 - 1
    EPOCHS = 50
    model = DCGAN()
    model.build()
    model.fit(EPOCHS,dataset)

    
    
