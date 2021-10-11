import tensorflow as tf
import numpy as np
import unit
import scipy.io as sio
import matplotlib.pyplot as plt
import os,json
from PIL import Image

class Model():

    def __init__(self,args,sess):
        self.args = args
        self.sess = sess
        info = sio.loadmat(os.path.join(self.args.tfrecords,'info.mat'))
        self.shape = info['shape']
        self.dim = info['dim']
        self.cube = args.cube
        self.tfrecords = args.tfrecords
        self.class_num = int(info['class_num'])
        self.data = info['data']
        self.imGIS = info['data_gt']
        self.total_train_num = info['total_train_num']
        self.result = args.result
        self.c_n = int(self.dim / args.c_r)
        self.data_path = args.data_path
        self.supervise_batch = args.supervise_batch # supervise train number
        self.ratio_cc = args.ratio_cc

        self.n_input = args.n_input
        self.n_output = args.n_output
        self.PhaseNumber = args.PhaseNumber
        self.numComponents = args.numComponents
        

        self.weight_learnable = args.weight_learnable
        self.global_step = tf.Variable(0,trainable=False)
        if args.use_lr_decay:
            self.lr = tf.train.exponential_decay(learning_rate=args.lr,
                                             global_step=self.global_step,
                                             decay_rate=args.decay_rete,
                                             decay_steps=args.decay_steps)
        else:
            self.lr = args.lr
        
        self.image = tf.placeholder(dtype=tf.float32, shape=(None, self.cube,self.cube,self.dim,1))
        #self.X_output = tf.placeholder(dtype=tf.float32, shape=(None, self.cube,self.cube,self.dim,1))
        #self.image = tf.placeholder(dtype=tf.float32, shape=(None, self.dim*self.cube*self.cube))
        #self.X_output = tf.placeholder(dtype=tf.float32, shape=(None, self.dim*self.cube*self.cube))
        self.label = tf.placeholder(dtype=tf.int64, shape=(None, 1))

        self.reconstruction= self.reconstruction
        self.classifier = self.classifier
        
        self.prediction,self.pre_symetric, self.feature= self.reconstruction(self.image)
        print("self.prediction",self.prediction.shape)
        self.pre_label = self.classifier(self.feature)
        self.model_name = os.path.join('model.ckpt')
        self.loss()
        self.summary_write = tf.summary.FileWriter(os.path.join(self.args.log),graph=tf.get_default_graph())
        self.saver = tf.train.Saver(max_to_keep=100)

    def compute_cost(self, prediction, image, PhaseNumber):
        #prediction=tf.reshape(prediction, shape=[-1, 220])
        print("prediction",prediction.shape, "image", image.shape)
        cost = tf.reduce_mean(tf.square(prediction - image))
        cost_sym = 0
        for k in range(PhaseNumber):
            cost_sym += tf.reduce_mean(tf.square(self.pre_symetric[k]))
        return [cost, cost_sym]

    def loss(self):
        with tf.name_scope('loss'):
            #pp = int(self.cube//2)
            pp=0
            #self.image =tf.reshape(self.image, shape=[-1, self.cube, self.cube, self.dim, 1])
            o_imge = self.image[:,pp:pp+3,pp:pp+3,:,:]  #(?, 1, 1, 220, 1)
            print("self.o_imge",o_imge.shape)
            self.o_imge = tf.layers.flatten(o_imge) #(?, 220)            
            #print(self.o_imge, self.prediction, self.pre_symetric)
            #self.loss_mse = tf.losses.mean_squared_error(self.o_imge,self.decode,scope='loss_mse')
            [cost, cost_sym] = self.compute_cost(self.prediction, self.o_imge, self.PhaseNumber)
            self.loss_mse = cost + 0.01*cost_sym
            
            self.label_ = self.label[0:self.args.supervise_batch,:]
            self.pre_label_ = self.pre_label[0:self.args.supervise_batch,:]
            self.loss_cross_entropy = tf.losses.sparse_softmax_cross_entropy(self.label_,self.pre_label_,scope='loss_cross_entropy')
            self.loss_cross_entropy = tf.reduce_mean(self.loss_cross_entropy)
            
            self.alpha = tf.Variable(initial_value=tf.constant(self.args.ratio_cc,dtype=tf.float32))# for mse
            self.beta = tf.Variable(initial_value=tf.constant(1,dtype=tf.float32))# for crossentropy

            if self.weight_learnable:
                self.loss_total = self.loss_mse*self.alpha + self.loss_cross_entropy*self.beta
            else:
                self.loss_total = self.loss_mse * self.args.ratio_cc + self.loss_cross_entropy
            
            tf.add_to_collection('losses',self.loss_total)

            tf.summary.scalar('loss_cross_entropy',self.loss_cross_entropy)
            tf.summary.scalar('loss_mse',self.loss_mse)
            tf.summary.scalar('loss_total',self.loss_total)
            tf.summary.scalar('learning_rate',self.lr)

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss_total,global_step=self.global_step)
        self.merged = tf.summary.merge_all()

    
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def add_con3d_weight_bias(self, w_shape, b_shape, order_no):
        Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer(), name='Weights_%d' % order_no)
        biases = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
        return [Weights, biases]

    def sampling(self, X_input):
        weights = self.weight_variable([self.n_output, self.n_input]) #shape=(200, 100)
        bias = self.bias_variable([self.n_input])
        print("X_input:",X_input.shape,"weights:",weights.shape)
        X_sampling = tf.matmul(X_input, weights) + bias
        return X_sampling

    def initial_reconstruction(self, X_sampling): 
        weights = self.weight_variable([self.n_input, self.n_output])
        bias = self.bias_variable([self.n_output])
        X0 = tf.matmul(X_sampling, weights) + bias
        return X0

    def ista_block(self, input_tensor, input_layers, layer_no):
        tau_value = tf.Variable(0.1, dtype=tf.float32)  
        lambda_step = tf.Variable(0.1, dtype=tf.float32)
        theta = lambda_step * tau_value
        soft_thr = tf.Variable(0.1, dtype=tf.float32)
        conv_size = 64
        filter_size = 3
        filetr_channel = 3

        if layer_no==0:   
            x2_ista=input_tensor
        else:
            x2_ista=tf.add(tf.add(tf.scalar_mul((1-lambda_step), input_tensor), tf.scalar_mul((lambda_step + theta), input_layers[-1])), (-tf.scalar_mul(theta, input_layers[-2])))

        [Weights0, bias0] = self.add_con3d_weight_bias([filter_size, filter_size, filetr_channel, 1, conv_size], [conv_size], 0)

        [Weights1, bias1] = self.add_con3d_weight_bias([filter_size, filter_size, filetr_channel, conv_size, conv_size], [conv_size], 1)
        #[Weights11, bias11] = self.add_con3d_weight_bias([filter_size, filter_size, filetr_channel, conv_size, conv_size], [conv_size], 11)
        [Weights11, bias11] = self.add_con3d_weight_bias([filter_size, filter_size, filetr_channel, conv_size ,1], [conv_size], 11)
        
        #[Weights2, bias2] = self.add_con3d_weight_bias([filter_size, filter_size, filetr_channel, conv_size, conv_size], [conv_size], 2)
        [Weights2, bias2] = self.add_con3d_weight_bias([filter_size, filter_size, filetr_channel, 1, conv_size], [conv_size], 2)
        [Weights22, bias22] = self.add_con3d_weight_bias([filter_size, filter_size, filetr_channel, conv_size, conv_size], [conv_size], 22)

        [Weights3, bias3] = self.add_con3d_weight_bias([filter_size, filter_size, filetr_channel, conv_size, 1], [1], 3)
        
        print(x2_ista)
        x3_ista = tf.nn.conv3d(x2_ista, Weights0, strides=[1, 1, 1, 1, 1], padding='SAME',name="x3_ista") 
        print(x3_ista)

        x4_ista = tf.nn.relu(tf.nn.conv3d(x3_ista, Weights1, strides=[1, 1, 1, 1, 1], padding='SAME'),name="x4_ista")
        print(x4_ista)

        x44_ista = tf.nn.conv3d(x4_ista, Weights11, strides=[1, 1, 1, 1, 1], padding='SAME',name="x44_ista")
        print(x44_ista) #(?, 1, 1, 220, 1)

        #tf.multip
        x5_ista = tf.multiply(tf.sign(x44_ista), tf.nn.relu(tf.abs(x44_ista) - soft_thr),name="x5_ista")
        print(x5_ista) #(?, 1, 1, 220, 1)

        x6_ista = tf.nn.relu(tf.nn.conv3d(x5_ista, Weights2, strides=[1, 1, 1, 1, 1], padding='SAME'),name="x6_ista")
        print(x6_ista) 

        x66_ista = tf.nn.conv3d(x6_ista, Weights22, strides=[1, 1, 1, 1, 1], padding='SAME',name="x66_ista")
        print(x66_ista)

        x7_ista = tf.nn.conv3d(x66_ista, Weights3, strides=[1, 1, 1, 1, 1], padding='SAME',name="x7_ista")
        print(x7_ista)

        x7_ista = x7_ista + x2_ista

        x3_ista_sym = tf.nn.relu(tf.nn.conv3d(x3_ista, Weights1, strides=[1, 1, 1, 1, 1], padding='SAME'),name="x3_ista_sym")
        x4_ista_sym = tf.nn.conv3d(x3_ista_sym, Weights11, strides=[1, 1, 1, 1, 1], padding='SAME',name="x4_ista_sym")
        x6_ista_sym = tf.nn.relu(tf.nn.conv3d(x4_ista_sym, Weights2, strides=[1, 1, 1, 1, 1], padding='SAME'),name="x6_ista_sym")
        x7_ista_sym = tf.nn.conv3d(x6_ista_sym, Weights22, strides=[1, 1, 1, 1, 1], padding='SAME',name="x7_ista_sym")

        x11_ista = x7_ista_sym - x3_ista

        return x7_ista, x11_ista, x44_ista


    def inference_ista(self, input_tensor, n, reuse): 
        input_tensor=tf.reshape(input_tensor, shape=[-1, 3, 3, self.numComponents, 1])
        layers = []
        layers_symetric = []
        #feature = np.zeros(shape=(1,1,220,1),dtype=np.float32)
        feature = tf.zeros([3,3,self.numComponents,1], dtype=tf.float32)
        layers.append(input_tensor)
        for i in range(n):
            with tf.variable_scope('conv_%d' %i, reuse=tf.AUTO_REUSE):
                conv1, conv1_sym, x44_ista = self.ista_block(input_tensor, layers, i)
                #print("x44_ista:",x44_ista) #shape=(?, 1, 1, 220, 1)
                feature = tf.add(feature, x44_ista) #(?, 1, 1, 220, 1)
                layers.append(conv1)
                layers_symetric.append(conv1_sym)
        return layers, layers_symetric, feature
    
    def reconstruction(self,image):
        print("reconstruction_image:",image.shape)
        image = tf.reshape(image, shape=[-1,self.dim*self.cube*self.cube])
        X_sampling = self.sampling(image)
        X0 = self.initial_reconstruction(X_sampling) #(?, 220)
        prediction, pre_symetric, feature = self.inference_ista(X0, self.PhaseNumber, reuse=False)
        prediction = prediction[-1]
        prediction = tf.layers.flatten(prediction)
        feature = tf.layers.flatten(feature)  #(?, 220)
        return prediction, pre_symetric, feature
    
    def classifier(self, feature):
        feature = tf.expand_dims(feature, 2)
        f_num = 16
        with tf.variable_scope('classifer', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('conv0'):
                conv0 = tf.layers.conv1d(feature, f_num, (8), strides=(1), padding='same')
                conv0 = tf.layers.batch_normalization(conv0)
                conv0 = tf.nn.relu(conv0)
                print(conv0)
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv1d(conv0, f_num * 2, (3), strides=(1), padding='same')
                conv1 = tf.layers.batch_normalization(conv1)
                conv1 = tf.nn.relu(conv1)
                print(conv1)
            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv1d(conv1, f_num * 4, (3), strides=(1), padding='same')
                conv2 = tf.layers.batch_normalization(conv2)
                conv2 = tf.nn.relu(conv2)
                print(conv2)
            with tf.variable_scope('global_info'):
                f_shape = conv2.get_shape()
                feature = tf.layers.conv1d(conv2, self.class_num, (int(f_shape[1])), (1))
                feature = tf.layers.flatten(feature)
                print(feature)
        return feature

    def load(self, checkpoint_dir):
        print("Loading model ...")
        model_name = os.path.join(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(model_name)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(model_name, ckpt_name))
            print("Load successful.")
            return True
        else:
            print("Load fail!!!")
            exit(0)

    def train(self,traindata,saedata,data_model):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.iterate_num = self.args.iter_num
        oa_list,aa_list,kappa_list,ac_llist,matrix_list = list(),list(),list(),list(),list()
        psnr_list = list()
        loss_list,mse_list,ce_list = list(),list(),list()
        for i in range(self.iterate_num):
            (train_data,train_label),(row,col,sae_data) = self.sess.run([traindata,saedata])
            if train_label.shape[0] != self.supervise_batch:
                continue
            feed_data = np.concatenate([train_data,sae_data])
            #print("feed_data:",feed_data.shape) 
            #print("train_label:",train_label.shape)
            _,summery,lr= self.sess.run([self.optimizer,self.merged,self.lr], feed_dict={self.image:feed_data,self.label_:train_label})

            if i % 1000 == 0:
                l_ce, l_mse, l_t = self.sess.run([self.loss_cross_entropy, self.loss_mse, self.loss_total],feed_dict={self.image: feed_data, self.label_: train_label})
                print('step:%d crossentropy:%f mse:%f total:%f lr:%f '%(i,l_ce, l_mse*self.ratio_cc, l_t,lr))
                loss_list.append(l_t)
                mse_list.append(l_mse)
                ce_list.append(l_ce)
                sio.savemat(os.path.join(self.result,'loss_list.mat'),{'loss_t':loss_list,
                                                                    'mse':mse_list,
                                                                    'ce':ce_list})
            if i % 10000 == 0:
                self.saver.save(self.sess, os.path.join(self.args.model, self.model_name), global_step=i)
                dataset_test = data_model.data_parse(
                    os.path.join(self.tfrecords, 'test_data.tfrecords'), type='test')  #(?, 1, 1, 220, 1)
                dataset_sae_test = data_model.data_parse(
                    os.path.join(self.tfrecords, 'sae_test_data.tfrecords'), type='sae_test') #(?, 1)
                oa, aa, kappa, ac_list, matrix = self.test(dataset_test)
                oa_list.append(oa)
                aa_list.append(aa)
                kappa_list.append(kappa)
                ac_llist.append(ac_list)
                matrix_list.append(matrix)
                sio.savemat(os.path.join(self.result, 'result_list.mat'),
                            {'oa': oa_list, 'aa': aa_list, 'kappa': kappa_list, 'ac_list': ac_llist,
                             'matrix': matrix_list})
                #psnr = self.get_decode_image(dataset_sae_test)
                #psnr_list.append(psnr)
                #sio.savemat(os.path.join(self.result, 'psnr_list.mat'), { 'psnr': psnr_list})
            self.summary_write.add_summary(summery,i)

    def test(self,testdata):
        acc_num,test_num = 0,0
        matrix = np.zeros((self.class_num,self.class_num),dtype=np.int32)
        try:
            while True:
                test_data, test_label = self.sess.run(testdata)
                # print(test_data.shape,test_label.shape)
                pre_label = self.sess.run(self.pre_label, feed_dict={self.image:test_data,self.label:test_label})
                pre_label = np.argmax(pre_label,1)
                pre_label = np.expand_dims(pre_label,1)
                acc_num += np.sum((pre_label==test_label))
                test_num += test_label.shape[0]
                print(acc_num,test_num,acc_num/test_num)
                for i in range(pre_label.shape[0]):
                    matrix[pre_label[i],test_label[i]]+=1
        except tf.errors.OutOfRangeError:
            print("test end!")

        ac_list = []
        for i in range(len(matrix)):
            ac = matrix[i, i] / sum(matrix[:, i])
            ac_list.append(ac)
            print(i+1,'class:','(', matrix[i, i], '/', sum(matrix[:, i]), ')', ac)
        print('confusion matrix:')
        print(np.int_(matrix))
        print('total right num:', np.sum(np.trace(matrix)))
        accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
        print('oa:', accuracy)
        # kappa
        kk = 0
        for i in range(matrix.shape[0]):
            kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
        pe = kk / (np.sum(matrix) * np.sum(matrix))
        pa = np.trace(matrix) / np.sum(matrix)
        kappa = (pa - pe) / (1 - pe)
        ac_list = np.asarray(ac_list)
        aa = np.mean(ac_list)
        oa = accuracy
        print('aa:',aa)
        print('kappa:', kappa)

        sio.savemat(os.path.join(self.result, 'result.mat'), {'oa': oa,'aa':aa,'kappa':kappa,'ac_list':ac_list,'matrix':matrix})
        return oa,aa,kappa,ac_list,matrix
    
    def out_mask_to_color_pic(self, data, palette_file='Indian_pines_Palette.json'):
        assert len(data.shape) == 2
        with open(palette_file, 'r') as fp:
            text = json.load(fp)
        color_pic = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
        for i in range(0, data.shape[0]):
            for j in range(0, data.shape[1]):
                assert str(data[i,j]) in list(text.keys())
                color_pic[i,j,:] = text[str(data[i,j])]
        return color_pic

    def save_decode_map(self,sae_data):
        data_gt = self.imGIS
        data_gt = self.out_mask_to_color_pic(data_gt)
        Image.fromarray(data_gt).save(os.path.join(self.result, 'groundtrouth.png'))
        print('Groundtruth map get finished')
        de_map = np.zeros(self.imGIS.shape,dtype=np.int32)
        try:
            while True:
                row,col,map_data = self.sess.run(sae_data)
                pre_label = self.sess.run(self.pre_label, feed_dict={self.image:map_data})
                pre_label = np.argmax(pre_label,1)
                for i in range(pre_label.shape[0]):
                    c,l = row[i],col[i]
                    de_map[c,l] = pre_label[i]+1
        except tf.errors.OutOfRangeError:
            print("test end!")
        de_map = self.out_mask_to_color_pic(de_map)
        Image.fromarray(de_map).save(os.path.join(self.result, 'decode_map.png'))
        print('decode map get finished')

    def get_decode_image(self,data):
        de_image = np.zeros(shape=self.shape[0]) #(145, 145, 220)
        try:
            while True:
                row,col,feed_data = self.sess.run(data)
                de_pixel = self.sess.run(self.prediction, feed_dict={self.image: feed_data})
                for k in range(de_pixel.shape[0]):
                    #print("de_pixel[k]:",de_pixel[k])
                    #print("de_pixel:",de_pixel[k].shape) #(220,)
                    #print(de_image[row[k],col[k],:].shape)#(1, 220)
                    de_image[row[k],col[k],:] = de_pixel[k]
        except tf.errors.OutOfRangeError:
            print("get decode image end!")
        psnr = unit.PSNR(self.data,de_image)
        print('reconstructed PSNR:',psnr)
        sio.savemat(os.path.join(self.result,'decode_image.mat'),{'decode_image':de_image,'psnr':psnr})
        return psnr

    def get_feature(self,data):
        # print(self.shape[0])
        shape = self.shape[0]
        shape[2] = self.c_n
        de_image = np.zeros(shape=shape)
        try:
            while True:
                row,col,feed_data = self.sess.run(data)
                de_pixel = self.sess.run(self.feature, feed_dict={self.image: feed_data})
                for k in range(de_pixel.shape[0]):
                    de_image[row[k],col[k],:] = de_pixel[k]
        except tf.errors.OutOfRangeError:
            print("get decode image end!")
        # psnr = unit.PSNR(self.data,de_image)
        # print('reconstructed PSNR:',psnr)
        sio.savemat(os.path.join(self.result,'feature.mat'),{'feature':de_image})
        # return psnr
