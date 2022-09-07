from __future__ import division
import cv2, keras
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import sys,os
import numpy as np


from utilitiestrain import preprocess_imagesandsaliencyforiqa,preprocess_label
import time

from modelfinal import TVdist,SGDNet
import tensorflow.keras.backend as K
import h5py, yaml
import math
from argparse import ArgumentParser

import warnings
warnings.filterwarnings("ignore")
# import keras.backend as K

# def mean_pred(y_true, y_pred):
#     return K.mean(y_pred)

# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy', mean_pred])
from scipy import stats
def evaluationmetrics(_y,_y_pred):
        sq = np.reshape(np.asarray(_y), (-1,))
        q = np.reshape(np.asarray(_y_pred), (-1,))

        srocc = stats.spearmanr(q, sq)[0]   #srocc is not always accurate. It need using matlab code to compute.
        krocc = stats.stats.kendalltau(sq, q)[0]
        plcc = stats.pearsonr(sq, q)[0]
        rmse = np.sqrt(((sq - q) ** 2).mean())
        mae = np.abs((sq - q)).mean()
        return srocc, krocc, plcc, rmse, mae

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def datasetgenerator(conf,log_dir,EXP_ID ='0', mostype = 'ss'):
    # im_dir = conf['im_dir']
    datainfo = conf['datainfo']
        #downsampling = conf['downsampling']
        #data_path = conf['data_path']
        # self.patch_size = conf['patch_size']
        # self.stride = conf['stride']
        # datainfo = conf['datainfo']
    Info = h5py.File(datainfo)
    index = Info['index'][:, int(EXP_ID) % 1000] # 
    # tindex = Info['index'][:, int(EXP_ID) % 1000] # 
    ref_ids = Info['ref_ids'][0, :] #
    test_ratio = conf['test_ratio']  #
    train_ratio = conf['train_ratio']
    trainindex = index[:int(math.ceil(train_ratio * len(index)))]
    testindex = index[int(math.ceil((1-test_ratio) * len(index))) :]
    print ('Training refs:')
    print (trainindex)
    print ('test refs:')
    print (testindex)
    # print len(testindex)
    train_index, val_index, test_index = [],[],[]
    for i in range(len(ref_ids)):
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            test_index.append(i) if (ref_ids[i] in testindex) else \
                val_index.append(i)
    val_index = test_index  # for 8,(2),2 training
    # from 0 to whole index
    train_index.sort()
    print ('Training indexs:')
    # train_index.sort()
    # print (train_index)    
    #print (len(test_index))
    if len(test_index)>0:
        ensure_dir(log_dir)
        testTfile = log_dir+ EXP_ID +'.txt'
        outfile = open(testTfile, "w")

        if mostype == 'DMOS':
            print(outfile, "\n".join(str(Info['DMOS'][0, i])  for i in test_index ))
        elif mostype == 'NMOS':
            print(outfile, "\n".join(str(Info['NMOS'][0, i])  for i in test_index ))
        else:
            print(outfile, "\n".join(str(Info['subjective_scores'][0, i])  for i in test_index ))         
        outfile.close() 
    return  train_index,val_index,test_index  
    
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,conf, batch_size=32, shuffle=True,mirror=False, mostype = 'ss', saliency = 'output'):
        'Initialization'
        # self.dim = dim
        self.batch_size = batch_size
        self.conf = conf
        # self.log_dir = log_dir
        # self.EXP_ID = EXP_ID
        self.list_IDs = list_IDs
        # self.n_channels = n_channels
        # self.n_classes = n_classes
        self.shuffle = shuffle
        self.mirror = mirror
        self.mostype = mostype
        self.saliency = saliency
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_IDs_temp.sort()
        # Generate data
        X, X2,Y  = self.__data_generation(list_IDs_temp)
        if self.saliency == 'input':
            return [X,X2], [Y]
        elif self.saliency == 'output':
            return [X], [Y,X2]
        else:
            return [X], [Y]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        # print ('shuffling!')
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty((self.batch_size, *self.dim, self.n_channels))
        # y = np.empty((self.batch_size), dtype=int)
        sim_dir = self.conf['sim_dir']
        im_dir = self.conf['im_dir']
        datainfo = self.conf['datainfo']
        shape_r =  self.conf['shape_r']  
        shape_c =  self.conf['shape_c']    
        Info = h5py.File(datainfo)
        index = list_IDs_temp
        # print len(index)
        # mos = Info['NMOS'][0, index]
        # mos = Info['DMOS'][0, index]
        if self.mostype == 'DMOS':
            mos = Info['DMOS'][0, index]
        elif self.mostype == 'NMOS':
            mos = Info['NMOS'][0, index]
        else:
            mos = Info['subjective_scores'][0, index]

        im_names = [Info[Info['image_name'][0, :][i]][()].tobytes()\
                                [::2].decode() for i in index]
        # print len(im_names)
        # print self.batch_size
        images = [os.path.join(im_dir, im_names[idx]) for idx in range(len(index))]
        simages = [os.path.join(sim_dir, im_names[idx]) for idx in range(len(index))]
        maps = [mos[idx] for idx in range(len(index))]    
        X,X2 = preprocess_imagesandsaliencyforiqa(images[0:len(index)], simages[0:len(index)], shape_r, shape_c, mirror=self.mirror,crop_h=shape_r , crop_w=shape_c)
        Y = preprocess_label(maps[0:len(index)])      
        #print(X.shape, X2.shape, Y)

        return X,X2,Y



if __name__ == '__main__':
    # phase = sys.argv[1]
    parser = ArgumentParser(description='PyTorch saliency guided CNNIQA')
    parser.add_argument('--batch_size', type=int, default=15,
                        help='input batch size for training (default: 15)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='0', type=str,
                        help='exp id (default: 1)')
    parser.add_argument('--database', default='LIVEc', type=str,
                        help='database name (default: LIVEc)')
    parser.add_argument('--phase', default='train', type=str,
                        help='phase (default: train')
    parser.add_argument('--pretrained', action='store_true',
                        help='use pretrained weight (default: False')   
    parser.add_argument('--fixed', action='store_true',
                        help='use fixed backbone (default: False')   
    parser.add_argument('--out2dim', type=int, default=1024,
                        help='number of epochs to train (default: 1024)')  
    parser.add_argument('--basemodel', default='resnet', type=str,
                        help='resnet or vgg (default: resnet)')      
    parser.add_argument('--saliency', default='output', type=str,
                        help='saliency information as input or output or none (default: output)')          
    parser.add_argument('--mostype', default='ss', type=str,
                        help='dataset label type: DMOS, NMOS, or raw scores (ss) (default: ss)')                         
    parser.add_argument('--CA', action='store_false',
                        help='use CA? (default: true')   
    parser.add_argument('--alpha', default='0.25', type=float,
                        help='alpha for saliency loss (default: 0.25)')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    print('phase: ' + args.phase)
    print('exp id: ' + args.exp_id)
    print('database: ' + args.database)
    print('lr: ' + str(args.lr))
    print('base_model: ' + args.basemodel)
    print('saliency information: ' + args.saliency)
    print('CA: ' + str(args.CA))
    print('mostype: ' + args.mostype)
    print('batch_size: ' + str(args.batch_size))
    config.update(config[args.database])

    log_dir = './data' + '/EXP{}-{}-testtxt/'.format(args.exp_id, args.database)

    b_s= args.batch_size
    crop_h =  config['shape_r']  
    # print (crop_h) #384
    crop_w =  config['shape_c']   
    model = SGDNet(basemodel = args.basemodel, saliency = args.saliency, CA = args.CA,fixed =args.fixed, img_cols=crop_w, img_rows=crop_h,out2dim=args.out2dim )

    print("Compile SGDNet Model")
    opt  = Adam(lr=args.lr)
    alpha = args.alpha
    if args.saliency == 'output':
        model.compile(optimizer=opt, loss= ['mae',TVdist], loss_weights=[1.0/ (1 + alpha),alpha/ (1 + alpha)])
        # model.compile(optimizer=opt, loss= ['mae',TVdist], loss_weights=[1.0, alpha])
    else:
        model.compile(optimizer=opt, loss= ['mae']) 

    model.summary()
    train_index,val_index,test_index = datasetgenerator(config,log_dir,args.exp_id, args.mostype)

    if args.phase == 'train':    
        if args.pretrained == True:
            print("Load weights SGDNet")
            weight_file = '../checkpoint/'+ 'saliencyoutput-alpha0.25-ss-Koniq10k-1024-EXP0-lr=0.0001-bs=19.33-0.1721-0.0817-0.1637-0.2054.pkl'
            model.load_weights(weight_file)  

        nb_imgs_train =  len(train_index) 
        nb_imgs_val =  len(val_index) 
        print("Training SGDNet")
        ensure_dir('../checkpoint/')
        checkpointdir= '../checkpoint/'+ 'saliency{}-alpha{}-{}-{}-{}-EXP{}-lr={}-bs={}'.format(args.saliency,str(args.alpha),args.mostype,args.database,str(args.out2dim),args.exp_id,str(args.lr),str(args.batch_size))
        print (checkpointdir)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7,verbose=1)

        train_generator = DataGenerator(train_index,config, b_s, shuffle=True, mirror= False, mostype= args.mostype, saliency= args.saliency)    
        val_generator = DataGenerator(val_index,config, 1, shuffle=False, mirror= False, mostype= args.mostype, saliency= args.saliency) 
        if args.saliency == 'output':
            model.fit_generator(generator=train_generator,epochs=args.epochs,steps_per_epoch= int(1* nb_imgs_train // b_s),
                                validation_data=val_generator, validation_steps= int(1*nb_imgs_val  // 1),
                                callbacks=[EarlyStopping(patience=50),
                                        ModelCheckpoint(checkpointdir+'.{epoch:02d}-{val_loss:.4f}-{loss:.4f}-{val_predictions_loss:.4f}-{val_saliency_loss:.4f}.pkl', save_best_only=True),
                                            reduce_lr])
        else:            
            model.fit_generator(generator=train_generator,epochs=args.epochs,steps_per_epoch= int(1* nb_imgs_train // b_s),
                                validation_data=val_generator, validation_steps= int(1*nb_imgs_val  // 1),
                                callbacks=[EarlyStopping(patience=30),
                                        ModelCheckpoint(checkpointdir+'.{epoch:02d}-{val_loss:.4f}-{loss:.4f}.pkl', save_best_only=True),
                                            reduce_lr])
    
    elif args.phase == "test":
        # path of output folder
        arg = 'saliencyoutput-alpha0.25-ss-Koniq10k-1024-EXP0-lr=0.0001-bs=19.33-0.1721-0.0817-0.1637-0.2054.pkl'
        print("Load weights SGDNet")
        weight_file = '../checkpoint/'+ arg
        model.load_weights(weight_file)         

        output_folder = 'TestResults/'+arg+ args.database+ '/'
        if os.path.isdir(output_folder) is False:
            os.makedirs(output_folder)

        nb_imgs_test = len(test_index) 
        totalrsults = [0] * nb_imgs_test
        output_folderfileavg = output_folder + 'results_avg' + '.txt'
        start_time0 = time.time()   
        repeat=1  # it should modify when you train/test on patches, otherwise keep it as default 1.
        # totalrsults=[]
        for i in range(repeat):
            output_folderfile = output_folder + 'results'+str(i) + '.txt'
        
            start_time = time.time()    
            test_generator = DataGenerator(test_index,config, 1, shuffle=False, mirror= False, mostype= args.mostype, saliency= args.saliency) 
            predictions = model.predict(test_generator, nb_imgs_test)
            if args.saliency == 'output':
                predictions0 =predictions[0]
            else:
                predictions0 =predictions
            #print len(predictions)        

            elapsed_time2 = time.time() - start_time            
            # print "test no. ", i             
            
            ("total model testing time: " , elapsed_time2)
            results =[]
            for pred in predictions0:
                results.append(float(pred)) 
                 
            outfile = open(output_folderfile, "w")
            print("\n".join(str(i) for i in results))
            outfile.write("\n".join(str(i) for i in results))
            outfile.close()
            totalrsults=[sum(x) for x in zip(results, totalrsults)]
            
        totalrsults=[x/repeat for x in totalrsults]
        outfile = open(output_folderfileavg, "w")
        outfile.write("\n".join(str(i) for i in totalrsults))
        outfile.close()
        
        elapsed_time = time.time() - start_time0   
        # print "test no. ", i         
        print ("total testing time: " , elapsed_time)
        
        with open(output_folderfileavg) as f:
            content = f.readlines()
        maps2 = [float(x.strip()) for x in content] 
        f.close()
        testTfile = log_dir+ args.exp_id +'.txt'
        with open(testTfile) as f:
            content = f.readlines()
        maps = [float(x.strip()) for x in content] 
        f.close() 
        
        #srocc, krocc, plcc, rmse, mae = evaluationmetrics(maps,maps2)
        #print("Testing Results  :SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f} "
        #          .format(srocc, krocc, plcc, rmse, mae))

    elif args.phase == "custom_test":
        # path of output folder
        arg = 'saliencyoutput-alpha0.25-ss-Koniq10k-1024-EXP0-lr=0.0001-bs=19.33-0.1721-0.0817-0.1637-0.2054.pkl'
        print("Load weights SGDNet")
        weight_file = '../checkpoint/'+ arg
        model.load_weights(weight_file)         

        output_folder = 'TestResults/'+"CustomTest"+ '/'
        if os.path.isdir(output_folder) is False:
            os.makedirs(output_folder)

        nb_imgs_test = len(test_index) 
        totalrsults = [0] * nb_imgs_test
        output_folderfileavg = output_folder + 'results_avg' + '.txt'
        start_time0 = time.time()   
        repeat=1  # it should modify when you train/test on patches, otherwise keep it as default 1.
        # totalrsults=[]
        for i in range(repeat):
            output_folderfile = output_folder + 'results'+str(i) + '.txt'
        
            start_time = time.time()    
            test_generator = DataGenerator(test_index,config, 1, shuffle=False, mirror= False, mostype= args.mostype, saliency= args.saliency) 
            predictions = model.predict(test_generator, nb_imgs_test)
            if args.saliency == 'output':
                predictions0 =predictions[0]
            else:
                predictions0 =predictions
            #print len(predictions)        

            elapsed_time2 = time.time() - start_time            
            # print "test no. ", i             
            
            ("total model custom testing time: " , elapsed_time2)
            results =[]
            for pred in predictions0:
                results.append(float(pred)) 
                 
            outfile = open(output_folderfile, "w")
            print("\n".join(str(i) for i in results))
            outfile.write("\n".join(str(i) for i in results))
            outfile.close()
            totalrsults=[sum(x) for x in zip(results, totalrsults)]
            
        totalrsults=[x/repeat for x in totalrsults]
        outfile = open(output_folderfileavg, "w")
        outfile.write("\n".join(str(i) for i in totalrsults))
        outfile.close()
        
        elapsed_time = time.time() - start_time0   
        # print "test no. ", i         
        print ("total testing time: " , elapsed_time)
        
    else:
        raise NotImplementedError