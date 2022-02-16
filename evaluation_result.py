# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 16:16:42 2021

@author: HP
"""
from keras.models import Sequential
from keras.layers import Conv2D,Activation, MaxPooling2D,Dense,Flatten
from tensorflow.keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_False_Positive_vs_True_Positive
import tensorflow as tf

def load_model():
    model1 = tf.keras.models.load_model('C:/Users/HP/spyder/Model/medical_diagnosis_cnn_model.h5')
    return model1
    
def Evaluate_CNN_Model():
    
    model = load_model()
     
    
    batch_size = 32
    test_datagen = ImageDataGenerator(rescale = 1.0/255.0, featurewise_center = True, featurewise_std_normalization= True)
    test_it = test_datagen.flow_from_directory('C:/Users/HP/Downloads/archive/Data/test',classes = ('normal','abnormal'), shuffle = False, batch_size=batch_size, target_size = (224,224))
    y_true = test_it.classes
    
    y_pred = model.predict_generator(test_it, steps = len(test_it), verbose = 1)
    
    y_pred_prob = y_pred[:,1]
    
    y_pred_binary = y_pred_prob > 0.5
    
    print('\nConfusion Matrix\n --------------------')
    print(confusion_matrix(y_true, y_pred_binary))
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    print('Accuracy: %f' % accuracy)
    
    precision = precision_score(y_true, y_pred_binary)
    print('Precision: %f' % precision)
    
    recall = recall_score(y_true, y_pred_binary)
    print('Recall: %f' % recall)
    
    f1 = f1_score(y_true, y_pred_binary)
    print('F1 score: %f' % f1)
    
    auc = roc_auc_score(y_true, y_pred_prob)
    print('ROC AUC: %f' % auc)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    
    plt_False_Positive_vs_True_Positive.plot(fpr, tpr, linestyle='--', label='')
    
    plt_False_Positive_vs_True_Positive.xlabel('False Positive Rate')
    plt_False_Positive_vs_True_Positive.ylabel('True Positive Rate')
    
    plt_False_Positive_vs_True_Positive.legend()
    
    plt_False_Positive_vs_True_Positive.show()
    
    
Evaluate_CNN_Model()