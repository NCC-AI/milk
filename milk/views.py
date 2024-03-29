import time
from multiprocessing import Process

from django.shortcuts import redirect, render, get_object_or_404
from django.views import generic

from .models import Progress, History
from .forms import ImageUploadForm, DirectoryPathForm
from .lib import predict, load_data
import numpy as np
from PIL import Image
import cv2

from sklearn.metrics import roc_curve, auc
from itertools import cycle
from scipy import interp

import os
import json
import matplotlib.pyplot as plt

from keras import backend as k

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_ROOT = os.path.join(BASE_DIR, 'milk/static')

class UploadView(generic.FormView):
    template_name = 'milk/upload.html'
    form_class = ImageUploadForm

    def form_valid(self, form):
        file = form.cleaned_data['file']
        img = Image.open(file).resize((28, 28)).convert('L')

        img_array = np.asarray(img) / 255
        img_array = img_array.reshape(1, 784)

        context = {
            'result': predict(img_array),
        }
        return render(self.request, 'milk/result.html', context)


class TrainView(generic.CreateView):
    model = Progress
    form_class = DirectoryPathForm
    template_name = 'milk/train.html'

    def form_valid(self, form):
        progress_instance = form.save()
        p = Process(target=update, args=(progress_instance.pk, progress_instance.target), daemon=True)
        p.start()

        return redirect('milk:progress', pk=progress_instance.pk)


class HomeView(generic.CreateView):
    """処理の開始ページ"""
    model = Progress
    form_class = DirectoryPathForm
    template_name = 'milk/home.html'

    def form_valid(self, form):
        print('test: ', form.instance.num)
        progress_instance = form.save()
        p = Process(target=update, args=(progress_instance.pk,), daemon=True)
        p.start()
        return redirect('milk:progress', pk=progress_instance.pk)


def update(pk, directory):
    """裏側で動いている時間のかかる処理"""
    progress = get_object_or_404(Progress, pk=pk)
    (x_train, y_train), (x_test, y_test), model = load_data(directory)

    progress.nb_train = len(x_train)

    nb_classes = y_train.shape[1]
    epochs = 30
    batch_size = 16

    for epoch in range(epochs):
        hist = model.fit(x_train, y_train,
                            epochs=1,
                            batch_size=batch_size,
                            validation_data=(x_test, y_test),
                            verbose=0
                            )

        y_prediction = model.predict(x_train)

        all_fpr, mean_tpr, auc = roc(y_train, y_prediction, nb_classes)

        confusion_visualize(x_train, np.argmax(y_train, axis=1), y_prediction, [i for i in range(nb_classes)])

        # acc, val_accを数値で受け取る
        progress.history_set.create(
            acc=hist.history['acc'][0],
            val_acc=hist.history['val_acc'][0],
            epochs=epoch,
            fpr=json.dumps(all_fpr),
            tpr=json.dumps(mean_tpr),
            auc=auc
        )


        progress.num = int( (epoch+1) * 100/epochs )  # progress.htmlのprogress barに表示する．Max100に広げる
        progress.save()

    model.save('model.h5')
    os.remove(MEDIA_ROOT+'/milk/model.png')

def progress(request, pk):
    """現在の進捗ページ"""
    progress = get_object_or_404(Progress, pk=pk)
    history = progress.history_set.all()
    if len(list(history.values_list('fpr'))) != 0:
        pca_str_list = json.loads(list(history.values_list('fpr'))[0][0])
        pca_int_list = [float(x) for x in pca_str_list]
        fpr = pca_int_list

        pca_str_list = json.loads(list(history.values_list('tpr'))[0][0])
        pca_int_list = [float(y) for y in pca_str_list]
        tpr = pca_int_list

    else:
        fpr = [0]
        tpr = [0]

    context = {
        'progress': progress,
        'history_list': history,
        'epochs': list(history.values_list('epochs', flat=True)),
        'acc_list': list(history.values_list('acc', flat=True)),
        'val_acc_list': list(history.values_list('val_acc', flat=True)),
        'fpr': fpr,
        'tpr': tpr,
        'auc': list(history.values_list('auc', flat=True))
    }
    return render(request, 'milk/progress.html', context)

def roc(y_test, y_prediction, num_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prediction[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= num_classes

    return list(all_fpr), list(mean_tpr), auc(all_fpr, mean_tpr)

def confusion_visualize(images, labels, predict_prob, class_names):
    _, height, width, _ = images.shape
    frame_percent = 2  # outframe
    frame = int((height + width) / 2 * (frame_percent/100)) + 1
    nb_classes = len(class_names)

    ffamily_title = cv2.FONT_HERSHEY_TRIPLEX
    linetype = cv2.LINE_AA
    
    result = np.zeros((height*nb_classes, width*nb_classes, 3))
    matrix = np.zeros((height*nb_classes, width*nb_classes, 3))

    max_predict_num = 0

    if np.max(images) <= 1:
        images *= 255
    predict_cls = np.argmax(predict_prob, axis=1)
    for true_index in range(nb_classes):
        for predict_index in range(nb_classes):
            index_range = np.where((labels==true_index) & (predict_cls==predict_index))
            predict_num = len(index_range[0])  # matrix number
            if max_predict_num < predict_num:
                max_predict_num = predict_num
            
            prob = predict_prob[index_range]
            one_picture = np.zeros((height, width, 3))
            one_picture_matrix = np.zeros((height, width, 3)).astype('uint8')
            if true_index == predict_index:
                one_picture[:, :, 1] = 255
                one_picture_matrix[:, :, 1] = 255
            else:
                one_picture[:, :, 2] = 255
                one_picture_matrix[:, :, 2] = 255

            one_picture_matrix[frame:-frame, frame:-frame] = 0
            if len(prob) == 0:
                one_picture[frame:-frame, frame:-frame] = np.zeros((height-2*frame, width-2*frame, 3))
                
            else:
                sort_range = np.argsort(prob[:, predict_index])[::-1]
                one_picture[frame:-frame, frame:-frame] = images[index_range[0][sort_range][0]][frame:-frame, frame:-frame]
                cv2.putText(one_picture_matrix, str(predict_num), (width//4, height//2), ffamily_title, 0.5, (255,255,255), 1, linetype)
                # one_picture_matrix[frame:-frame, frame:-frame, 0] = predict_num

            result[height*true_index:height*(true_index+1), width*predict_index:width*(predict_index+1)] = one_picture
            matrix[height*true_index:height*(true_index+1), width*predict_index:width*(predict_index+1)] = one_picture_matrix.copy()
        
        # matrix[:, :, 0] *= (255/predict_num)

    cv2.imwrite(MEDIA_ROOT+'/milk/confusion_visualize.png', result)
    cv2.imwrite(MEDIA_ROOT+'/milk/confusion_matrix.png', matrix)
