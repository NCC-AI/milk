import time
from multiprocessing import Process

from django.shortcuts import redirect, render, get_object_or_404
from django.views import generic

from .models import Progress, History
from .forms import ImageUploadForm, DirectoryPathForm
from .lib import predict, load_data
import numpy as np
from PIL import Image

import os
import json
import matplotlib.pyplot as plt

from keras import backend as k

print(os.getcwd())
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

        """
        context = {
            'target_directory': target_dir,
            'model': model.layers,
            'files': files
        }
        """
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

    epochs = 30
    batch_size = 16

    for epoch in range(epochs):
        hist = model.fit(x_train, y_train,
                            epochs=1,
                            batch_size=batch_size,
                            validation_data=(x_test, y_test),
                            verbose=0
                            )

        pca_x, pca_y = pca(model, x_train, layer_id=-1)
        

        # print('pca_x: ', pca_x)
        # print('pca_y: ', pca_y)

        # acc, val_accを数値で受け取る
        progress.history_set.create(
            acc=hist.history['acc'][0],
            val_acc=hist.history['val_acc'][0],
            epochs=epoch,
            pca_x=json.dumps(pca_x),
            pca_y=json.dumps(pca_y)
        )


        progress.num = int( (epoch+1) * 100/epochs )  # progress.htmlのprogress barに表示する．Max100に広げる
        progress.save()

    model.save('model.h5')

def progress(request, pk):
    """現在の進捗ページ"""
    progress = get_object_or_404(Progress, pk=pk)
    history = progress.history_set.all()
    if len(list(history.values_list('pca_x'))) != 0:
        pca_str_list = json.loads(list(history.values_list('pca_x'))[0][0])
        pca_int_list = [float(x) for x in pca_str_list]
        pca_x = pca_int_list

        pca_str_list = json.loads(list(history.values_list('pca_y'))[0][0])
        pca_int_list = [float(y) for y in pca_str_list]
        pca_y = pca_int_list

    else: 
        pca_x = [0]
        pca_y = [0]

    context = {
        'progress': progress,
        'history_list': history,
        'epochs': list(history.values_list('epochs', flat=True)),
        'acc_list': list(history.values_list('acc', flat=True)),
        'val_acc_list': list(history.values_list('val_acc', flat=True)),
        'pca_x': pca_x,
        'pca_y': pca_y
    }
    return render(request, 'milk/progress.html', context)

def pca(model, images, layer_id=-2):
    # layer_id = -2 , which means just before final output
    get_fc_layer_output = k.function([model.layers[0].input, k.learning_phase()],
                                    [model.layers[layer_id].output])

    # output in test mode = 0
    features = get_fc_layer_output([images, 0])[0]
    

    # Convert the data set to the main component based on the analysis result
    transformed = fit_transform(features)

    return list(transformed[:, 0].astype('str')), list(transformed[:, 1].astype('str'))

def fit_transform(x):
    n_components = 2

    # 平均を0にする
    x = x - x.mean(axis=0)
    cov_ = np.cov(x, rowvar=False)

    # 固有値と固有ベクトルを求めて固有値の大きい順にソート
    l, v = np.linalg.eig(cov_)
    l_index = np.argsort(l)[::-1]
    v_ = v[:,l_index] # 列ベクトルなのに注意

    # components_（固有ベクトル行列を途中まで取り出す）を作る
    components_ = v_[:,:n_components].T

    # データとcomponents_をかける
    # 上と下で二回転置してるのアホ・・・
    T = (np.mat(x)*(np.mat(components_.T))).A

    # 出力
    return T
