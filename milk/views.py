import time
from multiprocessing import Process

from django.shortcuts import redirect, render, get_object_or_404
from django.views import generic

from .models import Progress
from .forms import ImageUploadForm, DirectoryPathForm
from .lib import predict, load_data
import numpy as np
from PIL import Image

import os
import matplotlib.pyplot as plt

print(os.getcwd())


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
    optimizer = 'adam'
    loss = 'categorical_crossentropy'
    metrics = ['acc']
    epochs = 30
    batch_size = 16

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    history = dict(acc=[], val_acc=[])
    for epoch in range(epochs):
        hist = model.fit(x_train, y_train,
                            epochs=1,
                            batch_size=batch_size,
                            validation_data=(x_test, y_test)
                            )
        # acc, val_accを数値で受け取る
        history['acc'].append(hist.history['acc'])
        history['val_acc'].append(hist.history['val_acc'])
        # accのグラフを描画
        plt.plot(history['acc'], "o-", label="accuracy")
        plt.plot(history['val_acc'], "o-", label="validation_accuracy")
        plt.title('model accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.ylim([0, 1.05])
        plt.legend(loc="lower right")
        plt.savefig(MEDIA_ROOT+'/milk/model_acc.png')
        # progress.htmlのprogress barに表示する．Max100に広げる
        progress.epochs = int( (epoch+1) * 100/30 )
        progress.save()
    model.save('model.h5')

def train_model(model, x_train, y_train, x_test, y_test):
    optimizer = 'adam'
    loss = 'categorical_crossentropy'
    metrics = ['acc']
    epochs = 30
    batch_size = 16
    callbacks = []

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        validation_data=(x_test, y_test)
                        )

def progress(request, pk):
    """現在の進捗ページ"""
    context = {
        'progress': get_object_or_404(Progress, pk=pk)
    }
    return render(request, 'milk/progress.html', context)
