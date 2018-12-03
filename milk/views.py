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
import matplotlib.pyplot as plt

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

        # acc, val_accを数値で受け取る
        progress.history_set.create(
            acc=hist.history['acc'][0],
            val_acc=hist.history['val_acc'][0],
            epochs=epoch
        )

        progress.num = int( (epoch+1) * 100/epochs )  # progress.htmlのprogress barに表示する．Max100に広げる
        progress.save()

    model.save('model.h5')

def progress(request, pk):
    """現在の進捗ページ"""
    progress = get_object_or_404(Progress, pk=pk)
    history = progress.history_set.all()
    context = {
        'progress': progress,
        'history_list': history,
        'epochs': list(history.values_list('epochs', flat=True)),
        'acc_list': list(history.values_list('acc', flat=True)),
        'val_acc_list': list(history.values_list('val_acc', flat=True))
    }
    return render(request, 'milk/progress.html', context)
