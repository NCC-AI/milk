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
    files, model = load_data(directory)
    progress.nb_train = len(files)
    print('nb_files', len(files))
    for i in range(1, 11):
        time.sleep(1)
        progress.num = i * 10  # 初回に10、次に20...最後は100が入る。進捗のパーセントに対応
        progress.save()


def progress(request, pk):
    """現在の進捗ページ"""
    context = {
        'progress': get_object_or_404(Progress, pk=pk)
    }
    return render(request, 'milk/progress.html', context)
