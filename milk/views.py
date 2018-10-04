import base64
from io import BytesIO
from django.shortcuts import render
from django.views import generic
from .forms import ImageUploadForm, DirectoryPathForm
from .lib import predict, load_data
import numpy as np
from PIL import Image


class UploadView(generic.FormView):
    template_name = 'milk/upload.html'
    form_class = ImageUploadForm

    def form_valid(self, form):
        # アップロードファイル本体を取得
        file = form.cleaned_data['file']
        # ファイルを、28*28にリサイズし、グレースケール(モノクロ画像)
        img = Image.open(file).resize((28, 28)).convert('L')

        # 学習時と同じ形に画像データを変換する
        img_array = np.asarray(img) / 255
        img_array = img_array.reshape(1, 784)

        # 推論した結果を、テンプレートへ渡して表示
        context = {
            'result': predict(img_array),
        }
        return render(self.request, 'milk/result.html', context)


class TrainView(generic.FormView):
    template_name = 'milk/train.html'
    form_class = DirectoryPathForm

    def form_valid(self, form):
        # アップロードファイル本体を取得
        target_dir = form.cleaned_data['target_directory']
        # files = load_data(target_dir)
        # img = Image.open(files[0])
        # print(img.size)

        context = {
             'target_directory': target_dir,
        }
        return render(self.request, 'milk/build_model.html', context)


class PaintView(generic.TemplateView):
    template_name = 'milk/paint.html'

    def post(self, request):
        base_64_string = request.POST['img-src'].replace(
            'data:image/png;base64,', '')
        file = BytesIO(base64.b64decode(base_64_string))

        # ファイルを、28*28にリサイズし、グレースケール(モノクロ画像)
        img = Image.open(file).resize((28, 28)).convert('L')

        # 学習時と同じ形に画像データを変換する
        img_array = np.asarray(img) / 255
        img_array = img_array.reshape(1, 784)

        # 推論した結果を、テンプレートへ渡して表示
        context = {
            'result': predict(img_array),
        }
        return render(self.request, 'milk/result.html', context)
