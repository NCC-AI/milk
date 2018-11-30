from django import forms
from .models import Progress


class DirectoryPathForm(forms.ModelForm):
    class Meta:
        model = Progress
        fields = ("num", "target")


class ImageUploadForm(forms.Form):
    file = forms.ImageField(label='画像ファイル')
