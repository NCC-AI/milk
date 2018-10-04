from django import forms


class DirectoryPathForm(forms.Form):
    target_directory = forms.CharField(
        label='データセットフォルダパス',
        max_length=200,
        required=True,
        widget=forms.TextInput()
    )


class ImageUploadForm(forms.Form):
    file = forms.ImageField(label='画像ファイル')
