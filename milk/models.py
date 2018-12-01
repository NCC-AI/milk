from django.db import models


class Progress(models.Model):
    """進捗を表すモデル"""
    num = models.IntegerField('進捗', default=0)
    target = models.CharField('フォルダ', max_length=200)
    nb_train = models.IntegerField('訓練枚数', default=0)
    epochs = models.IntegerField('epoch', default=0)

    def __str__(self):
        return self.num
