from django.db import models


class Progress(models.Model):
    """進捗を表すモデル"""
    num = models.IntegerField('進捗', default=0)
    target = models.CharField('フォルダ', max_length=200)

    def __str__(self):
        return self.num
