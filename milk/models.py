from django.db import models


class Progress(models.Model):
    """進捗を表すモデル"""
    num = models.IntegerField('進捗', default=0)
    target = models.CharField('フォルダ', max_length=200)
    nb_train = models.IntegerField('訓練枚数', default=0)


class History(models.Model):
    """model.fitのhistory"""
    progress = models.ForeignKey(Progress, on_delete=models.CASCADE)
    epochs = models.IntegerField('epoch', default=0)
    acc = models.FloatField('accuracy', default=0)
    val_acc = models.FloatField('validation_accuracy', default=0)

    fpr = models.CharField(max_length=5000)
    tpr = models.CharField(max_length=5000)
    auc = models.FloatField('auc', default=0)
