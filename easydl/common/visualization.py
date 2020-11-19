import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(predictor, xs, ys):
    '''
    plot 2D decision boundary for a classifier.
    ``predictor`` shoule be a function, which can be called with ``predictor(data)``, where data has (n_sample, 2) shape
    ``xs`` should be a sequence representing the grid's x coordinates
    ``ys`` should be a sequence representing the grid's y coordinates

    example usage::

    plot_decision_boundary(func, np.arange(-6.0, 8.0, 0.1), np.arange(-8.0, 8.0, 0.1))

    '''
    xx, yy = np.meshgrid(xs, ys, indexing='ij')
    data = np.asarray([xx.flatten(), yy.flatten()]).T
    label = predictor(data)
    label.resize(xx.shape)
    plt.contourf(xx, yy, label, alpha=0.3)


def extended_confusion_matrix(y_true, y_pred, true_labels=None, pred_labels=None):
    '''
    compute confusion matrix(extended). classes in ``y_true`` can be different with classes in ``y_pred``.
    this is useful if you have 12 classes and want to divide them into 2 classes etc
    '''

    if not true_labels:
        true_labels = sorted(list(set(list(y_true))))
    true_label_to_id = {x : i for (i, x) in enumerate(true_labels)}
    if not pred_labels:
        pred_labels = true_labels
    pred_label_to_id = {x : i for (i, x) in enumerate(pred_labels)}
    confusion_matrix = np.zeros([len(true_labels), len(pred_labels)])
    for (true, pred) in zip(y_true, y_pred):
        confusion_matrix[true_label_to_id[true]][pred_label_to_id[pred]] += 1.0
    return confusion_matrix


def plot_confusion_matrix(cm, true_classes,pred_classes=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    pred_classes = pred_classes or true_classes
    if normalize:
        cm = cm.astype(np.float) / np.sum(cm, axis=1, keepdims=True)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    true_tick_marks = np.arange(len(true_classes))
    plt.yticks(true_tick_marks, true_classes)
    pred_tick_marks = np.arange(len(pred_classes))
    plt.xticks(pred_tick_marks, pred_classes, rotation=45)


    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()