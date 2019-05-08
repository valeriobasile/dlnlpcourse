import numpy as np

def precision_recall(pred, gold, c):
    g = set([i for i, e in enumerate(gold) if e == c])
    p = set([i for i, e in enumerate(pred) if e == c])
    tp = len(g.intersection(p))
    if len(p) > 0:
        precision = float(tp)/float(len(p))
    else:
        precision = 0.0
    if len(g) > 0:
        recall = float(tp)/float(len(g))
    else:
        recall = 0.0

    try:
        fscore = (precision * recall * 2.0) / (precision + recall)
    except:
        fscore = 0.0
    return precision, recall, fscore

def evaluate_model(model, X_test, y_test):
    score = model.evaluate(X_test, y_test,
                               batch_size=100, verbose=0)
    pred = model.predict_classes(X_test)
    gold = [np.argmax(i) for i in y_test]

    macrof = 0.0
    for label in np.unique(y_test):
        p, r, f = precision_recall(pred, gold, label)
        macrof += f
        print ("{0}: p={1:.3f}, r={2:.3f}, f={3:.3f}".format(label, p, r, f))
    macrof /= len(np.unique(y_test))
    print ("macro F1-score: {0:.3f}".format(macrof))
    print ("accuracy: {0:.3f}".format(score[1]))
    
