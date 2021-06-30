import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer

def evaluation_attribution(yte, yte_):
    if yte.ndim>1 or yte_.ndim>1:
        #not an attribution-like response (must be a single array of single-labels)
        return -1,-1,-1
    acc = (yte == yte_).mean()
    f1Macro = f1_score(yte, yte_, average='macro')
    f1micro = f1_score(yte, yte_, average='micro')
    return acc, f1Macro, f1micro


def evaluation_verification(yte, yte_):
    if yte.ndim==1 and yte_.ndim==1:
        #from single-label response representation to binary response representation
        mlb = MultiLabelBinarizer()
        yte = mlb.fit_transform(yte.reshape(-1,1))
        yte_ = mlb.transform(yte_.reshape(-1,1))

    acc = (yte==yte_).mean()
    Mf1 = f1_score(yte, yte_, average='macro')
    mf1 = f1_score(yte, yte_, average='micro')

    # acc = []
    # f1 = []
    # authors = np.unique(yte)
    # yte = np.array(yte)
    # yte_ = np.array(yte_)
    # for author in authors:
    #     yte_v = yte == author
    #     yte_v_ = yte_ == author
    #     acc_v = (yte_v == yte_v_).mean()
    #     acc.append(acc_v)
    #     f1.append(f1_score(yte_v,yte_v_, average='binary'))
    #
    # acc = np.mean(acc)
    # f1 = np.mean(f1)

    return acc, Mf1, mf1


# PAN-style evaluation
def evaluation(y_pred, y_prob, y_true):
    y_pred_array = np.array(y_pred)
    y_prob_array = np.array(y_prob)
    y_true_array = np.array(y_true)

    acc = (y_pred_array == y_true_array).mean()
    f1 = f1_score(y_true_array, y_pred_array)
    auc = roc_auc_score(y_true_array, y_prob_array)
    pan_eval = acc * auc

    print('Accuracy = {:.3f}'.format(acc))
    print('F1 = {:.3f}'.format(f1))
    print('AUC = {:.3f}'.format(auc))
    print('Acc*AUC = {:.3f}'.format(pan_eval))
    print('true:', y_true)
    print('pred:', y_pred)

    return pan_eval