from jit_DExtended_model import DeepJITExtended
from jit_utils import mini_batches_DExtended
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import torch 
from tqdm import tqdm

def evaluation_model(data, params):
    cc2ftr, pad_msg, pad_code, labels, dict_msg, dict_code = data
    batches = mini_batches_DExtended(X_ftr=cc2ftr, X_msg=pad_msg, X_code=pad_code, Y=labels)

    params.vocab_msg, params.vocab_code = len(dict_msg), len(dict_code)
    if len(labels.shape) == 1:
        params.class_num = 1
    else:
        params.class_num = labels.shape[1]
    params.embedding_ftr = cc2ftr.shape[1]

    # set up parameters
    params.cuda = (not params.no_cuda) and torch.cuda.is_available()
    del params.no_cuda
    params.filter_sizes = [int(k) for k in params.filter_sizes.split(',')]

    model = DeepJITExtended(args=params)
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(params.load_model))

    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        all_predict, all_label = list(), list()
        for i, (batch) in enumerate(tqdm(batches)):
            ftr, pad_msg, pad_code, label = batch
            if torch.cuda.is_available():
                ftr = torch.tensor(ftr).cuda()
                pad_msg, pad_code, labels = torch.tensor(pad_msg).cuda(), torch.tensor(
                    pad_code).cuda(), torch.cuda.FloatTensor(label)
            else:
                ftr = torch.tensor(ftr).long()
                pad_msg, pad_code, label = torch.tensor(pad_msg).long(), torch.tensor(pad_code).long(), torch.tensor(
                    labels).float()
            if torch.cuda.is_available():
                predict = model.forward(ftr, pad_msg, pad_code)
                predict = predict.cpu().detach().numpy().tolist()
            else:
                predict = model.forward(ftr, pad_msg, pad_code)
                predict = predict.detach().numpy().tolist()
            all_predict += predict
            all_label += labels.tolist()
    #print(all_label.shape)
    #print(all_predict.shape)
    # Apply threshold: if value is greater than 0.5 set to 1 else set to 0
    all_predict = [1 if x > 0.5 else 0 for x in all_predict]
    auc_score = roc_auc_score(y_true=all_label,  y_score=all_predict)
    accuracy = accuracy_score(y_true=all_label, y_pred=all_predict)
    precision = precision_score(y_true=all_label, y_pred=all_predict)
    recall = recall_score(y_true=all_label, y_pred=all_predict)
    f1 = f1_score(y_true=all_label, y_pred=all_predict)
    matthews = matthews_corrcoef(y_true=all_label, y_pred=all_predict)
    print('Test data -- Accuracy:', accuracy)
    print('Test data -- Precision:', precision)
    print('Test data -- Recall:', recall)
    print('Test data -- F1:', f1)
    print('Test data -- AUC score:', auc_score)
    print('Test data -- Matthews:', matthews)