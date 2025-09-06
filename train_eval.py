import json
import pickle
import torch.nn as nn
import logging
from dataloader import Dataloader
from MTIFGNN import MTIF
from lib.utils import *
from args import parse_args
from sklearn.metrics import roc_curve
from dataset.utils_abide import get_ids

root_folder = './dataset'

args, unknown = parse_args()

if args.device == 0:
    device = torch.device("cuda:0")
else:
    device = torch.device("cuda:1")

print('  Loading dataset ...')
dl = Dataloader()
_, _, nonimg = dl.load_data()
subject_IDs = get_ids()

with open(f'{root_folder}/save/bas_cv', 'rb') as f:
    X, y, idxs_train, idxs_val, idxs_test = pickle.load(f)

with open(f'{root_folder}/save/site_info_cv', 'rb') as f:
    y_site, idxs_train_site, idxs_val_site, idxs_test_site = pickle.load(f)

# overall results
corrects = np.zeros(10, dtype=np.int32)
accs = np.zeros(10, dtype=np.float32)
validation_accs = np.zeros(10, dtype=np.float32)
aucs = np.zeros(10, dtype=np.float32)
f1_scores = np.zeros(10)
sensitivities = np.zeros(10)
specificities = np.zeros(10)
roc_data = []
folds_detailed_results = ""
# 在全局范围定义两个列表，用于保存所有折中健康和患者的受试者ID
all_hc_ids = set()
all_nd_ids = set()
all_y_true = []
all_y_pred_prob = []


# 10 折交叉验证
for fold in range(10):
    print("\r\n========================= Fold {} ============================".format(fold))
    train_ind = idxs_train[fold]
    val_ind = idxs_val[fold]
    test_ind = idxs_test[fold]

    # extract node features
    node_ftr = dl.get_node_features(train_ind)

    # get PAE inputs
    edge_index, edgenet_input = dl.get_PEWE_inputs(nonimg)

    edge_index1, edge_input1 = construct_knn_graph(node_ftr)

    # normalization for PAE
    edgenet_input = (edgenet_input - edgenet_input.mean(axis=0)) / edgenet_input.std(axis=0)

    # build network (using default parameters)
    model = MTIF(input_dim=2000, embedding_dim=2000, gcn_input_dim=2000, gcn_hidden_filters=16, gcn_layers=3,
                 dropout=0.2, num_classes=2, non_image_dim=3, edge_dropout=0.5, config=args)
    model = model.to(device)

    # build loss, optimizer, metric
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # 学习率 指数衰减
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    features_cuda = torch.tensor(node_ftr, dtype=torch.float32).to(device)
    edge_index = torch.tensor(edge_index, dtype=torch.long).to(device)
    edgenet_input = torch.tensor(edgenet_input, dtype=torch.float32).to(device)
    edge_index1 = edge_index1.clone().detach().to(device)
    edge_input1 = edge_input1.clone().detach().to(device)
    labels = torch.tensor(y, dtype=torch.long).to(device)
    y_site = torch.tensor(y_site, dtype=torch.long).to(device)
    hyper_parameter_folder = f'./cv_models/{args.epochs}-{args.lr}'
    if not os.path.exists(hyper_parameter_folder):
        os.makedirs(hyper_parameter_folder)
    fold_model_path = f'{hyper_parameter_folder}/fold-{fold}.pth'

    # 创建日志文件夹
    log_dir = "./logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 配置 logging，保存到文件，同时输出到控制台
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{log_dir}/1.txt"),  # 保存到文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )


    def train():
        logging.info("  Number of training samples %d" % len(train_ind))
        logging.info("  Start training...\r\n")

        best_acc = 0
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                label_logits = model(features_cuda, edge_index,
                                                                                     edgenet_input, edge_index1,
                                                                                     edge_input1)

                loss_label = loss_fn(label_logits[train_ind], labels[train_ind])

                loss = loss_label
                loss.backward()
                optimizer.step()  # 进行参数更新！！！
                lr_scheduler.step()  # 进行学习率更新！！（这个只能对学习率更新，并不能更新参数！！！）
            correct_train, label_acc_train = accuracy(label_logits[train_ind], labels[train_ind])
            model.eval()
            with torch.set_grad_enabled(False):
                label_logits = model(features_cuda, edge_index, edgenet_input, edge_index1,
                                                           edge_input1)
            label_loss_val = loss_fn(label_logits[val_ind], labels[val_ind])
            correct_val, label_acc_val = accuracy(label_logits[val_ind], labels[val_ind])
            logging.info(
                "Epoch: {},\ttrain loss: {:.5f},\ttrain acc: {:.5f},\tval loss: {:.5f},\tval acc: {:.5f}".format(
                    epoch, loss_label.item(), label_acc_train.item(), label_loss_val.item(), label_acc_val.item()))
            if label_acc_val > best_acc:
                best_acc = label_acc_val
                torch.save(model.state_dict(), fold_model_path)
        validation_accs[fold] = best_acc
        logging.info("  Fold {} best accuracy of val dataset: {:.5f}".format(fold, best_acc))


    def test():
        # global folds_detailed_results
        logging.info("  Number of testing samples %d" % len(test_ind))
        logging.info('  Start testing...')

        model.load_state_dict(torch.load(fold_model_path))
        model.eval()
        label_logits = model(features_cuda, edge_index, edgenet_input, edge_index1, edge_input1)
        corrects[fold], accs[fold] = accuracy(label_logits[test_ind], labels[test_ind])
        y_true = labels[test_ind].detach().cpu().numpy()
        aucs[fold], pos_probs = auc(label_logits[test_ind].detach().cpu().numpy(), y_true)
        y_pred_prob = pos_probs
        y_pred = (y_pred_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_true, y_pred)
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        f1_scores[fold] = metrics['F1']
        sensitivities[fold] = metrics['Sensitivity']
        specificities[fold] = metrics['Specificity']

        logging.info("  Fold {} test accuracy {:.5f}, test auc {:.5f}".format(fold, accs[fold], aucs[fold]))
        logging.info(
            f"  Fold {fold} test accuracy {accs[fold]:.5f}, AUC {aucs[fold]:.5f}, Sensitivity {metrics['Sensitivity']:.5f}, Specificity {metrics['Specificity']:.5f}, F1 {metrics['F1']:.5f}")



    train()
    test()
logging.info("\r\n========================== Finish ==========================")
n_samples = 0
for i in range(len(idxs_test)):
    n_samples += len(idxs_test[i])
acc_nfold = np.sum(corrects) / n_samples
logging.info("=> Average test accuracy in {}-fold CV: {:.5f} ± {:.5f}".format(10, acc_nfold, np.std(corrects)))
logging.info("=> Average test auc in {}-fold CV: {:.5f} ± {:.5f}".format(10, np.mean(aucs), np.std(aucs)))
logging.info(f"=> Average Test F1 Score: {np.mean(f1_scores):.5f} ± {np.std(f1_scores):.5f}")
logging.info(f"=> Average Sensitivity: {np.mean(sensitivities):.5f} ± {np.std(sensitivities):.5f}")
logging.info(f"=> Average Specificity: {np.mean(specificities):.5f} ± {np.std(specificities):.5f}")


