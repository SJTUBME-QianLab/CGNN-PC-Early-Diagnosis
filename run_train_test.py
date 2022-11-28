import os
from time import gmtime, strftime, localtime

seed_ = 2
run_name = strftime("%Y%m%d_%H%M%S", localtime())
model = '3hos_renji'
lr = 0.001
epoch = 50
batchsize = 2560

lr_gcn = 0.01
epoch_gcn = 60
batchsize_gcn = 20
weight_bg = 0.1

epoch_baseline = 140
epoch_CCM = 140
epoch_spatial = 20
epoch_feature = 25
epoch_AMGNN = 50


# CNN
for fold_index_ in range(0, 5):
    os.system("python patch_predict.py "
              "--gpu 1 "
              "--run_name '' "
              "--lr {lr} "
              "--seed {seed} "
              "--batch_size_train {batchsize} "
              "--batch_size_val {batchsize} "
              "--n_epoch 30 "
              "--epoch {epoch} "
              "--optimizer Adam "
              "--save_name {model}_epoch{epoch}_BatchSize{batchsize}_lr{lr}_seed{seed} "
              "--fold_index {fold_index} ".format(fold_index=fold_index_, run_name1=run_name, model=model,
                                                  epoch=epoch, batchsize=batchsize, lr=lr, seed=seed_))


# GCN
for fold_index_ in range(0, 5):
    os.system("python CGNN_ours.py "
              "--gpu 1 "
              "--epoch {epoch_gcn} "
              "--lr {lr_gcn} "
              "--weight_bg {weight_bg} "
              "--result_path ./result/{model}_epoch{epoch}_BatchSize{batchsize}_lr{lr}_seed{seed}/ "
              "--result_path_gcn CGNN/gcn_N50_ch16_8_gauss30_cat_bg{weight_bg}_BCE_lr{lr_gcn}_epoch{epoch_gcn}_bs{batchsize_gcn}_{run_name1} "
              "--fold_index {fold_index}".format(fold_index=fold_index_, run_name1=run_name, model=model,
                                                 epoch=epoch, batchsize=batchsize, lr=lr, seed=seed_, lr_gcn=lr_gcn,
                                                 epoch_gcn=epoch_gcn, batchsize_gcn=batchsize_gcn, weight_bg=weight_bg))

