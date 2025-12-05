import os.path
import torch.optim as optim
from utl.val_kin import val_kin_gender
import os.path
import torch
from torch.utils.data import DataLoader
from utl.dataset_train import KinshipDataset
from utl.dataset_test import KinshipDataset as KinshipDataset_test
from collections import OrderedDict
from utl.logger import create_logger
import datetime
from timm.utils import AverageMeter
import time
import multiprocessing
from torch.utils.tensorboard import SummaryWriter
from models.swinT_Difference_contrast_sample_balance_fuzzy_gender import KinClassifier
from utl.losses import contrastive_loss

fold = {
    'fold1': dict(train_fold=['2', '3', '4', '5'], test_fold=['1', ]),
    'fold2': dict(train_fold=['1', '3', '4', '5'], test_fold=['2', ]),
    'fold3': dict(train_fold=['1', '2', '4', '5'], test_fold=['3', ]),
    'fold4': dict(train_fold=['1', '2', '3', '5'], test_fold=['4', ]),
    'fold5': dict(train_fold=['1', '2', '3', '4'], test_fold=['5', ]),
}
fold_all = {
    'all': dict(train_fold=['1', '2', '3', '4', '5'], test_fold=['1', '2', '3', '4', '5'])}


def remove_file(file_path):
    try:
        os.remove(file_path)
        print(f"文件 {file_path} 移除成功")
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在")
    except Exception as e:
        print(f"移除文件 {file_path} 时出错: {e}")


def main(fold_name_, data_name_, text_list_, batch_size_, text_list_test, type_lr, num_epochs,
         learning_rate, time__):
    multiprocessing.freeze_support()
    pretrain_path = 'D:\other\\'
    pretrain = None  #'simmim_pretrain.pth'
    # 定义一些超参数
    batch_size = batch_size_
    # 需要修改的参数
    model_ = 'swin_tiny_patch4_window4_64'
    model_type = 'SwinT'

    becl_qz = 0.4
    cl_qz = 0.2
    loss_name = 'bcel-' + str(becl_qz) + '_cl-' + str(cl_qz) + '_' + str(batch_size_) + type_lr + '_lr_' + str(
        learning_rate)

    save_path = ("./output/" + model_ + '/' + data_name_ + '_' + loss_name + '/' + str(
        time__) + '/' + fold_name_)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建日志
    logger = create_logger(output_dir=save_path, name=f"train")
    logger.info(f"Full config saved to {save_path}")
    logger.info(f"model_: {model_},pretrain:{pretrain},model_type:{model_type},batch_size:{batch_size},"
                f"num_epochs:{num_epochs}")

    train_dataset = KinshipDataset(img_path="", data_name=[data_name_, ], txt=text_list_,
                                   transform=None,
                                   fold=fold[fold_name_]['train_fold'], color='rgb_hsv_lab')  # 创建训练数据集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, )

    test_dataset = KinshipDataset_test(img_path='', data_name=[data_name_], txt=text_list_test,
                                       transform=None,
                                       fold=fold[fold_name_]['test_fold'], color='rgb_hsv_lab')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # 创建模型和优化器
    model = KinClassifier().to(device)
    logger.info(str(model))

    if pretrain:
        weight_path = pretrain_path + pretrain
        checkpoint = torch.load(weight_path, weights_only=False)  # 加载预训练权重
        # 与预训练模型参数名不一样，使用手动映射
        pretrained_dict = OrderedDict()  # 创建一个有序空字典
        for kk, v in checkpoint['models'].items():
            if kk.split('.')[0] == 'encoder':
                name = 'feature_extractor.' + kk[len('encoder.'):]
                pretrained_dict[name] = v
        missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)  # 模型和参数不匹配，得重新检查

        print('missing_keys:', missing_keys)
        print('unexpected_keys:', unexpected_keys)

    epoch_ii = 0

    files_pth = [file for file in os.listdir(save_path) if file[-3:] == 'pth']
    if len(files_pth) > 0:
        pretrain = sorted(files_pth, key=lambda x: int(x.split('.')[0]))[-1]
        epoch_ii = int(pretrain.split('.')[0])
        pretrain_path = save_path
        print(f'接着{pretrain}训练')
        weight_path = pretrain_path + '/' + pretrain
        checkpoint = torch.load(weight_path)  # 加载预训练权重
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)  # 模型和参数不匹配，得重新检查
        print('missing_keys:', missing_keys)
        print('unexpected_keys:', unexpected_keys)
    optimizer_model = optim.Adam(model.parameters(), lr=learning_rate)  # 创建优化器

    logger.info("Start training")
    # 开始训练循环
    end = time.time()
    num_steps = len(train_loader)

    # 创建TensorBoard写入器
    writer = SummaryWriter(save_path + '/tensorboard')

    bce_loss = torch.nn.CrossEntropyLoss()

    for epoch_i in range(1 + epoch_ii, num_epochs + 1):
        loss_meter = AverageMeter()
        contrast_loss_meter = AverageMeter()
        bce_loss_meter = AverageMeter()
        gender_bce_loss_meter = AverageMeter()
        batch_time = AverageMeter()
        for index_i, data in enumerate(train_loader):  # 遍历训练数据

            img1, img2, kinship_label, gender_p, gender_c, _ = data
            img1 = img1.to(device)
            img2 = img2.to(device)

            gender_label = torch.cat((gender_p, gender_c), dim=0).to(device)

            kinship_label = torch.cat((kinship_label, 1 - kinship_label)).to(device)
            x1, x2, cl_x, cl_gender = model(img1, img2, train=True)
            contrast_loss = contrastive_loss(x1, x2)

            bcel_loss = bce_loss(cl_x, kinship_label)
            bcel_loss_gender = bce_loss(cl_gender, gender_label)

            optimizer_model.zero_grad()
            loss = cl_qz * contrast_loss + becl_qz * bcel_loss + (1 - cl_qz - becl_qz) * bcel_loss_gender
            loss.backward()

            # 在此处添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer_model.step()
            batch_time.update(time.time() - end)
            end = time.time()
            loss_meter.update(loss.item(), img1.size(0))
            bce_loss_meter.update(becl_qz * bcel_loss.item(), img1.size(0))
            contrast_loss_meter.update(cl_qz * contrast_loss.item(), img1.size(0))
            gender_bce_loss_meter.update((1 - cl_qz - becl_qz) * bcel_loss_gender.item(), img1.size(0))

            etas = batch_time.avg * (num_steps - index_i)

            if index_i % int(len(train_loader) / 10) == 0 or index_i == len(train_loader) - 1 or True:
                lr = optimizer_model.param_groups[0]['lr']
                logger.info(
                    f'Train: [{epoch_i}/{num_epochs}][{index_i + 1}/{num_steps}]\t'
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                    f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                    f'bce_loss {bce_loss_meter.val:.4f} ({bce_loss_meter.avg:.4f})\t'
                    f'contrast_loss {contrast_loss_meter.val:.4f} ({contrast_loss_meter.avg:.4f})\t'
                    f'gender_loss {gender_bce_loss_meter.val:.4f} ({gender_bce_loss_meter.avg:.4f})')

        writer.add_scalar('Loss_total/train', loss_meter.avg, epoch_i)
        writer.add_scalar('Loss_bce/train', bce_loss_meter.avg, epoch_i)
        writer.add_scalar('Loss_contrast/train', contrast_loss_meter.avg, epoch_i)
        writer.add_scalar('Loss_gender/train', gender_bce_loss_meter.avg, epoch_i)
        writer.add_scalar('lr', optimizer_model.param_groups[0]['lr'], epoch_i)

        if epoch_i % 1 == 0:  # or epoch_i > (num_epochs - 10) or epoch_i < 21:
            torch.save(model.state_dict(), save_path + '/' + str(epoch_i) + ".pth")  # 保存模型到指定位置
            if epoch_i - 1 != 20:
                remove_file(save_path + '/' + str(epoch_i - 1) + ".pth")
                print("Model saved.")

        val_kin_gender(model, test_loader, epoch_i, device, save_path)

    # 关闭TensorBoard写入器
    writer.close()


if __name__ == '__main__':
    data_name = 'KFWI'
    batch_size = 16
    epoch_nums = 50
    type_lr = 'fix'
    lr = 4e-5
    train_lists = ['D:\other\KinFaceW-I/txt_list/all_label\list_path.txt']  # 样本列表

    test_lists = train_lists

    for time_ in range(1, 4):
        # time_ = 1

        for k in range(len(train_lists)):
            train_list = [train_lists[k]]

            test_list = [test_lists[k]]

            for fold_name in fold:
                main(fold_name, data_name, train_list, batch_size, test_list, type_lr, epoch_nums,
                     lr, time_)
