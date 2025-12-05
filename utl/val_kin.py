import torch
from sklearn.metrics import roc_curve
import numpy as np
import torch.nn.functional as F


def val_kin_gender(model_, test_loader_, epoch_, device_, save_file):
    model_.eval()
    with torch.no_grad():
        # 定义一个列表来存储预测结果和真实标签
        y_predict = []
        y_predict2 = []
        y_true = []  # 真实标签，即亲属关系标签

        gender_y_predict = []
        gender_y_predict2 = []
        gender_y_true = []  # 真实标签，即亲属关系标签

        type_list = []

        # 预测
        for index_i, data in enumerate(test_loader_):
            img1, img2, kinship_label, type_p, gender_p, gender_c = data
            img1 = img1.to(device_)
            img2 = img2.to(device_)
            kinship_label = kinship_label.to(device_)
            gender_label = torch.cat((gender_p, gender_c), dim=0)

            img1 = img1.to(device_)
            img2 = img2.to(device_)
            _, _, x, gender_x = model_(img1, img2, train=False)
            x = F.softmax(x, dim=1)
            pre2 = x[:, 1]
            pre = torch.argmax(x, dim=1)
            y_predict.extend(pre.tolist())
            y_predict2.extend(pre2.tolist())
            # 将预测结果添加到列表中
            y_true.extend(kinship_label.tolist())  # 将真实标签添加到列表中
            type_list.extend(list(type_p))

            gender = F.softmax(gender_x, dim=1)
            gender_pre2 = gender[:, 1]
            gender_pre = torch.argmax(gender, dim=1)
            gender_y_predict.extend(gender_pre.tolist())
            gender_y_predict2.extend(gender_pre2.tolist())
            # 将预测结果添加到列表中
            gender_y_true.extend(gender_label.tolist())  # 将真实标签添加到列表中

        fd_index = [i for i in range(len(y_predict)) if int(type_list[i] == 'fd')]
        fs_index = [i for i in range(len(y_predict)) if int(type_list[i] == 'fs')]
        md_index = [i for i in range(len(y_predict)) if int(type_list[i] == 'md')]
        ms_index = [i for i in range(len(y_predict)) if int(type_list[i] == 'ms')]

        fd_num, fs_num, md_num, ms_num = len(fd_index), len(fs_index), len(md_index), len(ms_index)
        fpr, tpr, thresholds_keras = roc_curve(y_true, y_predict2, pos_label=1)

        fnr = 1 - tpr  # 计算FNR
        abs_diffs_ = np.abs(fpr - fnr)  # 计算FPR和FNR之间的绝对差值
        min_index_ = np.argmin(abs_diffs_)  # 找到最小差值的索引
        threshold_min = thresholds_keras[min_index_]

        accuracy_min = sum([int(y_predict2[i] >= threshold_min) == y_true[i] for i in range(len(y_predict))]) / len(
            y_predict)
        accuracy_eer_fd = sum([int(y_predict2[i] >= threshold_min) == y_true[i] for i in range(len(y_predict)) if
                               type_list[i] == 'fd']) / fd_num
        accuracy_eer_fs = sum([int(y_predict2[i] >= threshold_min) == y_true[i] for i in range(len(y_predict)) if
                               type_list[i] == 'fs']) / fs_num
        accuracy_eer_md = sum([int(y_predict2[i] >= threshold_min) == y_true[i] for i in range(len(y_predict)) if
                               type_list[i] == 'md']) / md_num
        accuracy_eer_ms = sum([int(y_predict2[i] >= threshold_min) == y_true[i] for i in range(len(y_predict)) if
                               type_list[i] == 'ms']) / ms_num

        with open(save_file + '/result_eer.txt', 'a') as ff:
            ff.write(
                f'{epoch_}\tacc_eer:{accuracy_min:.5f}\tacc_eer_fd:{accuracy_eer_fd:.5f}\tacc_eer_fs:{accuracy_eer_fs:.5f}\t'
                f'acc_eer_md:{accuracy_eer_md:.5f}\tacc_eer_ms:{accuracy_eer_ms:.5f}\tthreshold_eer:{threshold_min:.8f}\t')

        fd_p, fd_t = [y_predict2[index] for index in fd_index], [y_true[index] for index in fd_index]
        fs_p, fs_t = [y_predict2[index] for index in fs_index], [y_true[index] for index in fs_index]
        md_p, md_t = [y_predict2[index] for index in md_index], [y_true[index] for index in md_index]
        ms_p, ms_t = [y_predict2[index] for index in ms_index], [y_true[index] for index in ms_index]

        t = [fd_t, fs_t, md_t, ms_t]
        p = [fd_p, fs_p, md_p, ms_p]
        num = [fd_num, fs_num, md_num, ms_num]
        type_ = ['fd', 'fs', 'md', 'ms']
        for ii in range(4):
            y_true_s, y_predict_s, num_, type_name = t[ii], p[ii], num[ii], type_[ii]
            fpr, tpr, thresholds_keras = roc_curve(y_true_s, y_predict_s, pos_label=1)
            fnr = 1 - tpr  # 计算FNR
            abs_diffs_ = np.abs(fpr - fnr)  # 计算FPR和FNR之间的绝对差值
            min_index_ = np.argmin(abs_diffs_)  # 找到最小差值的索引
            threshold_min = thresholds_keras[min_index_]
            accuracy_eer = sum(
                [int(y_predict_s[i] >= threshold_min) == y_true_s[i] for i in range(len(y_predict_s))]) / num_

            print(epoch_, 'aee_eer_split_' + type_name, accuracy_eer, 'threshold_eer:', threshold_min)
            with open(save_file + '/result_eer.txt', 'a') as ff:
                ff.write(
                    f'acc_eer_split_{type_name}:{accuracy_eer:.5f}\t')
        with open(save_file + '/result_eer.txt', 'a') as ff:
            ff.write(f'\n')

        # 计算性别正确率
        accuracy_min = sum([gender_y_predict[i] == gender_y_true[i] for i in range(len(gender_y_predict))]) / len(
            gender_y_predict)

        with open(save_file + '/gender_result_0.5.txt', 'a') as ff:
            ff.write(
                f'{epoch_}\tacc_:{accuracy_min:.5f}\n')

        fpr, tpr, thresholds_keras = roc_curve(gender_y_true, gender_y_predict2, pos_label=1)

        fnr = 1 - tpr  # 计算FNR
        abs_diffs_ = np.abs(fpr - fnr)  # 计算FPR和FNR之间的绝对差值
        min_index_ = np.argmin(abs_diffs_)  # 找到最小差值的索引
        threshold_min = thresholds_keras[min_index_]
        accuracy_min = sum([int(gender_y_predict2[i] >= threshold_min) == gender_y_true[i] for i in
                            range(len(gender_y_predict))]) / len(
            gender_y_predict)

        with open(save_file + '/gender_result_eer.txt', 'a') as ff:
            ff.write(
                f'{epoch_}\tacc_eer:{accuracy_min:.5f} threshold_eer:{threshold_min:.8f}\n')

