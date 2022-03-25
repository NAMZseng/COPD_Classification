import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize

# diabetes = pd.read_excel('./result/2D_densenet/test_50epoch_lr_dir_person.xlsx')
# n_class = 4
# list1 = diabetes['label_gt']
# list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3']])
#
# y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))
# auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
# fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())
#
# # for i, value in enumerate(thersholds):
# #     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# plt.plot(fpr, tpr, '-', label='Original chest HRCT images(2D)(AUC = {0:.2f})'.format(auc), lw=1)
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('ROC Curve')
# plt.legend(loc="lower right", prop={'size': 7.5})
#
# # ------------------------------------------------------------------
# diabetes = pd.read_excel('./result/2D_densenet/test_cut_num_precise_50epoch_dir_person.xlsx')
# n_class = 4
# list1 = diabetes['label_gt']
# list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3']])
#
# y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))
# auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
# fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())
#
# # for i, value in enumerate(thersholds):
# #     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# plt.plot(fpr, tpr, '-', label='Fine selection (HRCT images,2D) (AUC = {0:.2f})'.format(auc), lw=1)
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('ROC Curve')
# plt.legend(loc="lower right", prop={'size': 7.5})
#
# # ------------------------------------------------------------------
# diabetes = pd.read_excel('./result/2D_densenet/test_cut_num_rough_50epoch_dir_person.xlsx')
# n_class = 4
# list1 = diabetes['label_gt']
# list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3']])
#
# y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))
# auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
# fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())
#
# # for i, value in enumerate(thersholds):
# #     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# plt.plot(fpr, tpr, '-', label='Rough selection (HRCT images,2D)(AUC = {0:.2f})'.format(auc), lw=1)
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('ROC Curve')
# plt.legend(loc="lower right", prop={'size': 7.5})
#
# # ------------------------------------------------------------------
# diabetes = pd.read_excel('./result/2D_densenet/test_multi_instance_30epoch_dir_person.xlsx')
# n_class = 4
# list1 = diabetes['label_gt']
# list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3']])
#
# y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))
# auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
# fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())
#
# # for i, value in enumerate(thersholds):
# #     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# plt.plot(fpr, tpr, '-', label='Multiple instance (HRCT images,2D)(AUC = {0:.2f})'.format(auc), lw=1)
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('ROC Curve')
# plt.legend(loc="lower right", prop={'size': 7.5})
#
# # ------------------------------------------------------------------
# diabetes = pd.read_excel('./result/2D_densenet/test_seg_cut_num_precise_50epoch_dir_2_person.xlsx')
# n_class = 4
# list1 = diabetes['label_gt']
# list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3']])
#
# y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))
# auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
# fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())
#
# # for i, value in enumerate(thersholds):
# #     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# plt.plot(fpr, tpr, '-', label='Original parenchayma images (2D)(AUC = {0:.2f})'.format(auc), lw=1)
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('ROC Curve')
# plt.legend(loc="lower right", prop={'size': 7.5})
#
# # ------------------------------------------------------------------
# diabetes = pd.read_excel('./result/2D_densenet/test_seg_rough_multi_30epoch_dir_person.xlsx')
# n_class = 4
# list1 = diabetes['label_gt']
# list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3']])
#
# y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))
# auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
# fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())
#
# # for i, value in enumerate(thersholds):
# #     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# plt.plot(fpr, tpr, '-', label='Multiple instance (parenchayma,2D)(AUC = {0:.2f})'.format(auc), lw=1)
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('ROC Curve')
# plt.legend(loc="lower right", prop={'size': 7.5})
#
# # ------------------------------------------------------------------
#
#
# diabetes = pd.read_excel('./result/3D_densenet/test_3d_50epoch_dir_0.2_step.xlsx')
# n_class = 4
# list1 = diabetes['label_gt']
# list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3']])
#
# y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))
# auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
# fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())
#
# # for i, value in enumerate(thersholds):
# #     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# plt.plot(fpr, tpr, '-', label='Original chest HRCT images(3D)(AUC = {0:.2f})'.format(auc), lw=1)
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('ROC Curve')
# plt.legend(loc="lower right", prop={'size': 7.5})
#
# # ------------------------------------------------------------------
# diabetes = pd.read_excel('./result/3D_densenet/test_3d_cut_num_precise_50epoch_dir_0.2_step_10valid_2.xlsx')
# n_class = 4
# list1 = diabetes['label_gt']
# list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3']])
#
# y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))
# auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
# fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())
#
# # for i, value in enumerate(thersholds):
# #     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# plt.plot(fpr, tpr, '-', label='Fine selection (HRCT images,3D) (AUC = {0:.2f})'.format(auc), lw=1)
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('ROC Curve')
# plt.legend(loc="lower right", prop={'size': 7.5})
#
# # ------------------------------------------------------------------
# diabetes = pd.read_excel('./result/3D_densenet/test_3d_cut_num_rough_50epoch_dir_0.2_step_5valid.xlsx')
# n_class = 4
# list1 = diabetes['label_gt']
# list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3']])
#
# y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))
# auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
# fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())
#
# # for i, value in enumerate(thersholds):
# #     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# plt.plot(fpr, tpr, '-', label='Rough selection (HRCT images,3D)(AUC = {0:.2f})'.format(auc), lw=1)
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('ROC Curve')
# plt.legend(loc="lower right", prop={'size': 7.5})
#
# # ------------------------------------------------------------------
# diabetes = pd.read_excel('./result/3D_densenet/test_3d_rough_multi_50epoch_dir_random.xlsx')
# n_class = 4
# list1 = diabetes['label_gt']
# list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3']])
#
# y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))
# auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
# fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())
#
# # for i, value in enumerate(thersholds):
# #     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# plt.plot(fpr, tpr, '-', label='Multiple instance (HRCT images,3D)(AUC = {0:.2f})'.format(auc), lw=1)
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('ROC Curve')
# plt.legend(loc="lower right", prop={'size': 7.5})
#
# # ------------------------------------------------------------------
# diabetes = pd.read_excel('./result/3D_densenet/test_3d_seg_cut_num_precise_50epoch_dir_0.2_step_1.xlsx')
# n_class = 4
# list1 = diabetes['label_gt']
# list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3']])
#
# y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))
# auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
# fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())
#
# # for i, value in enumerate(thersholds):
# #     print("%f %f %f" % (fpr[i], tpr[i], value))
#
# plt.plot(fpr, tpr, '-', label='Original parenchayma images (3D)(AUC = {0:.2f})'.format(auc), lw=1)
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
# plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# # plt.title('ROC Curve')
# plt.legend(loc="lower right", prop={'size': 7.5})
#
# # ------------------------------------------------------------------
# diabetes = pd.read_excel('./result/3D_densenet/test_3d_seg_rough_multi_50epoch_dir_random.xlsx')
# diabetes = pd.read_excel('./result/efficientv2/efficientv2_1316.xlsx')
# diabetes = pd.read_excel('./result/resnet18/resnet18_1316_0.77.xlsx')
diabetes = pd.read_excel('./result/resnet_3D/resnet34_img_lrf_3_19.xlsx')
n_class = 4
list1 = diabetes['label_gt']
list2 = np.array(diabetes[['p0', 'p1', 'p2', 'p3']])

y_one_hot = label_binarize(y=list1, classes=np.arange(n_class))
auc = metrics.roc_auc_score(y_one_hot.ravel(), list2.ravel())
fpr, tpr, thersholds = metrics.roc_curve(y_one_hot.ravel(), list2.ravel())

# for i, value in enumerate(thersholds):
#     print("%f %f %f" % (fpr[i], tpr[i], value))

plt.plot(fpr, tpr, '-', label='3D Resnet34_img_lrf(AUC = {0:.2f})'.format(auc), lw=1)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# plt.title('ROC Curve')
plt.legend(loc="lower right", prop={'size': 7.5})

# ------------------------------------------------------------------


# diabetes = pd.read_excel('./result/3D_densenet/test_3d_seg_cut_size_cut_num_precise_50epoch_dir_0.2_step.xlsx')


plt.show()
