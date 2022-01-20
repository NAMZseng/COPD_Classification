import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from train import count_person_result

# count_person_result('./result/2D_densenet/test_cut_num_rough_50epoch_dir.xlsx',
#                     './result/2D_densenet/test_cut_num_rough_50epoch_dir_person.xlsx')
# diabetes = pd.read_excel('./result/2D_densenet/test_50epoch_lr_dir_person.xlsx')
# diabetes = pd.read_excel('./result/2D_densenet/test_cut_num_precise_50epoch_dir_person.xlsx')
# diabetes = pd.read_excel('./result/2D_densenet/test_cut_num_rough_50epoch_dir_person.xlsx')
# diabetes = pd.read_excel('./result/2D_densenet/test_multi_instance_30epoch_dir_person.xlsx')
# diabetes = pd.read_excel('./result/2D_densenet/test_seg_cut_num_precise_50epoch_dir_2_person.xlsx')

# diabetes = pd.read_excel('./result/3D_densenet/test_3d_50epoch_dir_0.2_step.xlsx')
# diabetes = pd.read_excel('./result/3D_densenet/test_3d_cut_num_precise_50epoch_dir_0.2_step_10valid_2.xlsx')
# diabetes = pd.read_excel('./result/3D_densenet/test_3d_cut_num_rough_50epoch_dir_0.2_step_5valid.xlsx')
# diabetes = pd.read_excel('./result/3D_densenet/test_3d_rough_multi_50epoch_dir_random.xlsx')
# diabetes = pd.read_excel('./result/3D_densenet/test_3d_seg_cut_num_precise_50epoch_dir_0.2_step_1.xlsx')
# diabetes = pd.read_excel('./result/3D_densenet/test_3d_seg_rough_multi_50epoch_dir_random.xlsx')
diabetes = pd.read_excel('./result/3D_densenet/test_3d_seg_cut_size_cut_num_precise_50epoch_dir_0.2_step.xlsx')

fact = diabetes['label_gt']
guess = diabetes['label-pre']

print("每个类别的精确率和召回率：\n", classification_report(y_true=fact, y_pred=guess))

# 混淆矩阵
classes = list(set(fact))
classes.sort()
confusion = confusion_matrix(y_true=fact, y_pred=guess)
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(second_index, first_index, confusion[first_index][second_index])

plt.show()
