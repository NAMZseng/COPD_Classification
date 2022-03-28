from torch.utils.tensorboard import SummaryWriter

log_file = open('./log/3D_resnet/resnet10_img_finetune_train.log', 'r')
writer = SummaryWriter(log_dir='/home/MHISS/zengnanrong/COPD/tensorboard/resnet_3D/resnet10_img_finetune_3_27')

for line in log_file:
    # Epoch 97. Train Loss: 0.139872, Train Acc: 0.476868, Valid Loss: 0.150181, Valid Acc: 0.340909, Time 00:05:02
    if 'Epoch' in line:
        line = line.split()
        writer.add_scalars('Loss', {'Train': float(line[4][:-1]), 'Valid': float(line[10][:-1])}, int(line[1][:-1]))
        writer.add_scalars('Accuracy', {'Train': float(line[7][:-1]), 'Valid': float(line[13][:-1])}, int(line[1][:-1]))

writer.close()
