import scipy.io as sio

hsi_path = r'E:\dataset\HS-LiDAR data\Trento\HSI.mat'
hsi_mat = sio.loadmat(hsi_path)  # (166,600,63)
hsi_data = hsi_mat['HSI']

lidar_path = r'E:\dataset\HS-LiDAR data\Trento\LiDAR.mat'
lidar_mat = sio.loadmat(lidar_path)
lidar_data = lidar_mat['LiDAR']

train_label = r'E:\dataset\HS-LiDAR data\Trento\TRLabel.mat'
train_mat = sio.loadmat(train_label)
train_data = train_mat['TRLabel']

test_label = r'E:\dataset\HS-LiDAR data\Trento\TSLabel.mat'
test_mat = sio.loadmat(test_label)
test_data = test_mat['TSLabel']
pass
