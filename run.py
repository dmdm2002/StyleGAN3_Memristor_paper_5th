import os

folders = ['SoftWareNoise', 'Memristor-BinaryGaussian-offline']

for folder in folders:
    os.system(
        f'python gen_cal_image_metrics.py --outdir=E:/5th/backup/ND/{folder}/ --trunc=1 --seeds=0 --network=E:/5th/backup/ND/{folder}')


