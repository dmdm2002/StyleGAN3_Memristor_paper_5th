import torch
import glob
import tqdm
import PIL.Image as Image
from torchvision.transforms import ToTensor, ToPILImage

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import FrechetInceptionDistance
from torchmetrics.image import KernelInceptionDistance


gen_images_path = 'E:/5th/ND/Memristor-BinaryGaussian-offline/2-fold/00000-stylegan3-r-B-gpus1-batch4-gamma6.6/best_output/*.png'
gen_images = glob.glob(gen_images_path)

# Preprocess
tf_toTensor = ToTensor()
images = [tf_toTensor(Image.open(p).convert("RGB")).unsqueeze(0).to(torch.uint8).to('cuda') for p in tqdm.tqdm(gen_images, desc="Load")]

ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
lpips = LearnedPerceptualImagePatchSimilarity().to('cuda')

fid = FrechetInceptionDistance(feature=2048).to('cuda')
kid = KernelInceptionDistance(feature=2048, subset_size=500, subsets=50).to('cuda')

gen1 = images[:500]
gen2 = images[500:]

for idx, img in enumerate(gen1):
    fid.update(img, real=True)
    kid.update(img, real=True)

    print(f'Gen-1-Real [{idx}/{len(gen1)}]')

for idx, img in enumerate(gen2):
    fid.update(img, real=False)
    kid.update(img, real=False)

    print(f'Gen-2-Fake [{idx}/{len(gen2)}]')

# for i in range(len(images)):
#     for j in range(i+1, len(images)):
#         ms_ssim.update(images[i], images[j])
#         lpips.update(images[i], images[j])
#
#         print(f'[{i}/{len(images)}] --> [{j}/{len(images)-(i+1)}]')


# print(f'ms_ssim: {ms_ssim.compute()}')
# print(f'LPIPS: {lpips.compute()}')
print(f'FID: {fid.compute()}')
print(f'KID: {kid.compute()}')