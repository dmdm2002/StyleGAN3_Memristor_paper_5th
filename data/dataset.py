import glob
import numpy as np
import PIL.Image as Image
import torch

from transforms import transform_image_handler


def shuffle_noise_row(noise):
    return np.random.shuffle(noise)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, noise_path, transform):
        super().__init__()

        self.image_list = glob.glob(f"{image_path}/*")
        self.noise_list = glob.glob(f"{noise_path}/*")

        self.transform = transform

    def __len__(self):
        return len(self.noise_list)

    def __getitem__(self, index):
        name = self.image_list[index].split('\\')[-1]

        image = self.transform(Image.open(self.image_list[index]))
        noise = torch.from_numpy(shuffle_noise_row(np.load(self.noise_list[index])))

        return image, noise, name


def get_loader(image_size=224,
               crop=False,
               jitter=False,
               noise=False,
               batch_size=16,
               image_path=None,
               noise_path=None,):

    transform = transform_image_handler(image_size=image_size,
                                        crop=crop,
                                        jitter=jitter,
                                        noise=noise)

    dataset = Dataset(image_path, noise_path, transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return loader

