
if __name__ == '__main__':
    print("Adjust the loop below to print the sample you want")
    import torch
    from torch.utils.data import DataLoader, random_split
    from torchvision import transforms

    from data.load_data import POST_FIRE_DIR
    from src.data_loader.dataset import WildfireDataset
    from src.model.unet import UNet
    from torch import optim
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np

    from src.data_loader.dataloader import get_mean_std, get_loader

    def adjust_image(image):
        rgb_image = image[:,:,[2, 1, 0]]
        # Normalize the values to range between 0 and 1
        # ABSOLUTELY BAD HACK
        if (np.min(rgb_image) != 0):
            rgb_image = (rgb_image + -1* np.min(rgb_image)) / (-2*np.min(rgb_image))
        #print(np.min(rgb_image))
     
        rgb_image = rgb_image.astype(np.float32)
        #print(rgb_image)
        #rgb_image /= np.max(rgb_image)

        # Apply some adjustments to enhance the image visibility
        gamma = 1.05
        rgb_image = np.clip(rgb_image ** (1/gamma), 0, 1)  # Apply gamma correction

        return rgb_image

    def print_sample(batch, sample_id):
        image = batch['image']
        mask = batch['mask']
        flip_img = image[sample_id].permute(1, 2, 0)
        pre_fire = np.array(flip_img)

        flip_mask = mask[sample_id].permute(1, 2, 0)

        pre_fire_rgb = adjust_image(pre_fire)

        fig = plt.figure(figsize=(20,6))
        ax1 = fig.add_subplot(1,3,1)
        ax1.imshow(pre_fire_rgb)

        ax1 = fig.add_subplot(1,3,2)
        ax1.imshow(flip_mask)

    dt = WildfireDataset(POST_FIRE_DIR, transforms=None)


    loader_args = dict(batch_size=1,
                       shuffle=0)

    trial_loader = get_loader(dt, is_train=1, loader_args=loader_args)
    #mean, std = get_mean_std(trial_loader)

    for batch in trial_loader:
        print_sample(batch, 0)
            
        break



 