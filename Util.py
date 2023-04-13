import torch

import torchvision.transforms as transforms

from PIL import Image

def save_filters(filters, img_width, img_height):

    margin = 5

    n = int(len(filters)**0.5)

    width = n * img_width + (n - 1) * margin

    height = n * img_height + (n - 1) * margin

    stitched_filters = Image.new('RGB', (width, height), (0, 0, 0))

    # fill the picture with our saved filters

    for i in range(n):

        for j in range(n):

            index = i * n + j

            if index < len(filters):

                img = filters[i * n + j]

                img = transforms.ToPILImage()(img)

                stitched_filters.paste(img, ((img_width + margin) * i, (img_height + margin) * j))

    # save the result to disk

    stitched_filters.save('stitched_filters_%dx%d.png' % (n, n))

# util function to convert a tensor into a valid image

def deprocess_image(x):

    # normalize tensor: center on 0., ensure std is 0.1

    x -= x.mean()

    x /= (x.std() + 1e-5)

    x *= 0.1

    # clip to [0, 1]

    x += 0.5

    x = torch.clamp(x, 0, 1)

    # convert to RGB array

    x *= 255

    x = x.permute(1, 2, 0).cpu().detach().numpy()

    x = np.clip(x, 0, 255).astype('uint8')

    return x

def normalize(x):

    # utility function to normalize a tensor by its L2 norm

    return x / (torch.sqrt(torch.mean(torch.square(x))) + 1e-5)

