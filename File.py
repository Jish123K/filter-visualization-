import torch

import torch.nn as nn

import torchvision.models as models

import torchvision.transforms as transforms

# Define regularizations

def blur_regularization(img, grads, size=(3, 3)):

    return nn.functional.avg_pool2d(img, kernel_size=size)

def decay_regularization(img, grads, decay=0.8):

    return decay * img

def clip_weak_pixel_regularization(img, grads, percentile=1):

    clipped = img

    threshold = torch.kthvalue(torch.abs(img).view(-1), int(percentile / 100.0 * img.numel()))

    clipped[torch.abs(img) < threshold] = 0

    return clipped

def gradient_ascent_iteration(loss_function, img):

    img.requires_grad_()

    optimizer = torch.optim.Adam([img], lr=0.1)

    for i in range(20):

        optimizer.zero_grad()

        loss_value, grads_value = loss_function(img)

        loss_value.backward()

        optimizer.step()

        img.data = clip_weak_pixel_regularization(img.data + 0.9 * grads_value, grads_value)

        img.data = decay_regularization(img.data, grads_value)

        img.data = blur_regularization(img.data, grads_value)

    return img.detach()

def visualize_filter(input_img, filter_index, layer, number_of_iterations=20):

    def loss_function(x):

        out = layer(x)

        return -out[0, filter_index].mean(), torch.autograd.grad(out, x)[0]

    img = input_img.clone()

    img = gradient_ascent_iteration(loss_function, img)

    return img

if __name__ == "__main__":

    # Configuration:

    img_size = 128

    num_filters = 16

    layer_name = 'features.29'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pre-trained model

    model = models.vgg16(pretrained=True).features.to(device)

    layer = model._modules[layer_name]

    # Prepare input image

    transform = transforms.Compose([

        transforms.Resize((img_size, img_size)),

        transforms.ToTensor(),

        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])

    input_img = transform(torch.randn(1, 3, img_size, img_size)).to(device)

    # Generate filter visualizations

    filter_indexes = range(num_filters)

    vizualizations = [None] * len(filter_indexes)

    for i, index in enumerate(filter_indexes):

        vizualizations[i] = visualize_filter(input_img, index, layer)

        # Save the visualizations to see the progress made so far

        save_filters(vizualizations, img_size, img_size)

