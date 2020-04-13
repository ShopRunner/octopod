from torchvision import transforms

imagenet_rgb_means = [0.485, 0.456, 0.406]
imagenet_rgb_std = [0.229, 0.224, 0.225]
resnet_img_size = 224

full_img_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(resnet_img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_rgb_means, imagenet_rgb_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((resnet_img_size, resnet_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_rgb_means, imagenet_rgb_std)
    ]),
}

cropped_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(resnet_img_size),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_rgb_means, imagenet_rgb_std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((resnet_img_size, resnet_img_size)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_rgb_means, imagenet_rgb_std)
    ]),
}
