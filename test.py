import torch
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt
from PIL import Image
#------------------------------------------------------------------
path = "best_model.pth" #Put your model path here
test_image = "test.jpg" #Put your image path here
#------------------------------------------------------------------
categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
              'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
              'train', 'tvmonitor']

def test():
    model = deeplabv3_resnet50(num_classes=21)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(test_image)
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        output = model(input_batch)["out"]
        pred = torch.argmax(output, dim=1).squeeze().numpy()
    
    ori_image_np = np.array(image)
    
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.imshow(ori_image_np)
    plt.title("Original")
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(pred, cmap='tab20')
    plt.title("Prediction")
    plt.axis('off')
    plt.show()
    
    for class_id in np.unique(pred):
        if class_id < len(categories):
            print(f"Class {class_id}: {categories[class_id]}")
if __name__ == '__main__':
    test()