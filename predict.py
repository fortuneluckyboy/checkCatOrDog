from PIL import Image
import torch
from torchvision import transforms, models

# 同じ前処理を再利用
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# モデル構築 & 重み読み込み
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location='cpu'))
model.eval()

# クラス名
class_names = ['cat', 'dog']

def predict_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]
if __name__ == "__main__":
    result = predict_image("./my_dataset/model/test.jpg")  # ← パスは実画像に置き換えて
    print(f"予測結果: {result}")

