import streamlit as st
import subprocess
from PIL import Image
import torch
from torchvision import transforms, models
import os
import shutil

PASSWORD = "nekoinu123"  # あなたが設定したいパスワードに変更OK
user_pw = st.text_input("パスワードを入力してください", type="password")

if user_pw != PASSWORD:
    st.warning("正しいパスワードを入力してください。")
    st.stop()



# クラス名（あなたのラベル順に合わせて）
class_names = ['cat', 'dog']

# モデル読み込み
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# 前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 推論関数
def predict(img: Image.Image) -> str:
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred = torch.max(output, 1)
        return class_names[pred.item()]

# Streamlit UI
st.title("猫か犬か判定 AI（+フィードバック付き）")

uploaded_file = st.file_uploader("画像を選んでください", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="アップロードされた画像", use_column_width=True)

    prediction = predict(img)
    st.subheader(f"AIの予測: {prediction}")

    # フィードバック受付
    feedback = st.radio("予測は正しい？", ["正しい", "間違っている"])

    if feedback == "間違っている":
        correct_label = st.selectbox("正解のクラスは？", class_names)
        if st.button("記録する"):
            # 保存先に分類してコピー（例: feedback_data/）
            save_path = f"my_dataset/train/{correct_label}/{uploaded_file.name}"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"正解として「{correct_label}」で記録しました ✅")
            # ★ ここで再学習スクリプトを呼び出す
            with st.spinner("再学習中..."):
                result = subprocess.run(["python3", "train_model.py"], capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("✅ 再学習が完了しました！")
                else:
                    st.error("❌ 再学習に失敗しました")
                    st.text(result.stderr)
    else:
        st.info("正解と判定されたため記録しませんでした。")
