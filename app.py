import streamlit as st
import subprocess
from PIL import Image
import torch
from torchvision import transforms, models
import os

# ===== 認証セクション =====
PASSWORD = "nekoinu123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    user_pw = st.text_input("パスワードを入力してください", type="password")
    if user_pw == PASSWORD:
        st.session_state.logged_in = True
        st.success("ログイン成功！")
        st.rerun()  # 再描画してパスワード欄を非表示に
    elif user_pw != "":
        st.warning("正しいパスワードを入力してください。")
    st.stop()

# クラス名（あなたのラベル順に合わせて）
class_names = ['cats', 'dogs']

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
            save_path = f"my_dataset/train/{correct_label}/{uploaded_file.name}"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            try:
                img = Image.open(uploaded_file).convert("RGB")
                img.save(save_path, format="JPEG")  # 拡張子に関係なくJPEGで保存
                st.success(f"正解として「{correct_label}」で記録しました ✅")

                # 再学習
                with st.spinner("再学習中..."):
                    result = subprocess.run(["python3", "train-model.py"], capture_output=True, text=True)
                    if result.returncode == 0:
                        st.success("✅ 再学習が完了しました！")
                    else:
                        st.error("❌ 再学習に失敗しました")
                        st.text(result.stderr)

            except Exception as e:
                st.error(f"画像の保存に失敗しました: {e}")
    else:
        st.info("正解と判定されたため記録しませんでした。")

# ==== 再スタートボタン ====
st.markdown("---")
if st.button("🔄 最初からやり直す"):
    for key in list(st.session_state.keys()):
        if key != "logged_in":
            del st.session_state[key]
    st.rerun()
