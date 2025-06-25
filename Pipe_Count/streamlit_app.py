    # Google DriveのファイルID（要設定）
    #file_id = "1VXTGRI_-s5l-JlBVmjZL1NbC2eWH7C9m""1-HPCm10U8CvnZrGvwaIsHp61ueYK1D5I"
import streamlit as st
import sys
import os

st.set_page_config(page_title="動作確認", page_icon="🔧")

st.title("🔧 Streamlit Cloud 動作確認")

st.write("**Python バージョン:**", sys.version)
st.write("**現在のディレクトリ:**", os.getcwd())

# 基本的なライブラリのみテスト
st.markdown("### 📦 ライブラリテスト")

try:
    import numpy as np
    st.success(f"✅ NumPy: {np.__version__}")
except Exception as e:
    st.error(f"❌ NumPy: {e}")

try:
    from PIL import Image
    st.success("✅ PIL/Pillow: OK")
except Exception as e:
    st.error(f"❌ PIL: {e}")

# ファイルアップロードテスト
uploaded_file = st.file_uploader("画像アップロードテスト", type=['png', 'jpg'])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロード成功", use_container_width=True)

st.success("✅ 基本機能は正常に動作しています")