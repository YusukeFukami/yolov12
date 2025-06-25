    # Google DriveのファイルID（要設定）
    #file_id = "1VXTGRI_-s5l-JlBVmjZL1NbC2eWH7C9m""1-HPCm10U8CvnZrGvwaIsHp61ueYK1D5I"

import streamlit as st
import os
import sys
import subprocess

# ページ設定（最初に実行）
st.set_page_config(
    page_title="パイプ検出アプリ",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 パイプ検出システム - デバッグ版")

# システム情報表示
st.markdown("### 🔍 システム診断")

# パッケージ確認
st.write("**Python バージョン:**", sys.version)
st.write("**作業ディレクトリ:**", os.getcwd())

# インストール済みパッケージ確認
with st.expander("📦 インストール済みパッケージ"):
    try:
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        st.text(result.stdout)
    except Exception as e:
        st.error(f"パッケージリスト取得エラー: {e}")

# ライブラリの存在確認
with st.expander("📚 システムライブラリ確認"):
    libs_to_check = [
        '/usr/lib/x86_64-linux-gnu/libGL.so.1',
        '/usr/lib/x86_64-linux-gnu/libglib-2.0.so.0',
        '/usr/lib/x86_64-linux-gnu/libSM.so.6',
        '/usr/lib/x86_64-linux-gnu/libXext.so.6',
        '/usr/lib/x86_64-linux-gnu/libXrender.so.1'
    ]
    
    for lib in libs_to_check:
        if os.path.exists(lib):
            st.success(f"✅ {lib}")
        else:
            st.error(f"❌ {lib} - 見つかりません")

# 環境変数設定
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

# 段階的インポートテスト
st.markdown("### 🧪 ライブラリインポートテスト")

import_status = {}

# 1. 基本ライブラリ
try:
    import numpy as np
    import_status['numpy'] = f"✅ バージョン: {np.__version__}"
except Exception as e:
    import_status['numpy'] = f"❌ エラー: {str(e)}"

try:
    from PIL import Image
    import_status['PIL'] = "✅ 成功"
except Exception as e:
    import_status['PIL'] = f"❌ エラー: {str(e)}"

# 2. OpenCV
try:
    import cv2
    import_status['OpenCV'] = f"✅ バージョン: {cv2.__version__}"
except Exception as e:
    import_status['OpenCV'] = f"❌ エラー: {str(e)}"

# 3. PyTorch
try:
    import torch
    import_status['PyTorch'] = f"✅ バージョン: {torch.__version__}"
except Exception as e:
    import_status['PyTorch'] = f"❌ エラー: {str(e)}"

# 4. Ultralytics
try:
    import ultralytics
    import_status['Ultralytics'] = f"✅ バージョン: {ultralytics.__version__}"
except Exception as e:
    import_status['Ultralytics'] = f"❌ エラー: {str(e)}"

# 5. YOLO
try:
    from ultralytics import YOLO
    import_status['YOLO'] = "✅ インポート成功"
    YOLO_AVAILABLE = True
except Exception as e:
    import_status['YOLO'] = f"❌ エラー: {str(e)}"
    YOLO_AVAILABLE = False

# インポート結果表示
for lib, status in import_status.items():
    if "✅" in status:
        st.success(f"{lib}: {status}")
    else:
        st.error(f"{lib}: {status}")

# ファイル確認
st.markdown("### 📄 設定ファイル確認")

col1, col2 = st.columns(2)

with col1:
    st.write("**packages.txt:**")
    if os.path.exists("packages.txt"):
        with open("packages.txt", "r") as f:
            st.code(f.read())
    else:
        st.error("packages.txt が見つかりません")

with col2:
    st.write("**requirements.txt:**")
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            st.code(f.read()[:500] + "...")  # 最初の500文字
    else:
        st.error("requirements.txt が見つかりません")

# 解決策提案
st.markdown("---")
st.markdown("### 💡 推奨される解決策")

if not YOLO_AVAILABLE:
    st.error("YOLOのインポートに失敗しています")
    
    st.markdown("""
    ### 🔧 Streamlit Cloud での対処法:
    
    1. **アプリの完全な再デプロイ**
       - Streamlit Cloud のダッシュボードでアプリを削除
       - 新規にアプリを作成してデプロイ
    
    2. **リソース制限の確認**
       - Streamlit Cloud の無料プランはメモリ制限あり
       - PyTorch + YOLO は大量のメモリを使用
    
    3. **代替案：軽量版の使用**
       - `torch` の代わりに `torch-cpu` を使用
       - より小さいYOLOモデル（yolov8n）を使用
    """)
    
    # より軽量なrequirements.txtの提案
    st.markdown("### 📝 軽量版 requirements.txt の提案:")
    st.code("""
# 基本
streamlit
numpy==1.24.3
opencv-python-headless==4.8.1.78
Pillow==10.0.1

# PyTorch CPU版（軽量）
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.0.1+cpu
torchvision==0.15.2+cpu

# YOLO
ultralytics==8.0.196

# 必須依存関係
gdown==4.7.1
PyYAML==6.0.1
matplotlib==3.7.2
pandas==2.0.3
    """)
else:
    st.success("✅ すべてのライブラリが正常にインポートされました！")
    
    # 簡単なテスト
    if st.button("🧪 YOLOモデルロードテスト"):
        try:
            with st.spinner("テスト中..."):
                model = YOLO('yolov8n.pt')  # 最小モデルでテスト
                st.success("✅ YOLOモデルのロードに成功！")
                st.write(f"モデルクラス数: {len(model.names)}")
        except Exception as e:
            st.error(f"モデルロードエラー: {e}")