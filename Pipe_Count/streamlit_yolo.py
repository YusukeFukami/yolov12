    # Google DriveのファイルID（要設定）
    #file_id = "1VXTGRI_-s5l-JlBVmjZL1NbC2eWH7C9m""1-HPCm10U8CvnZrGvwaIsHp61ueYK1D5I"

import streamlit as st
import os
import sys

# OpenCVインポート前の環境設定
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

# エラーハンドリング付きインポート
try:
    from PIL import Image
    import numpy as np
    import gdown
    import json
    from datetime import datetime
    import io
    
    # Ultralytics YOLOのインポート
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    
except ImportError as e:
    YOLO_AVAILABLE = False
    error_msg = str(e)

# ページ設定
st.set_page_config(
    page_title="パイプ検出アプリ",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 パイプ検出システム")
st.markdown("カスタムYOLOv12モデルによるパイプ検出")

# エラーチェック
if not YOLO_AVAILABLE:
    st.error("❌ システムエラー：必要なライブラリがロードできません")
    st.markdown("""
    ### 🔧 解決方法:
    
    1. **packages.txt** ファイルを作成して以下を追加:
    ```
    libgl1-mesa-glx
    libglib2.0-0
    libsm6
    libxext6
    libxrender-dev
    libgomp1
    ```
    
    2. **requirements.txt** を確認
    
    3. **Streamlit Cloud** でアプリを再デプロイ
    """)
    st.stop()

# 設定
DEFAULT_FILE_ID = "1-HPCm10U8CvnZrGvwaIsHp61ueYK1D5I"  # カスタムモデルのFile ID

# サイドバー
st.sidebar.title("⚙️ 設定")

# モデル設定
file_id = st.sidebar.text_input(
    "Google Drive File ID",
    value=DEFAULT_FILE_ID,
    help="カスタムパイプ検出モデルのFile ID"
)

# 推論パラメータ
confidence = st.sidebar.slider("信頼度閾値", 0.01, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("IoU閾値", 0.1, 1.0, 0.45, 0.05)

# キャッシュクリア
if st.sidebar.button("🗑️ モデルキャッシュをクリア"):
    if os.path.exists("model.pt"):
        os.remove("model.pt")
        st.sidebar.success("キャッシュをクリアしました")
        st.rerun()

@st.cache_resource
def load_model(file_id):
    """モデルをロード"""
    model_path = "model.pt"
    
    try:
        # ダウンロード
        if not os.path.exists(model_path):
            with st.spinner("モデルをダウンロード中..."):
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, model_path, quiet=False)
                
                if not os.path.exists(model_path):
                    st.error("ダウンロード失敗")
                    return None
                
                st.success(f"ダウンロード完了: {os.path.getsize(model_path)/1024/1024:.1f}MB")
        
        # モデル読み込み
        with st.spinner("モデルを読み込み中..."):
            model = YOLO(model_path)
            
        # モデル検証
        if hasattr(model, 'names'):
            names = model.names
            if len(names) == 2 and 0 in names and 1 in names:
                st.success("✅ パイプ検出モデル読み込み完了")
                st.info(f"クラス: {names}")
                return model
            else:
                st.error(f"❌ 期待と異なるモデル（クラス数: {len(names)}）")
                st.write(f"検出されたクラス: {list(names.values())[:10]}...")
                return None
        
        return model
        
    except Exception as e:
        st.error(f"モデル読み込みエラー: {e}")
        return None

# メイン処理
st.markdown("---")

# モデル読み込み
model = load_model(file_id)

if model is None:
    st.error("モデルが正しくロードされていません")
    st.info("File IDを確認し、キャッシュをクリアしてから再試行してください")
else:
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "画像をアップロード",
        type=['png', 'jpg', 'jpeg'],
        help="パイプが写った画像を選択"
    )
    
    if uploaded_file:
        # 画像読み込み
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("元画像")
            st.image(image, use_container_width=True)
        
        # 検出実行
        if st.button("🔍 検出実行", type="primary"):
            with st.spinner("検出中..."):
                try:
                    # 推論
                    results = model(image, conf=confidence, iou=iou)
                    
                    # 結果表示
                    with col2:
                        st.subheader("検出結果")
                        
                        # アノテーション付き画像
                        annotated = results[0].plot()
                        st.image(annotated, use_container_width=True)
                        
                        # 検出数
                        if results[0].boxes is not None:
                            num_detections = len(results[0].boxes)
                            st.metric("検出数", num_detections)
                            
                            # 詳細
                            for i, box in enumerate(results[0].boxes):
                                conf = box.conf.item()
                                st.write(f"パイプ #{i+1}: 信頼度 {conf:.2%}")
                        else:
                            st.info("検出されませんでした")
                            
                except Exception as e:
                    st.error(f"検出エラー: {e}")

# 使用方法
with st.expander("📚 使用方法"):
    st.markdown("""
    ### 基本的な使い方
    1. 画像をアップロード
    2. 必要に応じて信頼度閾値を調整
    3. 「検出実行」ボタンをクリック
    
    ### トラブルシューティング
    - **検出されない場合**: 信頼度閾値を下げる（0.1〜0.2）
    - **誤検出が多い場合**: 信頼度閾値を上げる（0.5以上）
    - **モデルエラー**: キャッシュをクリアして再ロード
    """)