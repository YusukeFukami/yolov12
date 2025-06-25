import streamlit as st
import torch
import gdown
import os
from PIL import Image
from ultralytics import YOLO

# ページ設定
st.set_page_config(
    page_title="パイプ検出アプリ",
    page_icon="🔧",
    layout="wide"
)

@st.cache_data
def download_model():
    """Google Driveからモデルをダウンロード"""
    model_path = "my_trained_model.pt"
    
    if not os.path.exists(model_path):
        st.info("学習済みモデルをダウンロード中...")
        
        # Google DriveのファイルID（要設定）
        file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("モデルダウンロード完了！")
        except Exception as e:
            st.error(f"ダウンロードエラー: {e}")
            return None
    
    return model_path

@st.cache_resource
def load_model():
    """YOLOモデルを読み込み"""
    model_path = download_model()
    if model_path and os.path.exists(model_path):
        return YOLO(model_path)
    return None

# メイン画面
st.title("🔧 パイプ検出アプリ")
st.write("画像をアップロードしてパイプを検出します")

# モデル読み込み
model = load_model()

if model is None:
    st.error("モデルが読み込めません。管理者に連絡してください。")
    st.stop()

# ファイルアップロード
uploaded_file = st.file_uploader(
    "画像をアップロード", 
    type=['jpg', 'jpeg', 'png'],
    accept_multiple_files=False
)

if uploaded_file is not None:
    # 画像表示
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("元画像")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("検出結果")
        
        # 推論実行
        with st.spinner("検出中..."):
            results = model(image)
            
            # 結果画像
            result_image = results[0].plot()
            st.image(result_image, use_column_width=True)
            
            # 検出統計
            detections = results[0].boxes
            if detections is not None:
                num_pipes = len(detections)
                st.success(f"🔧 検出されたパイプ数: {num_pipes}")
                
                # 信頼度表示
                if len(detections) > 0:
                    confidences = detections.conf.cpu().numpy()
                    avg_conf = confidences.mean()
                    st.metric("平均信頼度", f"{avg_conf:.2f}")
            else:
                st.warning("パイプが検出されませんでした")

# 使用方法
with st.expander("使用方法"):
    st.markdown("""
    1. 「画像をアップロード」ボタンから画像を選択
    2. 自動的にパイプ検出が実行されます
    3. 右側に検出結果が表示されます
    
    **対応形式**: JPG, JPEG, PNG
    """)