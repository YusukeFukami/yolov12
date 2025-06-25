    # Google DriveのファイルID（要設定）
    #file_id = "1VXTGRI_-s5l-JlBVmjZL1NbC2eWH7C9m"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ページ設定
st.set_page_config(
    page_title="パイプ検出アプリ",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 パイプ検出システム")

# 解決策選択
st.sidebar.title("🔧 設定")
model_choice = st.sidebar.radio(
    "使用するモデル:",
    ["標準YOLOv8（推奨）", "カスタムYOLOv12（実験的）"],
    index=0
)

@st.cache_resource
def load_yolo8_model():
    """標準YOLOv8モデル（確実に動作）"""
    try:
        st.info("🤖 標準YOLOv8モデルを読み込み中...")
        model = YOLO("yolov8n.pt")
        st.success("✅ 標準モデル読み込み完了！")
        return model
    except Exception as e:
        st.error(f"❌ 標準モデル読み込みエラー: {e}")
        return None

@st.cache_resource  
def load_custom_model():
    """カスタムYOLOv12モデル（互換性問題あり）"""
    import gdown
    import os
    
    file_id = "1VXTGRI_-s5l-JlBVmjZL1NbC2eWH7C9m"
    model_path = "my_trained_model.pt"
    
    try:
        # ダウンロード
        if not os.path.exists(model_path):
            st.info("🔄 カスタムモデルをダウンロード中...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
        
        # モデル読み込み
        st.info("🤖 カスタムYOLOv12モデルを読み込み中...")
        model = YOLO(model_path)
        st.success("✅ カスタムモデル読み込み完了！")
        return model
        
    except Exception as e:
        st.error(f"❌ カスタムモデル読み込みエラー: {e}")
        return None

def display_results(results, original_image):
    """検出結果を表示"""
    detections = results[0]
    num_detections = len(detections.boxes) if detections.boxes is not None else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 検出結果")
        st.metric("検出されたオブジェクト数", num_detections)
        
        if num_detections > 0:
            boxes = detections.boxes
            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
            confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
            
            st.subheader("🎯 検出詳細")
            for i, (cls, conf) in enumerate(zip(classes, confidences)):
                class_name = results[0].names[int(cls)]
                st.write(f"**{i+1}.** {class_name} (信頼度: {conf:.2f})")
    
    with col2:
        st.subheader("🖼️ 検出画像")
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="検出結果", use_container_width=True)

# モデル読み込み
if model_choice == "標準YOLOv8（推奨）":
    st.markdown("### 🟢 標準YOLOv8モデル（汎用物体検出）")
    st.info("✅ 確実に動作します（80種類の物体を検出）")
    model = load_yolo8_model()
else:
    st.markdown("### 🟡 カスタムYOLOv12モデル（パイプ専用）")
    st.warning("⚠️ 互換性問題で動作しない可能性があります")
    model = load_custom_model()

if model is None:
    st.error("❌ モデルが読み込まれていません")
    st.stop()

# 画像アップロード
st.subheader("📤 画像をアップロード")
uploaded_file = st.file_uploader(
    "画像を選択してください", 
    type=['png', 'jpg', 'jpeg']
)

if uploaded_file is not None:
    # 画像表示
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_container_width=True)
    
    # 推論実行
    with st.spinner("🔍 物体を検出中..."):
        try:
            # シンプルな推論
            results = model(image)
            
            # 結果表示
            display_results(results, image)
            st.success("✅ 検出完了！")
            
        except AttributeError as attr_err:
            if "qkv" in str(attr_err):
                st.error("❌ YOLOv12互換性エラー：qkv問題")
                st.markdown("""
                ### 💡 **解決策**
                1. **左サイドバー**で「標準YOLOv8」を選択
                2. **カスタムモデル**はローカル環境で実行してください
                """)
            else:
                st.error(f"❌ 推論エラー: {attr_err}")
                
        except Exception as e:
            st.error(f"❌ 画像処理エラー: {e}")

else:
    st.info("👆 画像ファイルをアップロードしてください")

# 説明
st.markdown("---")
if model_choice == "標準YOLOv8（推奨）":
    st.markdown("""
    ### ✅ **標準YOLOv8モデル**
    - **検出対象**: 人、車、動物など80種類
    - **動作**: 確実にStreamlit Cloudで動作
    - **用途**: 汎用物体検出の動作確認
    """)
else:
    st.markdown("""
    ### ⚠️ **カスタムYOLOv12モデル**
    - **検出対象**: パイプ専用
    - **問題**: Streamlit CloudでAttention機能非対応
    - **解決策**: ローカル環境での実行を推奨
    """)