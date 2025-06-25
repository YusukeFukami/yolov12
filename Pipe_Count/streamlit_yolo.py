    # Google DriveのファイルID（要設定）
    #file_id = "1VXTGRI_-s5l-JlBVmjZL1NbC2eWH7C9m""1-HPCm10U8CvnZrGvwaIsHp61ueYK1D5I"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import gdown
import os

# ページ設定
st.set_page_config(
    page_title="パイプ検出アプリ",
    page_icon="🔧",
    layout="wide"
)

st.title("🔧 パイプ検出システム")
st.markdown("**変換済みYOLOv12モデル**を使用した高精度パイプ検出アプリです")

# 解決策選択
st.sidebar.title("🔧 モデル設定")
model_choice = st.sidebar.radio(
    "使用するモデル:",
    ["変換済みパイプ検出（推奨）", "標準YOLOv8（汎用）", "元YOLOv12（実験的）"],
    index=0,
    help="変換済みモデルが最適です"
)

@st.cache_resource
def load_converted_model():
    """変換済みパイプ検出モデル（YOLOv8互換）"""
    
    # 変換されたモデルのGoogle Drive ID（実際のIDに置き換えてください）
    file_id = "1-HPCm10U8CvnZrGvwaIsHp61ueYK1D5I"  # ← converted_pipe_model.ptのファイルID
    model_path = "converted_pipe_model.pt"
    
    try:
        # ダウンロード
        if not os.path.exists(model_path):
            st.info("🔄 変換済みパイプ検出モデルをダウンロード中...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
            
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024 * 1024)
                st.success(f"✅ ダウンロード完了: {file_size:.1f}MB")
            else:
                st.error("❌ ダウンロードに失敗しました")
                return None
        
        # モデル読み込み（YOLOv8互換として）
        st.info("🤖 変換済みパイプ検出モデルを読み込み中...")
        model = YOLO(model_path)
        st.success("✅ パイプ検出モデル読み込み完了！")
        
        # モデル情報表示
        if hasattr(model, 'names'):
            # クラス名をわかりやすく表示
            class_names = list(model.names.values())
            if class_names == ['0', '1']:
                display_names = ['背景', 'パイプ']
                st.info(f"🎯 検出クラス: {display_names}")
            else:
                st.info(f"🎯 検出クラス: {class_names}")
        
        return model
        
    except Exception as e:
        st.error(f"❌ 変換モデル読み込みエラー: {e}")
        st.warning("🔄 標準モデルにフォールバック中...")
        return YOLO("yolov8n.pt")

@st.cache_resource
def load_yolo8_model():
    """標準YOLOv8モデル（汎用物体検出）"""
    try:
        st.info("🤖 標準YOLOv8モデルを読み込み中...")
        model = YOLO("yolov8n.pt")
        st.success("✅ 標準モデル読み込み完了！")
        return model
    except Exception as e:
        st.error(f"❌ 標準モデル読み込みエラー: {e}")
        return None

@st.cache_resource  
def load_original_yolov12():
    """元のYOLOv12モデル（互換性問題あり）"""
    
    file_id = "YOUR_ORIGINAL_YOLOV12_FILE_ID"  # 元のYOLOv12モデルID
    model_path = "original_yolov12_model.pt"
    
    try:
        # ダウンロード
        if not os.path.exists(model_path):
            st.info("🔄 元YOLOv12モデルをダウンロード中...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
        
        # モデル読み込み
        st.info("🤖 元YOLOv12モデルを読み込み中...")
        model = YOLO(model_path)
        st.success("✅ 元YOLOv12モデル読み込み完了！")
        return model
        
    except Exception as e:
        st.error(f"❌ 元YOLOv12モデル読み込みエラー: {e}")
        return None

def display_results(results, original_image):
    """検出結果を表示する関数"""
    
    # 検出結果の統計
    detections = results[0]
    num_detections = len(detections.boxes) if detections.boxes is not None else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 検出結果")
        st.metric("検出されたオブジェクト数", num_detections)
        
        if num_detections > 0:
            # 検出されたクラスと信頼度
            boxes = detections.boxes
            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else []
            confidences = boxes.conf.cpu().numpy() if boxes.conf is not None else []
            
            st.subheader("🎯 検出詳細")
            for i, (cls, conf) in enumerate(zip(classes, confidences)):
                class_name = results[0].names[int(cls)]
                
                # クラス名をわかりやすく表示
                if class_name in ['0', '1']:
                    display_name = '背景' if class_name == '0' else 'パイプ'
                else:
                    display_name = class_name
                    
                st.write(f"**{i+1}.** {display_name} (信頼度: {conf:.2f})")
        else:
            st.info("パイプが検出されませんでした")
    
    with col2:
        st.subheader("🖼️ 検出画像")
        # バウンディングボックス付きの画像を表示
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="検出結果", use_container_width=True)

# モデル読み込み
if model_choice == "変換済みパイプ検出（推奨）":
    st.markdown("### 🟢 変換済みパイプ検出モデル（YOLOv8互換）")
    st.info("✅ YOLOv12→YOLOv8変換でStreamlit Cloud対応")
    model = load_converted_model()
elif model_choice == "標準YOLOv8（汎用）":
    st.markdown("### 🟡 標準YOLOv8モデル（汎用物体検出）")
    st.info("✅ 確実に動作します（80種類の物体を検出）")
    model = load_yolo8_model()
else:  # 元YOLOv12（実験的）
    st.markdown("### 🔴 元YOLOv12モデル（パイプ専用・実験的）")
    st.warning("⚠️ 互換性問題で動作しない可能性があります")
    model = load_original_yolov12()

if model is None:
    st.error("❌ モデルが読み込まれていません")
    st.stop()

# モデル情報表示
with st.expander("🔧 モデル情報", expanded=False):
    try:
        st.text(f"モデルタイプ: {type(model.model).__name__}")
        if hasattr(model, 'names'):
            st.text(f"クラス数: {len(model.names)}")
            class_list = list(model.names.values())
            if len(class_list) <= 10:
                st.text(f"全クラス: {class_list}")
            else:
                st.text(f"クラス例: {class_list[:10]}...")
        st.text("✅ モデル読み込み成功")
    except Exception as info_err:
        st.warning(f"⚠️ モデル情報取得エラー: {info_err}")

# ファイルアップロード
st.subheader("📤 画像をアップロード")
uploaded_file = st.file_uploader(
    "パイプが写った画像を選択してください", 
    type=['png', 'jpg', 'jpeg'],
    help="対応形式: PNG, JPG, JPEG（最大200MB）"
)

# 画像処理と検出
if uploaded_file is not None:
    try:
        # 画像を読み込み
        image = Image.open(uploaded_file)
        st.image(image, caption="アップロードされた画像", use_container_width=True)
        
        # 推論設定
        with st.sidebar.expander("⚙️ 検出設定", expanded=False):
            confidence_threshold = st.slider(
                "信頼度閾値", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.25, 
                step=0.05,
                help="この値以上の信頼度のオブジェクトのみ表示"
            )
            
            max_detections = st.slider(
                "最大検出数", 
                min_value=1, 
                max_value=100, 
                value=50,
                help="検出するオブジェクトの最大数"
            )
        
        # YOLO推論実行
        with st.spinner("🔍 パイプを検出中..."):
            try:
                # 推論実行（設定パラメータ付き）
                results = model(
                    image,
                    conf=confidence_threshold,
                    max_det=max_detections,
                    verbose=False
                )
                
                # 結果を表示
                display_results(results, image)
                st.success("✅ 検出完了！")
                
            except AttributeError as attr_err:
                if "qkv" in str(attr_err):
                    st.error("❌ YOLOv12互換性エラー：qkv問題")
                    st.markdown("""
                    ### 💡 **解決策**
                    1. **左サイドバー**で「変換済みパイプ検出（推奨）」を選択
                    2. **カスタムモデル**はローカル環境で実行してください
                    """)
                else:
                    st.error(f"❌ 推論エラー: {attr_err}")
                    
            except Exception as e:
                st.error(f"❌ 推論処理エラー: {e}")
                st.info("💡 画像サイズまたは形式を確認してください")
        
    except Exception as e:
        st.error(f"❌ 画像処理エラー: {e}")
        st.error("💡 画像形式を確認してください（JPG, PNG, JPEG対応）")

else:
    st.info("👆 画像ファイルをアップロードしてください")

# 使用方法とヘルプ
st.markdown("---")

# 動的ヘルプ表示
if model_choice == "変換済みパイプ検出（推奨）":
    st.markdown("""
    ### ✅ **変換済みパイプ検出モデル**
    - **検出対象**: パイプ専用（カスタム訓練済み）
    - **動作**: YOLOv8互換形式でStreamlit Cloud対応
    - **用途**: 高精度パイプ検出の実用運用
    - **変換**: YOLOv12 → YOLOv8互換変換済み
    
    ### 📋 使用方法
    1. **画像アップロード**: パイプが写った画像を選択
    2. **設定調整**: 左サイドバーで信頼度閾値を調整
    3. **自動検出**: 変換されたモデルが自動でパイプを検出
    4. **結果確認**: 検出されたパイプの位置と信頼度を確認
    """)
elif model_choice == "標準YOLOv8（汎用）":
    st.markdown("""
    ### ✅ **標準YOLOv8モデル**
    - **検出対象**: 人、車、動物など80種類
    - **動作**: 確実にStreamlit Cloudで動作
    - **用途**: 汎用物体検出の動作確認
    
    ### 📋 使用方法
    1. **画像アップロード**: 任意の画像を選択
    2. **自動検出**: 80種類の物体を自動検出
    3. **結果確認**: 検出された物体の種類と信頼度を確認
    """)
else:
    st.markdown("""
    ### ⚠️ **元YOLOv12モデル**
    - **検出対象**: パイプ専用（カスタム訓練済み）
    - **問題**: Streamlit CloudでAttention機能非対応
    - **解決策**: ローカル環境での実行を推奨
    - **用途**: 動作確認・デバッグ用
    
    ### 💡 **推奨事項**
    互換性問題を避けるため「変換済みパイプ検出（推奨）」の使用を推奨します。
    """)

st.markdown("""
### 🔧 **トラブルシューティング**
- **モデル読み込みエラー**: Google Drive URLとファイルIDを確認
- **推論エラー**: 画像形式（JPG/PNG）と画像サイズを確認
- **qkvエラー**: 「変換済みパイプ検出」モデルを使用
- **検出されない**: 信頼度閾値を下げる（0.1-0.3）

### 📞 **サポート**
問題が解決しない場合は、管理者にお問い合わせください。
""")