    # Google DriveのファイルID（要設定）
    #file_id = "1VXTGRI_-s5l-JlBVmjZL1NbC2eWH7C9m"

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
st.markdown("**YOLOv12カスタムモデル**を使用したパイプ検出アプリです")

@st.cache_resource
def load_model():
    # Google DriveファイルID（実際のIDに置き換えてください）
    file_id = "1VXTGRI_-s5l-JlBVmjZL1NbC2eWH7C9m"
    model_path = "my_trained_model.pt"
    
    try:
        # ファイルが存在しない場合ダウンロード
        if not os.path.exists(model_path):
            st.info("🔄 カスタムYOLOv12モデルをダウンロード中...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
            
            # ダウンロード確認
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024 * 1024)
                st.success(f"✅ ダウンロード完了: {file_size:.1f}MB")
            else:
                st.error("❌ ダウンロードに失敗しました")
                return None
        
        # モデル読み込み（詳細なエラー情報付き）
        st.info("🤖 YOLOv12カスタムモデルを読み込み中...")
        
        # 互換性チェック付きでロード
        try:
            model = YOLO(model_path)
            st.success("✅ カスタムモデル読み込み完了！")
            return model
            
        except AttributeError as attr_err:
            st.error(f"❌ モデル互換性エラー: {attr_err}")
            st.error("🔧 YOLOv12モデルとStreamlit Cloud環境の互換性問題です")
            
            # フォールバック：標準モデル
            st.info("🔄 標準YOLOv8モデルにフォールバック中...")
            fallback_model = YOLO("yolov8n.pt")
            st.warning("⚠️ 一時的に標準モデルを使用します（汎用物体検出）")
            return fallback_model
            
    except Exception as e:
        st.error(f"❌ モデル読み込みエラー: {e}")
        st.error("📋 詳細: ダウンロードまたはファイル破損の可能性があります")
        
        # フォールバック：標準モデル
        try:
            st.info("🔄 標準YOLOv8モデルで代替実行...")
            fallback_model = YOLO("yolov8n.pt")
            st.warning("⚠️ カスタムモデルの代わりに標準モデルを使用します")
            return fallback_model
        except Exception as fallback_err:
            st.error(f"❌ フォールバックも失敗: {fallback_err}")
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
                st.write(f"**{i+1}.** {class_name} (信頼度: {conf:.2f})")
    
    with col2:
        st.subheader("🖼️ 検出画像")
        # バウンディングボックス付きの画像を表示
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="検出結果", use_container_width=True)

# モデル読み込み
model = load_model()

# モデル情報表示
if model is not None:
    try:
        # モデル情報を取得
        with st.expander("🔧 モデル情報", expanded=False):
            st.text(f"モデルタイプ: {type(model.model).__name__}")
            if hasattr(model, 'names'):
                st.text(f"クラス数: {len(model.names)}")
                st.text(f"検出クラス: {list(model.names.values())[:10]}...")  # 最初の10クラス表示
            st.text("✅ モデル読み込み成功")
            
    except Exception as info_err:
        st.warning(f"⚠️ モデル情報取得エラー: {info_err}")
        
else:
    st.error("❌ モデルが読み込まれていません")
    st.stop()

# ファイルアップロード
st.subheader("📤 画像をアップロード")
uploaded_file = st.file_uploader(
    "パイプが写った画像を選択してください", 
    type=['png', 'jpg', 'jpeg'],
    help="対応形式: PNG, JPG, JPEG"
)

# 画像処理と検出
if uploaded_file is not None:
    try:
        # 画像を読み込み
        image = Image.open(uploaded_file)
        st.image(image, caption="アップロードされた画像", use_container_width=True)
        
        # モデルが読み込まれているかチェック
        if model is None:
            st.error("❌ モデルが読み込まれていません。ページを再読み込みしてください。")
            st.stop()
        
        # YOLO推論実行（詳細なエラー処理付き）
        with st.spinner("🔍 パイプを検出中..."):
            try:
                # 画像をnumpy配列に変換（互換性向上）
                img_array = np.array(image)
                
                # 推論実行
                results = model(img_array)
                
                # 結果を表示
                display_results(results, image)
                
            except AttributeError as attr_err:
                st.error(f"❌ 推論中にモデル互換性エラー: {attr_err}")
                st.error("🔧 YOLOv12の特定機能がStreamlit Cloud環境でサポートされていません")
                st.info("💡 解決策：ローカル環境での実行、またはモデル変換が必要です")
                
            except Exception as inference_err:
                st.error(f"❌ 推論エラー: {inference_err}")
                st.info("💡 画像形式または推論パラメータを確認してください")
        
    except Exception as e:
        st.error(f"❌ 画像処理エラー: {e}")
        st.error("💡 画像形式を確認してください（JPG, PNG, JPEG対応）")

else:
    st.info("👆 上記から画像ファイルをアップロードしてください")

# フッター
st.markdown("---")
st.markdown(
    """
    ### 📋 使用方法
    1. **画像アップロード**: パイプが写った画像を選択
    2. **自動検出**: YOLOv12モデルが自動でパイプを検出
    3. **結果確認**: 検出されたパイプの位置と信頼度を確認
    
    ### ⚠️ 注意事項
    - カスタムYOLOv12モデルがStreamlit Cloud環境で互換性問題を起こす場合があります
    - その場合は自動的に標準YOLOv8モデル（汎用物体検出）に切り替わります
    - 最適なパイプ検出にはローカル環境での実行を推奨します
    """
)