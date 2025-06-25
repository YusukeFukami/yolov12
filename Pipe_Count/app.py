import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io

# ページ設定
st.set_page_config(
    page_title="パイプ検出アプリ",
    page_icon="🔧",
    layout="centered"
)

# タイトルと説明
st.title("📸 パイプ検出アプリ")
st.write("カメラで撮影または画像をアップロードして、パイプの本数を自動検出します。")

# モデルのパスを設定（環境に応じて変更してください）
MODEL_PATH = './my_trained_model.pt'  # ローカルに配置

# セッション状態の初期化
if 'model' not in st.session_state:
    try:
        st.session_state.model = YOLO(MODEL_PATH)
        st.success("モデルを正常に読み込みました！")
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {e}")
        st.stop()

# 画像入力方法の選択
input_method = st.radio(
    "画像の入力方法を選択してください：",
    ["📷 カメラで撮影", "📁 ファイルをアップロード"]
)

# 画像を格納する変数
image = None

if input_method == "📷 カメラで撮影":
    # カメラ入力（iPhoneなどのモバイルデバイスでも動作）
    camera_image = st.camera_input("撮影してください")
    if camera_image is not None:
        image = Image.open(camera_image)
        
else:
    # ファイルアップロード
    uploaded_file = st.file_uploader(
        "画像をアップロードしてください",
        type=['jpg', 'jpeg', 'png']
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

# 画像が存在する場合の処理
if image is not None:
    # 画像をnumpy配列に変換
    image_np = np.array(image)
    
    # BGRからRGBに変換（OpenCVの処理のため）
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_np
    
    # 検出ボタン
    if st.button("🔍 検出開始", type="primary", use_container_width=True):
        with st.spinner("検出中..."):
            try:
                # YOLOで推論
                results = st.session_state.model(image_bgr)
                
                # 検出された本数を取得
                detected_count = len(results[0].boxes) if results[0].boxes is not None else 0
                
                # 結果画像の作成
                result_image = image_np.copy()
                
                # バウンディングボックスを描画
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box)
                        # 赤色でバウンディングボックスを描画
                        cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                
                # 合計本数をテキストで追加
                text = f"TOTAL: {detected_count}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2.5
                color = (255, 0, 0)  # 赤色
                thickness = 6
                
                # テキストサイズを取得
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = 30
                text_y = 80
                
                # 白い背景の矩形を描画
                padding = 15
                cv2.rectangle(result_image,
                            (text_x - padding, text_y - text_size[1] - padding),
                            (text_x + text_size[0] + padding, text_y + padding),
                            (255, 255, 255), -1)
                
                # テキストを描画
                cv2.putText(result_image, text, (text_x, text_y), 
                           font, font_scale, color, thickness)
                
                # 結果の表示
                st.success(f"✅ 検出完了！")
                
                # メトリクスの表示
                col1, col2, col3 = st.columns(3)
                with col2:
                    st.metric("検出されたパイプ数", f"{detected_count} 本")
                
                # 結果画像の表示
                st.image(result_image, caption="検出結果", use_column_width=True)
                
                # 結果画像をダウンロード可能にする
                result_pil = Image.fromarray(result_image)
                buf = io.BytesIO()
                result_pil.save(buf, format="JPEG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📥 結果画像をダウンロード",
                    data=byte_im,
                    file_name=f"detection_result_{detected_count}pipes.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")
    
    # 元画像のプレビュー
    with st.expander("元画像を表示"):
        st.image(image, caption="入力画像", use_column_width=True)

# フッター
st.markdown("---")
st.markdown("🔧 パイプ検出システム | Powered by YOLO & Streamlit")

# サイドバーに使い方を表示
with st.sidebar:
    st.header("📖 使い方")
    st.markdown("""
    1. **カメラで撮影**または**画像をアップロード**を選択
    2. 画像を入力
    3. **検出開始**ボタンをクリック
    4. 結果を確認
    5. 必要に応じて結果画像をダウンロード
    
    ### 💡 ヒント
    - iPhoneやAndroidから直接カメラで撮影できます
    - 明るい場所で撮影すると検出精度が向上します
    - パイプが重ならないように撮影してください
    """)
    
    st.header("⚙️ 設定")
    confidence = st.slider("信頼度しきい値", 0.0, 1.0, 0.5, 0.05)
    if st.button("しきい値を適用"):
        if hasattr(st.session_state.model, 'conf'):
            st.session_state.model.conf = confidence
            st.success(f"信頼度しきい値を{confidence}に設定しました")