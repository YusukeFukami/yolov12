    # Google DriveのファイルID（要設定）
    #file_id = "1VXTGRI_-s5l-JlBVmjZL1NbC2eWH7C9m""1-HPCm10U8CvnZrGvwaIsHp61ueYK1D5I"

import streamlit as st
from PIL import Image
import numpy as np
import gdown
import os
import json
import time
from datetime import datetime
import io

# 安全なultralticsインポート
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Ultralytics インポートエラー: {e}")
    st.error("📋 OpenCVまたは依存関係の問題です")
    st.info("💡 管理者に連絡してください：requirements.txtの確認が必要")
    ULTRALYTICS_AVAILABLE = False

# ページ設定
st.set_page_config(
    page_title="パイプ検出アプリ",
    page_icon="🔧",
    layout="wide"
)

# カスタムCSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .detection-box {
        background-color: #e3f2fd;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔧 パイプ検出システム")
st.markdown("**YOLOv12ベース**の高精度パイプ検出アプリケーション")

# Ultralytics可用性チェック
if not ULTRALYTICS_AVAILABLE:
    st.error("❌ システムエラー：YOLOライブラリが利用できません")
    st.markdown("""
    ### 🔧 **技術的問題**
    - **OpenCV依存関係エラー**が発生しています
    - **管理者による修正**が必要です
    
    ### 📞 **解決手順**
    1. `requirements.txt`の確認
    2. `packages.txt`の追加
    3. Streamlit Cloudの再起動
    """)
    st.stop()

# セッション状態の初期化
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# サイドバー設定
st.sidebar.title("⚙️ 検出設定")

# Google Drive設定
st.sidebar.markdown("### 📥 モデル設定")
use_custom_id = st.sidebar.checkbox("カスタムFile IDを使用", value=False)

if use_custom_id:
    file_id = st.sidebar.text_input(
        "Google Drive File ID:",
        value="1-HPCm10U8CvnZrGvwaIsHp61ueYK1D5I",
        help="Google DriveのファイルIDを入力"
    )
else:
    # デフォルトのFile ID
    file_id = "1-HPCm10U8CvnZrGvwaIsHp61ueYK1D5I"

# 推論設定
st.sidebar.markdown("### 🎯 推論パラメータ")
confidence_threshold = st.sidebar.slider(
    "信頼度閾値", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.25, 
    step=0.05,
    help="この値以上の信頼度のオブジェクトのみ表示"
)

max_detections = st.sidebar.slider(
    "最大検出数", 
    min_value=1, 
    max_value=100, 
    value=50,
    help="検出するオブジェクトの最大数"
)

iou_threshold = st.sidebar.slider(
    "IoU閾値（重複除去）", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.45, 
    step=0.05,
    help="重複検出を除去するための閾値"
)

# 表示設定
st.sidebar.markdown("### 🎨 表示設定")
show_labels = st.sidebar.checkbox("ラベル表示", value=True)
show_conf = st.sidebar.checkbox("信頼度表示", value=True)
line_thickness = st.sidebar.slider("線の太さ", 1, 10, 3)

# バッチ処理設定
st.sidebar.markdown("### 📊 バッチ処理")
save_results = st.sidebar.checkbox("検出結果を保存", value=False)
auto_process = st.sidebar.checkbox("自動処理モード", value=False)

@st.cache_resource
def load_pipe_detection_model(file_id):
    """変換済みパイプ検出モデルをロード"""
    
    model_path = "pipe_detection_model.pt"
    
    try:
        # ダウンロード
        if not os.path.exists(model_path):
            st.info("🔄 パイプ検出モデルをダウンロード中...")
            url = f"https://drive.google.com/uc?id={file_id}"
            
            with st.spinner("ダウンロード中..."):
                gdown.download(url, model_path, quiet=False)
            
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024 * 1024)
                st.success(f"✅ ダウンロード完了: {file_size:.1f}MB")
            else:
                st.error("❌ ダウンロードに失敗しました")
                return None
        else:
            st.info("📂 既存のモデルファイルを使用")
        
        # モデル読み込み
        st.info("🤖 パイプ検出モデルを読み込み中...")
        model = YOLO(model_path)
        st.success("✅ モデル読み込み完了！")
        
        # モデル情報表示
        with st.expander("📋 モデル情報", expanded=False):
            if hasattr(model, 'names'):
                class_names = list(model.names.values())
                if class_names == ['0', '1']:
                    st.write("**検出クラス:** パイプ")
                    st.write("**モデルタイプ:** YOLOv12 → YOLOv8変換済み")
                else:
                    st.write(f"**検出クラス:** {class_names}")
            
            st.write(f"**モデルファイル:** {model_path}")
            st.write(f"**ファイルサイズ:** {os.path.getsize(model_path) / (1024 * 1024):.1f}MB")
        
        return model
        
    except Exception as e:
        st.error(f"❌ モデル読み込みエラー: {e}")
        st.info("💡 File IDが正しいか確認してください")
        return None

def display_results(results, original_image, image_name="検出結果"):
    """検出結果を表示する関数"""
    
    # 検出結果の統計
    detections = results[0]
    num_detections = len(detections.boxes) if detections.boxes is not None else 0
    
    # 検出履歴に追加
    detection_record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'image_name': image_name,
        'num_detections': num_detections,
        'confidence_threshold': confidence_threshold
    }
    st.session_state.detection_history.append(detection_record)
    
    # レイアウト
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 検出結果")
        
        # メトリクス表示
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("検出数", num_detections)
        with metric_col2:
            if num_detections > 0:
                avg_conf = np.mean([box.conf.item() for box in detections.boxes])
                st.metric("平均信頼度", f"{avg_conf:.2%}")
        
        if num_detections > 0:
            # 検出されたオブジェクトの詳細
            st.markdown("### 🎯 検出詳細")
            
            boxes = detections.boxes
            for i, box in enumerate(boxes):
                conf = box.conf.item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                width = x2 - x1
                height = y2 - y1
                
                # 検出ボックスの情報表示
                st.markdown(f"""
                <div class="detection-box">
                    <strong>パイプ #{i+1}</strong><br>
                    信頼度: <strong>{conf:.2%}</strong><br>
                    サイズ: {width:.0f} × {height:.0f} px<br>
                    位置: ({x1:.0f}, {y1:.0f})
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("パイプが検出されませんでした")
            st.caption("💡 信頼度閾値を下げてみてください")
    
    with col2:
        st.subheader("🖼️ 検出画像")
        
        # バウンディングボックス付きの画像を作成
        annotated_image = results[0].plot(
            labels=show_labels,
            conf=show_conf,
            line_width=line_thickness
        )
        
        st.image(annotated_image, caption=image_name, use_container_width=True)
        
        # 画像ダウンロードボタン
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            # 検出結果画像のダウンロード
            img_pil = Image.fromarray(annotated_image)
            buf = io.BytesIO()
            img_pil.save(buf, format='PNG')
            
            st.download_button(
                label="📥 検出結果画像",
                data=buf.getvalue(),
                file_name=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                mime="image/png"
            )
        
        with col_dl2:
            # 検出データのダウンロード
            if num_detections > 0:
                detection_data = {
                    'timestamp': datetime.now().isoformat(),
                    'image_name': image_name,
                    'detections': []
                }
                
                for i, box in enumerate(detections.boxes):
                    detection_data['detections'].append({
                        'id': i + 1,
                        'confidence': float(box.conf.item()),
                        'bbox': box.xyxy[0].tolist()
                    })
                
                st.download_button(
                    label="📊 検出データ (JSON)",
                    data=json.dumps(detection_data, indent=2, ensure_ascii=False),
                    file_name=f"detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# メインコンテンツ
st.markdown("---")

# モデル読み込み
model = load_pipe_detection_model(file_id)

if model is None:
    st.error("❌ モデルが読み込まれていません")
    st.info("💡 サイドバーでFile IDを確認してください")
else:
    st.session_state.model_loaded = True
    
    # タブ表示
    tab1, tab2, tab3 = st.tabs(["📤 画像アップロード", "📊 検出履歴", "📚 使用方法"])
    
    with tab1:
        # ファイルアップロード
        st.subheader("📸 検出する画像を選択")
        
        # 複数ファイルアップロード対応
        uploaded_files = st.file_uploader(
            "パイプが写った画像をアップロード", 
            type=['png', 'jpg', 'jpeg'],
            help="対応形式: PNG, JPG, JPEG（最大200MB）",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.markdown(f"### 📁 {len(uploaded_files)}枚の画像がアップロードされました")
            
            # 処理オプション
            col1, col2, col3 = st.columns(3)
            with col1:
                process_all = st.button("🚀 すべて処理", type="primary", use_container_width=True)
            with col2:
                if len(uploaded_files) > 1:
                    selected_index = st.selectbox(
                        "個別処理",
                        range(len(uploaded_files)),
                        format_func=lambda x: uploaded_files[x].name
                    )
                else:
                    selected_index = 0
            with col3:
                process_selected = st.button("🔍 選択画像を処理", use_container_width=True)
            
            # 一括処理
            if process_all or (auto_process and uploaded_files):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()
                
                all_results = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"処理中: {uploaded_file.name} ({idx+1}/{len(uploaded_files)})")
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    
                    try:
                        image = Image.open(uploaded_file)
                        
                        # 推論実行
                        results = model(
                            image,
                            conf=confidence_threshold,
                            iou=iou_threshold,
                            max_det=max_detections,
                            verbose=False
                        )
                        
                        all_results.append({
                            'file_name': uploaded_file.name,
                            'results': results,
                            'image': image
                        })
                        
                        # 結果表示
                        with results_container.expander(f"🖼️ {uploaded_file.name}", expanded=(idx==0)):
                            display_results(results, image, uploaded_file.name)
                            
                    except Exception as e:
                        st.error(f"❌ {uploaded_file.name} の処理エラー: {e}")
                
                status_text.text("✅ すべての画像の処理が完了しました！")
                progress_bar.empty()
                
                # バッチ結果の保存
                if save_results and all_results:
                    st.markdown("---")
                    st.subheader("💾 バッチ処理結果の保存")
                    
                    # 結果をZIPファイルにまとめる
                    zip_buffer = io.BytesIO()
                    import zipfile
                    
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for result in all_results:
                            # 画像を保存
                            img_buffer = io.BytesIO()
                            annotated = result['results'][0].plot()
                            Image.fromarray(annotated).save(img_buffer, format='PNG')
                            zip_file.writestr(
                                f"detected_{result['file_name']}", 
                                img_buffer.getvalue()
                            )
                    
                    st.download_button(
                        label="📦 全結果をダウンロード (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
            
            # 個別処理
            elif process_selected and uploaded_files:
                uploaded_file = uploaded_files[selected_index]
                
                try:
                    image = Image.open(uploaded_file)
                    
                    # 画像プレビュー
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption=f"元画像: {uploaded_file.name}", use_container_width=True)
                    
                    # 推論実行
                    with st.spinner("🔍 パイプを検出中..."):
                        results = model(
                            image,
                            conf=confidence_threshold,
                            iou=iou_threshold,
                            max_det=max_detections,
                            verbose=False
                        )
                    
                    # 結果表示
                    st.markdown("---")
                    display_results(results, image, uploaded_file.name)
                    
                except Exception as e:
                    st.error(f"❌ 画像処理エラー: {e}")
                    st.info("💡 画像形式を確認してください")
    
    with tab2:
        st.subheader("📜 検出履歴")
        
        if st.session_state.detection_history:
            # サマリー表示
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("総処理数", len(st.session_state.detection_history))
            with col2:
                total_detections = sum(record['num_detections'] for record in st.session_state.detection_history)
                st.metric("総検出数", total_detections)
            with col3:
                avg_detections = total_detections / len(st.session_state.detection_history)
                st.metric("平均検出数", f"{avg_detections:.1f}")
            
            # 履歴テーブル
            st.markdown("### 📊 詳細履歴")
            import pandas as pd
            history_df = pd.DataFrame(st.session_state.detection_history)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
            
            # エクスポートボタン
            col1, col2 = st.columns(2)
            with col1:
                history_json = json.dumps(st.session_state.detection_history, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📥 履歴をダウンロード (JSON)",
                    data=history_json,
                    file_name=f"detection_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                if st.button("🗑️ 履歴をクリア", type="secondary"):
                    st.session_state.detection_history = []
                    st.rerun()
        else:
            st.info("まだ検出履歴がありません")
    
    with tab3:
        st.subheader("📚 使用方法")
        
        st.markdown("""
        ### 🔧 **パイプ検出システムについて**
        
        このアプリケーションは、YOLOv12で学習したカスタムモデルを使用して、
        画像内のパイプを高精度で検出します。
        
        ### 🚀 **基本的な使い方**
        
        1. **画像のアップロード**
           - 「画像アップロード」タブで画像を選択
           - PNG、JPG、JPEG形式に対応
           - 複数画像の一括処理も可能
        
        2. **検出パラメータの調整**
           - **信頼度閾値**: 検出の確実性を調整（0.1〜1.0）
           - **IoU閾値**: 重複検出の除去を調整
           - **最大検出数**: 1画像あたりの最大検出数
        
        3. **結果の確認**
           - 検出されたパイプがバウンディングボックスで表示
           - 各検出の信頼度とサイズが表示
           - 結果画像とデータのダウンロードが可能
        
        ### 💡 **高度な機能**
        
        - **バッチ処理**: 複数画像を一括で処理
        - **自動処理モード**: アップロード時に自動で検出開始
        - **検出履歴**: 過去の検出結果を記録・エクスポート
        - **カスタムFile ID**: 異なるモデルファイルの使用
        
        ### ⚙️ **推奨設定**
        
        - **一般的な検出**: 信頼度閾値 0.25、IoU閾値 0.45
        - **高精度検出**: 信頼度閾値 0.5以上
        - **より多くの検出**: 信頼度閾値 0.1〜0.2
        
        ### 🔧 **トラブルシューティング**
        
        - **検出されない場合**
          - 信頼度閾値を下げてみる
          - 画像の品質や照明を確認
          - パイプが明確に写っているか確認
        
        - **誤検出が多い場合**
          - 信頼度閾値を上げる
          - IoU閾値を調整して重複を除去
        
        ### 📞 **サポート**
        
        技術的な問題が発生した場合は、管理者にお問い合わせください。
        """)

# フッター
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        🔧 パイプ検出システム v1.0 | Powered by YOLOv12 (YOLOv8互換形式)
    </div>
    """, 
    unsafe_allow_html=True
)