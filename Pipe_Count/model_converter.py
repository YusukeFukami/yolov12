
    #file_id = "1VXTGRI_-s5l-JlBVmjZL1NbC2eWH7C9m"  # 変換後のファイルIDに置き換え
    
# YOLOv12モデルをYOLOv8互換形式に変換
# ローカル環境で実行してください

# YOLOv12モデルをYOLOv8互換形式に変換
# ローカル環境で実行してください

from ultralytics import YOLO
import torch
import os
import numpy as np

def convert_yolov12_to_yolov8_compatible(input_model_path, output_model_path):
    """
    YOLOv12モデルをStreamlit Cloud互換形式に変換
    """
    try:
        print(f"🔄 {input_model_path} を読み込み中...")
        
        # YOLOv12モデルを読み込み
        original_model = YOLO(input_model_path)
        
        print("✅ 元モデル読み込み完了")
        print(f"📊 クラス数: {len(original_model.names)}")
        print(f"🏷️ クラス名: {list(original_model.names.values())}")
        
        # テスト推論で動作確認
        print("🧪 元モデルのテスト推論中...")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        test_results = original_model(test_image, verbose=False)
        print("✅ 元モデル正常動作確認")
        
        # 方法1: TorchScript変換（推奨）
        print("🔄 TorchScript形式に変換中...")
        try:
            torchscript_path = output_model_path.replace('.pt', '_torchscript.pt')
            original_model.export(format='torchscript', imgsz=640, optimize=True)
            
            # 生成されたファイルを探す
            for file in os.listdir('.'):
                if file.endswith('_torchscript.torchscript'):
                    os.rename(file, torchscript_path)
                    print(f"✅ TorchScript変換完了: {torchscript_path}")
                    break
        except Exception as ts_err:
            print(f"⚠️ TorchScript変換失敗: {ts_err}")
        
        # 方法2: 重み抽出＋新モデル作成
        print("🔄 重み抽出による互換モデル作成中...")
        try:
            # 元モデルの重みを抽出
            state_dict = original_model.ckpt
            
            # YOLOv8ベースモデルを作成
            base_model = YOLO("yolov8n.pt")
            
            # クラス数を合わせる
            if len(original_model.names) != len(base_model.names):
                print(f"🔧 クラス数調整: {len(base_model.names)} → {len(original_model.names)}")
            
            # 互換性のある重みのみ転送
            base_state_dict = base_model.ckpt
            compatible_weights = {}
            
            for key in base_state_dict.keys():
                if key in state_dict:
                    try:
                        # サイズが一致する場合のみ転送
                        if base_state_dict[key].shape == state_dict[key].shape:
                            compatible_weights[key] = state_dict[key]
                            print(f"✅ 転送: {key}")
                        else:
                            compatible_weights[key] = base_state_dict[key]
                            print(f"⚠️ サイズ不一致、元の重み使用: {key}")
                    except:
                        compatible_weights[key] = base_state_dict[key]
                        print(f"⚠️ エラー、元の重み使用: {key}")
                else:
                    compatible_weights[key] = base_state_dict[key]
                    print(f"📋 新規追加: {key}")
            
            # クラス名情報を追加
            compatible_weights['names'] = original_model.names
            
            # 新しいモデルとして保存
            torch.save(compatible_weights, output_model_path)
            print(f"✅ 互換モデル作成完了: {output_model_path}")
            
        except Exception as weight_err:
            print(f"❌ 重み抽出変換失敗: {weight_err}")
            return False
        
        # 方法3: ONNX変換（バックアップ）
        print("🔄 ONNX形式変換中...")
        try:
            onnx_path = output_model_path.replace('.pt', '.onnx')
            original_model.export(format='onnx', imgsz=640, optimize=True)
            print(f"✅ ONNX変換完了: {onnx_path}")
        except Exception as onnx_err:
            print(f"⚠️ ONNX変換失敗: {onnx_err}")
        
        return True
        
    except Exception as e:
        print(f"❌ 変換エラー: {e}")
        return False

def test_converted_models():
    """変換されたモデルをテスト"""
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    test_files = [
        "converted_pipe_model.pt",
        "converted_pipe_model_torchscript.pt", 
        "converted_pipe_model.onnx"
    ]
    
    working_models = []
    
    for model_file in test_files:
        if os.path.exists(model_file):
            try:
                print(f"🧪 {model_file} をテスト中...")
                
                if model_file.endswith('.onnx'):
                    print(f"⚠️ ONNX形式はYOLOクラスで直接読み込み不可")
                    continue
                
                model = YOLO(model_file)
                results = model(test_image, verbose=False)
                
                print(f"✅ {model_file} 正常動作確認")
                working_models.append(model_file)
                
            except Exception as e:
                print(f"❌ {model_file} テスト失敗: {e}")
    
    return working_models

if __name__ == "__main__":
    print("🚀 YOLOv12 → Streamlit Cloud互換 変換開始\n")
    
    # ファイルパス設定
    input_path = "my_trained_model.pt"  # ダウンロードしたYOLOv12モデル
    output_path = "converted_pipe_model.pt"  # 出力ファイル
    
    # 入力ファイル確認
    if not os.path.exists(input_path):
        print(f"❌ {input_path} が見つかりません")
        print("💡 まず以下を実行してください:")
        print("   gdown https://drive.google.com/uc?id=YOUR_FILE_ID -O my_trained_model.pt")
        exit(1)
    
    # 変換実行
    print(f"📁 入力: {input_path}")
    print(f"📁 出力: {output_path}")
    print("-" * 50)
    
    success = convert_yolov12_to_yolov8_compatible(input_path, output_path)
    
    if success:
        print("\n" + "="*50)
        print("🧪 変換されたモデルをテスト中...")
        working_models = test_converted_models()
        
        if working_models:
            print(f"\n✅ 変換成功！動作するモデル:")
            for model in working_models:
                print(f"   📄 {model}")
            
            print(f"\n🎯 次のステップ:")
            print(f"1. {working_models[0]} をGoogle Driveにアップロード")
            print("2. Google DriveでファイルIDを取得")
            print("3. Streamlit CloudでそのファイルIDを使用")
            print("4. YOLOv8互換として読み込み")
            
        else:
            print("❌ 全ての変換モデルが動作しませんでした")
    else:
        print("❌ 変換に失敗しました")