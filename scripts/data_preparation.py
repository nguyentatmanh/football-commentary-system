import os
import cv2
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

def create_merged_dataset(images_dir, output_dataset_dir, ball_model_path, pitch_model_path, player_model_path):
    """
    Chạy 3 mô hình riêng biệt trên thư mục ảnh thô, gộp nhãn và TỰ ĐỘNG chia train/val (80/20).
    """
    # Tạo cấu trúc thư mục chuẩn YOLO
    train_images = os.path.join(output_dataset_dir, "images", "train")
    val_images = os.path.join(output_dataset_dir, "images", "val")
    train_labels = os.path.join(output_dataset_dir, "labels", "train")
    val_labels = os.path.join(output_dataset_dir, "labels", "val")
    
    for folder in [train_images, val_images, train_labels, val_labels]:
        os.makedirs(folder, exist_ok=True)
    
    print("Loading models...")
    model_ball = YOLO(ball_model_path)
    model_pitch = YOLO(pitch_model_path)
    model_player = YOLO(player_model_path)
    print("Models loaded successfully!")

    image_paths = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png")) + list(Path(images_dir).glob("*.jpeg"))
    # Xáo trộn danh sách ảnh để chia train/val ngẫu nhiên
    random.shuffle(image_paths)
    
    # Tỉ lệ 80% train, 20% val
    split_index = int(len(image_paths) * 0.8)
    train_paths = image_paths[:split_index]
    val_paths = image_paths[split_index:]
    
    def process_images(paths, img_out_dir, lbl_out_dir, split_name):
        count = 0
        for img_path in paths:
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            merged_labels = []
            
            def get_bboxes(model, class_id_to_assign):
                results = model(img, verbose=False)
                bboxes = []
                for result in results:
                    for box in result.boxes:
                        x_c, y_c, bw, bh = box.xywhn[0].tolist()
                        conf = box.conf[0].item()
                        if conf > 0.3:
                            bboxes.append(f"{class_id_to_assign} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")
                return bboxes

            # Sinh nhãn
            merged_labels.extend(get_bboxes(model_ball, 0))
            merged_labels.extend(get_bboxes(model_pitch, 1))
            merged_labels.extend(get_bboxes(model_player, 2))
            
            # Copy ảnh và ghi nhãn nếu có ít nhất 1 vật thể
            if len(merged_labels) > 0:
                shutil.copy2(str(img_path), os.path.join(img_out_dir, img_path.name))
                label_path = os.path.join(lbl_out_dir, img_path.stem + ".txt")
                with open(label_path, "w") as f:
                    f.write("\n".join(merged_labels))
                count += 1
                
            if count % 100 == 0:
                print(f"[{split_name}] Processed {count} images...")
                
        return count

    print(f"\nBắt đầu gán nhãn tập TRAIN ({len(train_paths)} ảnh dự kiến)...")
    train_count = process_images(train_paths, train_images, train_labels, "TRAIN")
    
    print(f"\nBắt đầu gán nhãn tập VAL ({len(val_paths)} ảnh dự kiến)...")
    val_count = process_images(val_paths, val_images, val_labels, "VAL")
    
    print(f"\nHoàn tất! Train: {train_count} ảnh | Val: {val_count} ảnh.")
    
    # Tự động nén thành file Zip
    print(f"\nĐang nén toàn bộ dataset thành file {output_dataset_dir}.zip để đưa lên Kaggle...")
    shutil.make_archive(output_dataset_dir, 'zip', output_dataset_dir)
    print("Nén THÀNH CÔNG! Bạn có thể lấy file zip upload lên Kaggle Dataset ngay.")

if __name__ == "__main__":
    IMAGES_DIR = "C:/thị giác máy tính/football-commentary-system/data/raw_images" 
    OUTPUT_DATASET_DIR = "C:/thị giác máy tính/football-commentary-system/football-merged-dataset"
    
    BALL_MODEL = "C:/thị giác máy tính/football-commentary-system/data/models/football-ball-detection.pt"
    PITCH_MODEL = "C:/thị giác máy tính/football-commentary-system/data/models/football-pitch-detection.pt"
    PLAYER_MODEL = "C:/thị giác máy tính/football-commentary-system/data/models/football-player-detection.pt"
    
    create_merged_dataset(IMAGES_DIR, OUTPUT_DATASET_DIR, BALL_MODEL, PITCH_MODEL, PLAYER_MODEL)
