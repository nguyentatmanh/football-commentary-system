import os
import shutil
# pyrefly: ignore [missing-import]
from roboflow import Roboflow

def download_and_gather_images(api_key, output_raw_dir):
    """
    Tải 3 dataset từ Roboflow và gộp tất cả ảnh vào thư mục raw_images.
    Chỉ lấy ảnh, bỏ qua nhãn cũ vì chúng ta sẽ dùng data_preparation.py để gán nhãn lại.
    """
    os.makedirs(output_raw_dir, exist_ok=True)
    
    rf = Roboflow(api_key=api_key)
    
    # Định nghĩa 3 project trên Roboflow (workspace, project_name, version)
    datasets_to_download = [
        {"workspace": "roboflow-jvuqo", "project": "football-field-detection-f07vi", "version": 12},
        {"workspace": "roboflow-jvuqo", "project": "football-players-detection-3zvbc", "version": 10},
        {"workspace": "roboflow-jvuqo", "project": "football-ball-detection-rejhg", "version": 2}
    ]
    
    downloaded_paths = []
    
    print("Bắt đầu tải datasets từ Roboflow...")
    for ds_info in datasets_to_download:
        print(f"\nĐang tải {ds_info['project']} v{ds_info['version']}...")
        project = rf.workspace(ds_info["workspace"]).project(ds_info["project"])
        version = project.version(ds_info["version"])
        # Tải dưới dạng yolov8 (hoặc bất kỳ format nào vì ta chỉ cần ảnh)
        dataset = version.download("yolov8")
        downloaded_paths.append(dataset.location)
        
    print("\nQuá trình tải hoàn tất! Bắt đầu gộp ảnh...")
    
    image_extensions = ('.jpg', '.jpeg', '.png')
    copied_count = 0
    
    # Duyệt qua các thư mục vừa tải về
    for ds_path in downloaded_paths:
        # Trong format yolov8, ảnh thường nằm trong thư mục train/images, valid/images, test/images
        for split in ['train', 'valid', 'test']:
            split_images_dir = os.path.join(ds_path, split, 'images')
            
            if not os.path.exists(split_images_dir):
                continue
                
            for filename in os.listdir(split_images_dir):
                if filename.lower().endswith(image_extensions):
                    src_file = os.path.join(split_images_dir, filename)
                    dst_file = os.path.join(output_raw_dir, filename)
                    
                    # Tránh ghi đè nếu có trùng tên (Roboflow thường băm tên ảnh nên hiếm khi trùng)
                    if os.path.exists(dst_file):
                        base, ext = os.path.splitext(filename)
                        dst_file = os.path.join(output_raw_dir, f"{base}_duplicate{ext}")
                        
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1
                    
    print(f"\n✅ Đã gộp thành công {copied_count} ảnh vào thư mục: {output_raw_dir}")
    print("Bây giờ bạn có thể chạy file scripts/data_preparation.py để tự động gán nhãn!")

if __name__ == "__main__":
    # LƯU Ý QUAN TRỌNG:
    # 1. Cài đặt thư viện: pip install roboflow
    # 2. Thay thế API_KEY của bạn dưới đây. 
    # (Bạn có thể lấy API Key bằng cách đăng nhập Roboflow -> Settings -> Roboflow API)
    
    ROBOFLOW_API_KEY = "eTy7H5hR1klp9iGpRyyK" 
    
    # Thư mục chứa ảnh gộp (Cùng đường dẫn với IMAGES_DIR trong data_preparation.py)
    OUTPUT_RAW_DIR = "C:/thị giác máy tính/football-commentary-system/data/raw_images"
    
    if ROBOFLOW_API_KEY == "ĐIỀN_API_KEY_CỦA_BẠN_VÀO_ĐÂY":
        print("LỖI: Bạn chưa điền Roboflow API Key!")
        print("Vui lòng lấy API Key trên trang web Roboflow và thay vào biến ROBOFLOW_API_KEY.")
    else:
        download_and_gather_images(ROBOFLOW_API_KEY, OUTPUT_RAW_DIR)
