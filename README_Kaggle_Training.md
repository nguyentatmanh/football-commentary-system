# Hướng dẫn Huấn luyện Mô hình YOLOv8 Gộp trên Kaggle

Để huấn luyện thành công mô hình nhận diện được cả 3 vật thể (Bóng, Sân, Cầu thủ) với ảnh kích thước 1280 và YOLOv8x, bạn cần làm tuần tự theo 3 giai đoạn sau đây:

---

## GIAI ĐOẠN 1: Chuẩn bị Dữ liệu (Thực hiện trên máy tính của bạn)

Mục tiêu của bước này là dùng 3 mô hình `.pt` cũ của bạn để quét qua kho ảnh và tự động tạo ra bộ nhãn (labels) mới, trong đó mỗi bức ảnh đều có đủ 3 nhãn (0: bóng, 1: sân, 2: cầu thủ).

1. **Gom ảnh:** Tạo một thư mục (ví dụ `C:\thị giác máy tính\raw_images`) và ném TẤT CẢ các bức ảnh bóng đá từ 3 dataset trên Roboflow vào đó.
2. **Chạy Script:** 
   - Mở file `scripts/data_preparation.py`.
   - Kéo xuống dưới cùng (dòng 52), sửa lại các đường dẫn `IMAGES_DIR` (thư mục ảnh vừa tạo), `OUTPUT_LABELS_DIR` (nơi lưu nhãn) và đường dẫn 3 file `.pt` cho đúng với máy bạn.
   - Chạy lệnh trong terminal: `python scripts/data_preparation.py`.
   - Kết quả: Trong `OUTPUT_LABELS_DIR` sẽ xuất hiện hàng ngàn file `.txt`.
3. **Cấu trúc lại thư mục:** Tạo một thư mục mới tên là `football-merged-dataset` và sắp xếp ảnh, nhãn theo đúng chuẩn YOLO như sau:
   ```text
   football-merged-dataset/
   ├── images/
   │   ├── train/  (Copy 80% số ảnh vào đây)
   │   └── val/    (Copy 20% số ảnh vào đây)
   └── labels/
       ├── train/  (Copy 80% file .txt tương ứng vào đây)
       └── val/    (Copy 20% file .txt tương ứng vào đây)
   ```
4. **Nén dữ liệu:** Nén thư mục `football-merged-dataset` thành file `football-merged-dataset.zip`.

---

## GIAI ĐOẠN 2: Đưa Dữ liệu lên Kaggle

1. Đăng nhập vào [Kaggle](https://www.kaggle.com/).
2. Nhấn nút **+ Create** góc trái màn hình -> Chọn **New Dataset**.
3. Kéo thả file `football-merged-dataset.zip` vào, đặt tên là `football-merged-dataset` và nhấn **Create**.
4. Tiếp tục tạo một Dataset mới nữa, lần này bạn tải file `data/dataset_merged.yaml` lên (hoặc bạn có thể tự tạo file này trực tiếp trong notebook Kaggle sau cũng được).

---

## GIAI ĐOẠN 3: Chạy Huấn Luyện (Training) trên Kaggle

1. Trở về trang chủ Kaggle, nhấn **+ Create** -> Chọn **New Notebook**.
2. **Import code:** Trên thanh menu của Notebook, chọn `File` -> `Import Notebook`, sau đó tải lên file `kaggle/train_merged_model.ipynb` mà tôi đã tạo sẵn cho bạn ở thư mục dự án.
3. **Bật Multi-GPU:** 
   - Nhìn sang menu bên phải màn hình (phần **Notebook options**).
   - Ở mục **Accelerator**, bấm chọn **GPU T4 x2**. (Kaggle sẽ khởi động lại phiên chạy).
4. **Kết nối Dữ liệu:**
   - Ở menu bên phải, mục **Data**, nhấn nút **+ Add Data**.
   - Tìm kiếm tên dataset `football-merged-dataset` mà bạn vừa tạo ở Giai đoạn 2 rồi nhấn dấu **+** để add nó vào Notebook.
5. **Cập nhật đường dẫn file YAML:**
   - Khi bạn Add data vào Kaggle, dữ liệu sẽ nằm ở đường dẫn `/kaggle/input/...`
   - Bạn cần mở file `dataset_merged.yaml` và sửa lại dòng `path` trỏ đúng vào thư mục dataset của bạn. 
   - *Ví dụ:* `path: /kaggle/input/football-merged-dataset/football-merged-dataset`
6. **BẮT ĐẦU TRAIN:**
   - Bấm nút **Run All** (hoặc chạy từng ô mã - Cell) trong Notebook.
   - Code sẽ tự động tải thư viện YOLO, bắt đầu quá trình train trên 2 GPU T4 với mô hình YOLOv8x lớn nhất.
   - Khi quá trình train 100 epochs chạy xong (mất khoảng vài tiếng), file trọng số hoàn hảo nhất sẽ được lưu tại đường dẫn: `/kaggle/working/Football_Merged/yolov8x_1280p/weights/best.pt`. Bạn có thể tải file này về máy để sử dụng!
