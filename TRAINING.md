# Hướng dẫn train model nail segmentation

Tài liệu này mô tả cách chạy pipeline train model (thay thế `segment_nail()` placeholder
trong `main.py`), theo đúng 5 giai đoạn đã đề ra trong `training_plan.md`. Xem `CLAUDE.md`
để hiểu kiến trúc tổng thể, xem `training_plan.md` để hiểu lý do/quyết định đằng sau từng bước.

## 0. Cài đặt môi trường (chạy 1 lần)

```
.venv\Scripts\pip.exe install -r requirements.txt
.venv\Scripts\pip.exe install tensorflow
```

Kiểm tra lại đã cài đủ chưa:

```
.venv\Scripts\pip.exe list
.venv\Scripts\python.exe -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Máy Windows native nên TensorFlow sẽ chạy **CPU-only** (không có bản GPU cho Windows native
từ TF > 2.10) — model nhỏ (~120K tham số) và dataset ít nên train CPU vẫn đủ nhanh, không cần Colab.

> Nếu gặp lỗi `THESE PACKAGES DO NOT MATCH THE HASHES` khi cài: thường do mạng chập chờn làm hỏng
> gói tải về giữa chừng, không hẳn bị can thiệp. Thử `pip cache purge` rồi cài lại.

## 1. Chuẩn bị dataset — `prepare_dataset.py`

Dùng chính `crop_finger()`/`detect_landmarks()` trong `nail_lib.py` để cắt từng ngón từ ảnh gốc,
đảm bảo dữ liệu train giống hệt phân phối ảnh lúc inference thật.

Hiện chỉ đăng ký nguồn **`nails_segmentation/`** (vpapenko, CC0-1.0, an toàn để dùng).
Thư mục **`nails/nails/`** (nghi là Golbstein/Fingernails-Segmentation, chưa rõ license)
**chưa được dùng** — cần quyết định về license trước khi thêm vào.

```
.venv\Scripts\python.exe prepare_dataset.py --sources vpapenko --out dataset
```

Kết quả: `dataset/{train,val,test}/{images,masks}/*.png` (crop 128×128, BGR) + `dataset/manifest.csv`.
Script sẽ in ra số ảnh gốc xử lý được, số bị bỏ (không detect được tay / mask rỗng), và số crop
mỗi split. Với 52 ảnh gốc, kỳ vọng khoảng ~150–250 crop hợp lệ (rất ít nên đừng ngạc nhiên nếu số
mIoU ở bước 4 dao động nhiều).

Các tham số hay chỉnh: `--crop-size`, `--val-frac`, `--test-frac`, `--min-fg-frac`, `--seed`.

## 2+3. Train baseline — `train.py`

Đọc crop từ bước 1, augment on-the-fly (flip/rotate/zoom + brightness/contrast/hue/saturation),
train mini U-Net (định nghĩa trong `nail_seg_model.py`) với loss BCE+Dice, early-stopping theo `val_iou`.

```
.venv\Scripts\python.exe train.py --data dataset --out models/nail_seg.keras
```

Theo dõi tiến trình qua console (in `val_iou` mỗi epoch) hoặc TensorBoard:

```
.venv\Scripts\python.exe -m tensorboard.main --logdir logs
```

Model tốt nhất (theo `val_iou`) được lưu tại `models/nail_seg.keras`. Các tham số hay chỉnh:
`--epochs`, `--batch-size`, `--base-filters`, `--lr`.

## 4. Đánh giá — `evaluate.py`

Tính **mIoU trên tập test** (ngưỡng chấp nhận ~0.85 theo `training_plan.md`), breakdown theo từng
ngón, xuất ảnh so sánh `ảnh gốc | ground-truth | dự đoán`. Đồng thời chạy thử end-to-end trên
`images/img1.jpg`/`img2.jpg` để so sánh trực quan với `output.jpg` (baseline GrabCut hiện tại).

```
.venv\Scripts\python.exe evaluate.py --model models/nail_seg.keras --data dataset/test --out eval_out
```

Xem kết quả tại `eval_out/` (`*_iou0.xx.png` là các grid so sánh, `e2e_img1.jpg`/`e2e_img2.jpg`
là kết quả full pipeline). Nếu mIoU trung bình đạt ~0.85 trở lên thì mới nên đi tiếp bước 5.

## 5. Export TFLite — `export_tflite.py`

Chỉ chạy khi bước 4 đạt gate. Convert sang TFLite float16 (mặc định), re-verify lại mIoU qua
interpreter TFLite để chắc chắn không bị tụt do quantize hay lệch kênh màu.

```
.venv\Scripts\python.exe export_tflite.py --model models/nail_seg.keras --data dataset/test --out models/nail_seg_fp16.tflite
```

Thêm `--try-int8` nếu muốn thử luôn bản int8 (nhẹ hơn, dùng `dataset/train` làm representative
dataset) — script tự so sánh độ tụt mIoU giữa fp16 và int8, giữ fp16 nếu int8 tệ hơn đáng kể.

Output: file `.tflite` trong `models/` + `models/nail_seg_spec.json` mô tả input/output shape,
chuẩn hoá `[0,1]`, và **thứ tự kênh màu BGR** (không phải RGB — quan trọng cho bên làm app mobile).

## Sau khi có model

Thay phần thân hàm `segment_nail()` trong `main.py` bằng lời gọi TFLite interpreter với
`models/nail_seg_fp16.tflite` — phần crop/detect landmark/map tọa độ ngược (`nail_lib.py`) giữ nguyên,
không cần đổi gì.

## Thứ tự chạy tóm tắt

```
prepare_dataset.py  →  train.py  →  evaluate.py  →  (đạt gate) export_tflite.py
```
