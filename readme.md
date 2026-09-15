# Nghiên cứu: Detect điểm bao quanh 5 móng tay (nail contour points) → xuất TFLite

**Yêu cầu gốc (từ anh lead, qua tin nhắn của Phúc, 2026-09-15):** App VR/AI cần nhận diện các điểm bao quanh 5 móng tay khi người dùng đưa tay vào camera, trả về dạng file model (.tflite hoặc format khác), để phía app đọc điểm và tự vẽ polygon (phục vụ AR thử sơn móng / nail art).

**Quyết định đã chốt với Phúc:**
- Hướng kỹ thuật: **Segmentation + contour extraction** (model xuất mask từng móng, code app tự lấy điểm biên bằng contour rồi vẽ polygon) — thay vì model xuất thẳng landmark cố định.
- Chưa có dataset riêng, cần research/đề xuất nguồn.
- Giai đoạn hiện tại: research & đề xuất phương án, chưa build.

## 1. Pipeline đề xuất

1. **Hand detection + crop từng ngón** — dùng MediaPipe Hand Landmarker (model .task/.tflite có sẵn, miễn phí, chạy tốt trên mobile) lấy 21 landmark bàn tay. Dùng landmark đầu ngón (index 4, 8, 12, 16, 20) và khớp ngay dưới (3, 7, 11, 15, 19) để tính bounding box + hướng của từng ngón, crop 5 vùng ảnh nhỏ quanh mỗi đầu ngón (có thể xoay theo hướng ngón).
2. **Segmentation model nhẹ** — mỗi vùng crop nhỏ (ví dụ 64×64 hoặc 96×96) đưa qua model segmentation nhị phân (móng / không phải móng). Vì input đã crop nhỏ và gần như center vào móng, model có thể rất nhẹ so với segment cả bàn tay lớn.
3. **Contour → polygon** — từ mask nhị phân, lấy điểm biên (contour, kiểu `cv2.findContours`), đơn giản hóa bằng Douglas-Peucker (`approxPolyDP`) còn ~8–16 điểm/móng, map tọa độ ngược về ảnh gốc (bù lại bước crop/xoay). Bước này chạy ở code app, không nằm trong chính file model.
4. **Output** — danh sách điểm (x, y normalized) cho từng móng trong 5 móng → app vẽ polygon.

## 2. Kiến trúc model tham khảo

Bài báo production thực tế gần nhất với đúng bài toán này: **"Nail Polish Try-On: Realtime Semantic Segmentation of Small Objects for Native and Browser Smartphone AR Applications"** (Banuba, arXiv:1906.02222). Kết quả: ~94.5 mIoU ở 29.8ms/frame trên iPad Pro, kiến trúc cho phép đánh đổi tốc độ/độ chính xác bằng cách chỉnh input resolution và độ sâu network. Họ còn dự đoán thêm **hướng gốc→đầu móng** (base-tip direction) cùng với mask, giúp căn polygon/texture đúng chiều — đáng tham khảo nếu cần polygon có hướng nhất quán chứ không chỉ đúng hình dạng.

Với hướng crop-theo-từng-ngón (mục 1), một **mini U-Net** hoặc **MobileNetV3-small encoder + decoder nhẹ** là đủ — nhỏ hơn nhiều so với model segment cả bàn tay. Có code baseline sẵn để tham khảo/fork:
- [Golbstein/Fingernails-Segmentation](https://github.com/Golbstein/Fingernails-Segmentation) — mini U-Net cho fingernail segmentation, có notebook train sẵn.
- [sumamCodes/Nail-Segmentation](https://github.com/sumamCodes/Nail-Segmentation)
- [sercant/mobile-segmentation](https://github.com/sercant/mobile-segmentation) — kiến trúc segmentation real-time tối ưu cho mobile nói chung.

## 3. Dataset

| Nguồn | Số ảnh | Nhãn | License | Ghi chú |
|---|---|---|---|---|
| [vpapenko/nails-segmentation-dataset](https://github.com/vpapenko/nails-segmentation-dataset) | nhỏ (chưa rõ số chính xác) | ảnh + mask | CC0 | Dùng bootstrap nhanh |
| [Roboflow: nail-segmentation-odcgv (AIIT nail competition)](https://universe.roboflow.com/aiit-nail-competition/nail-segmentation-odcgv) | 379 | **polygon instance segmentation** (1 class "nail") | CC BY 4.0 | Có sẵn polygon, export được COCO/YOLO-seg qua Roboflow — phù hợp nhất nếu train trực tiếp trên polygon |
| Dataset kèm [Golbstein/Fingernails-Segmentation](https://github.com/Golbstein/Fingernails-Segmentation) | — | mask | — | Kèm code train |
| Tự chụp theo tutorial MakeML/[Medium (Alexey Korotkov)](https://medium.com/@lekorotkov/nails-semantic-segmentation-for-ios-tutorial-2e0471c27820) | 50 (tự chụp) | mask | — | Pipeline đầy đủ: chụp → train (MakeML) → export .tflite → app iOS demo sẵn trên GitHub |
| [Hugging Face: nngeek195/nail-segmentation-v1](https://huggingface.co/nngeek195/nail-segmentation-v1) | — | U-Net **PyTorch** train sẵn | — | Chưa có bản TFLite/ONNX, phải tự convert; chưa rõ độ đa dạng data train |

**Nhận định:** các dataset public đều khá nhỏ (50–379 ảnh) và chưa chắc đa dạng tông da / ánh sáng / móng đã sơn màu / góc chụp giống điều kiện thực tế của app. Đủ để dựng demo/prototype nhanh, nhưng để đạt chất lượng production nên tự thu thập + gán nhãn thêm ảnh tay thật (CVAT hoặc Roboflow) ở giai đoạn 2.

## 4. Format xuất

- **.tflite** hợp lý nếu app đích chạy Android (TensorFlow Lite runtime), hỗ trợ quantize int8 để nhẹ/nhanh trên thiết bị.
- Nếu cần cả iOS, convert thêm sang **CoreML** (`coremltools`) hoặc dùng **ONNX** làm định dạng trung gian.
- Bước contour→polygon nên để ở code app, không cần nhúng vào trong chính graph model.

## 5. Đề xuất các bước tiếp theo

- **Giai đoạn 1 (demo nhanh, ~2–3 ngày):** dùng dataset Roboflow (polygon sẵn) hoặc vpapenko + code baseline Golbstein, train 1 bản mini U-Net nhỏ, export .tflite, dựng thử pipeline đầy đủ (MediaPipe crop → segment → contour → polygon) để anh lead xem demo chạy thật, đánh giá hướng đi trước khi đầu tư dataset riêng.
- **Giai đoạn 2:** nếu hướng ổn nhưng độ chính xác chưa đủ, thu thập + gán nhãn ảnh tay thật (đa dạng tông da, móng sơn màu, góc chụp), retrain, tối ưu quantize theo tốc độ thực tế trên thiết bị đích.

## 6. Prototype đã chạy thử (2026-09-15)

Đã viết `nail_points_demo.py` — bản demo chạy được ngay bằng thư viện + model có sẵn (**chưa cần train gì**), để chứng minh pipeline trước khi đầu tư train model thật:

- **Hand detection:** MediaPipe Hand Landmarker Tasks API (`mediapipe.tasks.python.vision.HandLandmarker`), model `hand_landmarker.task` tải sẵn từ Google (qua GitHub mirror vì `storage.googleapis.com` bị chặn trong sandbox — máy có internet thường sẽ tải thẳng từ trang chính thức của Google được).
- **Crop từng ngón:** dùng landmark đầu ngón (4/8/12/16/20) + khớp dưới (3/7/11/15/19) để tính hướng ngón, xoay ảnh cho ngón thẳng đứng rồi crop. **Lưu ý quan trọng phát hiện được:** landmark đầu ngón nằm NGAY TRÊN móng (thậm chí giữa móng nếu là móng giả dài), không phải dưới móng — nên vùng crop phải mở rộng đáng kể về CẢ hai phía quanh landmark tip, không chỉ về phía gốc ngón.
- **Segmentation (placeholder cho model thật):** thử rect-seeded GrabCut trước — **không ổn định**, cùng 1 ảnh crop nhưng chạy nhiều lần cho kết quả khác nhau (do bước khởi tạo GMM/k-means nội bộ của GrabCut có random). Đã sửa bằng cách: lấy viền ngoài crop làm mẫu màu da (deterministic), tính khoảng cách màu Lab mỗi pixel tới mẫu đó, Otsu-threshold để có mask khởi tạo, rồi mới đưa vào GrabCut refine (`GC_INIT_WITH_MASK`) — ổn định qua nhiều lần chạy.
- **Kết quả test** trên ảnh tay thật (móng đã sơn đen, do anh Phúc gửi để test AR overlay hiện tại): **detect được 5/5 móng**, polygon bám khá sát viền móng thật, tốt hơn rõ rệt so với overlay hình chữ nhật cố định đang dùng trong app hiện tại.
- **Giới hạn đã biết của bản demo này (không phải giới hạn của hướng đi, mà của việc dùng CV cổ điển thay vì model đã train):**
  - Mới test trên 1 ảnh, móng đã sơn màu tối (tương phản mạnh với da) — cần test thêm với móng tự nhiên/màu nhạt, nhiều tông da, nhiều góc chụp.
  - GrabCut + color-distance là heuristic, không tổng quát hoá tốt bằng model đã train trên dataset đa dạng — đây chính là lý do nên đi tiếp hướng train mini U-Net như đề xuất ở mục 2, dùng chính pipeline crop-theo-ngón này làm bước tiền xử lý.
  - Chưa export .tflite ở bước demo này (vì bước segmentation hiện là CV cổ điển, không phải model). Khi có model U-Net train xong, chỉ cần thay hàm `segment_nail()` bằng gọi TFLite interpreter là dùng được y nguyên phần crop + contour + polygon.

File demo + kết quả (`nail_points_demo.py`, `nail_points_demo_result.jpg`, `nail_points_demo_result.json`, `models/hand_landmarker.task`) đã gửi cho Phúc qua chat.

## Sources
- [Nail Polish Try-On: Realtime Semantic Segmentation of Small Objects (arXiv:1906.02222)](https://arxiv.org/abs/1906.02222)
- [Golbstein/Fingernails-Segmentation](https://github.com/Golbstein/Fingernails-Segmentation)
- [vpapenko/nails-segmentation-dataset](https://github.com/vpapenko/nails-segmentation-dataset)
- [Roboflow: nail-segmentation-odcgv](https://universe.roboflow.com/aiit-nail-competition/nail-segmentation-odcgv)
- [Nails Semantic Segmentation iOS App Tutorial (Medium)](https://medium.com/@lekorotkov/nails-semantic-segmentation-for-ios-tutorial-2e0471c27820)
- [Hugging Face: nngeek195/nail-segmentation-v1](https://huggingface.co/nngeek195/nail-segmentation-v1)
- [sumamCodes/Nail-Segmentation](https://github.com/sumamCodes/Nail-Segmentation)
- [sercant/mobile-segmentation](https://github.com/sercant/mobile-segmentation)
- [MediaPipe Hand Landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)