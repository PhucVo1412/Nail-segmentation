# Kế hoạch train/finetune model segmentation móng tay → TFLite

**Ngày viết:** 2026-09-16
**Phạm vi:** chỉ train model segmentation (thay thế `segment_nail()` classical-CV trong `main.py`) + export `.tflite`. Không đụng vào phần app mobile (AR overlay, apply mẫu nail design) — team khác lo phần đó, ta chỉ giao file `.tflite` + spec input/output.
**Ràng buộc:** tài nguyên hạn hẹp → toàn bộ plan tối ưu theo hướng **$0 chi phí trả phí**, chỉ tốn effort người + compute free-tier.

---

## 0. Nguyên tắc tối ưu chi phí (đọc trước khi bắt tay làm)

1. **Tận dụng data công khai + augmentation trước, tự chụp/gán nhãn sau.** Chỉ leo lên bước tốn công (Giai đoạn 6) nếu bước rẻ (Giai đoạn 1–5) không đạt ngưỡng chấp nhận được.
2. **Train đúng phân phối lúc inference.** Model thật sẽ chỉ nhìn thấy crop-theo-ngón nhỏ (do `crop_finger()` cắt ra), không phải ảnh cả bàn tay — nên data train cũng phải là crop, dùng **chính** hàm `crop_finger()` hiện có để tạo, tránh lệch phân phối train/inference (bug ngầm rất khó bắt sau này).
3. **Model nhỏ + transfer learning** → converge nhanh với ít ảnh, ít epoch, không cần GPU mạnh, train được trên free-tier.
4. **Compute free-tier** (Google Colab / Kaggle free GPU) — không thuê cloud GPU trả phí.
5. **Train thẳng bằng TensorFlow/Keras** để export TFLite không vướng — né đường PyTorch → ONNX → TFLite hay gãy op giữa chừng (thời gian debug cũng là chi phí).
6. **Gate từng giai đoạn bằng số liệu cụ thể**, không làm tràn lan — dừng sớm nếu đạt yêu cầu, không tối ưu quá tay.

---

## 1. Kiến trúc đề xuất

- **Input:** crop vuông 96×96 hoặc 128×128 RGB (crop đã center vào móng nhờ `crop_finger()`).
- **Output:** mask nhị phân cùng kích thước (1 channel, sigmoid).
- **Option A — khuyến nghị, thử trước:** mini U-Net tự viết, train from scratch, ~50–200K params. Bài toán đã được "dọn sẵn" bởi bước crop (object gần giữa khung, ít nhiễu nền) nên không cần backbone nặng.
- **Option B — nếu A chưa đủ chính xác:** encoder `MobileNetV3Small` pretrained ImageNet (có sẵn trong `tf.keras.applications`) + decoder U-Net nhẹ vài lớp upsample với skip connection. Transfer learning giúp generalize tốt hơn khi data còn ít.
- **Mobile inference:** gộp 5 crop (5 ngón) thành 1 batch, gọi TFLite interpreter 1 lần/frame thay vì 5 lần, đỡ overhead.

---

## 2. Giai đoạn 0 — Refactor nhỏ trước khi train (~0.5 ngày)

Tách phần logic dùng chung ra khỏi `main.py` thành `nail_lib.py`:
- `detect_landmarks`, `crop_finger`, `local_to_image_points`, `FINGERS`
- các hằng số `UP_FACTOR` / `DOWN_FACTOR` / `WIDTH_FACTOR`

Để cả `main.py` (inference) và script chuẩn bị dataset (training) **import cùng một hàm crop**, thay vì copy-paste hai bản dễ trôi lệch nhau theo thời gian. Đây là bước rẻ nhất nhưng quan trọng nhất để đảm bảo train đúng phân phối inference (nguyên tắc #2).

---

## 3. Giai đoạn 1 — Chuẩn bị dataset (~1–2 ngày, $0)

**Nguồn public (đã khảo sát trong [readme.md](readme.md#3-dataset)):**

| Nguồn | Ảnh | Nhãn | License | Dùng được cho sản phẩm thương mại? |
|---|---|---|---|---|
| [vpapenko/nails-segmentation-dataset](https://github.com/vpapenko/nails-segmentation-dataset) | nhỏ | ảnh + mask | CC0-1.0 (đã verify) | ✅ Có — public domain, không ràng buộc gì |
| [Roboflow nail-segmentation-odcgv](https://universe.roboflow.com/aiit-nail-competition/nail-segmentation-odcgv) | 379 | polygon | CC BY 4.0 (đã verify) | ✅ Có, nhưng **phải ghi công** nguồn gốc (tên tác giả, link, license) — lưu lại trong repo/credits sản phẩm |
| [Golbstein/Fingernails-Segmentation](https://github.com/Golbstein/Fingernails-Segmentation) | — | mask | **Không có LICENSE file** (đã verify — không có badge/file license nào trên repo) | ⚠️ Chưa rõ — public repo không đồng nghĩa được cấp phép sử dụng tự do. Cần liên hệ tác giả xin phép trước khi dùng, hoặc **bỏ nguồn này** nếu 2 nguồn trên đã đủ data cho baseline |

Roboflow polygon cần rasterize thành mask (`cv2.fillPoly`) hoặc export thẳng dạng mask/COCO qua Roboflow.

**Lưu ý:** ưu tiên build baseline chỉ từ vpapenko + Roboflow (đã rõ ràng pháp lý). Chỉ cân nhắc thêm Golbstein nếu thiếu data *và* đã xin phép được tác giả.

**Cách tạo cặp (crop_img, crop_mask) — điểm mấu chốt tiết kiệm effort:**

Viết `prepare_dataset.py`: với mỗi ảnh gốc + mask gốc trong dataset public:
1. Chạy `detect_landmarks()` (MediaPipe) trên ảnh gốc → 21 landmark.
2. Với mỗi ngón, gọi `crop_finger()` **y hệt** logic production để lấy `crop`, `M`, `offset`.
3. Áp **cùng** `M` (warpAffine) + cùng `offset`/kích thước crop lên ảnh **mask** gốc → crop mask khớp pixel-perfect với crop ảnh, không cần biết đây là ngón nào (thumb/index/...) — chỉ cần vùng crop và mask tương ứng thẳng hàng.
4. Lưu cặp `(crop_img, crop_mask)`.

**Lọc:**
- Bỏ ảnh MediaPipe không detect được tay.
- Bỏ crop có mask gần như rỗng (0 pixel nail) — coi là crop lệch/không có móng trong khung, trừ khi chủ đích giữ một ít mẫu "negative" đã verify tay.

**Ước tính:** ~300–700 ảnh gốc public → ~1000–2500 crop hợp lệ (không phải ảnh nào cũng lộ đủ 5 ngón có nhãn).

**Split train/val/test theo ẢNH GỐC**, không theo crop — nhiều crop từ cùng 1 ảnh không được lẫn giữa train và val/test (tránh leakage).

---

## 4. Giai đoạn 2 — Augmentation (song song Giai đoạn 1, $0)

Áp dụng trên crop (không phải ảnh gốc):
- Xoay nhẹ ±15°, brightness/contrast jitter, hue/saturation jitter (bù đa dạng tông da & ánh sáng mà dataset public không có nhiều)
- Horizontal flip, scale jitter nhẹ
- Cutout/occlusion nhỏ ngẫu nhiên (giả lập phản chiếu ánh sáng che một phần móng)

Dùng `tf.keras.layers` augmentation hoặc `albumentations` — nhân effective dataset lên 5–10x, miễn phí, không cần thêm ảnh thật.

---

## 5. Giai đoạn 3 — Train baseline (~0.5–1 ngày compute, $0 — Colab free T4)

- Loss: **BCE + Dice** (chuẩn cho segmentation object nhỏ/mất cân bằng lớp foreground-background).
- Optimizer Adam, mixed precision, early stopping theo val IoU.
- Batch size có thể lớn vì ảnh nhỏ (96–128px) → train nhanh.
- Log bằng TensorBoard (miễn phí, local) là đủ, không cần dịch vụ trả phí.
- Nếu máy hiện tại có GPU: kiểm tra trước bằng `tf.config.list_physical_devices('GPU')`, có thể train local luôn không cần Colab.

---

## 6. Giai đoạn 4 — Đánh giá & gate quyết định (~0.5 ngày)

- Tính **mIoU** trên tập test giữ riêng (không đụng lúc train).
- Test end-to-end thực tế: thay `segment_nail()` trong `main.py` bằng model mới, chạy lại trên `images/img1.jpg`, `images/img2.jpg`, so sánh chất lượng polygon với bản GrabCut hiện tại (đã có `output.jpg`/`output.json` làm baseline so sánh trực quan).

**Gate quyết định:**
- mIoU ≥ ~0.85 trên test set → đi tiếp Giai đoạn 5 (export TFLite).
- Thấp hơn rõ rệt → thử Option B (MobileNetV3-Small) trước khi nghĩ đến Giai đoạn 6 (tốn công thu thập + gán nhãn thêm — chỉ làm khi thực sự cần).

---

## 7. Giai đoạn 5 — Export TFLite + quantize (~0.5 ngày, $0)

- `tf.lite.TFLiteConverter.from_keras_model(model)`.
- Thử 2 mức quantize, so mIoU trước/sau để chọn:
  - **float16**: an toàn về accuracy, nhẹ ~2x so với float32.
  - **full-integer int8**: nhanh/nhẹ nhất cho mobile, cần `representative_dataset` (lấy vài trăm crop từ tập train).
- Benchmark latency bằng TFLite interpreter chạy trên CPU máy hiện tại — chỉ là proxy tốc độ vì không có device mobile thật để đo trực tiếp; **ghi rõ cho team mobile cần tự đo lại trên device đích**.
- Output bàn giao: `models/nail_seg.tflite` + file spec input/output (xem mục 8).

---

## 8. Giai đoạn 6 — CHỈ làm nếu Giai đoạn 4 không đạt (không tính trước vào effort/cost)

- Tự chụp thêm ảnh tay đa dạng tông da/ánh sáng/móng đã sơn màu — bắt đầu nhỏ (~100–200 ảnh mới) nhắm đúng nhóm còn yếu (đọc từ lỗi cụ thể ở Giai đoạn 4), không chụp đại trà.
- Gán nhãn **bán tự động bằng SAM/SAM2** (chạy free trên Colab): chỉ cần click/box mỗi móng thay vì vẽ polygon thủ công toàn bộ — nhanh hơn nhiều so với gán nhãn tay 100%, rồi review nhanh lại.
- Retrain **từ checkpoint** Option A/B đã có sẵn (không train lại từ đầu) để tiết kiệm compute.

## 9. (Optional, không ưu tiên chi phí) Auxiliary head hướng gốc→đầu móng

Như paper Banuba (arXiv:1906.02222, xem [readme.md](readme.md#2-kiến-trúc-model-tham-khảo)) — dự đoán thêm hướng base→tip cùng mask để polygon có hướng nhất quán. Bỏ qua ở bản đầu để tối ưu chi phí, chỉ làm nếu team mobile thực sự cần polygon có hướng ổn định (vd. để align texture nail design).

---

## 10. Bàn giao cho team mobile

- File `.tflite` + spec: kích thước input, cách normalize (`[0,1]` hay `[-1,1]`), thứ tự channel (RGB), kích thước/threshold output mask.
- Ghi rõ: model **chỉ nhận crop-theo-ngón đã chuẩn hoá góc xoay** giống hệt `crop_finger()` trong `main.py` — team mobile cần implement lại đúng bước crop này (native code Android/iOS), không phải chạy thẳng trên ảnh full-frame.
- Cung cấp lại logic `local_to_image_points()` (map crop-local → ảnh gốc) để tham khảo port sang mobile, dùng vẽ polygon đúng vị trí trên ảnh preview camera.

---

## 11. Ước tính effort & chi phí tổng

| Giai đoạn | Effort người | Compute | Chi phí |
|---|---|---|---|
| 0. Refactor `nail_lib.py` | 0.5 ngày | — | $0 |
| 1. Chuẩn bị dataset (crop + mask) | 1–2 ngày | CPU | $0 |
| 2. Augmentation | song song GĐ1 | CPU | $0 |
| 3. Train baseline | 0.5–1 ngày | Colab free T4 | $0 |
| 4. Đánh giá + gate | 0.5 ngày | CPU | $0 |
| 5. Export + quantize TFLite | 0.5 ngày | CPU | $0 |
| **Tổng (nếu đạt gate ở GĐ4)** | **~3–4.5 ngày** | free-tier | **$0** |
| 6. (Chỉ nếu cần) chụp + gán nhãn thêm + retrain | +1–3 ngày | Colab free (SAM) | $0 tiền, tốn thời gian người |

---

## 12. Rủi ro & phương án dự phòng

- **MediaPipe landmark fail trên ảnh dataset lạ góc/độ phân giải thấp** → hạ ngưỡng confidence hoặc loại bỏ ảnh đó; không ảnh hưởng nhiều vì dư data (mất một ít không sao).
- **int8 quantize giảm mạnh accuracy** → dùng float16 thay thế, vẫn nhẹ hơn đáng kể so với float32 gốc, an toàn accuracy hơn.
- **Data public thiên lệch tông da/ánh sáng** → nếu Giai đoạn 4 lộ rõ bias (model tệ hẳn trên 1 nhóm), ưu tiên Giai đoạn 6 chụp bổ sung đúng nhóm thiếu thay vì chụp đại trà tốn effort không hiệu quả.
- **Model tốt trên desktop nhưng chậm trên mobile thật** → vì không đo được trên device thật ở giai đoạn train, cần team mobile benchmark sớm bản TFLite đầu tiên (ngay sau Giai đoạn 5) để phát hiện sớm nếu cần giảm thêm kích thước input/model, tránh làm xong hết pipeline mới phát hiện không kịp real-time.
