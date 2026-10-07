# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | NguyenTatDat |
| MSSV | 2A202602578 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/Gaohonggg/K4-L3-DAY21-NguyenTatDat-2A202602578-CI-CD-for-AI-Systems |
| Ngày nộp | Chưa nộp |

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Run 3 đạt F1 cao nhất, vượt 0.65. So với run 1, recall tăng 0.6048→0.6371, precision giảm 0.8621→0.8144. Chênh lệch nhỏ trên 500 mẫu chưa chứng minh ưu thế ổn định; ba cấu hình chưa cô lập ảnh hưởng từng tham số.

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Luôn đoán thu nhập thấp đạt accuracy 0.752 trên holdout nhưng F1 lớp dương bằng 0. Gate dùng F1 nhị phân với target=1, kết hợp precision và recall; weighted chịu trọng số lớp đông, macro trung bình hai lớp. Nếu tìm khách hàng tiềm năng, bỏ sót gây mất cơ hội nên ưu tiên recall; chi phí tùy ứng dụng.

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow cần `pkg_resources`. | Setuptools 84 không còn API này. | Đã hạ và cố định setuptools 80.9.0. |
| Không tạo được bucket S3. | IAM user thiếu quyền `s3:CreateBucket`. | Gắn policy giới hạn bucket lab; DVC đã push thành công. |
| CI lỗi fingerprint SSH. | Fingerprint ED25519 khác host key ECDSA được Action chọn. | Lấy fingerprint ECDSA trực tiếp từ EC2, cập nhật variable và chạy lại Release. |

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (22.361 mẫu) | 0.7149 | 0.8740 |
| Bước 3 (44.722 mẫu) | 0.7354 | 0.8820 |

**Nhận xét:** Cùng holdout 500 mẫu và siêu tham số, F1 tăng 0.0205, accuracy tăng 0.0080; precision tăng 0.8144→0.8283, recall tăng 0.6371→0.6613. Cải thiện này chưa khái quát cho mọi dữ liệu. Commit `1b0fa90` chỉ đổi con trỏ DVC, kích hoạt pipeline. API xác nhận deployment và SHA-256 khớp manifest S3.

## 5. Phần Bonus Đã Thực Hiện

- [ ] Bonus 1: Chưa tích hợp DagsHub.
- [x] Bonus 2: Quét 0.1–0.9, bước 0.05: batch 2 đạt F1 0.7537 tại 0.30, so với 0.7354 tại 0.5. Tối ưu trên holdout đã quan sát; gate giữ mặc định.
- [x] Bonus 3: CI tạo confusion matrix, precision/recall từng lớp trong `detail.txt`, upload cùng `report.json`. Nhận xét chi phí sai lầm ở mục 2.
- [ ] Bonus 4: Chưa tích hợp so sánh model hiện hành.
- [x] Bonus 5: Batch 2 có tỷ lệ dương 24.7842%, không cảnh báo; tests xác nhận chỉ cảnh báo khi lệch hơn 5 điểm phần trăm.
