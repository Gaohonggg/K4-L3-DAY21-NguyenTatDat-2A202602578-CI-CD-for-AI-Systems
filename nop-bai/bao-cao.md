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

**Lý do:** Run 3 đạt F1 cao nhất tại ngưỡng mặc định, vượt 0.65; run 1 đạt accuracy cao nhất. Recall tăng 0.6048→0.6371, precision giảm 0.8621→0.8144. Chênh lệch F1 nhỏ trên 500 mẫu chưa chứng minh ưu thế ổn định. Giảm learning rate thường cần thêm cây; ba cấu hình chưa cô lập ảnh hưởng từng tham số.

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Lớp thu nhập cao chiếm khoảng 24.8%; luôn đoán thu nhập thấp vẫn đạt accuracy 0.752 nhưng F1 lớp dương bằng 0. F1 kết hợp precision và recall. Dùng F1 nhị phân với target=1: weighted chịu trọng số lớp đông, macro trung bình hai lớp; cả hai khác chỉ số gate yêu cầu. Nếu tìm khách hàng tiềm năng, bỏ sót gây mất cơ hội nên ưu tiên recall; chi phí cần xác định theo ứng dụng.

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow cần `pkg_resources`. | Setuptools 84 không còn API này. | Đã hạ và cố định setuptools 80.9.0. |
| Không tạo được bucket S3. | IAM user thiếu quyền `s3:CreateBucket`. | Gắn policy giới hạn bucket lab; DVC đã push thành công. |

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | Chưa chạy CI | Chưa chạy CI |
| Bước 3 (thêm `train_batch2`) | Chưa thực hiện | Chưa thực hiện |

**Nhận xét:** Sẽ bổ sung kết quả thực tế sau hai lần chạy CI.

## 5. Phần Bonus Đã Thực Hiện

- [ ] Bonus 1: Chưa tích hợp DagsHub.
- [x] Bonus 2: Quét 0.1–0.9, bước 0.05: run 3 đạt F1 0.7368 tại 0.30, so với 0.7149 tại 0.5. Tối ưu trên holdout đã quan sát; gate giữ mặc định.
- [ ] Bonus 3: Đã tạo `detail.txt`; còn tích hợp và upload artifact trên CI.
- [ ] Bonus 4: Chưa tích hợp so sánh model hiện hành.
- [x] Bonus 5: Tỷ lệ dương 24.7708%, không cảnh báo; tests xác nhận chỉ cảnh báo khi lệch hơn 5 điểm phần trăm.
