# BÁO CÁO DỮ LIỆU ĐỒ THỊ XE MÁY TP.HCM (DATA REPORT V9.2)

## 1. TỔNG QUAN ĐỒ THỊ ĐƯỜNG BỘ
- **Tệp GraphML:** `data/graph/hcm_motorbike.graphml`
- **Số nút mạng lưới:** 602 nút
- **Số cạnh có hướng:** 11408 cạnh
- **Thành phần liên thông:** 100% thuộc Thành phần liên thông mạnh lớn nhất (Largest SCC). Mọi cặp nút đều có đường đi có hướng đến được nhau.
- **Quy tắc cầu qua sông Sài Gòn:** 6 công trình cầu chính (Cầu Ba Son, Cầu Thủ Thiêm, Cầu Sài Gòn, Cầu Phú Mỹ, Cầu Kênh Tẻ, Cầu Bình Triệu) với phân làn xe máy và tốc độ thiết kế riêng.
- **Phạt quay đầu (U-turn penalty):** Áp dụng 200m tương đương tại các nút giao không có dải mở quay đầu.

## 2. CƠ SỞ DỮ LIỆU ĐIỂM GIAO NHẬN
- **Số kho hàng (Depots):** 20 kho phân bố trên 4 nhóm vùng (Central: 5 kho, Inner Ring: 5 kho, Peripheral: 5 kho, Riverside: 5 kho).
- **Số cửa hàng (Stores):** 570 cửa hàng bao phủ đầy đủ 18 quận/huyện TP.HCM.
- **Tỷ lệ nguồn dữ liệu:** 70% `osm_poi` thực tế, 30% `synthetic_osm_node` trên nút giao thông OSM.
- **Khoảng cách snap:** 100% điểm được snap vào nút SCC với cự ly snap trung bình 18.5 m (tất cả đều < 50 m, ngưỡng tối đa cho phép là 150 m).

## 3. PHÂN CHIA ĐỊA CHÍNH (DISTRICTS SPLIT)
- **Tập Train (Huấn luyện):** 12 quận/huyện (375 cửa hàng, 13 kho).
- **Tập Validation (Kiểm định):** 3 quận (75 cửa hàng, 3 kho).
- **Tập Test (Giữ riêng đánh giá):** 4 quận (120 cửa hàng, 4 kho).

## 4. MA TRẬN CHI PHÍ & TÍNH BẤT ĐỐI XỨNG
- **Kích thước ma trận:** 590 x 590 (float32).
- **Đường chéo:** Hoàn toàn bằng 0.0 m.
- **Giá trị vô cực (Inf) / Không xác định (NaN):** 0 (Không tồn tại).
- **Tỷ lệ cặp điểm bất đối xứng ($C_{ij} \neq C_{ji}$ chênh lệch > 10m):** **87.26%** (303,230 / 347,510 cặp).

## 5. KỊCH BẢN ĐÁNH GIÁ CỐ ĐỊNH
- **Validation Scenarios:** 30 kịch bản cố định trong `data/scenarios/val_scenarios.json`.
- **Holdout Test Scenarios:** 50 kịch bản cố định trong `data/scenarios/test_scenarios.json`.
