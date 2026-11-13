# TÀI LIỆU YÊU CẦU SẢN PHẨM (PRD)
## DỰ ÁN: MP-MOTO-CVRP (V9.3) — HỆ THỐNG HUẤN LUYỆN & ĐIỀU PHỐI ĐỘI XE MÁY GIAO HÀNG NỘI THÀNH TP.HCM BẰNG DEEP REINFORCEMENT LEARNING

---

| Thông tin dự án | Chi tiết |
| :--- | :--- |
| **Tên dự án** | MP-MOTO-CVRP (Motorcycle Fleet Capacity-Constrained Vehicle Routing with Deep Q-Learning) |
| **Phiên bản** | V9.3 (Apple Minimalist Tab Design & Authentic OpenStreetMap Geo-Data) |
| **Đơn vị thực hiện** | Nhóm 10 — Trường Đại học FPT TP. Hồ Chí Minh (Tháng 11/2026) |
| **Thành viên** | Phan Tấn Phát, Hà Trọng Khiêm, Nguyễn Trí Anh |
| **Trạng thái** | Hoàn thiện Production Ready (Mã nguồn & Báo cáo Thực nghiệm) |

---

## 1. TỔNG QUAN & MỤC TIÊU DỰ ÁN (EXECUTIVE SUMMARY)

### 1.1. Bối cảnh Logistics Xe Máy Đô Thị TP. Hồ Chí Minh
Tại TP. Hồ Chí Minh, giao hàng chặng cuối (last-mile delivery) chủ yếu được thực hiện bằng **đội xe máy**. Khác với xe tải hay ô tô tại các nước phương Tây, xe máy tại TP.HCM phải đối mặt với:
- Mạng lưới hẻm chằng chịt, đường một chiều, các dải phân cách cứng và cầu vượt chỉ dành riêng hoặc cấm xe máy.
- Giới hạn tải trọng đồng nhất nghiêm ngặt ($Q \in [40, 80]\text{ kg}$), số kiện hàng đóng thùng có kích thước hạn chế.
- Đội xe máy $K$ chiếc xuất phát từ các kho trung chuyển (Hub / Depot) tại các cửa ngõ hoặc trung tâm quận, phục vụ hàng chục điểm bán lẻ (Retail Stores / POIs) và quay về kho khép kín.

### 1.2. Mục tiêu Tổng Quát
Hệ thống **MP-MOTO-CVRP** được thiết kế thành một nền tảng liên thông khép kín gồm 3 phân hệ:
1. **Phân hệ Dữ liệu (Data Foundation):** Đồ thị đường bộ xe máy TP.HCM có hướng từ OpenStreetMap (OSM) với thành phần liên thông mạnh duy nhất (Largest Strongly Connected Component - SCC) gồm 602 nút giao và 11.408 cạnh có hướng; cơ sở dữ liệu 20 kho trung chuyển và 570 cửa hàng trải rộng trên 19 quận/huyện TP.HCM với tên đường thực tế; ma trận cự ly và thời gian di chuyển chuẩn xác.
2. **Phân hệ Huấn luyện Học tăng cường (RL Training Pipeline):** Huấn luyện tác tử Double Deep Q-Network (DDQN) với cơ chế lọc hành động bất khả thi (Action Masking) để chọn 1 trong 4 toán tử Local Search (`2-opt`, `intra-relocate`, `inter-relocate`, `inter-swap`) nhằm tối ưu hóa cự ly lộ trình.
3. **Phân hệ Ứng dụng Chủ Kho (Warehouse Dispatch App):** Giao diện Web tương tác đạt chuẩn **Apple Human Interface Guidelines (Minimalist Tab Design)**, cho phép chủ kho chọn kho, lọc cửa hàng theo 1 quận hoặc đa quận, tùy biến số kiện hàng, số xe $K$, tải trọng $Q$, giải và theo dõi lộ trình theo nguyên tắc Single Source of Truth Geometry (`legs[]`).

---

## 2. NGUYÊN TẮC BẤT BIẾN CỐT LÕI (CORE INVARIANTS)

Bất kỳ thành phần nào của hệ thống đều phải tuân thủ tuyệt đối 4 bất biến sau:

```text
+-----------------------------------------------------------------------------------------+
|                                4 BẤT BIẾN CỦA DỰ ÁN                                     |
+-----------------------------------------------------------------------------------------+
| 1. BẢO TOÀN TẢI TRỌNG (Feasibility): Mọi tuyến r phải có sum(demands_kg) <= Q.          |
| 2. PHỤC VỤ DUY NHẤT (Uniqueness): Mỗi khách hàng 1..n xuất hiện đúng một lần.          |
| 3. KHÉP KÍN DEPOT (Cycle Closure): Tuyến đường luôn bắt đầu và kết thúc tại 0.         |
| 4. TRUY NGUYÊN & TRUNG THỰC (Scientific Honesty): Dữ liệu thực, đo đạc khách quan.     |
+-----------------------------------------------------------------------------------------+
```

### Chi tiết các cam kết:
- **Nguyên tắc Single Source of Truth:** Backend chịu trách nhiệm 100% tính toán lộ trình trên đồ thị có hướng OSM, tái tạo chuỗi nút `node_path` và tọa độ chi tiết `geometry` cho từng chặng `legs[]`. Frontend tuyệt đối không tự tính đường chim bay hay suy đoán tọa độ.
- **Không sinh nghiệm vi phạm tải trọng:** Bộ lọc Feasibility Gate 4 tầng (Kiểm tra số điểm, kiểm tra tải trọng kiện lẻ, kiểm tra tổng tải đội xe, kiểm tra thuật toán xếp thùng First-Fit-Decreasing) chặn đứng mọi kịch bản không khả thi ngay trước khi giải.
- **Tính trung thực khoa học:** Phản ánh chính xác kết quả đo đạc: các mô hình DRL (`DDQN Base`, `DDQN Finetuned`) được đối chiếu công bằng với baseline Heuristic (`CW + Greedy`) trên cùng tập dữ liệu và thời gian thực thi.

---

## 3. LỘ TRÌNH TRIỂN KHAI THEO TỪNG GIAI ĐOẠN (PROJECT PHASES)

### Phase 0: Phân Hệ Nền Tảng Dữ Liệu Địa Lý (Data Foundation)
- Trích xuất đồ thị đường bộ xe máy TP.HCM từ OpenStreetMap (`scripts/build_dataset.py`).
- Bổ sung các hành lang kết nối liên quận (Arterial Corridor Connectors) qua các ranh giới tự nhiên (kênh rạch, sân bay Tân Sơn Nhất), đảm bảo **100% đồ thị nằm trong 1 SCC duy nhất (602 nodes, 11.408 edges)**.
- Xây dựng cơ sở dữ liệu `depots.json` (20 kho bãi phân bố tại Inner City, Outer Ring, Logistics Hub).
- Xây dựng cơ sở dữ liệu `stores.json` (570 cửa hàng phân bổ đều trên 19 quận/huyện TP.HCM, mỗi điểm gắn chính xác tên đường thực tế từ OSM và tọa độ nằm trọn trong Bounding Box của quận đó).
- Tính toán trước ma trận khoảng cách (`dist_m.npy`) và thời gian (`time_s.npy`) cho 590 điểm (20 kho + 570 cửa hàng) bằng thuật toán Dijkstra đa nguồn có hướng.

### Phase 1: Môi Trường CVRP & Pipeline Huấn Luyện DRL
- Xây dựng môi trường Gym `MotorbikeCVRPEnv` (`src/cvrp/motorbike_env.py`):
  - Khởi tạo nghiệm bằng Clarke-Wright Savings có kiểm tra ràng buộc dung lượng tải.
  - 4 toán tử Local Search: `0: 2-opt`, `1: intra-relocate`, `2: inter-relocate`, `3: inter-swap`.
  - Bộ trích xuất 28 đặc trưng trạng thái đồ thị ($F28$).
  - Bộ mặt nạ hành động khả thi (Action Masking) để ngăn tác tử chọn các toán tử không thể cải thiện.
- Huấn luyện 2 mô hình mạng nơ-ron DDQN (`src/agents/train_motorbike_cvrp.py`):
  - **DDQN Gốc (Base Stage-1):** Mạng MLP huấn luyện trên phân phối dữ liệu tổng hợp cơ sở (`models/base_ddqn_model.pt`).
  - **DDQN Cải tiến (Finetuned Stage-2):** Huấn luyện nâng cao với Curriculum Learning trên các kịch bản thực tế TP.HCM kết hợp Action Masking (`models/finetuned_ddqn_model.pt`).

### Phase 2: Phân Hệ Dịch Vụ Điều Phối (FastAPI Backend Service)
- Xây dựng bộ điều phối `src/web/dispatch_motorbike.py`:
  - Quản lý bộ đệm ma trận `MatrixManager` cho phép truy xuất cự ly $O(1)$ và tái tạo tọa độ polyline `reconstruct_leg_geometry`.
  - Hỗ trợ 3 chế độ giải thuật qua tham số `solver_mode`: `cw_greedy`, `ddqn_base`, `ddqn_finetuned`.
  - Tích hợp endpoint snap tọa độ người dùng vào tim đường OSM trong phạm vi $\le 150\text{ m}$.
  - Cung cấp kết quả điều phối chi tiết: danh sách xe, điểm dừng, tải trọng, cự ly, thời gian ước tính (ETA), chuỗi tọa độ `legs[]`, giá trị Q-values dự đoán và giải trình phân bổ.

### Phase 3: Giao Diện Người Dùng Tối Giản Chuẩn Apple (Minimalist Web App)
- Tái cấu trúc giao diện theo **Apple Human Interface Guidelines**:
  - **Tầng 1 (Underline Tabs):** 3 tab chính với nhãn 1–2 từ: **`Điều phối`**, **`Lộ trình`**, **`Đối chiếu`**. Tab kích hoạt có gạch chân xanh hệ thống `#007AFF` 2px sắc nét.
  - **Tầng 2 (Segmented Control):** Thanh trượt pill bo góc trên nền xám `#F2F2F7` cho phép chuyển đổi tức thì giữa 3 mô hình (`CW + Greedy`, `DDQN Gốc`, `DDQN Cải tiến`).
  - **Loại bỏ hoàn toàn AI-Slop:** Không sử dụng emoji trang trí hay icon nhiều màu; toàn bộ iconography sử dụng vector SVG đơn sắc (`Monoline Monochrome`, `stroke="currentColor"`).
  - **Bảng đối chiếu 3 mô hình:** Tự động chạy đo đạc song song 3 mô hình khi người dùng chuyển sang tab `Đối chiếu`, hiển thị bảng phẳng không viền dọc với các chỉ số: cự ly, % tiết kiệm, số xe, độ trễ xử lý (ms).

---

## 4. ĐẶC TẢ YÊU CẦU CHỨC NĂNG (FUNCTIONAL REQUIREMENTS)

### 4.1. Quản lý Điểm Xuất Phát (Depot Management)
- **FR-01:** Cho phép chọn kho xuất phát từ danh sách 20 kho chuẩn hóa theo vùng (Inner, Outer, Hub).
- **FR-02:** Cho phép người dùng nhấp trực tiếp trên bản đồ TP.HCM để đặt kho tùy ý; hệ thống tự động snap vào nút giao thông OSM gần nhất ($\le 150\text{ m}$) và hiển thị khoảng cách snap thực tế.

### 4.2. Quản lý Cửa Hàng & Kiện Hàng (Store & Cargo Management)
- **FR-03 (Bộ lọc Quận):**
  - Chế độ **Chỉ chọn 1 quận (Strict Single-District)**: Nhấp quận nào thì danh sách chuyển sang quận đó, tự động chọn mẫu các điểm trong quận.
  - Chế độ **Gộp nhiều quận (Multi-District)**: Bật công tắc "Đa quận" để chọn đồng thời nhiều quận (ví dụ: Quận 1 + Bình Thạnh + Quận 3).
- **FR-04 (Điều chỉnh Kiện hàng & Tải trọng):**
  - Mỗi cửa hàng có bộ nút tăng/giảm (`+`/`−`) số kiện hàng.
  - Công thức tính reactive thời gian thực: $\text{Tổng kiện} = \sum n_i$, $\text{Tổng tải (kg)} = \sum (n_i \times w_{pkg})$ với $w_{pkg} = 5.0\text{ kg}$.
  - Chặn không cho tăng số kiện nếu một cửa hàng vượt quá tải trọng xe $Q$.

### 4.3. Cấu hình Đội Xe & Cổng Kiểm Soát Khả Thi (Fleet & Feasibility Gate)
- **FR-05:** Thanh trượt số lượng xe máy sẵn sàng $K \in [1, 15]$.
- **FR-06:** Segmented buttons chọn tải trọng đồng nhất $Q \in \{40, 50, 60, 80\}\text{ kg}$.
- **FR-07 (Multi-Tier Feasibility Gate):** Tự động kiểm tra và khóa nút điều phối nếu:
  1. Số cửa hàng được chọn $< 2$.
  2. Bất kỳ điểm nào có $\text{demand} > Q$.
  3. Tổng tải $\sum d_i > K \times Q$.
  4. Thuật toán First-Fit-Decreasing cần số xe vượt quá $K$.

### 4.4. Động Cơ Giải & Tái Tạo Lộ Trình (Solver & Routing)
- **FR-08:** Hỗ trợ 3 mô hình giải thuật:
  - `cw_greedy`: Clarke-Wright + Greedy Local Search.
  - `ddqn_base`: Base Stage-1 Double DQN.
  - `ddqn_finetuned`: Stage-2 Finetuned Double DQN.
- **FR-09:** Trả về polyline chuẩn xác theo tim đường từ mảng `legs[]` có hướng.

### 4.5. Trực Quan Hóa & Mô Phỏng Lộ Trình (Visualization & Simulation)
- **FR-10:** Tự động chuyển sang tab `Lộ trình` sau khi điều phối thành công.
- **FR-11:** Hiển thị danh sách xe hoạt động và xe chờ lệnh (`Chờ lệnh`, mờ `45%`), kèm biểu đồ thanh ngang tải trọng.
- **FR-12:** Hiển thị 4 thanh monoline 3px dự đoán Q-values cho 4 toán tử.
- **FR-13 (Mô phỏng Xe máy):** Marker xe máy di chuyển mượt mà bám sát tim đường theo thời gian thực tế với các nút điều khiển tốc độ ($1\times, 2\times, 5\times$).

### 4.6. Đối Chiếu Hiệu Năng 3 Mô Hình (Head-to-Head Benchmarking)
- **FR-14:** Tại tab `Đối chiếu`, tự động gọi đồng thời 3 mô hình trên cùng cấu hình bài toán và xuất bảng so sánh chi tiết: Cự ly (km), Tiết kiệm (%), Số xe dùng, Độ trễ tính toán (ms), đánh dấu mô hình chiến thắng.

---

## 5. ĐẶC TẢ GIAO DIỆN LẬP TRÌNH ỨNG DỤNG (API SPECIFICATION)

| Phương thức | Đường dẫn API | Tham số đầu vào | Mô tả chức năng |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/motorbike/depots` | Không | Lấy danh sách 20 kho trung chuyển TP.HCM |
| `GET` | `/api/motorbike/stores` | `district` (tùy chọn, hỗ trợ danh sách phân tách bằng dấu phẩy) | Lấy danh sách 570 cửa hàng theo quận/huyện |
| `GET` | `/api/motorbike/meta` | Không | Lấy metadata đồ thị (số nút, số cạnh, tải trọng hỗ trợ) |
| `POST` | `/api/motorbike/snap` | `{"lat": float, "lon": float}` | Snap tọa độ vào nút mạng lưới xe máy ($\le 150\text{ m}$) |
| `POST` | `/api/dispatch-motorbike` | `DispatchRequest` (depot, stores, fleet, options) | Giải bài toán CVRP và trả về lộ trình chi tiết |

---

## 6. TIÊU CHÍ NGHIỆM THU HỆ THỐNG (ACCEPTANCE CRITERIA)

| Mã AC | Tiêu chí nghiệm thu | Phương pháp kiểm tra | Kết quả đạt được |
| :--- | :--- | :--- | :--- |
| **AC-01** | Đồ thị giao thông xe máy là 1 SCC duy nhất, không có nút cô lập | `test_geometric_invariants.py` | 602 nodes, 11.408 edges kết nối 100% |
| **AC-02** | Dữ liệu 570 cửa hàng gắn đúng tên đường và quận thực tế | Bounding box spatial check | 100% cửa hàng thuộc đúng quận khai báo |
| **AC-03** | Feasibility Gate 4 tầng chặn đứng mọi trường hợp vi phạm | `test_motorbike_cvrp.py` | 100% trường hợp quá tải bị từ chối với mã 400 |
| **AC-04** | 3 mô hình giải thuật hoạt động độc lập và ổn định | Test suite API dispatch | `cw_greedy`, `ddqn_base`, `ddqn_finetuned` pass 100% |
| **AC-05** | Giao diện điều hướng tối giản chuẩn Apple, không có AI-slop | Kiểm tra giao diện trực quan | 3 tab underline, monoline icons, zero emoji |
| **AC-06** | Mô phỏng xe máy bám sát polyline có sai số $\le 5\text{ m}$ | `test_simulation_fidelity.py` | Độ lệch tối đa $\le 3.2\text{ m}$ |
| **AC-07** | Bộ kiểm thử tự động toàn diện pass 100% | `python tests/run_all_tests.py` | 9/9 test modules PASSED (100%) |
