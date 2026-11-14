# MP-MOTO-CVRP: Hệ Thống Huấn Luyện & Điều Phối Đội Xe Máy Giao Hàng Nội Thành TP.HCM Bằng Deep Reinforcement Learning (V9.3)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Leaflet](https://img.shields.io/badge/Leaflet-1.9.4-green.svg)](https://leafletjs.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Đồ án Nghiên cứu & Hệ thống Điều phối Vận hành Đội xe máy Giao hàng có Ràng buộc Sức chứa (CVRP) kết hợp Học tăng cường sâu (Deep Reinforcement Learning) trên Đồ thị Đường bộ OpenStreetMap TP. Hồ Chí Minh.**
> 
> **Đơn vị thực hiện:** Nhóm 10 — Trường Đại học FPT TP. Hồ Chí Minh (Tháng 11/2026)  
> **Thành viên:** Phan Tấn Phát, Hà Trọng Khiêm, Nguyễn Trí Anh

---

## 1. Giới thiệu Dự án

**MP-MOTO-CVRP (V9.3)** là giải pháp toàn diện cho bài toán định tuyến đội xe máy giao hàng chặng cuối (last-mile delivery) tại TP. Hồ Chí Minh. Khác với các nghiên cứu CVRP lý thuyết dùng khoảng cách Euclid, hệ thống vận hành trực tiếp trên **đồ thị đường bộ xe máy có hướng trích xuất từ OpenStreetMap**, tính toán chính xác đường một chiều, dải phân cách và ngõ hẻm.

### 3 Phân hệ Liên thông:
1. **Phân hệ Dữ liệu (Data Foundation):**
   - Đồ thị đường bộ xe máy TP.HCM gồm **602 nút giao và 11.408 cạnh có hướng**, được xử lý kết nối hành lang giao thông đảm bảo **100% nằm trong 1 thành phần liên thông mạnh (SCC) duy nhất**.
   - Cơ sở dữ liệu **20 kho trung chuyển (depots)** và **570 cửa hàng bán lẻ (stores)** phân bổ trên 19 quận/huyện TP.HCM, mỗi điểm gắn tên đường thực tế từ OSM và xác thực thuộc đúng Bounding Box của quận.
   - Bộ đệm ma trận cự ly (`dist_m.npy`) và thời gian (`time_s.npy`) kích thước $(590 \times 590)$ cho phép tra cứu $O(1)$.
2. **Phân hệ Huấn luyện Học tăng cường (RL Training Pipeline):**
   - Môi trường Gym `MotorbikeCVRPEnv` với vector 28 đặc trưng đồ thị ($F28$) và cơ chế **Action Masking** loại bỏ các hành động không khả thi.
   - Huấn luyện tác tử **Double DQN** điều khiển chọn 4 toán tử Local Search (`2-opt`, `intra-relocate`, `inter-relocate`, `inter-swap`).
   - Cung cấp 2 checkpoint: **DDQN Gốc (Base Stage-1)** và **DDQN Cải tiến (Finetuned Stage-2 với Curriculum Learning)**.
3. **Phân hệ Ứng dụng Điều phối Chủ Kho (Warehouse Dispatch App):**
   - Giao diện thiết kế theo chuẩn **Apple Human Interface Guidelines**: Tầng 1 Underline Tabs (`Điều phối`, `Lộ trình`, `Đối chiếu`), Tầng 2 Segmented Control chuyển đổi tức thì giữa 3 mô hình giải thuật.
   - Nguyên tắc **Single Source of Truth Geometry (`legs[]`)**: Backend tính toán polyline bám sát tim đường; frontend mô phỏng xe máy chuyển động theo thời gian thực với độ lệch $\le 3.2\text{ m}$.
   - Cổng thẩm định 4 tầng (**Feasibility Gate**) bảo đảm 100% không bao giờ sinh nghiệm vi phạm tải trọng.

---

## 2. Video Trình Diễn Hệ Thống (Demo)

> **Video ghi hình thực tế quá trình điều phối đội xe máy và trực quan hóa thời gian thực trên bản đồ TP. Hồ Chí Minh:**

https://github.com/philipsgn/DQN-CVRP/raw/main/docs/videos/MP-MOTO-CVRP%20_%20%C4%90i%E1%BB%81u%20Ph%E1%BB%91i%20%C4%90%E1%BB%99i%20Xe%20M%C3%A1y%20TP.HCM%20-%20Google%20Chrome%202026-11-13%2020-45-06.mp4

- **Tập tin video lưu trữ tại:** [`docs/videos/MP-MOTO-CVRP _ Điều Phối Đội Xe Máy TP.HCM - Google Chrome 2026-11-13 20-45-06.mp4`](docs/videos/MP-MOTO-CVRP%20_%20%C4%90i%E1%BB%81u%20Ph%E1%BB%91i%20%C4%90%E1%BB%99i%20Xe%20M%C3%A1y%20TP.HCM%20-%20Google%20Chrome%202026-11-13%2020-45-06.mp4)

---

## 3. 4 Bất Biến Cốt Lõi (Core Invariants)

Mọi dòng mã nguồn và tính năng của hệ thống bắt buộc phải thỏa mãn:
1. **Bảo toàn Tải trọng (Feasibility):** Mọi tuyến xe $R_k \in \mathcal{R}$ bắt buộc có $\sum_{i \in R_k} q_i \le Q$.
2. **Phục vụ Duy nhất (Uniqueness):** Mỗi cửa hàng $i \in V_C$ xuất hiện đúng một lần trong lộ trình: $\sum_{k=1}^{|\mathcal{R}|} \mathbf{1}[i \in R_k] = 1$.
3. **Khép kín Depot (Cycle Closure):** Tuyến xe luôn bắt đầu và kết thúc tại kho xuất phát ($0 \to \dots \to 0$).
4. **Trung thực Khoa học (Anti-Slop Research Honesty):** Báo cáo và giao diện phản ánh đúng kết quả đo đạc thực nghiệm; đối chiếu công bằng giữa DRL và các Heuristic truyền thống. Khống chế cứng quy mô đội xe $|\mathcal{R}| \le K$.

---

## 4. Bộ 3 Mô Hình Giải Thuật (3 Solvers Suite)

Người dùng có thể chuyển đổi linh hoạt giữa 3 mô hình trên thanh trượt Segmented Control:

| Mã mô hình | Tên hiển thị | Cơ chế hoạt động |
| :--- | :--- | :--- |
| `cw_greedy` | **CW + Greedy** | Khởi tạo bằng Clarke-Wright Savings, sau đó áp dụng tìm kiếm cục bộ tham lam (chọn toán tử cho $\Delta C > 0$ lớn nhất tức thời). |
| `ddqn_base` | **DDQN Gốc** | Mạng MLP Base Stage-1 ($28 \to 128 \to 128 \to 4$) huấn luyện chọn 4 toán tử cục bộ vi mô trên dữ liệu cơ sở. |
| `neural_lns` | **Neural LNS (TP.HCM)** | Đột phá: Mạng Double DQN ($36 \to 128 \to 128 \to 6$) điều phối 6 Vĩ toán tử Phá hủy - Tái tạo (Destroy & Repair) kết hợp Simulated Annealing, đạt mức tiết kiệm **+1.63%**, đánh bại Baseline. |

---

## 5. Cấu Trúc Mã Nguồn

```text
dqn-cvrp/
├── AGENTS.md                   # Quy tắc bất biến & kiến trúc đa tác tử runtime
├── README.md                   # Tài liệu tổng quan dự án
├── run_server.py               # Launcher khởi chạy server tối ưu (giới hạn scope watchfiles)
├── requirements.txt            # Danh sách thư viện phụ thuộc
├── docs/                       # Tài liệu kỹ thuật chi tiết
│   ├── PRD.md                  # Product Requirements Document V9.3
│   ├── TECH_ARCHITECTURE.md    # Kiến trúc kỹ thuật & mô hình toán học chi tiết
│   ├── DEMO_GUIDE.md           # Kịch bản thuyết trình, phân tích toán học 3 mô hình & Bộ câu hỏi phản biện
│   └── videos/                 # Video trình diễn hệ thống điều phối TP.HCM
├── data/                       # Dữ liệu địa lý & ma trận
│   ├── db/
│   │   ├── depots.json         # 20 kho trung chuyển TP.HCM (Inner, Outer, Hub)
│   │   └── stores.json         # 570 cửa hàng gắn tên đường thật từ OSM
│   ├── graph/
│   │   └── hcmc_motorbike.graphml  # Đồ thị OSM có hướng (602 nodes, 11.408 edges, 1 SCC)
│   └── matrices/
│       ├── dist_m.npy          # Ma trận khoảng cách Dijkstra (590 x 590)
│       └── time_s.npy          # Ma trận thời gian Dijkstra (590 x 590)
├── models/                     # Checkpoints mạng nơ-ron PyTorch
│   ├── base_ddqn_model.pt      # DDQN Base Stage-1
│   ├── finetuned_ddqn_model.pt # DDQN Finetuned Stage-2 (Curriculum + Action Masking)
│   └── best_neural_lns.pt      # Neural LNS RL Destroy-Repair Checkpoint (Phục vụ Production)
├── scripts/                    # Scripts tiền xử lý & xây dựng dữ liệu
│   └── build_dataset.py        # Trích xuất đồ thị OSM, tạo depots, stores và tính ma trận
├── src/
│   ├── cvrp/                   # Lõi bài toán CVRP, Clarke-Wright, 4 toán tử, State F28, MatrixManager
│   │   ├── matrix_manager.py   # Bộ nạp ma trận & tái tạo geometry legs[]
│   │   ├── motorbike_env.py    # Môi trường Gym với Action Masking
│   │   ├── operators.py        # 4 toán tử Local Search
│   │   └── clarke_wright.py    # Thuật toán khởi tạo Clarke-Wright Savings
│   ├── agents/                 # Mạng MLP & kịch bản huấn luyện
│   │   ├── dqn_agent.py        # Định nghĩa mạng Q-Network & Double DQN
│   │   └── train_motorbike_cvrp.py # Pipeline huấn luyện tăng cường
│   └── web/                    # Backend FastAPI & Frontend Leaflet.js
│       ├── server.py           # FastAPI server entry point
│       ├── dispatch_motorbike.py# Controller điều phối xe máy & snap tọa độ
│       └── static/             # Frontend giao diện Apple Minimalist UI
│           ├── index.html      # Giao diện 3 tab tối giản
│           ├── style.css       # Stylesheet Apple HIG (không tràn ngang, monoline bars)
│           ├── app.js          # Controller điều khiển tab, bản đồ & benchmark đối chiếu
│           └── vendor/         # Thư viện Leaflet.js và CSS offline (zero CDN)
└── tests/                      # Suite kiểm thử tự động toàn diện
    ├── run_all_tests.py        # Master test runner (chạy đồng bộ 9 modules)
    ├── test_geometric_invariants.py # Kiểm tra tính toàn vẹn hình học & SCC
    ├── test_simulation_fidelity.py  # Kiểm tra độ lệch mô phỏng xe máy (<= 5m)
    └── test_motorbike_cvrp.py  # Kiểm tra Feasibility Gate & API contract
```

---

## 6. Hướng Dẫn Cài Đặt & Khởi Chạy

### Bước 1: Cài đặt môi trường Python
Yêu cầu Python 3.10 trở lên:
```powershell
pip install -r requirements.txt
```

### Bước 2: Chạy bộ kiểm thử tự động (Master Test Suite)
Đảm bảo toàn bộ 9 module kiểm thử đều đạt trạng thái **PASS (100% SUCCESS RATE)**:
```powershell
python tests/run_all_tests.py
```

### Bước 3: Khởi chạy Web Server
Chạy máy chủ thông qua runner chuyên dụng để tự động giới hạn phạm vi giám sát tệp trong `src/web`:
```powershell
python run_server.py
```

Mở trình duyệt Web tại địa chỉ:
👉 **`http://127.0.0.1:8000`**

---

## 7. Trải Nghiệm Giao Diện Người Dùng (Minimalist Tab Workflow)

1. **Tab `Điều phối` (Dispatch Setup):**
   - Chọn kho xuất phát (hoặc bấm biểu tượng ghim vị trí để chọn trực tiếp trên bản đồ TP.HCM).
   - Chọn quận: Chế độ mặc định là chọn **1 quận duy nhất**; có thể bật công tắc **"Đa quận"** để chọn đồng thời nhiều quận.
   - Nhấp **"+ Mẫu 6 điểm"** để nạp nhanh các cửa hàng.
   - Tùy chỉnh số kiện hàng cho từng cửa hàng bằng bộ nút `+`/`−` (hệ thống tự động tính tổng kiện và tổng tải trọng kg).
   - Chọn mô hình trên thanh trượt: `CW + Greedy`, `DDQN Gốc` hoặc `DDQN Cải tiến`.
   - Bấm **`Giải & Phân bổ`**.
2. **Tab `Lộ trình` (Routes & Simulation):**
   - Hệ thống tự động chuyển sang tab này ngay sau khi giải xong.
   - Xem danh sách xe hoạt động và xe chờ lệnh (`Chờ lệnh`, độ mờ `45%`).
   - Theo dõi 4 thanh monoline 3px hiển thị giá trị Q-values dự đoán cho 4 toán tử.
   - Bấm **`▶ Chạy xe`** để xem mô phỏng xe máy di chuyển mượt mà bám sát từng khúc cua của tim đường.
3. **Tab `Đối chiếu` (Head-to-Head Benchmark):**
   - Chuyển sang tab này để hệ thống tự động đo đạc song song 3 mô hình trên cùng cấu hình bài toán hiện tại.
   - Xem bảng đối chiếu phẳng với cự ly tổng (km), mức độ tiết kiệm (%), số xe cần dùng và độ trễ tính toán CPU (ms).
