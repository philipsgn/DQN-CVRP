# CẨM NANG THUYẾT TRÌNH & BÁO CÁO CHUYÊN KHẢO KỸ THUẬT TOÀN DIỆN
## DỰ ÁN: MP-MOTO-CVRP — ĐIỀU PHỐI ĐỘI XE MÁY GIAO HÀNG NỘI THÀNH TP.HCM BẰNG DEEP REINFORCEMENT LEARNING
**Đơn vị thực hiện:** Nhóm 10 — Trường Đại học FPT TP. Hồ Chí Minh (Khóa luận Tốt nghiệp / Nghiên cứu Kỹ thuật)  
**Nhóm tác giả:** Phan Tấn Phát, Hà Trọng Khiêm, Nguyễn Trí Anh  
**Phiên bản hệ thống:** V9.3 (Apple Minimalist Tab Design & Authentic OpenStreetMap Directed Geo-Data)

---

## MỤC LỤC TỔNG QUAN
- [PHẦN I: THÔNG TIN DỰ ÁN & ĐẶT VẤN ĐỀ LOGISTICS XE MÁY TP.HCM](#phần-i-thông-tin-dự-án--đặt-vấn-đề-logistics-xe-máy-tphcm)
- [PHẦN II: KỊCH BẢN THUYẾT TRÌNH & TRÌNH DIỄN THỰC NGHIỆM (TALK TRACK)](#phần-ii-kịch-bản-thuyết-trình--trình-diễn-thực-nghiệm-talk-track)
- [PHẦN III: PHÂN TÍCH CHUYÊN SÂU TỪNG MÔ HÌNH GIẢI THUẬT (MODEL DEEP-DIVE)](#phần-iii-phân-tích-chuyên-sâu-từng-mô-hình-giải-thuật-model-deep-dive)
- [PHẦN IV: BÓC TÁCH TOÁN HỌC VECTOR ĐẶC TRƯNG TRẠNG THÁI (F28 & F36)](#phần-iv-bóc-tách-toán-học-vector-đặc-trưng-trạng-thái-f28--f36)
- [PHẦN V: BÓC TÁCH TOÀN BỘ KHÔNG GIAN HÀNH ĐỘNG (ACTIONS & OPERATORS)](#phần-v-bóc-tách-toàn-bộ-không-gian-hành-động-actions--operators)
- [PHẦN VI: CHÍNH SÁCH (POLICY), BỘ LỌC VÀ CÁC QUY TẮC BẤT BIẾN (RULES & INVARIANTS)](#phần-vi-chính-sách-policy-bộ-lọc-và-các-quy-tắc-bất-biến-rules--invariants)
- [PHẦN VII: KIẾN TRÚC ĐA TÁC TỬ & ĐƯỜNG ĐI PHỐI HỢP CỦA CÁC AGENTS](#phần-vii-kiến-trúc-đa-tác-tử--đường-đi-phối-hợp-của-các-agents)
- [PHẦN VIII: TỔNG HỢP SỐ LIỆU ĐỊNH LƯỢNG THEO TỪNG PHASE & BỘ CÂU HỎI PHẢN BIỆN HỘI ĐỒNG (DEFENSE Q&A)](#phần-viii-tổng-hợp-số-liệu-định-lượng-theo-từng-phase--bộ-câu-hỏi-phản-biện-hội-đồng-defense-qa)

---

## PHẦN I: THÔNG TIN DỰ ÁN & ĐẶT VẤN ĐỀ LOGISTICS XE MÁY TP.HCM

### 1.1. Bối cảnh Logistics Chặng Cuối (Last-Mile Delivery) tại TP. Hồ Chí Minh
Tại TP. Hồ Chí Minh, giao hàng thương mại điện tử và bán lẻ chặng cuối phụ thuộc gần như tuyệt đối vào **đội xe máy** thay vì ô tô tải như các nước phương Tây. Xe máy phải đối mặt với các đặc thù khắt khe của hạ tầng giao thông đô thị:
1. **Hệ thống đường một chiều và hẻm nhánh chằng chịt:** Khoảng cách di chuyển từ điểm $A$ đến $B$ trên thực tế lệch rất lớn so với chiều ngược lại từ $B$ về $A$ ($d(A, B) \ne d(B, A)$). Điều này biến bài toán CVRP đối xứng lý thuyết thành **Asymmetric CVRP (ACVRP)** trên đồ thị có hướng.
2. **Ràng buộc tải trọng thùng xe máy ($Q$):** Thùng hàng xe máy tiêu chuẩn chỉ chở được tối đa từ $40\text{ kg}$ đến $80\text{ kg}$ (trung bình $50\text{ kg}$, tương đương 10 kiện hàng chuẩn $5\text{ kg}$). Việc phân bổ quá tải dù chỉ $1\text{ kg}$ sẽ vi phạm luật giao thông đường bộ và gây mất an toàn.
3. **Quy mô đội xe sẵn sàng ($K$):** Mỗi kho hàng chỉ sở hữu hoặc thuê một số lượng xe máy hữu hạn $K$. Giải thuật định tuyến tuyệt đối không được phép sinh số tuyến vượt quá $K$ ($K_{\text{used}} \le K$).

### 1.2. Đồ thị Đường bộ Xe máy TP.HCM (Authentic OpenStreetMap Foundation)
Hệ thống không sử dụng dữ liệu giả định hay khoảng cách Euclidean "đường chim bay", mà được xây dựng trên nền tảng dữ liệu không gian thực:
- **Đồ thị có hướng (Directed Graph):** Trích xuất từ OpenStreetMap (OSM) với bộ lọc dành riêng cho xe máy (`highway in ['primary', 'secondary', 'tertiary', 'residential', 'living_street']`).
- **Liên thông mạnh duy nhất (Strongly Connected Component - 1 SCC):** Đồ thị gồm **602 nút giao thông chính** và **11.408 cạnh có hướng**, được tích hợp các hành lang liên quận qua rào cản tự nhiên (kênh rạch, sân bay), bảo đảm $100\%$ mọi điểm đều có đường đi có hướng đến mọi điểm khác.
- **Cơ sở dữ liệu Địa danh Chuẩn xác:**
  - **20 Kho trung chuyển (`depots.json`):** Phân bổ tại Nội thành (Bến Thành, Đa Kao), Vành đai (Bến xe Miền Tây, Tân Bình), và Hub Cửa ngõ (Cát Lái, Hiệp Phước, Depot FPT High-Tech Park).
  - **570 Cửa hàng bán lẻ (`stores.json`):** Phân bổ đều trên 19 quận/huyện, mỗi điểm gắn tên đường thật từ cạnh đồ thị OSM (`edge['name']`) và tọa độ nằm chính xác bên trong ranh giới hành chính của quận đó.
  - **Ma trận Cự ly & Thời gian:** Kích thước $590 \times 590$ (`dist_m.npy`, `time_s.npy`) tính toán trước bằng thuật toán Dijkstra có hướng, cho phép tra cứu $O(1)$ và tái tạo tọa độ tim đường chính xác từng mét.

---

## PHẦN II: KỊCH BẢN THUYẾT TRÌNH & TRÌNH DIỄN THỰC NGHIỆM (TALK TRACK)

Kịch bản được thiết kế cho phần thuyết trình và demo trực tiếp kéo dài **10–12 phút** trước Hội đồng chấm Đồ án / Giảng viên / Chuyên gia phản biện.

```text
+---------------------------------------------------------------------------------------------------------+
|                                    LƯU TRÌNH DEMO TRỰC TIẾP TRÊN WEB UI                                 |
+---------------------------------------------------------------------------------------------------------+
| Cảnh 1: Khởi tạo Kho & Snap Tọa độ OSM -> Cảnh 2: Lọc Quận & Thẩm định Tải Feasibility Gate ->          |
| Cảnh 3: Phân bổ 3 Mô hình trên Segmented Control -> Cảnh 4: Đối chiếu Benchmark Head-to-Head ->         |
| Cảnh 5: Mô phỏng Di chuyển Bám tim đường OSM & Trình diễn Q-Values HUD.                                |
+---------------------------------------------------------------------------------------------------------+
```

---

### CẢNH 1: MỞ ĐẦU & KHỞI TẠO ĐIỂM XUẤT PHÁT (DEPOT SELECTION & OSM SNAP)
- **Mục tiêu trình diễn:** Giới thiệu ngôn ngữ thiết kế tối giản chuẩn Apple (Minimalist Tab Design), kho xuất phát và khả năng snap GPS tim đường OSM.
- **Thao tác trên UI:**
  1. Mở trình duyệt tại `http://127.0.0.1:8000`.
  2. Chỉ vào tiêu đề `MP-MOTO-CVRP` (được thiết kế thoáng đãng, không có icon che chữ, không có emoji thừa).
  3. Tại mục **Kho xuất phát (Depot)**, chọn `[CENTRAL] Kho Trung Tâm Bến Thành (Quận 1)`.
  4. Bấm nút **`+ Chọn trên bản đồ`**, nhấp chuột vào một vị trí bất kỳ gần Bến Vân Đồn (Quận 4). Quan sát popup hiển thị tọa độ thực và khoảng cách tự động snap vào tim đường OSM gần nhất ($\le 150\text{ m}$).
- **Lời thoại của người thuyết trình (Presenter Speech):**
  > *"Kính thưa Thầy Cô và Hội đồng, đây là hệ thống MP-MOTO-CVRP phiên bản V9.3. Giao diện được nhóm tái cấu trúc toàn diện theo triết lý thiết kế tối giản của Apple Human Interface Guidelines, loại bỏ hoàn toàn các yếu tố rườm rà.  
  > Điểm đặc biệt đầu tiên: hệ thống được xây dựng trên đồ thị đường bộ có hướng thực tế của OpenStreetMap tại TP. Hồ Chí Minh gồm 602 nút và hơn 11 nghìn cạnh. Khi chủ kho nhấp chọn một điểm bất kỳ trên bản đồ, mô-đun MatrixManager sẽ lập tức snap vị trí này vào tim đường gần nhất với sai số dưới 150 mét, đảm bảo xe máy luôn xuất phát từ một làn đường có hướng hợp lệ."*

---

### CẢNH 2: CẤU HÌNH CỬA HÀNG, ĐA QUẬN & CỔNG THẨM ĐỊNH TẢI TRỌNG (FEASIBILITY GATE)
- **Mục tiêu trình diễn:** Chứng minh tính toàn vẹn dữ liệu địa lý (tên đường thật theo quận), chế độ Đơn quận / Đa quận, và Cổng thẩm định khả thi 4 tầng (Feasibility Gate).
- **Thao tác trên UI:**
  1. Tại hàng chip quận, nhấp chọn `Quận 1`: Danh sách bên dưới lập tức chỉ hiển thị các cửa hàng nằm thực sự tại Quận 1 (Hai Bà Trưng, Lê Thánh Tôn,...).
  2. Bật công tắc `Đa quận: BẬT`: Chip `TẤT CẢ` tắt, chọn đồng thời 3 quận: `Bình Chánh`, `Bình Thạnh`, `Bình Tân`. Quan sát nút sampler hiển thị `+ 6 điểm / quận (18 điểm)` và danh sách chỉ nạp các điểm thuộc đúng 3 quận này.
  3. Chỉ vào hàng tổng: `Tổng: X kiện • Y.0 kg`. Bấm nút tăng (`+`) hoặc giảm (`−`) kiện hàng tại một cửa hàng, chứng minh phép tính kiện $\times 5\text{ kg}$ đồng bộ tuyệt đối với tổng tải trọng.
  4. **Thử nghiệm phá vỡ (Stress Test Feasibility Gate):** Tăng tải trọng lên vượt quá sức chứa tổng của đội xe (ví dụ: kéo slider $K = 1\text{ xe}$, $Q = 40\text{ kg}$ trong khi tổng tải là $120\text{ kg}$). Bấm nút **`Phân bổ`**.
  5. Hệ thống từ chối ngay lập tức và hiển thị thông báo lỗi rõ ràng từ Cổng thẩm định 4 tầng. Kéo lại $K = 7\text{ xe}$, $Q = 50\text{ kg}$ để trạng thái chuyển về `100% Feasible`.
- **Lời thoại của người thuyết trình (Presenter Speech):**
  > *"Thưa Thầy Cô, một lỗi phổ biến ở các hệ thống logistics là tình trạng 'râu ông nọ cắm cằm bà kia' - ví dụ địa chỉ đường Hai Bà Trưng nhưng lại bị gán tọa độ ở Bình Chánh. Trong MP-MOTO-CVRP, 570 cửa hàng đều được kiểm tra chéo không gian (Spatial Point-in-Polygon Check), đảm bảo 100% tên đường và tọa độ thuộc đúng ranh giới hành chính của quận đó.  
  > Đồng thời, hệ thống áp dụng nguyên tắc Bất biến thứ nhất: Tuyệt đối không sinh nghiệm bất khả thi. Cổng Feasibility Gate 4 tầng kết hợp giải thuật Bin Packing First-Fit-Decreasing sẽ chặn đứng mọi kịch bản thiếu xe hoặc quá tải trước khi thuật toán giải được phép kích hoạt."*

---

### CẢNH 3: BỘ 3 MÔ HÌNH GIẢI THUẬT & SINGLE SOURCE OF TRUTH GEOMETRY
- **Mục tiêu trình diễn:** Trực quan hóa thanh trượt Segmented Control chuyển đổi 3 mô hình, nguyên tắc Single Source of Truth Geometry (`legs[]`), và chỉ số HUD Telemetry.
- **Thao tác trên UI:**
  1. Trên thanh Segmented Control, chọn mô hình `Neural LNS`.
  2. Nhấp nút **`Phân bổ`** (nút màu xanh hệ thống `#007AFF`).
  3. Quan sát bản đồ: Bản đồ tự động fit bounds, vẽ các tuyến đường đa sắc màu bám sát từng khúc cua, tim đường của TP.HCM.
  4. Chỉ vào Capsule Telemetry ở góc trên bản đồ:
     - **Tổng cự ly:** ví dụ `16.34 km`.
     - **Tiết kiệm CW:** ví dụ `+1.63%`.
     - **Đội xe điều động:** `7 / 7 xe` (chứng minh $K_{\text{used}} \le K$, không bị lỗi phân mảnh $8/7$ xe).
     - **Makespan:** Thời gian hoàn tất của xe về kho muộn nhất.
     - **Badge:** `100% Feasible` xanh lá.
- **Lời thoại của người thuyết trình (Presenter Speech):**
  > *"Khi bấm Phân bổ, toàn bộ phép tính tối ưu diễn ra tại Backend. Kết quả trả về danh sách các chặng `legs[]` chứa danh sách nút giao OSM và chuỗi tọa độ chi tiết của tim đường có hướng. Frontend Leaflet chỉ việc render trực tiếp chuỗi tọa độ này. Nhóm cam kết nguyên tắc Single Source of Truth: Không có bất kỳ đoạn đường chim bay hay nội suy giả mạo nào trên bản đồ.  
  > Đặc biệt, nhờ cơ chế Route Consolidation, số xe điều động luôn tuân thủ nghiêm ngặt $K_{\text{used}} \le K$, giải quyết dứt điểm lỗi tràn xe của các giải thuật heuristic cổ điển."*

---

### CẢNH 4: ĐỐI CHIẾU HIỆU NĂNG HEAD-TO-HEAD (TAB `ĐỐI CHIẾU`)
- **Mục tiêu trình diễn:** Đối chiếu trực tiếp 3 mô hình trên cùng 1 kịch bản, chứng minh tính trung thực khoa học và sự vượt trội của Neural LNS.
- **Thao tác trên UI:**
  1. Nhấp sang Tab 3: **`Đối chiếu`**.
  2. Hệ thống lập tức kích hoạt chạy song song 3 mô hình trên đúng tập điểm xuất phát, cửa hàng và tải trọng vừa cấu hình.
  3. Phân tích bảng dữ liệu đối chiếu phẳng (Clean Minimalist Table):
     - `CW + Greedy`: Cự ly $16.57\text{ km}$ (+0.22%), 7 xe, độ trễ $16\text{ ms}$.
     - `DDQN Gốc`: Cự ly $16.61\text{ km}$ (+0.00%), 7 xe, độ trễ $131\text{ ms}$.
     - `Neural LNS (TP.HCM)`: Cự ly **$16.34\text{ km}$ (+1.63%)**, 7 xe, độ trễ $240\text{ ms}$.
- **Lời thoại của người thuyết trình (Presenter Speech):**
  > *"Kính thưa Hội đồng, đây là bảng đối chiếu Head-to-Head được chạy trực tiếp tại thời gian thực.  
  > Chúng ta thấy rất rõ một kết quả trung thực khoa học: Mô hình DDQN Gốc khi chỉ học chọn các toán tử cục bộ vi mô (như 2-opt, relocate) chỉ đạt kết quả ngang bằng Clarke-Wright ($16.61\text{ km}$ vs $16.57\text{ km}$) do dư địa cải thiện sau CW chỉ từ 0.5% đến 1.2% và không thể vượt qua bẫy cực trị địa phương.  
  > Để giải quyết bế tắc này, nhóm đã nghiên cứu và phát triển thành công mô hình thứ 3: Neural LNS. Bằng cách kết hợp Reinforcement Learning với 6 vĩ toán tử Phá hủy - Tái tạo (Destroy & Repair) và thuật toán luyện kim Simulated Annealing, Neural LNS đã rút ngắn cự ly xuống còn 16.34 km, tiết kiệm thêm 1.63% quãng đường di chuyển và đánh bại cả hai baseline!"*

---

### CẢNH 5: MÔ PHỎNG DI CHUYỂN BÁM TIM ĐƯỜNG & HUD Q-VALUES (TAB `LỘ TRÌNH`)
- **Mục tiêu trình diễn:** Trực quan hóa công cụ mô phỏng di chuyển với sai số tim đường $\le 3.2\text{ m}$ và thanh Q-values monoline.
- **Thao tác trên UI:**
  1. Nhấp sang Tab 2: **`Lộ trình`**.
  2. Bấm nút **`▶ Chạy xe`**: Các biểu tượng xe máy bắt đầu lăn bánh đồng loạt từ Kho Bến Thành, di chuyển dọc theo tim đường đến các cửa hàng để giao kiện và quay về kho.
  3. Bấm tăng tốc độ mô phỏng (`2x`, `4x`).
  4. Chỉ vào bảng danh sách xe: Hiển thị cự ly từng tuyến, số kiện giao, tải trọng thùng xe và thanh Q-values monoline 3px trực quan.
- **Lời thoại của người thuyết trình (Presenter Speech):**
  > *"Tại Tab Lộ trình, người điều hành kho có thể bấm Chạy xe để theo dõi diễn tiến giao hàng mô phỏng. Động cơ mô phỏng tính toán quãng đường tích lũy và nội suy vị trí tuyến tính bám sát tim đường OSM với độ lệch tối đa chỉ 3.2 mét.  
  > Đồng thời, các thanh Q-values hiển thị độ tin cậy của mô hình DRL đối với từng tuyến đường. Khi toàn bộ các xe hoàn tất và quay về kho an toàn, trạng thái giao hàng chuyển sang Hoàn thành.  
  > Sau đây, em xin phép chuyển sang phần phân tích chuyên sâu về kiến trúc toán học, vector đặc trưng và các thuật toán cốt lõi của hệ thống."*

---

## PHẦN III: PHÂN TÍCH CHUYÊN SÂU TỪNG MÔ HÌNH GIẢI THUẬT (MODEL DEEP-DIVE)

```text
+---------------------------------------------------------------------------------------------------------+
|                                    BỘ 3 MÔ HÌNH GIẢI THUẬT TRONG HỆ THỐNG                               |
+---------------------------------------------------------------------------------------------------------+
| 1. CW + Greedy: Clarke-Wright Savings kết hợp tìm kiếm cục bộ tham lam (Greedy Local Search).           |
| 2. DDQN Gốc: Mạng MLP Base Stage-1 huấn luyện chọn 4 toán tử cục bộ vi mô.                             |
| 3. Neural LNS: RL Destroy & Repair học chính sách điều phối phá hủy - tái tạo ma trận TP.HCM.          |
+---------------------------------------------------------------------------------------------------------+
```

---

### 3.1. Mô hình 1: Heuristic Baselines (`CW + Greedy`, `Cyclic`, `Random`)

#### A. Thuật toán Tiết kiệm Clarke-Wright (Clarke-Wright Savings Heuristic)
Clarke-Wright là thuật toán nền tảng dùng để sinh nghiệm ban đầu khả thi $R_{\text{init}}$. Ban đầu, giả định mỗi khách hàng $i$ được phục vụ bởi một xe riêng biệt theo lộ trình khép kín $(0 \to i \to 0)$, tổng chi phí ban đầu là $\sum 2 \cdot d(0, i)$.
Khoản tiết kiệm khi gộp hai khách hàng $i$ và $j$ vào cùng một tuyến $(0 \to i \to j \to 0)$ được tính bằng:

$$s_{ij} = d(0, i) + d(j, 0) - d(i, j)$$

Do ma trận khoảng cách TP.HCM là **bất đối xứng** ($d(i, j) \ne d(j, i)$), công thức tính savings có hướng bắt buộc xét tính thứ tự từ đuôi tuyến này nối vào đầu tuyến kia.

#### B. Tìm kiếm Cục bộ Tham lam (Greedy Local Search)
Tại mỗi bước $t$, Greedy duyệt qua tất cả 4 toán tử cục bộ (`2-opt`, `intra-relocate`, `inter-relocate`, `inter-swap`), sinh tối đa 128 ứng viên hợp lệ cho mỗi toán tử, tính toán mức giảm cự ly:

$$\Delta C = C_{\text{current}} - C_{\text{candidate}}$$

Toán tử nào cho $\Delta C^* = \max \Delta C > 0$ lớn nhất sẽ được chọn và áp dụng. Nếu cả 4 toán tử đều có $\Delta C \le 0$, thuật toán dừng lại do đã chạm cực trị địa phương.

- **Ưu điểm:** Tốc độ thực thi cực nhanh (chỉ mất $10–20\text{ ms}$ cho bài toán 20–30 điểm).
- **Nhược điểm:** Hoàn toàn bế tắc khi rơi vào cực tiểu cục bộ; không có cơ chế thoát hiểm.

---

### 3.2. Mô hình 2: DDQN Gốc (Stage-1 Pre-trained on Synthetic Distributions)

#### A. Kiến trúc Mạng Nơ-ron & Không gian Trạng thái
- **Cấu trúc mạng:** Multi-Layer Perceptron (MLP) gồm 3 tầng fully-connected:
  $$\text{Input}(28) \longrightarrow \text{FC}(128) + \text{ReLU} \longrightarrow \text{FC}(128) + \text{ReLU} \longrightarrow \text{Output}(4)$$
- **Nhiệm vụ của mạng:** Nhận vector trạng thái $s_t \in \mathbb{R}^{28}$ biểu diễn hình học và tải trọng của nghiệm hiện tại, dự đoán vector $Q(s_t, \cdot) = [Q_0, Q_1, Q_2, Q_3]$ tương ứng với giá trị kỳ vọng của 4 toán tử cục bộ vi mô.

#### B. Phân tích Nguyên nhân Hạn chế Học thuật (Anti-Slop Research Honesty)
Trong các bài báo quốc tế và thực nghiệm thực tế, DDQN chọn toán tử cục bộ sau Clarke-Wright **không thể vượt trội áp đảo** so với Greedy. Nhóm chứng minh 3 nguyên nhân khách quan:
1. **Dư địa cải thiện sau CW cực kỳ hẹp:** Clarke-Wright đã tối ưu hóa từ $85\% - 90\%$ bài toán. Dư địa còn lại cho Local Search chỉ dao động trong khoảng $0.56\% - 1.23\%$.
2. **Không gian hành động vi mô (Micro Actions) bị chia cắt:** Toán tử `2-opt` hay `relocate` chỉ có thể đảo hoặc dời $1–2$ khách hàng liền kề. Nếu 2 tuyến đường bị Clarke-Wright gom sai cụm lớn, các toán tử vi mô không thể di chuyển đồng thời một nhóm khách hàng mà không làm phát sinh chi phí tạm thời cao.
3. **Chi phí suy luận mạng nơ-ron:** Thời gian forward-pass qua PyTorch CPU tốn khoảng $0.27\text{ ms}$ mỗi bước so với $0.05\text{ ms}$ của phép duyệt Greedy, khiến tỷ lệ hiệu năng/thời gian của DDQN gốc không đạt kỳ vọng thương mại.

---

### 3.3. Mô hình 3: Neural LNS (RL Destroy & Repair Fine-tuned TP.HCM)

Để giải quyết triệt để sự bế tắc của DDQN Gốc, nhóm đã thiết kế kiến trúc **Neural Large Neighborhood Search (Neural LNS)**:

```mermaid
flowchart LR
    State["Vector Trạng Thái F36<br/>(36 Chiều: Tải, Không Gian, Nhiệt Độ, Lịch Sử)"] --> Agent["Neural Agent (Double DQN)<br/>MLP 36 -> 128 -> 128 -> 6"]
    Agent --> Action["Chọn 1 trong 6 Vĩ Toán Tử Phá Hủy - Tái Tạo<br/>(Macro Destroy & Repair Actions)"]
    Action --> Destroy["Pha Phá Hủy (Destroy):<br/>Rút q khách hàng (Worst-Cost / Shaw / Squeeze)"]
    Destroy --> Repair["Pha Tái Tạo (Repair):<br/>Tái chèn vào tuyến tối ưu (Greedy / Regret-2)"]
    Repair --> Anneal{"Kiểm Tra Chấp Nhận:<br/>Simulated Annealing P = exp(dC / Tt)"}
    Anneal -- "Chấp nhận" --> UpdateSol["Cập nhật Tuyến Mới & Làm Nguội Tt"]
    Anneal -- "Từ chối" --> RevertSol["Giữ Tuyến Cũ & Ghi Nhận Streak"]
```

#### A. Kiến trúc Mạng Nơ-ron & Không gian Trạng thái Mở rộng
- **Cấu trúc mạng:**
  $$\text{Input}(36) \longrightarrow \text{FC}(128) + \text{ReLU} \longrightarrow \text{FC}(128) + \text{ReLU} \longrightarrow \text{Output}(6)$$
- Tác tử không chọn toán tử cục bộ đổi chỗ 1 điểm, mà quyết định **chiến lược vĩ mô** phá hủy một cụm từ $20\% - 40\%$ số điểm giao và tái cấu trúc lại toàn bộ mạng lưới.

#### B. Cơ chế Chấp nhận Luyện kim (Simulated Annealing Acceptance)
Thay vì chỉ chấp nhận khi $\Delta C > 0$ một cách bảo thủ, Neural LNS tích hợp cơ chế làm nguội Simulated Annealing. Nếu nước đi làm tăng cự ly ($\Delta C = C_{\text{current}} - C_{\text{candidate}} < 0$), nghiệm candidate vẫn được chấp nhận với xác suất:

$$P(\text{accept}) = \exp\left( \frac{\Delta C}{T_t} \right)$$

Trong đó nhiệt độ $T_t$ giảm dần theo từng bước tìm kiếm:

$$T_t = T_0 \cdot \alpha^t, \quad T_0 = 0.05 \cdot C_{\text{init}}, \quad \alpha = 0.985$$

Cơ chế này cho phép thuật toán tạm thời suy thoái cự ly trong khoảng $1.5\% - 2.0\%$ ở các bước đầu để nhảy ra khỏi "hố sâu" cực tiểu địa phương, sau đó tái cấu trúc tuyến về trạng thái tối ưu toàn cục hơn.

#### C. Kết quả Vượt trội Định lượng
Trên tập dữ liệu 24 điểm giao hàng thực tế tại TP.HCM (tổng tải $315\text{ kg}$, $K=7\text{ xe}$, $Q=50\text{ kg}$):
- `CW + Greedy`: $16.57\text{ km}$ (+0.22% savings).
- `DDQN Gốc`: $16.61\text{ km}$ (+0.00% savings).
- `Neural LNS (TP.HCM)`: **$16.34\text{ km}$ (+1.63% savings)**, rút ngắn thêm $230\text{ mét}$ hành trình bám tim đường trong khi vẫn bảo đảm $100\%$ dung lượng tải và $K_{\text{used}} = 7/7\text{ xe}$.

---

## PHẦN IV: BÓC TÁCH TOÁN HỌC VECTOR ĐẶC TRƯNG TRẠNG THÁI (F28 & F36)

Một đóng góp kỹ thuật quan trọng của đồ án là việc ánh xạ cấu trúc đồ thị định tuyến phức tạp thành các vector số học có số chiều cố định, không phụ thuộc vào số lượng khách hàng $N$ thay đổi.

---

### 4.1. Bóc Tách Chi Tiết Vector Trạng Thái $F28$ (28 Chiều - Dành Cho DDQN Gốc)

Vector $s_t \in \mathbb{R}^{28}$ được chia thành 2 nhóm đặc trưng:

#### Nhóm 1: 12 Đặc Trưng Toàn Cục (Global Structural Features: $f_0 \dots f_{11}$)
1. **$f_0$ - Tỷ lệ tải trọng trung bình ($\mu_{\text{load}}$):**
   $$f_0 = \frac{1}{|\mathcal{R}|} \sum_{k=1}^{|\mathcal{R}|} \frac{\text{Load}(R_k)}{Q} \in [0, 1]$$
   Phản ánh mức độ lấp đầy thùng xe trung bình của toàn đội xe máy.
2. **$f_1$ - Độ lệch chuẩn tải trọng ($\sigma_{\text{load}}$):**
   $$f_1 = \sqrt{\frac{1}{|\mathcal{R}|} \sum_{k=1}^{|\mathcal{R}|} \left(\frac{\text{Load}(R_k)}{Q} - f_0\right)^2}$$
   Phản ánh sự mất cân bằng tải giữa các xe (giá trị cao báo hiệu có xe chở đầy, xe chở rỗng).
3. **$f_2$ - Tỷ lệ tải xe nặng nhất ($\max \text{Load} / Q$):** Đo lường nguy cơ vi phạm sức chứa khi điều chuyển khách.
4. **$f_3$ - Tỷ lệ tải xe nhẹ nhất ($\min \text{Load} / Q$):** Đo lường tiềm năng giải thể gom tuyến.
5. **$f_4$ - Tỷ lệ xe hoạt động chuẩn hóa ($|\mathcal{R}| / K$):** Tỷ lệ số xe đang được điều động so với tổng số xe sẵn sàng $K$.
6. **$f_5$ - Cự ly tuyến trung bình chuẩn hóa:** Cự ly trung bình mỗi xe chia cho bán kính phục vụ cực đại.
7. **$f_6$ - Độ lệch chuẩn cự ly giữa các tuyến:** Mức độ chênh lệch quãng đường giữa tài xế chạy xa nhất và gần nhất.
8. **$f_7$ - Tỷ lệ cự ly tuyến dài nhất so với toàn mạng:** Đo lường độ nghẽn makespan của đội xe.
9. **$f_8$ - Tiến độ ngân sách tìm kiếm ($t / T_{\max}$):**
   $$f_8 = \frac{t}{T_{\max}} \in [0, 1]$$
   Cho phép mạng nơ-ron biết mình đang ở giai đoạn đầu (khám phá) hay giai đoạn cuối (hội tụ).
10. **$f_9$ - Tỷ lệ cải thiện tích lũy:**
    $$f_9 = \frac{C_{\text{init}} - C_t}{C_{\text{init}}} \in [0, 1]$$
11. **$f_{10}$ - Tỷ lệ bước cải thiện thành công gần đây:** Tỷ lệ số bước có $\Delta C > 0$ trong 10 bước qua.
12. **$f_{11}$ - Hệ số mật độ điểm dừng:** Số khách hàng trung bình trên một kilomet di chuyển.

#### Nhóm 2: 16 Đặc Trưng Lịch Sử Toán Tử (Operator Dynamics Features: $f_{12} \dots f_{27}$)
Gồm 4 chỉ số thống kê $\times$ 4 toán tử cục bộ ($a \in \{0, 1, 2, 3\}$):
- **$f_{12}, f_{16}, f_{20}, f_{24}$ - Tần suất được chọn trong 10 bước gần nhất ($freq_a$):** Đo lường mức độ ưu tiên gần đây của toán tử $a$.
- **$f_{13}, f_{17}, f_{21}, f_{25}$ - Tỷ lệ thành công lịch sử ($succ_a$):** Số lần $\Delta C > 0$ chia cho tổng số lần toán tử $a$ được gọi.
- **$f_{14}, f_{18}, f_{22}, f_{26}$ - Mức cải thiện cự ly trung bình chuẩn hóa ($\overline{\Delta C}_a$):** Hiệu quả giảm cự ly trên mỗi bước thành công.
- **$f_{15}, f_{19}, f_{23}, f_{27}$ - Chuỗi không cải thiện chuẩn hóa ($streak_a / \text{patience}$):**
  $$f = \min\left(1.0, \; \frac{streak_a}{30}\right)$$
  Báo hiệu toán tử này đã cạn kiệt tiềm năng cải thiện trong lân cận hiện tại.

---

### 4.2. Bóc Tách Chi Tiết Vector Trạng Thái $F36$ (36 Chiều - Dành Cho Neural LNS)

Vector trạng thái $S_t \in \mathbb{R}^{36}$ trong Neural LNS nâng cấp toàn diện góc nhìn của tác tử về không gian bài toán:

| Khoảng chỉ số | Tên nhóm đặc trưng | Số chiều | Ý nghĩa vật lý & Công thức toán học |
| :--- | :--- | :---: | :--- |
| **$f_0 \dots f_7$** | Cấu trúc Tải trọng Tuyến | 8 | Tải trung bình, $\sigma_{\text{load}}$, $\min/\max$ tải, số tuyến có tải dưới $50\%$, hệ số biến thiên tải trọng $CV = \sigma / \mu$, tỷ lệ xe trống. |
| **$f_8 \dots f_{11}$** | Hình học & Bất đối xứng OSM | 4 | Bán kính bao quanh cụm khách hàng, khoảng cách trung bình đến Depot, **tỷ lệ bất đối xứng cự ly trung bình** $\frac{d(u, v)}{d(v, u)}$, mức độ phân cụm không gian. |
| **$f_{12} \dots f_{17}$** | Tiến độ & Trạng thái Annealing | 6 | Tiến độ ngân sách $t / T_{\max}$, tỷ lệ nhiệt độ hiện tại $T_t / T_0$, tỷ lệ chấp nhận nước đi thoái biến tích lũy, số bước liên tiếp không cải thiện kỷ lục tốt nhất ($streak_{\text{best}}$). |
| **$f_{18} \dots f_{35}$** | Động thái 6 Vĩ toán tử LNS | 18 | Gồm $3\text{ chỉ số} \times 6\text{ vĩ toán tử}$: Tần suất chọn gần đây (6), tỷ lệ thành công (6), biên độ delta cost chuẩn hóa (6). |

---

## PHẦN V: BÓC TÁCH TOÀN BỘ KHÔNG GIAN HÀNH ĐỘNG (ACTIONS & OPERATORS)

---

### 5.1. Bốn Toán Tử Cục Bộ Vi Mô (Micro Operators - DDQN Gốc)

```mermaid
flowchart TD
    subgraph MicroOps["4 Toán Tử Local Search Cục Bộ"]
        Op0["Action 0: 2-opt<br/>Đảo ngược đoạn tuyến<br/>(Bảo toàn tải trọng 100%)"]
        Op1["Action 1: Intra-relocate<br/>Dời điểm trong cùng 1 tuyến<br/>(Bảo toàn tải trọng 100%)"]
        Op2["Action 2: Inter-relocate<br/>Chuyển điểm từ Tuyến A sang B<br/>(Bắt buộc kiểm tra Load_B + q_i <= Q)"]
        Op3["Action 3: Inter-swap<br/>Đổi chỗ điểm giữa Tuyến A và B<br/>(Kiểm tra Load_A - qi + qj <= Q và Load_B - qj + qi <= Q)"]
    end
```

1. **`Action 0 (2-opt)`:** Chọn một tuyến $R_k$, cắt bỏ 2 cạnh không kề nhau và nối lại theo chiều đảo ngược:
   $$(v_1 \to v_2 \dots v_p \to v_{p+1}) \Longrightarrow (v_1 \to v_p \dots v_2 \to v_{p+1})$$
   *Ràng buộc tải trọng:* Nghiễm nhiên bảo toàn vì tập khách hàng trên tuyến không đổi.
2. **`Action 1 (Intra-relocate)`:** Rút một khách hàng $v_i$ ra khỏi tuyến $R_k$ và chèn vào vị trí $j$ trên cùng tuyến đó.
   *Ràng buộc tải trọng:* Nghiễm nhiên bảo toàn.
3. **`Action 2 (Inter-relocate)`:** Rút khách hàng $v_i$ từ tuyến $R_A$ và chèn vào tuyến $R_B$.
   *Ràng buộc tải trọng bắt buộc:*
   $$\text{Load}(R_B) + q_{v_i} \le Q$$
   Nếu điều kiện không thỏa mãn, ứng viên lập tức bị loại bỏ mà không cần tính toán chi phí.
4. **`Action 3 (Inter-swap)`:** Hoán đổi đồng thời khách hàng $v_i \in R_A$ và $v_j \in R_B$.
   *Ràng buộc tải trọng bắt buộc:*
   $$\begin{cases} \text{Load}(R_A) - q_{v_i} + q_{v_j} \le Q \\ \text{Load}(R_B) - q_{v_j} + q_{v_i} \le Q \end{cases}$$

---

### 5.2. Sáu Vĩ Toán Tử Phá Hủy & Tái Tạo (Macro Actions - Neural LNS)

Tại mỗi bước, tác tử Neural LNS phá hủy một tập con $\mathcal{D} \subset V_C$ gồm $q$ khách hàng ($q \in [2, \max(3, 0.35 \cdot N)]$) bằng 1 trong 3 cơ chế Destroy, sau đó tái chèn bằng 1 trong 2 cơ chế Repair:

#### A. Các Cơ Chế Phá Hủy (Destroy Mechanisms)
1. **Worst-Cost Removal:** Tính toán chi phí tiết kiệm khi rút khách hàng $i$ ra khỏi tuyến:
   $$\text{cost\_saving}(i) = d(\text{pred}(i), i) + d(i, \text{succ}(i)) - d(\text{pred}(i), \text{succ}(i))$$
   Khách hàng có chi phí đóng góp lớn nhất (làm tuyến bị gãy khúc, đi vòng xa) sẽ bị ưu tiên rút ra trước.
2. **Shaw Relatedness Removal:** Rút các khách hàng có mức độ tương đồng cao với một khách hàng hạt nhân $r$:
   $$\mathcal{R}(i, j) = \phi \cdot \frac{d(i, j)}{\max d} + \chi \cdot \frac{|q_i - q_j|}{\max q}$$
   Với trọng số $\phi = 0.8, \chi = 0.2$. Cơ chế này bóc tách toàn bộ một cụm địa lý đang bị phân bổ sai tuyến.
3. **Route-Squeeze Removal:** Chọn tuyến có tổng tải trọng nhỏ nhất trong mạng lưới và rút toàn bộ khách hàng trên tuyến đó vào tập $\mathcal{D}$, giải phóng hoàn toàn 1 xe.

#### B. Các Cơ Chế Tái Tạo (Repair Mechanisms)
1. **Greedy Best Insertion:** Lần lượt lấy từng khách hàng trong $\mathcal{D}$ và chèn vào vị trí có chi phí gia tăng $\Delta C$ nhỏ nhất trên toàn mạng mà vẫn thỏa mãn $\le Q$.
2. **Regret-2 Insertion:** Tính toán độ chênh lệch chi phí giữa vị trí chèn tốt nhất (thứ 1) và vị trí chèn tốt nhì (thứ 2):
   $$\Delta_{\text{regret}}(i) = \Delta C_i^{(2)} - \Delta C_i^{(1)}$$
   Khách hàng có $\Delta_{\text{regret}}$ lớn nhất (tức nếu không chèn ngay vào vị trí tốt nhất thì sẽ phải chịu thiệt hại rất nặng ở vị trí khác) sẽ được ưu tiên chèn trước.
3. **2-opt Polish:** Sau khi tái chèn bằng Regret-2, tự động chạy một lượt quét nhanh 2-opt trên các tuyến vừa được chỉnh sửa để nắn thẳng các cạnh bị chéo.

#### C. Bảng Phối Hợp 6 Vĩ Toán Tử (6 Macro Actions Suite)
| Action ID | Tên Vĩ Toán Tử | Cơ chế Phá hủy (Destroy) | Cơ chế Tái tạo (Repair) | Tinh chỉnh Cục bộ |
| :---: | :--- | :--- | :--- | :---: |
| **`0`** | `Worst-Greedy` | Worst-Cost Removal | Greedy Best Insertion | Không |
| **`1`** | `Worst-Regret2` | Worst-Cost Removal | Regret-2 Insertion | Không |
| **`2`** | `Shaw-Greedy` | Shaw Relatedness Removal | Greedy Best Insertion | Không |
| **`3`** | `Shaw-Regret2` | Shaw Relatedness Removal | Regret-2 Insertion | Không |
| **`4`** | `Route-Squeeze` | Route-Squeeze (Giải thể tuyến nhẹ) | Regret-2 Insertion | Không |
| **`5`** | `Worst-Regret2-Polish`| Worst-Cost Removal | Regret-2 Insertion | **2-opt Polish** |

---

## PHẦN VI: CHÍNH SÁCH (POLICY), BỘ LỌC VÀ CÁC QUY TẮC BẤT BIẾN (RULES & INVARIANTS)

---

### 6.1. Phương Trình Cập Nhật Double Deep Q-Learning
Thuật toán Q-Learning chuẩn thường gặp hiện tượng thiên kiến thổi phồng giá trị (Overestimation Bias) do sử dụng cùng một bộ tham số để vừa chọn vừa đánh giá hành động. Hệ thống MP-MOTO-CVRP áp dụng **Double DQN (DDQN)** với 2 mạng nơ-ron tách biệt: mạng chính $\theta$ (Policy Network) và mạng mục tiêu $\theta^-$ (Target Network):

$$Y_t^{\text{DoubleQ}} = r_t + \gamma \cdot Q\left(s_{t+1}, \; \arg\max_{a'} Q(s_{t+1}, a'; \theta); \; \theta^-\right)$$

Hàm mất mát Huber Loss được tối ưu hóa bằng Adam Optimizer:

$$\mathcal{L}(\theta) = \mathbb{E}\left[ \text{Huber}\left( Y_t^{\text{DoubleQ}} - Q(s_t, a_t; \theta) \right) \right]$$

### 6.2. Cơ Chế Lọc Hành Động Khả Thi (Action Masking)
Khi ở trạng thái $s_t$, môi trường xây dựng vector nhị phân $\mathbf{m}_t \in \{0, 1\}^A$.
- Nếu số tuyến đường $|\mathcal{R}| < 2$, các hành động liên tuyến (`inter-relocate`, `inter-swap`, `Route-Squeeze`) bị gắn $m_t[a] = 0$.
- Khi đưa qua mạng nơ-ron, mặt nạ được áp dụng theo công thức:
  $$\widetilde{Q}(s, a) = \begin{cases} Q(s, a) & \text{nếu } m_t[a] = 1 \\ -\infty & \text{nếu } m_t[a] = 0 \end{cases}$$
- Tác tử hành động theo chính sách suy luận: $a_t^* = \arg\max_a \widetilde{Q}(s, a)$, bảo đảm $100\%$ không bao giờ chọn hành động vô nghĩa.

### 6.3. Cơ Chế Khống Chế Cứng Đội Xe (Hard Fleet Cap: $K_{\text{used}} \le K$)
Khắc phục hiện tượng phân mảnh tải trọng của Clarke-Wright khiến sinh thừa xe ($8/7$ xe):
- **Thuật toán Route Consolidation:**
  1. Nếu $|\mathcal{R}| > K$, sắp xếp các tuyến theo tổng tải tăng dần: $\text{Load}(R_1) \le \text{Load}(R_2) \le \dots \le \text{Load}(R_m)$.
  2. Rút toàn bộ khách hàng khỏi tuyến nhẹ nhất $R_1$.
  3. Phân bổ các khách hàng này vào $m-1$ tuyến còn lại bằng phương pháp **Regret-2 Insertion**, chỉ chấp nhận các vị trí chèn có $\text{Load} \le Q$.
  4. Lặp lại cho đến khi $|\mathcal{R}| \le K$. Nếu không thể gom được nữa mà không vượt $Q$, Cổng Feasibility Gate sẽ báo lỗi trước bài toán.
- **Cam kết:** Nghiệm xuất xưởng luôn thỏa mãn $K_{\text{used}} \le K$.

### 6.4. Bốn Quy Tắc Bất Biến Cốt Lõi (`AGENTS.md`)
1. **BẢO TOÀN TẢI TRỌNG (Feasibility Invariant):** Mọi tuyến xe $r$ bắt buộc có $\sum_{i \in r} q_i \le Q$. Mô-đun `ConstraintValidatorAgent` thẩm định độc lập sau mỗi bước.
2. **PHỤC VỤ DUY NHẤT (Uniqueness Invariant):** Mỗi khách hàng $1 \dots n$ xuất hiện đúng một lần duy nhất trong toàn bộ mạng lưới tuyến.
3. **KHÉP KÍN DEPOT (Cycle Closure Invariant):** Mọi tuyến đường luôn bắt đầu và kết thúc tại Depot $0$ ($0 \to v_1 \dots v_p \to 0$).
4. **TRUY NGUYÊN & TRUNG THỰC KHOA HỌC (Scientific Honesty Invariant):** Đo đạc khách quan, báo cáo đúng số liệu thực tế, giải thích nguyên nhân khoa học của từng mô hình mà không ngụy tạo kết quả.

---

## PHẦN VII: KIẾN TRÚC ĐA TÁC TỬ & ĐƯỜNG ĐI PHỐI HỢP CỦA CÁC AGENTS

Hệ thống vận hành dựa trên sự phối hợp tuần tự nhịp nhàng của **6 Tác tử Chuyên trách (Specialized Agents)**:

```mermaid
sequenceDiagram
    autonumber
    actor Dispatcher as Người Điều Phối / Web Client
    participant Orch as 1. OrchestratorAgent
    participant Extractor as 2. FeatureExtractorAgent
    participant Policy as 3. PolicySelectorAgent
    participant Exec as 4. OperatorExecutorAgent
    participant Valid as 5. ConstraintValidatorAgent
    participant Vis as 6. GeospatialVisualizerAgent

    Dispatcher->>Orch: Yêu cầu giải bài toán (Depot, Stores, K, Q, SolverMode)
    Orch->>Orch: Khởi tạo nghiệm Clarke-Wright có kiểm tra tải trọng
    Orch->>Orch: Thực hiện Route Consolidation nếu K_used > K
    Orch->>Valid: Thẩm định nghiệm ban đầu
    Valid-->>Orch: PASS (100% Feasible, K_used <= K)
    
    loop Vòng lặp Tối ưu hóa (Neural LNS / DDQN / Greedy)
        Orch->>Extractor: Trích xuất trạng thái từ lộ trình hiện tại
        Extractor-->>Policy: Cung cấp vector F36 (hoặc F28) & Action Mask
        Policy->>Policy: Forward-pass mạng MLP, chọn Action tối ưu a_t
        Policy-->>Orch: Quyết định Action a_t & vector Q-values
        Orch->>Exec: Yêu cầu thực thi Action a_t trên lộ trình
        Exec->>Exec: Phá hủy (Destroy) & Tái tạo (Repair) nghiệm ứng viên
        Exec->>Exec: Đánh giá Simulated Annealing P = exp(dC / Tt)
        Exec-->>Orch: Nghiệm mới, delta_cost, accepted (True/False)
        alt Được chấp nhận (accepted == True)
            Orch->>Valid: Thẩm định độc lập nghiệm mới
            Valid-->>Orch: PASS (100% Feasible)
            Orch->>Orch: Cập nhật nghiệm kỷ lục (best_solution)
        else Bị từ chối
            Orch->>Orch: Giữ nguyên nghiệm hiện tại, tăng streak
        end
    end

    Orch->>Vis: Chuyển giao nghiệm tối ưu cuối cùng
    Vis->>Vis: MatrixManager tái tạo chuỗi tọa độ tim đường OSM legs[]
    Vis-->>Dispatcher: Trả về JSON hoàn chỉnh: routes, legs, telemetry, q_values
```

### Chi tiết Phân công Nhiệm vụ của 6 Tác tử:
1. **`OrchestratorAgent` (Tác tử Điều phối Hệ thống):** Quản lý vòng đời giải thuật, tiếp nhận request từ FastAPI, điều tiết ngân sách tìm kiếm (`budget_steps`), thực hiện ép gom tuyến nếu số xe vượt $K$, và tổng hợp kết quả cuối cùng.
2. **`FeatureExtractorAgent` (Tác tử Trích xuất Đặc trưng):** Đọc đồ thị tuyến đường và lịch sử tìm kiếm, tính toán ma trận tương quan và xuất ra vector số học chuẩn $F28$ hoặc $F36$, kiểm tra điều kiện sinh mặt nạ `action_mask`.
3. **`PolicySelectorAgent` (Tác tử Ra Quyết định Chính sách):** Nạp trọng số mạng nơ-ron PyTorch (`best_neural_lns.pt` hoặc `base_ddqn_model.pt`), thực hiện suy luận forward-pass có áp dụng Action Masking để chọn chiến lược tối ưu.
4. **`OperatorExecutorAgent` (Tác tử Thực thi Toán tử):** Sở hữu thư viện giải thuật biến đổi cấu trúc tuyến: tính toán savings, thực thi phá hủy Worst-Cost/Shaw, tái tạo Greedy/Regret-2, và kiểm tra điều kiện chấp nhận Simulated Annealing.
5. **`ConstraintValidatorAgent` (Tác tử Thẩm định Độc lập):** Đóng vai trò kiểm toán viên độc lập. Tính lại từ đầu tổng tải trọng của từng tuyến đường dựa trên nhu cầu kiện hàng gốc, xác nhận không có bất kỳ tuyến nào vượt $Q$ và mỗi khách hàng xuất hiện đúng 1 lần.
6. **`GeospatialVisualizerAgent` (Tác tử Trực quan hóa Địa lý):** Chịu trách nhiệm truy xuất đồ thị có hướng OSM, ráp nối chuỗi đỉnh `node_path` thành các đối tượng `legs[]` chứa danh sách tọa độ GPS chi tiết cho Leaflet.js vẽ polyline bám sát tim đường.

---

## PHẦN VIII: TỔNG HỢP SỐ LIỆU ĐỊNH LƯỢNG THEO TỪNG PHASE & BỘ CÂU HỎI PHẢN BIỆN HỘI ĐỒNG (DEFENSE Q&A)

---

### 8.1. Bảng Tổng Hợp Quá Trình Phát Triển Qua 6 Giai Đoạn (Project Phases)

| Giai đoạn (Phase) | Trọng tâm kỹ thuật | Kết quả đạt được & Chỉ số định lượng |
| :--- | :--- | :--- |
| **Phase 0: Graph & Data Foundation** | Xây dựng đồ thị đường xe máy TP.HCM từ OSM; tạo CSDL 20 kho, 570 cửa hàng. | Đồ thị **602 nút, 11.408 cạnh có hướng**, **1 SCC duy nhất**; $100\%$ tên đường và quận chuẩn xác; ma trận $590 \times 590$ Dijkstra tra cứu $O(1)$. |
| **Phase 1: Gym Environment & Micro Ops** | Xây dựng môi trường `MotorbikeCVRPEnv`, 4 toán tử cục bộ, vector $F28$. | $100\%$ tuân thủ Gymnasium API; kiểm tra tải trọng $\le Q$ nghiêm ngặt; tích hợp Action Masking. |
| **Phase 2: DRL Pre-training Pipeline** | Huấn luyện mạng DDQN Base Stage-1 trên phân phối dữ liệu chuẩn. | Huấn luyện 5.000 episodes; lưu mô hình `models/base_ddqn_model.pt`; chứng minh hiện tượng hội tụ sau Clarke-Wright. |
| **Phase 3: Web Engine & SSOT Geometry** | FastAPI backend; tái tạo polyline `legs[]`; động cơ mô phỏng xe. | Sai số GPS snap $\le 150\text{ m}$; sai số mô phỏng bám tim đường $\le 3.2\text{ m}$; triệt tiêu hoàn toàn đường chim bay. |
| **Phase 4: Apple Minimalist UI Design** | Tái cấu trúc giao diện theo Apple HIG; thanh 3 tab, bộ lọc đơn quận/đa quận. | Loại bỏ $100\%$ AI-slop và emoji; thanh Segmented Control chuyển 3 mô hình; đồng bộ 2 chiều kiện $\leftrightarrow$ kg. |
| **Phase 5: Neural LNS & Hard Fleet Cap** | Xây dựng môi trường LNS, vector $F36$, 6 vĩ toán tử, Simulated Annealing. | Đạt mức tiết kiệm **+1.63%**; đánh bại baseline; ép số xe điều động triệt để $K_{\text{used}} \le K$ (7/7 xe). |

---

### 8.2. Bảng Đối Chiếu Hiệu Năng Thực Nghiệm 3 Mô Hình
*(Đo đạc thực tế trên cùng một bài toán thực tế TP.HCM: 24 điểm giao hàng, tổng tải $315\text{ kg}$, $K=7\text{ xe}$, $Q=50\text{ kg}$, Kho Trung tâm Bến Thành)*:

| Tiêu chí đo đạc | 1. Heuristic (`CW + Greedy`) | 2. DRL Cơ bản (`DDQN Gốc`) | 3. Đột phá (`Neural LNS TP.HCM`) | Kết luận & Ý nghĩa |
| :--- | :---: | :---: | :---: | :--- |
| **Tổng cự ly di chuyển** | $16.57\text{ km}$ | $16.61\text{ km}$ | **$16.34\text{ km}$** | **Neural LNS tốt nhất** (rút ngắn $230\text{ m}$ tim đường). |
| **Tỷ lệ tiết kiệm so với CW** | $+0.22\%$ | $+0.00\%$ | **$+1.63\%$** | **Neural LNS vượt trội hoàn toàn** (+1.41% so với Greedy). |
| **Số xe điều động ($K_{\text{used}}$)** | $7\text{ xe}$ | $7\text{ xe}$ | **$7\text{ xe}$** | Tuân thủ tuyệt đối $K_{\text{used}} \le 7$, không bị lỗi $8/7$ xe. |
| **Tải trọng xe lớn nhất** | $50.0\text{ kg}$ / $50.0\text{ kg}$ | $50.0\text{ kg}$ / $50.0\text{ kg}$ | **$50.0\text{ kg}$ / $50.0\text{ kg}$** | $100\%$ Feasible, thùng xe tận dụng tối đa dung lượng. |
| **Độ trễ tính toán (Latency)** | **$16\text{ ms}$** | $131\text{ ms}$ | $240\text{ ms}$ | Tất cả đều $< 0.5\text{ giây}$, đáp ứng điều phối thời gian thực. |
| **Khả năng thoát cực tiểu địa phương** | Rất kém (Dừng ngay khi $\Delta C \le 0$) | Kém (Bị giới hạn bởi toán tử vi mô) | **Rất tốt** (Nhờ Simulated Annealing & 6 Vĩ toán tử) | Thoát bẫy thành công để tìm ra cấu trúc tuyến tốt hơn. |

---

### 8.3. Bộ 6 Câu Hỏi Phản Biện Học Thuật Thường Gặp Của Hội Đồng (Defense Q&A)

#### Câu hỏi 1: Tại sao nhóm không gọi trực tiếp Google Maps API hay OSRM public mà phải tốn công trích xuất đồ thị OSM 602 nút?
> **Trả lời:**  
> *"Dạ thưa Thầy Cô, có 3 lý do kỹ thuật then chốt:  
> Thứ nhất, trong quá trình huấn luyện Reinforcement Learning với hàng ngàn episodes, việc gọi API ngoài sẽ vi phạm giới hạn tốc độ (Rate Limit), phát sinh chi phí khổng lồ và độ trễ mạng làm gián đoạn quá trình học.  
> Thứ hai, mạng lưới hẻm và đường một chiều xe máy tại TP.HCM trên Google Maps thường được tối ưu cho ô tô, dẫn đến cự ly không chính xác cho xe máy.  
> Thứ ba, việc sở hữu trọn vẹn đồ thị có hướng 602 nút và ma trận $590 \times 590$ cục bộ cho phép hệ thống tra cứu khoảng cách với độ phức tạp $O(1)$ (dưới 1 micro giây), giúp thuật toán LNS đánh giá hàng trăm ứng viên trong chớp mắt mà không phụ thuộc vào Internet."*

#### Câu hỏi 2: Tại sao mô hình DDQN Gốc lại có kết quả gần như tương đương Clarke-Wright? Có phải do mạng nơ-ron chưa học được gì không?
> **Trả lời:**  
> *"Dạ thưa Thầy Cô, đây chính là phát hiện khoa học trung thực mà nhóm đã bảo vệ trong báo cáo: Mạng nơ-ron không hề bị lỗi mà đã học được cách chọn toán tử tốt nhất trong phạm vi cho phép. Tuy nhiên, thuật toán Clarke-Wright vốn đã là một heuristic rất mạnh, giải quyết được 85–90% bài toán. Dư địa còn lại cho các toán tử cục bộ vi mô (chỉ dời hoặc đảo 1–2 điểm) là cực kỳ nhỏ (chỉ 0.5% – 1.2%).  
> Khi lân cận cục bộ đã cạn kiệt, mọi nước đi vi mô đều làm tăng cự ly, khiến mạng DDQN gốc không thể tìm thấy nước đi cải thiện. Đây là hiện tượng 'bẫy cực tiểu địa phương' kinh điển của Local Search mà chính các công trình nghiên cứu quốc tế gần đây cũng đã chỉ ra."*

#### Câu hỏi 3: Mô hình Neural LNS đã làm thế nào để vượt qua bẫy cực trị địa phương đó?
> **Trả lời:**  
> *"Dạ thưa Thầy Cô, Neural LNS vượt qua bẫy nhờ 2 vũ khí cốt lõi:  
> 1. Thay đổi không gian hành động từ 'vi mô' sang 'vĩ mô': Thay vì chỉ dời 1 điểm, mô hình rút ra đồng thời một cụm từ 20% đến 40% số điểm (Destroy) rồi tái chèn thông minh bằng Regret-2 (Repair). Việc này giúp cấu trúc các tuyến được sắp xếp lại trên diện rộng.  
> 2. Tích hợp cơ chế Luyện kim Simulated Annealing: Cho phép thuật toán chấp nhận các nước đi suy thoái nhẹ ở giai đoạn đầu với xác suất $P = \exp(\Delta C / T_t)$. Nhờ đó, tác tử có thể tạm thời chịu lỗ một đoạn đường nhỏ để 'leo ra khỏi hố lũng' cực tiểu địa phương và rơi vào một vùng nghiệm mới tối ưu hơn."*

#### Câu hỏi 4: Tại sao số xe điều động trong các thuật toán VRP thường bị vượt quá $K$ (ví dụ $8/7$ xe) và nhóm khắc phục như thế nào?
> **Trả lời:**  
> *"Dạ thưa Thầy Cô, thuật toán Clarke-Wright gộp tuyến dựa trên tiêu chí cự ly tiết kiệm lớn nhất $s_{ij} > 0$ chứ không tối ưu hóa việc nhét đầy thùng xe. Khi các khoản savings dương cạn kiệt, các tuyến có thể bị phân mảnh tải trọng (ví dụ mỗi xe chỉ chở 25–40 kg trên sức chứa 50 kg), dẫn đến số tuyến sinh ra nhiều hơn số xe $K$ của chủ kho dù tổng tải vẫn đủ.  
> Nhóm đã khắc phục triệt để bằng thuật toán Route Consolidation: Nếu sau bước ban đầu số tuyến $> K$, hệ thống tự động bốc toàn bộ khách hàng trên tuyến nhẹ nhất và tái phân bổ vào các tuyến còn lại theo tiêu chí Regret-2 Insertion có kiểm tra nghiêm ngặt $\le Q$. Quy trình này ép số tuyến giảm xuống đúng $K$ mà không bao giờ vượt quá tải trọng xe."*

#### Câu hỏi 5: Làm thế nào nhóm đảm bảo 100% không có xe nào chở quá tải trọng $Q$?
> **Trả lời:**  
> *"Dạ thưa Thầy Cô, hệ thống bảo vệ tính khả thi bằng cơ chế phòng thủ 3 lớp (3-Layer Defense):  
> Lớp 1 (Giao diện): Cổng Feasibility Gate 4 tầng kiểm tra số điểm, kiện lẻ, tổng tải và giải thuật xếp thùng Bin-Packing FFD ngay khi người dùng chỉnh dữ liệu.  
> Lớp 2 (Toán tử): Mọi toán tử chuyển tuyến (`inter-relocate`, `inter-swap`, `Repair`) đều bắt buộc kiểm tra điều kiện $\text{Load} \le Q$ trước khi tính chi phí; nếu vượt $Q$, ứng viên bị hủy ngay lập tức.  
> Lớp 3 (Thẩm định độc lập): Tác tử `ConstraintValidatorAgent` đóng vai trò kiểm toán viên độc lập, tính lại toàn bộ tải trọng sau mỗi bước. Nếu có bất kỳ vi phạm nào, toàn bộ lượt giải sẽ bị hủy bỏ ngay lập tức."*

#### Câu hỏi 6: Giao diện Web hiển thị bản đồ có đảm bảo là đường đi thực tế không hay chỉ vẽ đường thẳng?
> **Trả lời:**  
> *"Dạ thưa Thầy Cô, nhóm cam kết tuân thủ tuyệt đối nguyên tắc Single Source of Truth Geometry:  
> Backend giải bài toán dựa trên ma trận Dijkstra có hướng, sau đó truy vết lại toàn bộ danh sách nút giao thông `node_path` trên đồ thị OSM. Tọa độ của từng khúc cua, tim đường được đóng gói thành mảng `legs[].geometry` gửi về Frontend.  
> Frontend Leaflet chỉ làm duy nhất một nhiệm vụ là vẽ mảng tọa độ này lên bản đồ. Tuyệt đối không có bất kỳ đường thẳng chim bay nào được tự ý vẽ ra."*

---

## PHẦN IX: HƯỚNG DẪN VẬN HÀNH & KHỞI CHẠY HỆ THỐNG

### 9.1. Khởi Chạy Web Server Demo
Tại thư mục gốc dự án, thực thi lệnh PowerShell:

```powershell
# Chạy Web Server FastAPI Production-Ready
C:\Python314\python.exe run_server.py
```
Khi màn hình hiển thị:
```text
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```
Mở trình duyệt Web tại: 👉 **`http://127.0.0.1:8000`**

### 9.2. Chạy Toàn Bộ Kiểm Thử Tự Động (Automated Test Suite)
Để xác nhận tính toàn vẹn của 10 mô-đun trước khi trình chiếu:

```powershell
C:\Python314\python.exe tests/run_all_tests.py
```
Xác nhận màn hình xuất hiện thông báo:
```text
==================================================
  ALL 10 TEST MODULES PASSED (100% SUCCESS RATE)  
==================================================
```

---
*Tài liệu được biên soạn phục vụ công tác báo cáo đồ án tốt nghiệp và chuyển giao công nghệ — Nhóm 10, ĐH FPT TP. Hồ Chí Minh.*
