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
Clarke-Wright là thuật toán nền tảng dùng để sinh nghiệm ban đầu khả thi $\mathcal{R}_{\text{init}}$. Ban đầu, giả định mỗi khách hàng $i \in V_C$ được phục vụ bởi một xe máy riêng biệt theo lộ trình khép kín $(0 \to i \to 0)$, tổng chi phí xuất phát là:

$$\mathcal{C}_{\text{init}} = \sum_{i=1}^n \Big( d(0, i) + d(i, 0) \Big)$$

Khoản tiết kiệm (Savings) khi hợp nhất hai khách hàng $i$ và $j$ từ hai tuyến riêng lẻ vào cùng một tuyến có hướng $(\dots \to i \to j \to \dots)$ được định nghĩa chuẩn xác theo ma trận cự ly có hướng OpenStreetMap:

$$s_{ij} = d(i, 0) + d(0, j) - d(i, j)$$

Do đồ thị giao thông thực tế TP.HCM mang tính chất **bất đối xứng** ($d(i, j) \ne d(j, i)$ do đường một chiều, dải phân cách và ngõ hẻm), công thức tính savings có hướng bắt buộc xét rõ tính thứ tự di chuyển:
- Nối đuôi tuyến $R_A = (0 \to \dots \to i \to 0)$ vào đầu tuyến $R_B = (0 \to j \to \dots \to 0)$ thành tuyến mới $R_{AB} = (0 \to \dots \to i \to j \to \dots \to 0)$.
- **Điều kiện ràng buộc dung lượng tải bắt buộc:**

$$\text{Load}(R_A) + \text{Load}(R_B) \le Q \iff \sum_{u \in R_A} q_u + \sum_{v \in R_B} q_v \le Q$$

#### B. Tìm kiếm Cục bộ Tham lam (Greedy Local Search)
Tại mỗi bước lặp $t$, giải thuật Greedy tiến hành duyệt vét cạn trên tập 4 toán tử cục bộ vi mô $\mathcal{A}_{\text{micro}} = \{0: \text{2-opt}, \; 1: \text{intra-relocate}, \; 2: \text{inter-relocate}, \; 3: \text{inter-swap}\}$, sinh tối đa $|\mathcal{M}_a| \le 128$ ứng viên khả thi cho mỗi toán tử, và đánh giá mức độ giảm cự ly:

$$\Delta \mathcal{C}(a, m) = \mathcal{C}(\mathcal{R}_{\text{current}}) - \mathcal{C}\big(\mathcal{R}_{\text{candidate}}(a, m)\big)$$

Nước đi có độ cải thiện cực đại $\Delta \mathcal{C}^*$ được xác định theo công thức:

$$(a^*, m^*) = \arg\max_{a \in \mathcal{A}_{\text{micro}}, \, m \in \mathcal{M}_a} \Delta \mathcal{C}(a, m)$$

Quy tắc cập nhật trạng thái nghiệm tuân thủ nguyên tắc cải thiện nghiêm ngặt:

$$\mathcal{R}_{t+1} = \begin{cases} \mathcal{R}_{\text{candidate}}(a^*, m^*) & \text{khi } \Delta \mathcal{C}(a^*, m^*) > 0 \\ \mathcal{R}_t & \text{khi } \Delta \mathcal{C}(a^*, m^*) \le 0 \end{cases}$$

- **Ưu điểm:** Tốc độ thực thi cực nhanh (chỉ mất $10–20\text{ ms}$ cho bài toán $20–30$ điểm).
- **Nhược điểm:** Hoàn toàn bế tắc khi rơi vào cực tiểu cục bộ; không có cơ chế thoát hiểm.

---

### 3.2. Mô hình 2: DDQN Gốc (Stage-1 Pre-trained on Synthetic Distributions)

#### A. Kiến trúc Mạng Nơ-ron & Không gian Trạng thái
- **Cấu trúc mạng:** Multi-Layer Perceptron (MLP) gồm 3 tầng liên kết đầy đủ (Fully-Connected):
  $$\mathbb{R}^{28} \xrightarrow{\mathbf{W}_1 \in \mathbb{R}^{128 \times 28}, \, \mathbf{b}_1} \text{FC}(128) + \text{ReLU} \xrightarrow{\mathbf{W}_2 \in \mathbb{R}^{128 \times 128}, \, \mathbf{b}_2} \text{FC}(128) + \text{ReLU} \xrightarrow{\mathbf{W}_3 \in \mathbb{R}^{4 \times 128}, \, \mathbf{b}_3} \text{Output}(\mathbb{R}^4)$$
- **Nhiệm vụ của mạng:** Nhận vector trạng thái $s_t \in \mathbb{R}^{28}$ biểu diễn thuộc tính hình học và tải trọng của lộ trình hiện tại, dự đoán vector $Q$-values kỳ vọng cho 4 toán tử vi mô:

$$\mathbf{Q}(s_t; \theta) = \big[ Q(s_t, a_0; \theta), \; Q(s_t, a_1; \theta), \; Q(s_t, a_2; \theta), \; Q(s_t, a_3; \theta) \big]^\top \in \mathbb{R}^4$$

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
  $$\mathbb{R}^{36} \xrightarrow{\mathbf{W}_1 \in \mathbb{R}^{128 \times 36}} \text{FC}(128) + \text{ReLU} \xrightarrow{\mathbf{W}_2 \in \mathbb{R}^{128 \times 128}} \text{FC}(128) + \text{ReLU} \xrightarrow{\mathbf{W}_3 \in \mathbb{R}^{6 \times 128}} \text{Output}(\mathbb{R}^6)$$
- Tác tử không chọn toán tử cục bộ đổi chỗ 1 điểm, mà quyết định **chiến lược vĩ mô** phá hủy một cụm từ $20\% - 40\%$ số điểm giao và tái cấu trúc lại toàn bộ mạng lưới.

#### B. Cơ chế Chấp nhận Luyện kim (Simulated Annealing Acceptance)
Thay vì chỉ chấp nhận khi $\Delta \mathcal{C} > 0$ một cách bảo thủ, Neural LNS tích hợp cơ chế làm nguội Simulated Annealing. Nếu nước đi làm tăng cự ly ($\Delta \mathcal{C} = \mathcal{C}_{\text{current}} - \mathcal{C}_{\text{candidate}} < 0$), nghiệm ứng viên $\mathcal{R}_{\text{candidate}}$ vẫn được chấp nhận với xác suất chuyển trạng thái:

$$P\big(\text{accept} \mid \Delta \mathcal{C}, T_t\big) = \begin{cases} 1.0 & \text{khi } \Delta \mathcal{C} \ge 0 \\ \exp\left( \dfrac{\Delta \mathcal{C}}{T_t} \right) & \text{khi } \Delta \mathcal{C} < 0 \end{cases}$$

Trong đó nhiệt độ $T_t$ suy giảm hình học theo từng bước tìm kiếm:

$$T_t = T_0 \cdot \alpha^t, \quad \text{với } T_0 = \beta \cdot \mathcal{C}_{\text{init}}, \; \beta = 0.05, \; \alpha = 0.985$$

Cơ chế này cho phép thuật toán tạm thời suy thoái cự ly trong khoảng $1.5\% - 2.0\%$ ở các bước đầu để nhảy ra khỏi "hố sâu" cực tiểu địa phương, sau đó tái cấu trúc tuyến về trạng thái tối ưu toàn cục hơn.

#### C. Kết quả Vượt trội Định lượng
Trên tập dữ liệu 24 điểm giao hàng thực tế tại TP.HCM (tổng tải $315\text{ kg}$, $K=7\text{ xe}$, $Q=50\text{ kg}$):
- `CW + Greedy`: $16.57\text{ km}$ ($+0.22\%$ savings).
- `DDQN Gốc`: $16.61\text{ km}$ ($+0.00\%$ savings).
- `Neural LNS (TP.HCM)`: **$16.34\text{ km}$ ($+1.63\%$ savings)**, rút ngắn thêm $230\text{ m}$ hành trình bám tim đường trong khi vẫn bảo đảm $100\%$ dung lượng tải và $K_{\text{used}} = 7/7\text{ xe}$.

---

## PHẦN IV: BÓC TÁCH TOÁN HỌC VECTOR ĐẶC TRƯNG TRẠNG THÁI (F28 & F36)

Một đóng góp kỹ thuật quan trọng của đồ án là việc ánh xạ cấu trúc đồ thị định tuyến không gian phức tạp thành các vector số học có số chiều cố định, không phụ thuộc vào quy mô số lượng khách hàng $N$ thay đổi.

---

### 4.1. Bóc Tách Chi Tiết Vector Trạng Thái $F28$ (28 Chiều — Dành Cho DDQN Gốc)

Vector trạng thái $s_t \in \mathbb{R}^{28}$ được phân rã thành 2 nhóm đặc trưng cấu trúc và động thái:

#### Nhóm 1: 12 Đặc Trưng Toàn Cục (Global Structural Features: $f_0 \dots f_{11}$)

1. **$f_0$ — Tỷ lệ tải trọng trung bình của đội xe ($\mu_{\text{load}}$):**
   $$f_0 = \frac{1}{|\mathcal{R}|} \sum_{k=1}^{|\mathcal{R}|} \frac{\text{Load}(R_k)}{Q} \in [0, 1]$$
   Đo lường mức độ lấp đầy thùng xe bình quân của toàn bộ các phương tiện đang vận hành.

2. **$f_1$ — Độ lệch chuẩn tỷ lệ tải trọng ($\sigma_{\text{load}}$):**
   $$f_1 = \sqrt{\frac{1}{|\mathcal{R}|} \sum_{k=1}^{|\mathcal{R}|} \left(\frac{\text{Load}(R_k)}{Q} - f_0\right)^2}$$
   Phản ánh sự mất cân đối tải giữa các xe (giá trị cao báo hiệu tình trạng xe chở quá đầy trong khi xe khác chở rỗng).

3. **$f_2$ — Tỷ lệ tải xe nặng nhất ($\max \text{Load} / Q$):**
   $$f_2 = \max_{k=1 \dots |\mathcal{R}|} \frac{\text{Load}(R_k)}{Q} \in [0, 1]$$
   Đo lường nguy cơ vi phạm sức chứa khi điều chuyển khách hàng sang tuyến này.

4. **$f_3$ — Tỷ lệ tải xe nhẹ nhất ($\min \text{Load} / Q$):**
   $$f_3 = \min_{k=1 \dots |\mathcal{R}|} \frac{\text{Load}(R_k)}{Q} \in [0, 1]$$
   Đo lường tiềm năng giải thể gom tuyến nhằm tiết kiệm chi phí cố định của một phương tiện.

5. **$f_4$ — Tỷ lệ điều động đội xe chuẩn hóa ($|\mathcal{R}| / K$):**
   $$f_4 = \frac{|\mathcal{R}|}{K} \in (0, 1]$$
   Tỷ lệ số xe đang lăn bánh thực tế so với tổng quy mô đội xe sẵn sàng $K$.

6. **$f_5$ — Cự ly tuyến bình quân chuẩn hóa ($\overline{\mathcal{C}}_{\text{norm}}$):**
   $$f_5 = \frac{1}{|\mathcal{R}| \cdot \mathcal{C}_{\text{init}}} \sum_{k=1}^{|\mathcal{R}|} \mathcal{C}(R_k)$$
   Cự ly trung bình mỗi xe chia cho tổng chi phí ban đầu để giữ giá trị trong miền xác định hữu hạn.

7. **$f_6$ — Độ lệch chuẩn cự ly giữa các tuyến ($\sigma_{\mathcal{C}}$):**
   $$f_6 = \frac{1}{\mathcal{C}_{\text{init}}} \sqrt{\frac{1}{|\mathcal{R}|} \sum_{k=1}^{|\mathcal{R}|} \left(\mathcal{C}(R_k) - \overline{\mathcal{C}}\right)^2}$$
   Đo lường mức độ công bằng về khối lượng công việc và thời gian di chuyển giữa các tài xế.

8. **$f_7$ — Tỷ số nghẽn Makespan ($\rho_{\text{makespan}}$):**
   $$f_7 = \frac{\max_{k} \mathcal{C}(R_k)}{\sum_{k=1}^{|\mathcal{R}|} \mathcal{C}(R_k)} \in \left[\frac{1}{|\mathcal{R}|}, \; 1\right]$$
   Tỷ lệ độ dài tuyến xe về kho muộn nhất so với tổng cự ly toàn mạng lưới.

9. **$f_8$ — Tiến độ ngân sách tìm kiếm cục bộ ($\tau_{\text{budget}}$):**
   $$f_8 = \frac{t}{T_{\max}} \in [0, 1]$$
   Cung cấp thông tin thời gian biểu để mạng nơ-ron điều chỉnh giữa khám phá (Exploration) và hội tụ (Exploitation).

10. **$f_9$ — Tỷ lệ cải thiện chi phí tích lũy ($\Delta \mathcal{C}_{\text{cum}}$):**
    $$f_9 = \frac{\mathcal{C}_{\text{init}} - \mathcal{C}_t}{\mathcal{C}_{\text{init}}} \in [0, 1]$$
    Đo lường tổng quãng đường đã tiết kiệm được kể từ nghiệm xuất phát Clarke-Wright.

11. **$f_{10}$ — Tỷ lệ bước cải thiện thành công gần đây ($\text{succ}_{\text{recent}}$):**
    $$f_{10} = \frac{1}{W} \sum_{\tau = t - W + 1}^t \mathbf{1}\big[\Delta \mathcal{C}_\tau > 0\big], \quad (W = 10)$$
    Tần suất tìm thấy nước đi giảm cự ly trong cửa sổ trượt $10$ bước gần nhất.

12. **$f_{11}$ — Hệ số mật độ điểm dừng trên hành trình ($\text{density}$):**
    $$f_{11} = \frac{N}{\sum_{k=1}^{|\mathcal{R}|} \mathcal{C}(R_k)}$$
    Số khách hàng phục vụ bình quân trên mỗi kilomet hành trình di chuyển.

---

#### Nhóm 2: 16 Đặc Trưng Động Thái Lịch Sử Toán Tử ($f_{12} \dots f_{27}$)

Bao gồm $4\text{ chỉ số thống kê} \times 4\text{ toán tử vi mô}$ ($a \in \{0: \text{2-opt}, \; 1: \text{intra}, \; 2: \text{inter-rel}, \; 3: \text{inter-swap}\}$):

1. **Tần suất được chọn trong cửa sổ gần nhất ($W=10$ bước):**
   $$f_{12 + 4a} = \text{freq}_a = \frac{1}{W} \sum_{\tau = t - W + 1}^t \mathbf{1}[a_\tau = a] \in [0, 1]$$

2. **Tỷ lệ thành công lịch sử tích lũy ($\text{succ}_a$):**
   $$f_{13 + 4a} = \text{succ}_a = \frac{\sum_{\tau=1}^t \mathbf{1}[a_\tau = a \land \Delta \mathcal{C}_\tau > 0]}{\max\left(1, \; \sum_{\tau=1}^t \mathbf{1}[a_\tau = a]\right)} \in [0, 1]$$

3. **Mức cải thiện cự ly trung bình chuẩn hóa ($\overline{\Delta \mathcal{C}}_a$):**
   $$f_{14 + 4a} = \overline{\Delta \mathcal{C}}_a = \frac{\sum_{\tau=1}^t \Delta \mathcal{C}_\tau \cdot \mathbf{1}[a_\tau = a \land \Delta \mathcal{C}_\tau > 0]}{\mathcal{C}_{\text{init}} \cdot \max\left(1, \; \sum_{\tau=1}^t \mathbf{1}[a_\tau = a \land \Delta \mathcal{C}_\tau > 0]\right)}$$

4. **Chuỗi không cải thiện chuẩn hóa theo ngưỡng kiên nhẫn ($\text{patience} = 30$):**
   $$f_{15 + 4a} = \text{streak\_norm}_a = \min\left(1.0, \; \frac{\text{streak}_a}{30}\right) \in [0, 1]$$
   Khi $\text{streak\_norm}_a \to 1.0$, tác tử nhận diện lân cận của toán tử $a$ đã cạn kiệt tiềm năng cải thiện.

---

### 4.2. Bóc Tách Chi Tiết Vector Trạng Thái $F36$ (36 Chiều — Dành Cho Neural LNS)

Vector trạng thái $S_t \in \mathbb{R}^{36}$ trong Neural LNS nâng cấp toàn diện góc nhìn không gian, bao gồm 4 khối đặc trưng chuyên sâu:

| Khối đặc trưng | Khoảng chỉ số | Chiều | Ý nghĩa vật lý & Biểu thức toán học chuẩn LaTeX |
| :--- | :---: | :---: | :--- |
| **1. Cấu trúc Tải trọng Tuyến** | $f_0 \dots f_7$ | $8$ | - $f_0 = \mu_{\text{load}} = \frac{1}{|\mathcal{R}|} \sum_k \frac{\text{Load}(R_k)}{Q}$<br/>- $f_1 = \sigma_{\text{load}} = \sqrt{\frac{1}{|\mathcal{R}|} \sum_k (\frac{\text{Load}(R_k)}{Q} - \mu_{\text{load}})^2}$<br/>- $f_2 = \min_k \frac{\text{Load}(R_k)}{Q}, \quad f_3 = \max_k \frac{\text{Load}(R_k)}{Q}$<br/>- $f_4 = \rho_{<50\%} = \frac{1}{|\mathcal{R}|} \sum_k \mathbf{1}\left[\frac{\text{Load}(R_k)}{Q} < 0.5\right]$<br/>- $f_5 = CV_{\text{load}} = \frac{\sigma_{\text{load}}}{\mu_{\text{load}} + \epsilon}$ (Hệ số biến thiên tải)<br/>- $f_6 = \rho_{\text{idle}} = \frac{K - |\mathcal{R}|}{K}$ (Tỷ lệ xe dự phòng chưa dùng)<br/>- $f_7 = \Delta \text{Load}_{\max} = f_3 - f_2$ (Biên độ chênh lệch tải cực đại) |
| **2. Hình học & Bất đối xứng OSM** | $f_8 \dots f_{11}$ | $4$ | - $f_8 = R_{\text{cluster}} = \frac{\max_{i \in V_C} d(0, i)}{\max_{(u, v)} d(u, v)}$ (Bán kính cụm phục vụ)<br/>- $f_9 = \overline{d}_0 = \frac{1}{N} \sum_{i \in V_C} \frac{d(0, i) + d(i, 0)}{2 \cdot \max d}$ (Khoảng cách trung bình Depot)<br/>- $f_{10} = \rho_{\text{asym}} = \frac{1}{|E_{\mathcal{R}}|} \sum_{(u, v) \in E_{\mathcal{R}}} \frac{\|d(u, v) - d(v, u)\|}{d(u, v) + d(v, u) + \epsilon}$ (Mức bất đối xứng cung đường OSM)<br/>- $f_{11} = \text{DB}_{\text{spatial}}$ (Hệ số phân tách cụm Davies-Bouldin chuẩn hóa) |
| **3. Tiến độ & Luyện kim Annealing** | $f_{12} \dots f_{17}$ | $6$ | - $f_{12} = \tau_{\text{budget}} = \frac{t}{T_{\max}}$ (Tiến độ ngân sách tìm kiếm)<br/>- $f_{13} = \frac{T_t}{T_0} = \alpha^t$ (Tỷ lệ nhiệt độ Simulated Annealing hiện tại)<br/>- $f_{14} = \rho_{\text{accept\_degrade}} = \frac{\sum \mathbf{1}[\text{accepted} \, \land \, \Delta \mathcal{C} < 0]}{\max(1, \, \sum \mathbf{1}[\Delta \mathcal{C} < 0])}$ (Tỷ lệ chấp nhận nước đi thoái biến)<br/>- $f_{15} = \text{streak\_norm}_{\text{best}} = \min(1.0, \, \frac{\text{streak}_{\text{best}}}{30})$ (Chuỗi không cải thiện Best-So-Far)<br/>- $f_{16} = \frac{\mathcal{C}_{\text{init}} - \mathcal{C}_{\text{best}}}{\mathcal{C}_{\text{init}}}$ (Tỷ lệ cải thiện nghiệm kỷ lục)<br/>- $f_{17} = \frac{\mathcal{C}_t - \mathcal{C}_{\text{best}}}{\mathcal{C}_{\text{best}}}$ (Khoảng cách giữa nghiệm hiện tại và kỷ lục) |
| **4. Động thái 6 Vĩ toán tử LNS** | $f_{18} \dots f_{35}$ | $18$ | Gồm $3\text{ chỉ số} \times 6\text{ vĩ toán tử Macro } a \in \{0 \dots 5\}$:<br/>- $f_{18 + 3a} = \text{freq}_a^{(10)} = \frac{1}{10} \sum_{\tau=t-9}^t \mathbf{1}[a_\tau = a]$<br/>- $f_{19 + 3a} = \text{succ}_a = \frac{\sum \mathbf{1}[a_\tau = a \, \land \, \text{accepted}]}{\max(1, \, \sum \mathbf{1}[a_\tau = a])}$<br/>- $f_{20 + 3a} = \overline{\Delta \mathcal{C}}_a = \frac{\sum \Delta \mathcal{C}_\tau \cdot \mathbf{1}[a_\tau = a \, \land \, \text{accepted}]}{\mathcal{C}_{\text{init}} \cdot \max(1, \, \sum \mathbf{1}[a_\tau = a \, \land \, \text{accepted}])}$ |

---

## PHẦN V: BÓC TÁCH TOÀN BỘ KHÔNG GIAN HÀNH ĐỘNG (ACTIONS & OPERATORS)

---

### 5.1. Bốn Toán Tử Cục Bộ Vi Mô (Micro Operators — DDQN Gốc)

```mermaid
flowchart TD
    subgraph MicroOps["4 Toán Tử Local Search Cục Bộ"]
        Op0["Action 0: 2-opt<br/>Đảo ngược đoạn tuyến<br/>(Bảo toàn tải trọng 100%)"]
        Op1["Action 1: Intra-relocate<br/>Dời điểm trong cùng 1 tuyến<br/>(Bảo toàn tải trọng 100%)"]
        Op2["Action 2: Inter-relocate<br/>Chuyển điểm từ Tuyến A sang B<br/>(Bắt buộc kiểm tra Load_B + q_i <= Q)"]
        Op3["Action 3: Inter-swap<br/>Đổi chỗ điểm giữa Tuyến A và B<br/>(Kiểm tra Load_A - qi + qj <= Q và Load_B - qj + qi <= Q)"]
    end
```

1. **`Action 0 (2-opt)`:** Chọn một tuyến $R_k = (0 \to \dots \to u \to v_i \to \dots \to v_j \to w \to \dots \to 0)$, đảo ngược phân đoạn từ $v_i$ đến $v_j$ thành $(0 \to \dots \to u \to v_j \dots v_i \to w \to \dots \to 0)$.
   - Mức biến thiên cự ly trên đồ thị có hướng OSM:
     $$\Delta \mathcal{C}_{\text{2-opt}} = d(u, v_i) + d(v_j, w) - d(u, v_j) - d(v_i, w) - \sum_{l=i}^{j-1} \Big( d(v_l, v_{l+1}) - d(v_{l+1}, v_l) \Big)$$
   - *Ràng buộc tải trọng:* $\sum_{v \in R_k} q_v$ không đổi $\implies$ Bảo toàn khả thi $100\%$.

2. **`Action 1 (Intra-relocate)`:** Rút một khách hàng $v_i$ ra khỏi vị trí giữa $(u, w)$ trên tuyến $R_k$ và tái chèn vào giữa $(x, y)$ trên cùng tuyến đó.
   - Mức biến thiên cự ly:
     $$\Delta \mathcal{C}_{\text{intra}} = \Big( d(u, v_i) + d(v_i, w) - d(u, w) \Big) + \Big( d(x, y) - d(x, v_i) - d(v_i, y) \Big)$$
   - *Ràng buộc tải trọng:* Bảo toàn khả thi $100\%$.

3. **`Action 2 (Inter-relocate)`:** Rút khách hàng $v_i$ khỏi tuyến $R_A$ (giữa $u_A$ và $w_A$) và chèn sang tuyến $R_B$ (giữa $x_B$ và $y_B$).
   - Mức biến thiên cự ly:
     $$\Delta \mathcal{C}_{\text{inter-rel}} = \Big( d(u_A, v_i) + d(v_i, w_A) - d(u_A, w_A) \Big) + \Big( d(x_B, y_B) - d(x_B, v_i) - d(v_i, y_B) \Big)$$
   - **Ràng buộc sức chứa bắt buộc:**
     $$\text{Load}(R_B) + q_{v_i} \le Q$$
     Nếu điều kiện vi phạm, ứng viên lập tức bị hủy bỏ không tính toán chi phí.

4. **`Action 3 (Inter-swap)`:** Hoán đổi vị trí của hai khách hàng $v_i \in R_A$ (giữa $u_A, w_A$) và $v_j \in R_B$ (giữa $x_B, y_B$).
   - Mức biến thiên cự ly:
     $$\Delta \mathcal{C}_{\text{inter-swap}} = \Big( d(u_A, v_i) + d(v_i, w_A) - d(u_A, v_j) - d(v_j, w_A) \Big) + \Big( d(x_B, v_j) + d(v_j, y_B) - d(x_B, v_i) - d(v_i, y_B) \Big)$$
   - **Ràng buộc sức chứa kép bắt buộc:**
     $$\begin{cases} \text{Load}(R_A) - q_{v_i} + q_{v_j} \le Q \\[4pt] \text{Load}(R_B) - q_{v_j} + q_{v_i} \le Q \end{cases}$$

---

### 5.2. Sáu Vĩ Toán Tử Phá Hủy & Tái Tạo (Macro Actions — Neural LNS)

Tại mỗi bước $t$, tác tử Neural LNS trích xuất một tập con $\mathcal{D} \subset V_C$ gồm $q$ khách hàng ($q \in [2, \; \max(3, \lfloor 0.35 \cdot N \rfloor)]$) thông qua 1 trong 3 cơ chế Phá hủy (Destroy), sau đó tái chèn thông minh bằng 1 trong 2 cơ chế Tái tạo (Repair):

#### A. Các Cơ Chế Phá Hủy (Destroy Mechanisms)

1. **Worst-Cost Removal:** Tính toán chi phí tiết kiệm cục bộ khi loại bỏ khách hàng $v_i$ ra khỏi tuyến hiện tại:
   $$\Delta \text{Saving}(v_i) = d\big(\text{pred}(v_i), v_i\big) + d\big(v_i, \text{succ}(v_i)\big) - d\big(\text{pred}(v_i), \text{succ}(v_i)\big)$$
   Sắp xếp giảm dần theo $\Delta \text{Saving}(v_i)$, ưu tiên loại bỏ các khách hàng làm tuyến bị gấp khúc hoặc đi vòng xa nhất.

2. **Shaw Relatedness Removal:** Rút các khách hàng có mức độ tương đồng địa lý và nhu cầu cao với một khách hàng hạt nhân $r \in V_C$:
   $$\mathcal{R}(i, j) = \phi \cdot \frac{d(i, j)}{\max_{(u, v)} d(u, v)} + \chi \cdot \frac{|q_i - q_j|}{\max_k q_k} + \psi \cdot \Big( 1 - \mathbf{1}\big[R(i) = R(j)\big] \Big)$$
   Với bộ trọng số chuẩn hóa: $\phi = 0.8, \; \chi = 0.1, \; \psi = 0.1$. Cơ chế này giúp bóc tách toàn bộ một cụm không gian đang bị phân mảnh giữa các xe.

3. **Route-Squeeze Removal:** Xác định tuyến có tổng tải trọng nhỏ nhất:
   $$R^* = \arg\min_{R_k \in \mathcal{R}} \text{Load}(R_k)$$
   Rút toàn bộ khách hàng trên $R^*$ vào tập phá hủy $\mathcal{D}$, giải phóng hoàn toàn phương tiện này nhằm đưa $|\mathcal{R}|$ về sát $K_{\min}$.

#### B. Các Cơ Chế Tái Tạo (Repair Mechanisms)

1. **Greedy Best Insertion:** Với mỗi khách hàng $v_i \in \mathcal{D}$, duyệt toàn bộ các vị trí chèn khả thi $p = (R_k, \text{pos})$ thỏa mãn $\text{Load}(R_k) + q_{v_i} \le Q$, chọn vị trí có độ tăng cự ly nhỏ nhất:
   $$p^*(v_i) = \arg\min_{p} \Delta \mathcal{C}(v_i, p)$$

2. **Regret-2 Insertion:** Tính toán thiệt hại cơ hội giữa vị trí chèn tối ưu nhất (thứ 1) và vị trí chèn tốt thứ nhì (thứ 2):
   $$\Delta_{\text{regret}}(v_i) = \Delta \mathcal{C}^{(2)}(v_i) - \Delta \mathcal{C}^{(1)}(v_i) = \min_{p \ne p_1^*} \Delta \mathcal{C}(v_i, p) - \min_p \Delta \mathcal{C}(v_i, p)$$
   Khách hàng có độ ân hận lớn nhất sẽ được ưu tiên chèn trước:
   $$v^* = \arg\max_{v_i \in \mathcal{D}} \Delta_{\text{regret}}(v_i)$$

3. **2-opt Polish:** Sau khi tái chèn bằng Regret-2, tự động thực thi một lượt quét nhanh 2-opt trên các tuyến vừa có khách hàng chèn vào để khử triệt để các giao cắt hình học.

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

### 6.1. Phương Trình Cập Nhật Double Deep Q-Learning & Hàm Mất Mát

Nhằm loại bỏ hiện tượng thổi phồng giá trị kỳ vọng (Overestimation Bias), hệ thống áp dụng thuật toán **Double DQN (DDQN)** với hai bộ tham số tách biệt: mạng đánh giá hành động $\theta$ (Online Policy Network) và mạng định lượng mục tiêu $\theta^-$ (Target Network):

$$Y_t^{\text{DoubleQ}} = r_t + \gamma \cdot Q\left(s_{t+1}, \; \arg\max_{a' \in \mathcal{A}} Q(s_{t+1}, a'; \theta); \; \theta^-\right)$$

Trong đó:
- $r_t$ là phần thưởng định lượng bước lặp:
  $$r_t = \begin{cases} \dfrac{\mathcal{C}_t - \mathcal{C}_{t+1}}{\mathcal{C}_{\text{init}}} \times 100 & \text{khi } \mathcal{C}_{t+1} < \mathcal{C}_t \\[6pt] -0.05 & \text{khi } \mathcal{C}_{t+1} \ge \mathcal{C}_t \end{cases}$$
- $\gamma = 0.99$ là hệ số chiết khấu tương lai.

Hàm mất mát Huber Loss mượt mà giúp huấn luyện ổn định trước các biến thiên đột ngột:

$$\mathcal{L}(\theta) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \ell_\delta\left( Y_i^{\text{DoubleQ}} - Q(s_i, a_i; \theta) \right)$$

$$\ell_\delta(e) = \begin{cases} \dfrac{1}{2} e^2 & \text{khi } |e| \le \delta \\[6pt] \delta \left(|e| - \dfrac{1}{2}\delta\right) & \text{khi } |e| > \delta \end{cases}, \quad (\delta = 1.0)$$

---

### 6.2. Cơ Chế Lọc Hành Động Khả Thi (Action Masking)

Trước khi tính toán suy luận mạng nơ-ron, môi trường sinh vector mặt nạ nhị phân $\mathbf{m}_t \in \{0, 1\}^{|\mathcal{A}|}$:
- Nếu số tuyến $|\mathcal{R}| < 2$, các hành động liên tuyến (`inter-relocate`, `inter-swap`, `Route-Squeeze`) bị khóa: $m_t[a] = 0$.
- Mặt nạ được áp dụng trực tiếp lên vector $Q$-values:

$$\widetilde{Q}(s_t, a) = \begin{cases} Q(s_t, a; \theta) & \text{khi } m_t[a] = 1 \\[4pt] -\infty & \text{khi } m_t[a] = 0 \end{cases}$$

Chính sách hành động suy luận (Greedy Execution Policy):

$$a_t^* = \arg\max_{a \in \mathcal{A}} \widetilde{Q}(s_t, a)$$

Bảo đảm $100\%$ không bao giờ chọn toán tử vi phạm cấu trúc bài toán.

---

### 6.3. Cơ Chế Khống Chế Cứng Đội Xe (Hard Fleet Cap: $K_{\text{used}} \le K$)

Khắc phục triệt để hiện tượng phân mảnh tải trọng làm sinh thừa xe ($8/7$ xe):
- **Thuật toán Route Consolidation:**
  1. Khi $|\mathcal{R}| > K$, sắp xếp các tuyến theo tổng tải tăng dần:
     $$\text{Load}(R_1) \le \text{Load}(R_2) \le \dots \le \text{Load}(R_m)$$
  2. Giải thể toàn bộ khách hàng khỏi tuyến nhẹ nhất $R_1$.
  3. Phân bổ các khách hàng này vào $m-1$ tuyến còn lại bằng phương pháp **Regret-2 Insertion**, chỉ chấp nhận các vị trí chèn thỏa mãn:
     $$\text{Load}(R_k) + q_v \le Q$$
  4. Lặp lại cho đến khi $|\mathcal{R}| \le K$.
- **Cam kết:** Nghiệm xuất xưởng luôn thỏa mãn $K_{\text{used}} \le K$.

---

### 6.4. Bốn Quy Tắc Bất Biến Cốt Lõi (`AGENTS.md`)

1. **BẢO TOÀN TẢI TRỌNG (Feasibility Invariant):**
   $$\sum_{i \in R_k} q_i \le Q, \quad \forall R_k \in \mathcal{R}$$
   Tác tử `ConstraintValidatorAgent` độc lập tái thẩm định $100\%$ sau mỗi bước.

2. **PHỤC VỤ DUY NHẤT (Uniqueness Invariant):**
   $$\sum_{k=1}^{|\mathcal{R}|} \mathbf{1}[i \in R_k] = 1, \quad \forall i \in V_C$$

3. **KHÉP KÍN DEPOT (Cycle Closure Invariant):**
   Mọi tuyến xe luôn bắt đầu và kết thúc tại Depot $0$:
   $$R_k = (0 \to v_{k,1} \to v_{k,2} \to \dots \to v_{k,p} \to 0)$$

4. **TRUY NGUYÊN & TRUNG THỰC KHOA HỌC (Scientific Honesty Invariant):**
   Số liệu thực nghiệm phản ánh đúng thực tế, đối chiếu công bằng giữa DRL và Heuristic.

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
