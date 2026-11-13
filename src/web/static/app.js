/**
 * MP-MOTO-CVRP (V9.3) — Warehouse Dispatch App Controller
 * 100% Single Source of Truth Geometry & Vehicle-Bound Motorcycle Simulation
 * Zero-Broken-Numbers Guard (No undefined, NaN, or Infinity on UI)
 * Apple Human Interface Guidelines: Minimalist Underline Tabs & Segmented Control
 * Two-Way Map Sync, Per-District Smart Sampler & Grouped Store Management
 */

// --- Application State ---
const state = {
  map: null,
  markersLayer: null,
  routesLayer: null,
  bikesLayer: null,
  depots: [],
  selectedDepot: null,
  allStores: [],
  displayedStores: [], // Stores currently loaded in sidebar list
  checkedStoreIds: new Set(), // Set of store IDs that are actively selected for routing
  removedStoreIds: new Set(), // Stores explicitly removed from the active list
  storePkgMap: {}, // store_id -> n_pkg
  storeMarkers: {}, // store_id -> L.marker
  storeVehicleColorMap: {}, // store_id -> hex color string (after dispatch)
  selectedDistricts: new Set(['ALL']), // Set of district strings
  loadSizeMode: '6', // '6' | '15' | '30' | '50' | 'all' | 'custom'
  searchQuery: '', // Full-store-picker search text
  kMinSuggested: null, // Auto-computed minimum fleet size
  isMultiDistrict: false, // Default: false (strict single-district mode)
  pickingDepotOnMap: false,
  pickingStoreOnMap: false,
  solverMode: 'neural_lns', // 'cw_greedy' | 'ddqn_base' | 'neural_lns'
  fleet: {
    k: 3,
    q_kg: 50
  },
  lastDispatchResult: null,
  benchmarkResults: null,
  simulation: {
    isPlaying: false,
    speedMultiplier: 1,
    elapsedSeconds: 0,
    animFrameId: null,
    vehicleMarkers: {}, // vehicle_id -> L.marker
    vehiclePaths: {},   // vehicle_id -> array of { lat, lon, cum_dist_m, cum_time_s }
    totalDuration_s: 0
  }
};

// Safe Number Formatter (Anti-Broken-Number Invariant)
function formatNumber(val, decimals = 1, suffix = '') {
  if (val === null || val === undefined || isNaN(val) || !isFinite(val)) {
    return '—';
  }
  return Number(val).toFixed(decimals) + (suffix ? ' ' + suffix : '');
}

// Get array of active stores selected for dispatch (no 6-store cap)
function getActiveSelectedStores() {
  const active = [];
  state.allStores.forEach((st) => {
    if (state.checkedStoreIds.has(st.id)) {
      const n_pkg = state.storePkgMap[st.id] || (st.typical_pkg_range ? st.typical_pkg_range[0] : 2);
      const w_pkg = 5.0;
      active.push({
        id: st.id,
        name: st.name,
        address: st.address,
        district: st.district,
        lat: st.lat,
        lon: st.lon,
        n_pkg: n_pkg,
        weight_per_pkg_kg: w_pkg,
        demand_kg: Number((n_pkg * w_pkg).toFixed(1)),
        snapped_node_id: st.snapped_node_id
      });
    }
  });
  return active;
}

// Stores available for the current district selection (ALL = full 570-store pool)
function getPoolForDistricts() {
  if (state.selectedDistricts.has('ALL') || state.selectedDistricts.size === 0) {
    return state.allStores;
  }
  return state.allStores.filter((st) => state.selectedDistricts.has(st.district));
}

// Stores currently visible in the picker (search results override the loaded subset)
function getVisibleStores() {
  const q = state.searchQuery.trim().toLowerCase();
  if (q) {
    return state.allStores.filter((st) => {
      if (state.removedStoreIds.has(st.id)) return false;
      const haystack = `${st.name || ''} ${st.address || ''} ${st.district || ''}`.toLowerCase();
      return haystack.includes(q);
    });
  }
  return state.displayedStores.filter((st) => !state.removedStoreIds.has(st.id));
}

// --- DOM Bootstrap ---
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  setupEventListeners();
  loadInitialData();
});

// --- Map Initialization ---
function initMap() {
  const container = L.DomUtil.get('map');
  if (container != null && container._leaflet_id) {
    container._leaflet_id = null;
  }

  state.map = L.map('map', {
    zoomControl: false,
    attributionControl: false
  }).setView([10.7820, 106.6980], 12);

  L.control.zoom({ position: 'bottomright' }).addTo(state.map);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    subdomains: ['a', 'b', 'c']
  }).addTo(state.map);

  state.routesLayer = L.layerGroup().addTo(state.map);
  state.markersLayer = L.layerGroup().addTo(state.map);
  state.bikesLayer = L.layerGroup().addTo(state.map);

  // Map click listener
  state.map.on('click', (e) => {
    if (state.pickingDepotOnMap) {
      handleCustomDepotPick(e.latlng.lat, e.latlng.lng);
    } else if (state.pickingStoreOnMap) {
      handleStorePickFromMap(e.latlng.lat, e.latlng.lng);
    }
  });
}

// --- Load Initial Data from Backend ---
async function loadInitialData() {
  try {
    // 1. Load Depots
    const resDepots = await fetch('/api/motorbike/depots');
    const dataDepots = await resDepots.json();
    state.depots = dataDepots.depots || [];
    renderDepotsDropdown();

    // 2. Load Stores
    const resStores = await fetch('/api/motorbike/stores');
    const dataStores = await resStores.json();
    state.allStores = dataStores.stores || [];

    // Initialize package map
    state.allStores.forEach((s) => {
      state.storePkgMap[s.id] = s.typical_pkg_range ? s.typical_pkg_range[0] : 2;
    });

    // 3. Render District Chips & flexible load controls
    renderDistrictChips();
    updateLoadSizeControls();

    const searchInput = document.getElementById('storeSearchInput');
    if (searchInput) {
      searchInput.placeholder = `Tìm tên cửa hàng, tên đường, quận... (toàn bộ ${state.allStores.length} điểm)`;
    }

    // 4. Populate default depot selection
    if (state.depots.length > 0) {
      selectDepot(state.depots[0].id);
    }

    // 5. Initial state: load a small starter batch (user can expand to all 570)
    loadStoresFromPool(6);

  } catch (err) {
    console.error('Failed to load initial data:', err);
    showToast('Không thể tải danh mục từ máy chủ. Vui lòng kiểm tra kết nối.');
  }
}

// --- Depots UI Handling ---
function renderDepotsDropdown() {
  const select = document.getElementById('depotSelect');
  if (!select) return;
  select.innerHTML = '';

  state.depots.forEach((d) => {
    const opt = document.createElement('option');
    opt.value = d.id;
    opt.textContent = `[${d.zone_type.toUpperCase()}] ${d.name} (${d.district})`;
    select.appendChild(opt);
  });

  select.addEventListener('change', (e) => {
    selectDepot(e.target.value);
  });
}

function selectDepot(depotId) {
  const depot = state.depots.find((d) => d.id === depotId);
  if (!depot) return;

  state.selectedDepot = depot;
  document.getElementById('depotSelect').value = depot.id;
  document.getElementById('depotAddressDisplay').textContent = `${depot.district} • Tọa độ: ${depot.lat.toFixed(4)}, ${depot.lon.toFixed(4)}`;
  document.getElementById('depotZoneDisplay').textContent = `Khu vực: ${depot.zone_type.toUpperCase()} • Nguồn: ${depot.source.toUpperCase()}`;
  document.getElementById('depotSnapBadge').textContent = `Snap: ${formatNumber(depot.snap_distance_m, 1, 'm')}`;

  renderDepotMarker();
  updateRealtimeFeasibility();
}

async function handleCustomDepotPick(lat, lon) {
  state.pickingDepotOnMap = false;
  const btn = document.getElementById('btnPickDepotMap');
  if (btn) btn.classList.remove('btn-highlight');

  try {
    const res = await fetch('/api/motorbike/snap', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ lat, lon })
    });
    const data = await res.json();
    if (!res.ok) {
      const reason = data.detail?.reason || 'Vị trí ngoài mạng lưới đường xe máy (> 150m)!';
      showToast(reason);
      return;
    }

    state.selectedDepot = {
      id: 'CUSTOM_DEPOT',
      name: `Kho Chỉ Định (${data.lat.toFixed(4)}, ${data.lon.toFixed(4)})`,
      district: 'TP.HCM',
      lat: data.lat,
      lon: data.lon,
      snapped_node_id: data.snapped_node_id,
      snap_distance_m: data.snap_distance_m,
      zone_type: 'custom',
      source: 'user_map'
    };

    document.getElementById('depotAddressDisplay').textContent = `Tọa độ đã snap: ${data.lat.toFixed(4)}, ${data.lon.toFixed(4)}`;
    document.getElementById('depotZoneDisplay').textContent = 'Khu vực: Tự chọn trên bản đồ';
    document.getElementById('depotSnapBadge').textContent = `Snap: ${formatNumber(data.snap_distance_m, 1, 'm')}`;

    renderDepotMarker();
    updateRealtimeFeasibility();
    showToast(`Đã gắn kho xuất phát (Snap: ${data.snap_distance_m.toFixed(1)}m vào tim đường).`);
  } catch (err) {
    console.error('Snap error:', err);
    showToast('Lỗi khi snap tọa độ.');
  }
}

function renderDepotMarker() {
  if (!state.selectedDepot) return;

  // Clear only existing depot marker
  state.markersLayer.eachLayer((layer) => {
    if (layer.options.icon && layer.options.icon.options.className.includes('custom-depot-marker')) {
      state.markersLayer.removeLayer(layer);
    }
  });

  const icon = L.divIcon({
    className: 'custom-depot-marker',
    html: `
      <div style="background: #1d1d1f; color: #ffffff; border: 2.5px solid #ffffff; border-radius: 50%; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);">
        K
      </div>
    `,
    iconSize: [32, 32],
    iconAnchor: [16, 16]
  });

  L.marker([state.selectedDepot.lat, state.selectedDepot.lon], { icon: icon })
    .addTo(state.markersLayer)
    .bindPopup(`<strong>Kho: ${state.selectedDepot.name}</strong><br><span style="font-size:11px; color:#64748b;">Điểm xuất phát điều phối</span>`)
    .openPopup();
}

// --- District Chips Rendering & Handling ---
function renderDistrictChips() {
  const container = document.getElementById('districtChips');
  if (!container) return;
  container.innerHTML = '';

  // 1. ALL Chip
  const isAllActive = state.selectedDistricts.has('ALL');
  const allChip = document.createElement('button');
  allChip.type = 'button';
  allChip.className = `district-chip ${isAllActive ? 'active' : ''}`;
  allChip.textContent = `TẤT CẢ (${state.allStores.length})`;
  allChip.addEventListener('click', () => handleDistrictChipClick('ALL'));
  container.appendChild(allChip);

  // 2. Individual District Chips
  const districts = [...new Set(state.allStores.map((s) => s.district))].sort();
  districts.forEach((dist) => {
    const count = state.allStores.filter((s) => s.district === dist).length;
    const isDistActive = !isAllActive && state.selectedDistricts.has(dist);
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = `district-chip ${isDistActive ? 'active' : ''}`;
    chip.textContent = `${dist} (${count})`;
    chip.addEventListener('click', () => handleDistrictChipClick(dist));
    container.appendChild(chip);
  });
}

// --- Flexible Load-Size Controls (6 / 15 / 30 / 50 / All / Custom) ---
function updateLoadSizeControls() {
  const pool = getPoolForDistricts();

  const allChip = document.getElementById('loadSizeAllChip');
  if (allChip) allChip.textContent = `Tất cả (${pool.length})`;

  const poolInfo = document.getElementById('loadPoolInfo');
  if (poolInfo) poolInfo.textContent = `${pool.length} điểm khả dụng`;

  const inputLoad = document.getElementById('inputLoadCount');
  if (inputLoad) inputLoad.max = Math.max(1, pool.length);

  document.querySelectorAll('.load-size-chip').forEach((chip) => {
    chip.classList.toggle('active', chip.getAttribute('data-count') === state.loadSizeMode);
  });
}

function getLoadCountValue() {
  if (state.loadSizeMode === 'all') return 'all';
  if (state.loadSizeMode === 'custom') {
    const input = document.getElementById('inputLoadCount');
    const parsed = parseInt(input ? input.value : '', 10);
    return Number.isFinite(parsed) && parsed >= 1 ? parsed : null;
  }
  const parsed = parseInt(state.loadSizeMode, 10);
  return Number.isFinite(parsed) && parsed >= 1 ? parsed : null;
}

function handleDistrictChipClick(district) {
  if (district === 'ALL') {
    state.selectedDistricts.clear();
    state.selectedDistricts.add('ALL');
  } else if (state.isMultiDistrict) {
    state.selectedDistricts.delete('ALL');
    if (state.selectedDistricts.has(district)) {
      state.selectedDistricts.delete(district);
      if (state.selectedDistricts.size === 0) {
        state.selectedDistricts.add('ALL');
      }
    } else {
      state.selectedDistricts.add(district);
    }
  } else {
    // Strict single district mode
    state.selectedDistricts.clear();
    state.selectedDistricts.add(district);
  }

  renderDistrictChips();
  updateLoadSizeControls();
  refreshStorePool();
}

// Unlock the entire selected pool in the picker (no 6-store truncation)
function refreshStorePool() {
  const pool = getPoolForDistricts();
  const poolIds = new Set(pool.map((s) => s.id));

  state.removedStoreIds.clear();
  state.checkedStoreIds = new Set(
    [...state.checkedStoreIds].filter((id) => poolIds.has(id))
  );
  state.displayedStores = [...pool];
  state.storeVehicleColorMap = {};
  state.searchQuery = '';
  const searchInput = document.getElementById('storeSearchInput');
  if (searchInput) searchInput.value = '';

  renderStoresList();
  renderStoreMarkers();
  updateSummaryMetrics();
  updateRealtimeFeasibility();
  fitMapToSelection();

  const label = state.selectedDistricts.has('ALL')
    ? 'toàn thành phố'
    : [...state.selectedDistricts].join(', ');
  showToast(`Đã mở khóa ${pool.length} cửa hàng (${label}) để tự do chọn.`);
}

// Load exactly M random stores (or the full pool when count === 'all')
function loadStoresFromPool(count) {
  const pool = getPoolForDistricts();
  if (pool.length === 0) {
    showToast('Không có cửa hàng nào trong vùng đang chọn.');
    return;
  }

  let target;
  if (count === 'all' || count >= pool.length) {
    target = [...pool];
  } else {
    const shuffled = [...pool].sort(() => 0.5 - Math.random());
    target = shuffled.slice(0, count);
  }

  target.forEach((st) => {
    // Random packages between 1 and 4
    state.storePkgMap[st.id] = Math.floor(Math.random() * 4) + 1;
  });

  state.removedStoreIds.clear();
  state.displayedStores = target;
  state.checkedStoreIds = new Set(target.map((s) => s.id));
  state.storeVehicleColorMap = {};
  state.searchQuery = '';
  const searchInput = document.getElementById('storeSearchInput');
  if (searchInput) searchInput.value = '';

  renderStoresList();
  renderStoreMarkers();
  updateSummaryMetrics();
  updateRealtimeFeasibility();
  fitMapToSelection();

  showToast(`Đã nạp ${target.length}/${pool.length} điểm vào đợt giao hàng.`);
}

function fitMapToSelection() {
  const bounds = L.latLngBounds();
  if (state.selectedDepot) {
    bounds.extend([state.selectedDepot.lat, state.selectedDepot.lon]);
  }
  const focusStores = getVisibleStores();
  if (focusStores.length > 0) {
    focusStores.forEach((st) => bounds.extend([st.lat, st.lon]));
    state.map.fitBounds(bounds, { padding: [60, 60], maxZoom: 14 });
  }
}

// --- Grouped Stores List Rendering ---
function renderStoresList() {
  const container = document.getElementById('storesListContainer');
  const countBadge = document.getElementById('storesCountBadge');
  if (!container) return;

  container.innerHTML = '';
  const visibleStores = getVisibleStores();
  const activeCount = state.checkedStoreIds.size;
  if (countBadge) countBadge.textContent = `${activeCount}/${visibleStores.length} điểm`;

  if (visibleStores.length === 0) {
    const isSearching = state.searchQuery.trim().length > 0;
    container.innerHTML = `
      <div class="empty-dest-notice">
        <div style="font-weight: 600; color: #1d1d1f;">${isSearching ? 'Không tìm thấy cửa hàng phù hợp' : 'Chưa có cửa hàng nào'}</div>
        <div style="font-size: 11px; color: #8e8e93; margin-top: 4px;">
          ${isSearching
            ? 'Thử từ khóa khác hoặc xóa ô tìm kiếm để xem danh sách hiện tại.'
            : 'Nhấp chọn một quận ở trên, chọn <strong>Quy mô nạp</strong> rồi bấm <strong>Nạp điểm</strong> để bắt đầu.'}
        </div>
      </div>
    `;
    updateSummaryMetrics();
    return;
  }

  // Group stores by district
  const groups = {};
  visibleStores.forEach((st) => {
    if (!groups[st.district]) groups[st.district] = [];
    groups[st.district].push(st);
  });

  const districtNames = Object.keys(groups);
  let globalIdx = 0;

  districtNames.forEach((dist) => {
    const distStores = groups[dist];
    const distActiveStores = distStores.filter((s) => state.checkedStoreIds.has(s.id));
    const distPkgs = distActiveStores.reduce((acc, s) => acc + (state.storePkgMap[s.id] || 2), 0);

    // If multi-district list (> 1 district), show sticky district header
    if (districtNames.length > 1) {
      const header = document.createElement('div');
      header.className = 'district-group-header';
      header.innerHTML = `
        <span>${dist.toUpperCase()} • ${distActiveStores.length}/${distStores.length} ĐIỂM</span>
        <span>${distPkgs} KIỆN</span>
      `;
      container.appendChild(header);
    }

    distStores.forEach((st) => {
      globalIdx++;
      const currentIdx = globalIdx;
      const isChecked = state.checkedStoreIds.has(st.id);
      const n_pkg = state.storePkgMap[st.id] || 2;
      const w_pkg = 5.0;
      const demand_kg = Number((n_pkg * w_pkg).toFixed(1));

      const row = document.createElement('div');
      row.className = `dest-item-row ${!isChecked ? 'inactive-store-row' : ''}`;
      row.setAttribute('data-id', st.id);

      row.innerHTML = `
        <input type="checkbox" class="store-row-checkbox" data-id="${st.id}" ${isChecked ? 'checked' : ''} title="Kích hoạt / Bỏ kích hoạt giao điểm này">
        <div class="dest-badge">#${currentIdx}</div>
        <div class="dest-details">
          <div class="dest-title" title="${st.name}">${st.name}</div>
          <div class="dest-sub" title="${st.address}">${st.district} • ${st.address}</div>
          <div class="dest-sub-weight mono">${n_pkg} kiện • ${demand_kg.toFixed(1)} kg</div>
        </div>
        <div class="ios-stepper">
          <button type="button" class="stepper-btn stepper-minus" title="Giảm 1 kiện">−</button>
          <span class="stepper-val mono">${n_pkg} kiện</span>
          <button type="button" class="stepper-btn stepper-plus" title="Tăng 1 kiện">+</button>
        </div>
        <button type="button" class="dest-delete-btn" title="Bỏ điểm này khỏi danh sách">✕</button>
      `;

      // Two-way Hover on row -> Highlight marker on map
      row.addEventListener('mouseenter', () => {
        const marker = state.storeMarkers[st.id];
        if (marker) {
          const el = marker.getElement();
          if (el) el.classList.add('marker-hovered');
          marker.openPopup();
        }
      });

      row.addEventListener('mouseleave', () => {
        const marker = state.storeMarkers[st.id];
        if (marker) {
          const el = marker.getElement();
          if (el) el.classList.remove('marker-hovered');
          marker.closePopup();
        }
      });

      // Checkbox toggle (Active/Inactive toggle)
      const chk = row.querySelector('.store-row-checkbox');
      chk.addEventListener('change', (e) => {
        e.stopPropagation();
        if (chk.checked) {
          state.checkedStoreIds.add(st.id);
          state.removedStoreIds.delete(st.id);
          // A store found via search may not be in the loaded subset yet
          if (!state.displayedStores.some((s) => s.id === st.id)) {
            state.displayedStores.push(st);
          }
          row.classList.remove('inactive-store-row');
        } else {
          state.checkedStoreIds.delete(st.id);
          row.classList.add('inactive-store-row');
        }
        updateStoreMarkerVisual(st.id);
        updateSummaryMetrics();
        updateRealtimeFeasibility();
      });

      // Stepper minus
      row.querySelector('.stepper-minus').addEventListener('click', (e) => {
        e.stopPropagation();
        let current = state.storePkgMap[st.id] || 2;
        if (current > 1) {
          state.storePkgMap[st.id] = current - 1;
          renderStoresList();
          updateSummaryMetrics();
          updateRealtimeFeasibility();
        }
      });

      // Stepper plus
      row.querySelector('.stepper-plus').addEventListener('click', (e) => {
        e.stopPropagation();
        let current = state.storePkgMap[st.id] || 2;
        const nextDemand = (current + 1) * w_pkg;
        if (nextDemand <= state.fleet.q_kg) {
          state.storePkgMap[st.id] = current + 1;
          renderStoresList();
          updateSummaryMetrics();
          updateRealtimeFeasibility();
        } else {
          showToast(`Kiện hàng tại '${st.name}' không được vượt quá tải trọng xe (${state.fleet.q_kg} kg)!`);
        }
      });

      // Delete item
      row.querySelector('.dest-delete-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        state.checkedStoreIds.delete(st.id);
        state.removedStoreIds.add(st.id);
        state.displayedStores = state.displayedStores.filter((item) => item.id !== st.id);
        renderStoresList();
        renderStoreMarkers();
        updateSummaryMetrics();
        updateRealtimeFeasibility();
        fitMapToSelection();
      });

      container.appendChild(row);
    });
  });

  updateSummaryMetrics();
}

// Single Source of Truth Reactive Summary Metrics
function updateSummaryMetrics() {
  const activeStores = getActiveSelectedStores();
  const totalPkgs = activeStores.reduce((acc, s) => acc + s.n_pkg, 0);
  const totalWeight = activeStores.reduce((acc, s) => acc + s.demand_kg, 0.0);

  const countBadge = document.getElementById('storesCountBadge');
  if (countBadge) countBadge.textContent = `${activeStores.length}/${getVisibleStores().length} điểm`;

  const pkgsEl = document.getElementById('totalPkgsDisplay');
  if (pkgsEl) pkgsEl.textContent = `${totalPkgs} kiện`;

  const weightEl = document.getElementById('totalWeightDisplay');
  if (weightEl) weightEl.textContent = `${totalWeight.toFixed(1)} kg`;
}

// Render 1-to-1 Markers for Stores with 1..N continuous numbering
function renderStoreMarkers() {
  // Clear previous store markers
  state.markersLayer.eachLayer((layer) => {
    if (!layer.options.icon || !layer.options.icon.options.className.includes('custom-depot-marker')) {
      state.markersLayer.removeLayer(layer);
    }
  });

  state.storeMarkers = {};

  // Group stores to match the exact order in sidebar
  const visibleStores = getVisibleStores();
  const groups = {};
  visibleStores.forEach((st) => {
    if (!groups[st.district]) groups[st.district] = [];
    groups[st.district].push(st);
  });

  let globalIdx = 0;
  Object.keys(groups).forEach((dist) => {
    groups[dist].forEach((st) => {
      globalIdx++;
      const currentIdx = globalIdx;
      const isChecked = state.checkedStoreIds.has(st.id);

      // Marker Color: assigned vehicle color > blue default > grey inactive
      let markerBg = '#007aff';
      if (!isChecked) {
        markerBg = '#8e8e93';
      } else if (state.storeVehicleColorMap[st.id]) {
        markerBg = state.storeVehicleColorMap[st.id];
      }

      const icon = L.divIcon({
        className: `custom-store-marker ${!isChecked ? 'marker-inactive' : ''}`,
        html: `
          <div style="background: ${markerBg}; color: #ffffff; border: 2px solid #ffffff; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);">
            ${currentIdx}
          </div>
        `,
        iconSize: [24, 24],
        iconAnchor: [12, 12]
      });

      // Large batches: keep markers lightweight (layerGroup + no per-marker animation)
      const marker = L.marker([st.lat, st.lon], {
        icon: icon,
        riseOnHover: visibleStores.length <= 50,
        keyboard: false
      }).addTo(state.markersLayer);
      const pkgCount = state.storePkgMap[st.id] || 2;
      marker.bindPopup(`<strong>#${currentIdx} ${st.name}</strong><br><span style="font-size:11px; color:#64748b;">${st.address}</span><br>Kiện: ${pkgCount} (${(pkgCount * 5.0).toFixed(1)} kg)`);

      // Two-way Click on Map -> Scroll to row in sidebar & highlight
      marker.on('click', () => {
        const row = document.querySelector(`.dest-item-row[data-id="${st.id}"]`);
        if (row) {
          row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          row.classList.remove('highlight-flash');
          void row.offsetWidth; // trigger reflow
          row.classList.add('highlight-flash');
        }
      });

      state.storeMarkers[st.id] = marker;
    });
  });
}

function updateStoreMarkerVisual(storeId) {
  const marker = state.storeMarkers[storeId];
  if (!marker) return;

  const isChecked = state.checkedStoreIds.has(storeId);
  const el = marker.getElement();
  if (el) {
    const inner = el.querySelector('div');
    if (isChecked) {
      el.classList.remove('marker-inactive');
      const color = state.storeVehicleColorMap[storeId] || '#007aff';
      if (inner) {
        inner.style.background = color;
        inner.style.borderColor = '#ffffff';
        inner.style.opacity = '1';
        inner.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.25)';
      }
    } else {
      el.classList.add('marker-inactive');
      if (inner) {
        inner.style.background = '#8e8e93';
        inner.style.borderColor = '#d1d1d6';
        inner.style.opacity = '0.55';
        inner.style.boxShadow = 'none';
      }
    }
  }
}

// Click to pick store from map
function handleStorePickFromMap(lat, lon) {
  state.pickingStoreOnMap = false;
  const btn = document.getElementById('btnPickStoreMap');
  if (btn) btn.classList.remove('btn-highlight');

  // Find closest store in allStores
  let closestStore = null;
  let minDistance = Infinity;

  state.allStores.forEach((st) => {
    const dLat = (st.lat - lat) * 111.0;
    const dLon = (st.lon - lon) * 111.0 * Math.cos(lat * Math.PI / 180.0);
    const distKm = Math.sqrt(dLat * dLat + dLon * dLon);
    if (distKm < minDistance) {
      minDistance = distKm;
      closestStore = st;
    }
  });

  if (!closestStore || minDistance > 3.0) {
    showToast('Không tìm thấy cửa hàng gần vị trí bạn vừa nhấp (< 3 km).');
    return;
  }

  // Check if already in displayedStores
  if (!state.displayedStores.some((s) => s.id === closestStore.id)) {
    state.displayedStores.push(closestStore);
  }
  state.removedStoreIds.delete(closestStore.id);
  state.checkedStoreIds.add(closestStore.id);
  state.storePkgMap[closestStore.id] = state.storePkgMap[closestStore.id] || 2;

  renderStoresList();
  renderStoreMarkers();
  updateSummaryMetrics();
  updateRealtimeFeasibility();

  // Scroll to new store
  const row = document.querySelector(`.dest-item-row[data-id="${closestStore.id}"]`);
  if (row) {
    row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    row.classList.add('highlight-flash');
  }

  showToast(`Đã thêm cửa hàng: ${closestStore.name} (${closestStore.district})`);
}

// Expand the K slider/input ceiling automatically for large batches
function updateFleetKControls(suggestedK) {
  state.kMinSuggested = suggestedK;

  const slider = document.getElementById('sliderK');
  const inputK = document.getElementById('inputK');
  const btnApply = document.getElementById('btnApplyKMin');

  const requiredMax = Math.max(50, suggestedK || 1, state.checkedStoreIds.size, state.fleet.k);
  if (slider) slider.max = requiredMax;
  if (inputK) inputK.max = requiredMax;

  if (btnApply) {
    if (suggestedK && suggestedK > 0) {
      btnApply.disabled = false;
      btnApply.textContent = `Đặt K = ${suggestedK} xe`;
    } else {
      btnApply.disabled = true;
      btnApply.textContent = 'Đặt K = ... xe';
    }
  }
}

function applySuggestedK() {
  if (!state.kMinSuggested) return;
  state.fleet.k = state.kMinSuggested;

  const slider = document.getElementById('sliderK');
  if (slider) {
    slider.max = Math.max(parseInt(slider.max, 10) || 50, state.fleet.k);
    slider.value = state.fleet.k;
  }
  const inputK = document.getElementById('inputK');
  if (inputK) inputK.value = state.fleet.k;
  const labelK = document.getElementById('labelK');
  if (labelK) labelK.textContent = `${state.fleet.k} xe`;

  updateRealtimeFeasibility();
  showToast(`Đã đặt K = ${state.fleet.k} xe (K_min tối ưu).`);
}

// --- Real-time Multi-Tier Feasibility Gate ---
function updateRealtimeFeasibility() {
  const banner = document.getElementById('feasibilityBanner');
  const msgEl = document.getElementById('feasibilityMessage');
  const btnDispatch = document.getElementById('btnDispatch');
  const dockBtnDispatch = document.getElementById('dockBtnDispatch');
  const activeStores = getActiveSelectedStores();

  if (activeStores.length < 2) {
    banner.className = 'feasibility-status invalid';
    msgEl.textContent = 'Cần chọn ít nhất 2 cửa hàng để bắt đầu phân bổ.';
    if (btnDispatch) btnDispatch.disabled = true;
    if (dockBtnDispatch) dockBtnDispatch.disabled = true;
    updateFleetKControls(null);
    return;
  }

  const demands = activeStores.map((s) => s.demand_kg);
  const totalDemand = demands.reduce((a, b) => a + b, 0.0);
  const fleetCap = state.fleet.k * state.fleet.q_kg;
  const kMinByWeight = Math.ceil(totalDemand / state.fleet.q_kg);

  // Fast Bin Packing FFD
  const sortedDemands = [...demands].sort((a, b) => b - a);
  const bins = [];
  for (const d of sortedDemands) {
    let placed = false;
    for (let b = 0; b < bins.length; b++) {
      if (bins[b] + d <= state.fleet.q_kg) {
        bins[b] += d;
        placed = true;
        break;
      }
    }
    if (!placed) bins.push(d);
  }

  const kMinSuggested = Math.max(kMinByWeight, bins.length);
  updateFleetKControls(kMinSuggested);

  // Check 1: Overweight item
  const overweightItem = activeStores.find((s) => s.demand_kg > state.fleet.q_kg);
  if (overweightItem) {
    banner.className = 'feasibility-status invalid';
    msgEl.textContent = `'${overweightItem.name}' (${overweightItem.demand_kg}kg) vượt tải trọng xe (${state.fleet.q_kg}kg)!`;
    if (btnDispatch) btnDispatch.disabled = true;
    if (dockBtnDispatch) dockBtnDispatch.disabled = true;
    updateFleetKControls(null);
    return;
  }

  // Check 2: Total capacity
  if (totalDemand > fleetCap) {
    banner.className = 'feasibility-status invalid';
    msgEl.textContent = `Tổng tải (${totalDemand.toFixed(1)}kg) > sức chứa (${fleetCap}kg). Cần ≥ ${kMinByWeight} xe!`;
    if (btnDispatch) btnDispatch.disabled = true;
    if (dockBtnDispatch) dockBtnDispatch.disabled = true;
    return;
  }

  // Check 3: Bin-Packing feasibility
  if (bins.length > state.fleet.k) {
    banner.className = 'feasibility-status invalid';
    msgEl.textContent = `Kẹt tải Bin-Packing: Cần ít nhất ${bins.length} xe để chứa hết kiện hàng!`;
    if (btnDispatch) btnDispatch.disabled = true;
    if (dockBtnDispatch) dockBtnDispatch.disabled = true;
    return;
  }

  // Passed all checks
  banner.className = 'feasibility-status valid';
  const fillPct = ((totalDemand / Math.max(1, fleetCap)) * 100).toFixed(0);
  msgEl.textContent = `Khả thi: ${totalDemand.toFixed(1)}kg / ${fleetCap}kg (${fillPct}%) • K_min = ${kMinSuggested} xe`;

  if (btnDispatch) btnDispatch.disabled = false;
  if (dockBtnDispatch) dockBtnDispatch.disabled = false;
}

// --- Solve & Dispatch Action ---
async function executeDispatch() {
  const activeStores = getActiveSelectedStores();
  if (!state.selectedDepot || activeStores.length < 2) {
    showToast('Vui lòng chọn kho xuất phát và ít nhất 2 cửa hàng!');
    return;
  }

  const btnDispatch = document.getElementById('btnDispatch');
  if (btnDispatch) {
    btnDispatch.disabled = true;
    btnDispatch.innerHTML = '<span class="btn-spinner"></span> Đang giải bài toán...';
  }

  try {
    const payload = {
      depot: {
        id: state.selectedDepot.id,
        lat: state.selectedDepot.lat,
        lon: state.selectedDepot.lon
      },
      stores: activeStores.map((s) => ({
        id: s.id,
        n_pkg: s.n_pkg,
        weight_per_pkg_kg: s.weight_per_pkg_kg
      })),
      fleet: {
        k: state.fleet.k,
        q_kg: state.fleet.q_kg
      },
      options: {
        solver_mode: state.solverMode,
        w_pkg_default_kg: 5.0,
        mode: 'min_distance'
      }
    };

    const res = await fetch('/api/dispatch-motorbike', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    if (!res.ok) {
      const errDetail = data.detail || {};
      const reason = errDetail.reason || 'Lỗi điều phối không xác định.';
      showToast(`Không thể phân bổ: ${reason}`);
      return;
    }

    state.lastDispatchResult = data;
    renderDispatchResults(data);

    // Auto switch to Tab "Lộ trình" for seamless Apple-like UX
    switchTab('routes');

    showToast(`Đã phân bổ ${data.k_used} xe: ${formatNumber(data.total_distance_km, 2, 'km')}`);

  } catch (err) {
    console.error('Dispatch error:', err);
    showToast('Lỗi kết nối đến máy chủ điều phối.');
  } finally {
    if (btnDispatch) {
      btnDispatch.disabled = false;
      btnDispatch.textContent = 'Giải & Phân bổ';
    }
  }
}

// --- Render Single Source of Truth Routes on Map ---
function renderDispatchResults(data) {
  // Update Top Telemetry Capsule
  document.getElementById('hudDistance').textContent = formatNumber(data.total_distance_km, 2, 'km');
  const savingPrefix = data.saving_vs_cw_pct >= 0 ? '+' : '';
  document.getElementById('hudSavings').textContent = `${savingPrefix}${formatNumber(data.saving_vs_cw_pct, 2, '%')}`;
  document.getElementById('hudVehicles').textContent = `${data.k_used} / ${state.fleet.k}`;
  document.getElementById('hudMakespan').textContent = formatNumber(data.makespan_min, 1, 'phút');
  
  const badgeEl = document.getElementById('solverBadge');
  if (badgeEl) {
    badgeEl.textContent = data.solver_name || 'DDQN Cải tiến';
  }

  // Clear previous route polylines & bike markers
  state.routesLayer.clearLayers();
  state.bikesLayer.clearLayers();
  stopSimulation();

  // Map stores to assigned vehicle colors
  state.storeVehicleColorMap = {};
  data.vehicles.forEach((veh) => {
    veh.stops.forEach((stop) => {
      if (stop.type === 'store') {
        state.storeVehicleColorMap[stop.id] = veh.color;
      }
    });
  });

  // Re-render store markers with assigned vehicle colors!
  renderStoreMarkers();

  // Prepare Simulation Paths
  state.simulation.vehiclePaths = {};
  state.simulation.vehicleMarkers = {};
  let maxTime_s = 0;

  const bounds = L.latLngBounds();
  bounds.extend([state.selectedDepot.lat, state.selectedDepot.lon]);

  data.vehicles.forEach((veh) => {
    if (!veh.legs || veh.legs.length === 0) return;

    const fullPolylineCoords = [];
    const pathSegments = [];
    let cumDist = 0;
    let cumTime = 0;

    veh.legs.forEach((leg) => {
      leg.geometry.forEach((pt, pIdx) => {
        fullPolylineCoords.push(pt);
        bounds.extend(pt);

        const stepDist = (pIdx > 0) ? (leg.distance_m / leg.geometry.length) : 0;
        const stepTime = (pIdx > 0) ? (leg.time_s / leg.geometry.length) : 0;
        cumDist += stepDist;
        cumTime += stepTime;

        pathSegments.push({
          lat: pt[0],
          lon: pt[1],
          cum_dist_m: cumDist,
          cum_time_s: cumTime
        });
      });
    });

    state.simulation.vehiclePaths[veh.vehicle_id] = pathSegments;
    if (cumTime > maxTime_s) maxTime_s = cumTime;

    // Draw Route Polyline
    L.polyline(fullPolylineCoords, {
      color: veh.color,
      weight: 4.0,
      opacity: 0.9,
      lineCap: 'round',
      lineJoin: 'round'
    }).addTo(state.routesLayer);

    // Initial Motorcycle Simulation Marker at Depot (Monoline Dot)
    const bikeIcon = L.divIcon({
      className: 'motorbike-sim-marker',
      html: `
        <div style="background: ${veh.color}; width: 22px; height: 22px; border-radius: 50%; border: 2px solid #ffffff; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 700; color: #ffffff; box-shadow: 0 3px 8px rgba(0,0,0,0.3);">
          ${veh.vehicle_id.replace('MOTO_', '')}
        </div>
      `,
      iconSize: [22, 22],
      iconAnchor: [11, 11]
    });

    const marker = L.marker([state.selectedDepot.lat, state.selectedDepot.lon], { icon: bikeIcon })
      .addTo(state.bikesLayer)
      .bindPopup(`<strong>${veh.vehicle_id}</strong><br>Tải trọng: ${veh.load_kg} / ${state.fleet.q_kg} kg`);

    state.simulation.vehicleMarkers[veh.vehicle_id] = marker;
  });

  state.simulation.totalDuration_s = maxTime_s;

  // Zoom map to fit routes
  if (bounds.isValid()) {
    state.map.fitBounds(bounds, { padding: [50, 50] });
  }

  // Render Post-Dispatch UI Panel
  renderPostDispatchPanel(data);

  // Update Q-Values Monoline Bars
  renderQValuesBars(data);

  // Enable Simulation Buttons
  const simBtn = document.getElementById('btnSimToggle');
  if (simBtn) simBtn.disabled = false;
  const dockSimBtn = document.getElementById('dockBtnSimToggle');
  if (dockSimBtn) dockSimBtn.disabled = false;
}

// --- Render Q-Values Monoline Bars ---
function renderQValuesBars(data) {
  const qSection = document.getElementById('qValuesSection');
  if (!qSection) return;

  const qVals = data.q_values || {};
  const statusEl = document.getElementById('qValuesStatus');
  if (statusEl) {
    statusEl.textContent = data.solver_name || 'DDQN Cải tiến';
  }

  const opKeys = ['2-opt', 'intra-relocate', 'inter-relocate', 'inter-swap'];
  const values = opKeys.map((k) => (qVals[k] !== undefined ? qVals[k] : null));

  const validVals = values.filter((v) => v !== null);
  const minVal = validVals.length > 0 ? Math.min(...validVals) : 0;
  const maxVal = validVals.length > 0 ? Math.max(...validVals) : 1;
  const span = Math.max(0.001, maxVal - minVal);

  opKeys.forEach((key, idx) => {
    const val = qVals[key];
    const valEl = document.getElementById(`qVal${idx}`);
    const barEl = document.getElementById(`qBar${idx}`);

    if (valEl && barEl) {
      if (val !== undefined && val !== null) {
        valEl.textContent = val.toFixed(3);
        const pct = Math.round(Math.max(8, Math.min(100, ((val - minVal) / span) * 90 + 10)));
        barEl.style.width = `${pct}%`;
      } else {
        valEl.textContent = '—';
        barEl.style.width = '0%';
      }
    }
  });
}

// --- Post-Dispatch Details & Reasoning ---
function renderPostDispatchPanel(data) {
  const summaryCount = document.getElementById('routesSummaryCount');
  if (summaryCount) {
    summaryCount.textContent = `${data.k_used} / ${state.fleet.k} xe`;
  }

  // Mini CSS Bar Chart for Cargo
  const chartContainer = document.getElementById('cargoBarsContainer');
  if (chartContainer) {
    chartContainer.innerHTML = '';
    data.vehicles.forEach((veh) => {
      const barRow = document.createElement('div');
      barRow.className = 'cargo-bar-row';
      const fillPct = Math.min(100, (veh.load_kg / state.fleet.q_kg) * 100);

      barRow.innerHTML = `
        <div class="cargo-bar-meta">
          <span style="color: ${veh.color}; font-weight: 600;">${veh.vehicle_id}</span>
          <span class="mono">${formatNumber(veh.load_kg, 1)} / ${state.fleet.q_kg} kg (${fillPct.toFixed(0)}%)</span>
        </div>
        <div class="cargo-track">
          <div class="cargo-fill" style="width: ${fillPct}%; background-color: ${veh.color};"></div>
        </div>
      `;
      chartContainer.appendChild(barRow);
    });
  }

  // Table Breakdown
  const tableBody = document.getElementById('fleetTableBody');
  if (tableBody) {
    tableBody.innerHTML = '';
    data.vehicles.forEach((veh) => {
      const tr = document.createElement('tr');
      const isIdle = (veh.load_kg === 0 && veh.load_pkgs === 0);
      const storeStops = veh.stops.filter((s) => s.type === 'store');

      if (isIdle) {
        tr.style.opacity = '0.45';
        tr.innerHTML = `
          <td><span class="veh-dot" style="background: ${veh.color};"></span>${veh.vehicle_id}</td>
          <td><span style="font-size: 11px; color: #8e8e93;">Chờ lệnh</span></td>
          <td class="mono">0.0 kg</td>
          <td class="mono">—</td>
        `;
      } else {
        tr.innerHTML = `
          <td><span class="veh-dot" style="background: ${veh.color};"></span>${veh.vehicle_id}</td>
          <td>${storeStops.length} điểm (${veh.load_pkgs} kiện)</td>
          <td class="mono">${formatNumber(veh.load_kg, 1)} kg</td>
          <td class="mono">${formatNumber(veh.distance_km, 2, 'km')}</td>
        `;
      }
      tableBody.appendChild(tr);
    });
  }

  // Reasoning List
  const reasoningContainer = document.getElementById('fleetReasoningContainer');
  if (reasoningContainer) {
    reasoningContainer.innerHTML = '';
    data.vehicles.forEach((veh) => {
      const item = document.createElement('div');
      item.className = 'reasoning-item';
      item.innerHTML = `
        <div class="reasoning-veh-tag" style="color: ${veh.color}; font-weight: 600;">${veh.vehicle_id}:</div>
        <div class="reasoning-text">${veh.reasoning}</div>
      `;
      reasoningContainer.appendChild(item);
    });
  }
}

// --- Head-to-Head Benchmark Tab Engine ---
async function runHeadToHeadBenchmark() {
  const tableBody = document.getElementById('benchmarkTableBody');
  if (!tableBody) return;

  const activeStores = getActiveSelectedStores();
  if (!state.selectedDepot || activeStores.length < 2) {
    tableBody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align: center; color: #8e8e93; padding: 24px 0;">
          Vui lòng chọn ít nhất 2 điểm giao ở tab <strong>Điều phối</strong> để đối chiếu.
        </td>
      </tr>
    `;
    return;
  }

  tableBody.innerHTML = `
    <tr>
      <td colspan="5" style="text-align: center; color: #8e8e93; padding: 24px 0;">
        <span class="btn-spinner"></span> Đang đo đạc và đối chiếu 3 mô hình...
      </td>
      </tr>
    `;

    const models = [
      { id: 'cw_greedy', name: 'CW + Greedy' },
      { id: 'ddqn_base', name: 'DDQN Gốc' },
      { id: 'neural_lns', name: 'Neural LNS' }
    ];

    try {
      const promises = models.map(async (m) => {
        const t0 = performance.now();
        const payload = {
          depot: {
            id: state.selectedDepot.id,
            lat: state.selectedDepot.lat,
            lon: state.selectedDepot.lon
          },
          stores: activeStores.map((s) => ({
            id: s.id,
            n_pkg: s.n_pkg,
            weight_per_pkg_kg: s.weight_per_pkg_kg
          })),
          fleet: {
            k: state.fleet.k,
            q_kg: state.fleet.q_kg
          },
          options: {
            solver_mode: m.id,
            w_pkg_default_kg: 5.0,
            mode: 'min_distance'
          }
        };

        const res = await fetch('/api/dispatch-motorbike', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        const latency_ms = Math.round(performance.now() - t0);
        return { id: m.id, name: m.name, data, latency_ms, ok: res.ok };
      });

      const results = await Promise.all(promises);
      state.benchmarkResults = results;

    const successful = results.filter((r) => r.ok && r.data && r.data.total_distance_km);
    if (successful.length === 0) {
      tableBody.innerHTML = `
        <tr>
          <td colspan="5" style="text-align: center; color: #ff3b30; padding: 24px 0;">
            Không thể lấy kết quả đối chiếu từ máy chủ.
          </td>
        </tr>
      `;
      return;
    }

    const minDistance = Math.min(...successful.map((r) => r.data.total_distance_km));

    tableBody.innerHTML = '';
    results.forEach((r) => {
      const tr = document.createElement('tr');
      if (!r.ok || !r.data) {
        tr.innerHTML = `
          <td><strong>${r.name}</strong></td>
          <td colspan="4" style="color: #ff3b30;">Lỗi xử lý</td>
        `;
        tableBody.appendChild(tr);
        return;
      }

      const dist = r.data.total_distance_km;
      const isBest = (dist === minDistance);
      const savings = r.data.saving_vs_cw_pct;
      const savingPrefix = savings >= 0 ? '+' : '';

      tr.innerHTML = `
        <td>
          <div style="font-weight: ${isBest ? '600' : '400'}; color: ${isBest ? '#007aff' : '#1d1d1f'};">
            ${r.name} ${isBest ? '★' : ''}
          </div>
        </td>
        <td class="mono" style="font-weight: ${isBest ? '600' : '400'};">${dist.toFixed(2)} km</td>
        <td class="mono" style="color: ${savings > 0 ? '#34c759' : '#8e8e93'};">${savingPrefix}${savings.toFixed(1)}%</td>
        <td class="mono">${r.data.k_used} xe</td>
        <td class="mono">${r.latency_ms} ms</td>
      `;
      tableBody.appendChild(tr);
    });

  } catch (err) {
    console.error('Benchmark error:', err);
    tableBody.innerHTML = `
      <tr>
        <td colspan="5" style="text-align: center; color: #ff3b30; padding: 24px 0;">
          Lỗi mạng trong quá trình đo đạc đối chiếu.
        </td>
      </tr>
    `;
  }
}

// --- Tab Switching Logic ---
function switchTab(targetTab) {
  // Update Tabs
  document.querySelectorAll('#mainTabs .tab-item').forEach((tab) => {
    if (tab.getAttribute('data-tab') === targetTab) {
      tab.classList.add('active');
    } else {
      tab.classList.remove('active');
    }
  });

  // Update Panes
  const panes = {
    dispatch: document.getElementById('tabPaneDispatch'),
    routes: document.getElementById('tabPaneRoutes'),
    benchmark: document.getElementById('tabPaneBenchmark')
  };

  Object.keys(panes).forEach((key) => {
    if (panes[key]) {
      if (key === targetTab) {
        panes[key].classList.add('active');
      } else {
        panes[key].classList.remove('active');
      }
    }
  });

  // Action on tab activation
  if (targetTab === 'benchmark') {
    runHeadToHeadBenchmark();
  }
}

// --- Simulation Engine ---
function toggleSimulation() {
  if (state.simulation.isPlaying) {
    pauseSimulation();
  } else {
    startSimulation();
  }
}

function startSimulation() {
  if (!state.lastDispatchResult) return;
  state.simulation.isPlaying = true;
  const btn = document.getElementById('btnSimToggle');
  if (btn) btn.textContent = '⏸ Tạm dừng';
  const dockLabel = document.getElementById('dockSimLabel');
  if (dockLabel) dockLabel.textContent = 'Tạm dừng';

  let lastTimestamp = performance.now();

  function stepAnimation(currentTimestamp) {
    if (!state.simulation.isPlaying) return;

    const deltaMs = currentTimestamp - lastTimestamp;
    lastTimestamp = currentTimestamp;

    const deltaSeconds = (deltaMs / 1000.0) * state.simulation.speedMultiplier * 20.0;
    state.simulation.elapsedSeconds += deltaSeconds;

    Object.keys(state.simulation.vehiclePaths).forEach((vehId) => {
      const path = state.simulation.vehiclePaths[vehId];
      const marker = state.simulation.vehicleMarkers[vehId];
      if (!path || path.length === 0 || !marker) return;

      const currentTargetTime = state.simulation.elapsedSeconds;
      const pos = interpolatePositionAlongPath(path, currentTargetTime);
      marker.setLatLng([pos.lat, pos.lon]);
    });

    if (state.simulation.elapsedSeconds >= state.simulation.totalDuration_s) {
      pauseSimulation();
      state.simulation.elapsedSeconds = 0;
      showToast('Đội xe máy đã hoàn tất toàn bộ lộ trình giao hàng.');
      return;
    }

    state.simulation.animFrameId = requestAnimationFrame(stepAnimation);
  }

  state.simulation.animFrameId = requestAnimationFrame(stepAnimation);
  showToast('Đang mô phỏng xe di chuyển theo tim đường...');
}

function pauseSimulation() {
  state.simulation.isPlaying = false;
  if (state.simulation.animFrameId) {
    cancelAnimationFrame(state.simulation.animFrameId);
    state.simulation.animFrameId = null;
  }
  const btn = document.getElementById('btnSimToggle');
  if (btn) btn.textContent = '▶ Chạy xe';
  const dockLabel = document.getElementById('dockSimLabel');
  if (dockLabel) dockLabel.textContent = 'Chạy xe';
}

function stopSimulation() {
  pauseSimulation();
  state.simulation.elapsedSeconds = 0;
}

function interpolatePositionAlongPath(path, t_seconds) {
  if (t_seconds <= 0) return path[0];
  if (t_seconds >= path[path.length - 1].cum_time_s) return path[path.length - 1];

  for (let i = 0; i < path.length - 1; i++) {
    const p1 = path[i];
    const p2 = path[i + 1];
    if (t_seconds >= p1.cum_time_s && t_seconds <= p2.cum_time_s) {
      const segDuration = p2.cum_time_s - p1.cum_time_s;
      const ratio = segDuration > 0 ? (t_seconds - p1.cum_time_s) / segDuration : 0;
      return {
        lat: p1.lat + ratio * (p2.lat - p1.lat),
        lon: p1.lon + ratio * (p2.lon - p1.lon)
      };
    }
  }
  return path[path.length - 1];
}

// --- Setup All Event Listeners ---
function setupEventListeners() {
  // Sidebar Toggle
  const toggleBtn = document.getElementById('sidebarToggle');
  const closeBtn = document.getElementById('sidebarCloseBtn');
  const sidebar = document.getElementById('sidebar');

  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      toggleBtn.classList.toggle('collapsed');
    });
  }

  if (closeBtn && sidebar) {
    closeBtn.addEventListener('click', () => {
      sidebar.classList.add('collapsed');
      if (toggleBtn) toggleBtn.classList.add('collapsed');
    });
  }

  // Underline Navigation Tabs (Tầng 1)
  document.querySelectorAll('#mainTabs .tab-item').forEach((tab) => {
    tab.addEventListener('click', () => {
      const target = tab.getAttribute('data-tab');
      switchTab(target);
    });
  });

  // Model Segmented Control (Tầng 2 - 3 Models)
  const segmentBtns = document.querySelectorAll('#modelSegmentedControl .segment-btn');
  segmentBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      segmentBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.solverMode = btn.getAttribute('data-model');
      
      const badge = document.getElementById('solverBadge');
      if (badge) badge.textContent = btn.textContent;

      showToast(`Chuyển sang mô hình: ${btn.textContent}`);

      // Auto re-dispatch if valid selection is already ready
      if (getActiveSelectedStores().length >= 2) {
        executeDispatch();
      }
    });
  });

  // Re-run Benchmark CTA
  const btnReRunBench = document.getElementById('btnReRunBenchmark');
  if (btnReRunBench) {
    btnReRunBench.addEventListener('click', () => {
      runHeadToHeadBenchmark();
    });
  }

  // Pick Depot Map Button
  const btnPickDepot = document.getElementById('btnPickDepotMap');
  if (btnPickDepot) {
    btnPickDepot.addEventListener('click', () => {
      state.pickingDepotOnMap = !state.pickingDepotOnMap;
      state.pickingStoreOnMap = false;
      const btnPickStore = document.getElementById('btnPickStoreMap');
      if (btnPickStore) btnPickStore.classList.remove('btn-highlight');

      if (state.pickingDepotOnMap) {
        btnPickDepot.classList.add('btn-highlight');
        showToast('Nhấp vào một vị trí trên bản đồ để đặt kho xuất phát...');
      } else {
        btnPickDepot.classList.remove('btn-highlight');
      }
    });
  }

  // Pick Store Map Button
  const btnPickStore = document.getElementById('btnPickStoreMap');
  if (btnPickStore) {
    btnPickStore.addEventListener('click', () => {
      state.pickingStoreOnMap = !state.pickingStoreOnMap;
      state.pickingDepotOnMap = false;
      const btnPickDep = document.getElementById('btnPickDepotMap');
      if (btnPickDep) btnPickDep.classList.remove('btn-highlight');

      if (state.pickingStoreOnMap) {
        btnPickStore.classList.add('btn-highlight');
        showToast('Nhấp vào một vị trí trên bản đồ để thêm cửa hàng gần nhất...');
      } else {
        btnPickStore.classList.remove('btn-highlight');
      }
    });
  }

  // Multi-District Toggle Button
  const btnToggleMulti = document.getElementById('btnToggleMultiDistrict');
  if (btnToggleMulti) {
    btnToggleMulti.addEventListener('click', () => {
      state.isMultiDistrict = !state.isMultiDistrict;
      const label = document.getElementById('multiDistrictLabel');
      if (label) label.textContent = state.isMultiDistrict ? 'Đa quận: BẬT' : 'Đa quận: TẮT';
      btnToggleMulti.classList.toggle('active', state.isMultiDistrict);

      // Re-render district chips with updated multi-district logic
      renderDistrictChips();
      updateLoadSizeControls();

      showToast(state.isMultiDistrict ? 'Đã bật chế độ gộp nhiều quận.' : 'Đã tắt chế độ gộp (Chỉ chọn 1 quận duy nhất).');
    });
  }

  // Select All Stores Button (selects all visible / search results, no cap)
  const btnSelectAll = document.getElementById('btnSelectAllStores');
  if (btnSelectAll) {
    btnSelectAll.addEventListener('click', () => {
      const visibleStores = getVisibleStores();
      visibleStores.forEach((st) => {
        state.checkedStoreIds.add(st.id);
        state.removedStoreIds.delete(st.id);
        if (!state.displayedStores.some((s) => s.id === st.id)) {
          state.displayedStores.push(st);
        }
      });
      renderStoresList();
      renderStoreMarkers();
      updateSummaryMetrics();
      updateRealtimeFeasibility();
      fitMapToSelection();
      showToast(`Đã kích hoạt toàn bộ ${visibleStores.length} điểm đang hiển thị.`);
    });
  }

  // Deselect All Stores Button
  const btnDeselectAll = document.getElementById('btnDeselectAllStores');
  if (btnDeselectAll) {
    btnDeselectAll.addEventListener('click', () => {
      state.checkedStoreIds.clear();
      renderStoresList();
      renderStoreMarkers();
      updateSummaryMetrics();
      updateRealtimeFeasibility();
      showToast('Đã bỏ kích hoạt tất cả cửa hàng.');
    });
  }

  // Flexible Load-Size Chips (6 / 15 / 30 / 50 / All)
  document.querySelectorAll('.load-size-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      state.loadSizeMode = chip.getAttribute('data-count');
      document.querySelectorAll('.load-size-chip').forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');

      const inputLoad = document.getElementById('inputLoadCount');
      if (inputLoad) {
        inputLoad.value = state.loadSizeMode === 'all' ? '' : state.loadSizeMode;
      }
    });
  });

  // Custom load count input
  const inputLoadCount = document.getElementById('inputLoadCount');
  if (inputLoadCount) {
    inputLoadCount.addEventListener('input', () => {
      if (inputLoadCount.value.trim() !== '') {
        state.loadSizeMode = 'custom';
        document.querySelectorAll('.load-size-chip').forEach((c) => c.classList.remove('active'));
      }
    });
    inputLoadCount.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        const btnLoad = document.getElementById('btnLoadStores');
        if (btnLoad) btnLoad.click();
      }
    });
  }

  // Load exactly the selected number of stores (or the full pool)
  const btnLoad = document.getElementById('btnLoadStores');
  if (btnLoad) {
    btnLoad.addEventListener('click', () => {
      const count = getLoadCountValue();
      if (count === null) {
        showToast('Vui lòng chọn hoặc nhập số lượng điểm hợp lệ (>= 1).');
        return;
      }
      loadStoresFromPool(count);
    });
  }

  // Full Store Picker Search (debounced)
  const storeSearchInput = document.getElementById('storeSearchInput');
  if (storeSearchInput) {
    let searchTimer = null;
    storeSearchInput.addEventListener('input', () => {
      if (searchTimer) clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        state.searchQuery = storeSearchInput.value;
        renderStoresList();
        renderStoreMarkers();
        updateSummaryMetrics();
        fitMapToSelection();
      }, 120);
    });
  }

  // Clear Stores Button
  const btnClear = document.getElementById('btnClearStores');
  if (btnClear) {
    btnClear.addEventListener('click', () => {
      state.checkedStoreIds.clear();
      state.removedStoreIds.clear();
      state.displayedStores = [];
      state.storeVehicleColorMap = {};
      state.searchQuery = '';
      const searchInput = document.getElementById('storeSearchInput');
      if (searchInput) searchInput.value = '';
      renderStoresList();
      renderStoreMarkers();
      updateSummaryMetrics();
      updateRealtimeFeasibility();
      state.routesLayer.clearLayers();
      state.bikesLayer.clearLayers();
      stopSimulation();
      showToast('Đã làm sạch toàn bộ danh sách cửa hàng.');
    });
  }

  // Fleet K Slider + Direct Numeric Input (auto-expanded ceiling)
  const sliderK = document.getElementById('sliderK');
  const inputK = document.getElementById('inputK');
  const labelK = document.getElementById('labelK');

  function syncFleetK(value) {
    const parsed = parseInt(value, 10);
    if (!Number.isFinite(parsed) || parsed < 1) return;
    state.fleet.k = parsed;
    if (sliderK) {
      sliderK.max = Math.max(parseInt(sliderK.max, 10) || 50, state.fleet.k);
      sliderK.value = state.fleet.k;
    }
    if (inputK) {
      inputK.max = Math.max(parseInt(inputK.max, 10) || 50, state.fleet.k);
      inputK.value = state.fleet.k;
    }
    if (labelK) labelK.textContent = `${state.fleet.k} xe`;
    updateRealtimeFeasibility();
  }

  if (sliderK) {
    sliderK.addEventListener('input', () => syncFleetK(sliderK.value));
  }
  if (inputK) {
    inputK.addEventListener('change', () => syncFleetK(inputK.value));
    inputK.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') syncFleetK(inputK.value);
    });
  }

  // One-click K_min suggestion
  const btnApplyKMin = document.getElementById('btnApplyKMin');
  if (btnApplyKMin) {
    btnApplyKMin.addEventListener('click', applySuggestedK);
  }

  // Capacity Segmented Control Buttons
  document.querySelectorAll('.cap-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.cap-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      state.fleet.q_kg = parseInt(btn.getAttribute('data-cap'));
      const labelQ = document.getElementById('labelQ');
      if (labelQ) labelQ.textContent = `${state.fleet.q_kg} kg`;
      updateRealtimeFeasibility();
    });
  });

  // Dispatch CTA
  const btnDispatch = document.getElementById('btnDispatch');
  if (btnDispatch) {
    btnDispatch.addEventListener('click', executeDispatch);
  }

  const dockBtnDispatch = document.getElementById('dockBtnDispatch');
  if (dockBtnDispatch) {
    dockBtnDispatch.addEventListener('click', executeDispatch);
  }

  // Simulation Controls
  const btnSimToggle = document.getElementById('btnSimToggle');
  if (btnSimToggle) {
    btnSimToggle.addEventListener('click', toggleSimulation);
  }

  const btnSimStop = document.getElementById('btnSimStop');
  if (btnSimStop) {
    btnSimStop.addEventListener('click', stopSimulation);
  }

  const dockBtnSimToggle = document.getElementById('dockBtnSimToggle');
  if (dockBtnSimToggle) {
    dockBtnSimToggle.addEventListener('click', toggleSimulation);
  }

  // Simulation Speed Pills (1x, 2x, 5x)
  document.querySelectorAll('.speed-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.speed-pill').forEach((p) => p.classList.remove('active'));
      pill.classList.add('active');
      state.simulation.speedMultiplier = parseInt(pill.getAttribute('data-speed'));
      showToast(`Tốc độ mô phỏng: ${state.simulation.speedMultiplier}x`);
    });
  });

  // Dock Reset Button
  const dockBtnReset = document.getElementById('dockBtnReset');
  if (dockBtnReset) {
    dockBtnReset.addEventListener('click', () => {
      location.reload();
    });
  }

  // Export JSON Report Button
  const btnExport = document.getElementById('btnExportReport');
  if (btnExport) {
    btnExport.addEventListener('click', () => {
      if (!state.lastDispatchResult) return;
      const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(state.lastDispatchResult, null, 2));
      const a = document.createElement('a');
      a.setAttribute('href', dataStr);
      a.setAttribute('download', `mp_moto_cvrp_report_${Date.now()}.json`);
      document.body.appendChild(a);
      a.click();
      a.remove();
      showToast('Đã tải xuống báo cáo kỹ thuật JSON.');
    });
  }
}

// --- Toast Notification ---
let toastTimer = null;
function showToast(msg) {
  const toast = document.getElementById('deliveryToast');
  const msgEl = document.getElementById('toastMessage');
  if (!toast || !msgEl) return;

  msgEl.textContent = msg;
  toast.classList.remove('hidden');

  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.add('hidden');
  }, 3500);
}
