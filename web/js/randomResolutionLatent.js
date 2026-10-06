// ComfyUI-RandomResolutionLatent · 前端扩展
//
// 做三件事（坑都是 TagPromptEditor 项目里踩平的，注释保留关键的）：
//  1. 给所有 widget 挂中文 label（canvas 绘制用 w.label ?? w.name，Python 侧不动名字）
//  2. 隐藏 seed / control_after_generate（官方 hidden 机制，序列化不受影响，
//     旧工作流 widgets_values 位置数组一个位都不动）
//  3. 自定义分辨率改成「格子模式」：原生 STRING 退居幕后当唯一真相源（隐藏但仍
//     序列化），DOM 格子编辑器负责看和改。底层存储仍是每行一个 1024x1024，
//     旧工作流 / API 格式完全兼容。

import { app } from "../../../scripts/app.js";

const NODE_NAMES = new Set(["ResolutionScheduler", "ResolutionPoolPreview"]);

// ---------------------------------------------------------------------------
// 中文显示名（只改 label，不改 name —— name 是序列化与 INPUT_TYPES 的键）
// ---------------------------------------------------------------------------
const LABELS = {
  pool_source: "分辨率来源",
  preset_group: "预设桶",
  custom_resolutions: "自定义分辨率",
  megapixels: "目标像素 (MP)",
  aspect_ratios: "宽高比列表",
  multiple_of: "对齐倍数",
  pick_mode: "挑选模式",
  start_index: "起始序号",
  batch_size: "单次张数 (batch)",
  latent_format: "Latent 通道",
  seed: "随机种子",
  advance_each_run: "每次排队换尺寸",
};

// ---------------------------------------------------------------------------
// 样式
// ---------------------------------------------------------------------------
let styleInjected = false;
function injectStyle() {
  if (styleInjected && document.getElementById("rrl-style")) return;
  const link = document.createElement("link");
  link.id = "rrl-style";
  link.rel = "stylesheet";
  link.href = new URL("../css/tiles.css", import.meta.url).href;
  document.head.appendChild(link);
  styleInjected = true;
}

// ---------------------------------------------------------------------------
// ⚠️ 关键补丁：把 widget.width 钉死成 undefined。
// 前端 1.53 的 WidgetLegacy 会在 draw() 里把 widget.width 写成当时的画布宽度，
// 而 DOM widget 容器宽度「widget.width 优先于 node.width」，被写脏后节点拉宽
// 下半 UI 不跟。恢复「从未设置」语义即可。
// ---------------------------------------------------------------------------
function lockWidgetWidth(widget) {
  if (!widget) return;
  try {
    Object.defineProperty(widget, "width", {
      configurable: true,
      enumerable: false,
      get: () => undefined,
      set: () => {},
    });
  } catch (e) {
    console.warn("[RandomResolutionLatent] 无法锁定 widget.width", e);
  }
}

// 隐藏原生 widget：不改动 widgets 数组的长度与顺序，只让它不画、不占高度。
// 序列化照旧（widget.value 仍随工作流保存），旧工作流零影响。
function hideWidget(widget) {
  if (!widget) return;
  widget.hidden = true;
  widget.computeSize = () => [0, -4];
}

// ---------------------------------------------------------------------------
// 尺寸解析（与后端 resolution_pool.py 的规则对齐）
// ---------------------------------------------------------------------------
const TOKEN_RE = /(\d{1,5})\s*([x×X*,，]|[:：])\s*(\d{1,5})/g;
const MIN_SIDE = 64;
const MAX_SIDE = 8192;
const STEP = 8;

function normalizeSide(value) {
  const snapped = Math.round(value / STEP) * STEP;
  return Math.max(MIN_SIDE, Math.min(MAX_SIDE, snapped));
}

function normalizeSize(w, h) {
  return { w: normalizeSide(w), h: normalizeSide(h) };
}

function megapixelsOf(node) {
  const w = node.widgets?.find((x) => x.name === "megapixels");
  const v = Number(w?.value);
  return Number.isFinite(v) && v > 0 ? v : 1.0;
}

/** 文本 -> [{w,h}]。支持 1024x1024 / 1024*1024 / 1024,1024 / 16:9（按目标像素换算），# 注释。 */
function parseSizes(text, mp) {
  const out = [];
  const seen = new Set();
  const clean = String(text || "")
    .split("\n")
    .map((l) => l.split("#")[0])
    .join("\n");
  TOKEN_RE.lastIndex = 0;
  let m;
  while ((m = TOKEN_RE.exec(clean))) {
    const a = parseInt(m[1], 10);
    const b = parseInt(m[3], 10);
    let size;
    if (m[2] === ":" || m[2] === "：") {
      // 比例 -> 按目标像素换算（与后端 size_from_aspect 一致）
      const ratio = Math.max(1e-6, a / Math.max(1, b));
      const total = Math.max(0.01, mp) * 1_000_000;
      size = normalizeSize(Math.sqrt(total * ratio), Math.sqrt(total / ratio));
    } else {
      size = normalizeSize(a, b);
    }
    const key = `${size.w}x${size.h}`;
    if (!seen.has(key)) {
      seen.add(key);
      out.push(size);
    }
  }
  return out;
}

function tilesToText(tiles) {
  return tiles.map((t) => `${t.w}x${t.h}`).join("\n");
}

// ---------------------------------------------------------------------------
// 格子渲染 / 高度 / 提交 —— 全部状态变化只走这一条路径
// （「同一份状态被两处渲染必须走同一条更新路径」，别再拆成两份实现）
// ---------------------------------------------------------------------------

function renderTiles(st) {
  const grid = st.gridEl;
  grid.textContent = "";
  st.countEl.textContent = st.tiles.length ? `· ${st.tiles.length} 种` : "";
  if (!st.tiles.length) {
    const empty = document.createElement("span");
    empty.className = "rrl-empty";
    empty.textContent = "还没有分辨率 —— 在下面输入宽高后点「添加」";
    grid.appendChild(empty);
  }
  st.tiles.forEach((t, i) => {
    const tile = document.createElement("span");
    tile.className = "rrl-tile";
    tile.title = `${t.w}×${t.h} · ${((t.w * t.h) / 1e6).toFixed(2)}MP`;
    const label = document.createElement("span");
    label.className = "rrl-tile-text";
    label.textContent = `${t.w}×${t.h}`;
    const del = document.createElement("button");
    del.type = "button";
    del.textContent = "×";
    del.title = "移除";
    del.addEventListener("click", () => {
      st.tiles.splice(i, 1);
      commitTiles(st);
    });
    tile.append(label, del);
    grid.appendChild(tile);
  });
}

function syncTilesHeight(st) {
  // 测 inner（内容），不是被 ComfyUI 强制设高的 wrap 外层
  const h = Math.max(96, Math.ceil(st.inner.getBoundingClientRect().height));
  if (Math.abs(h - st.height) < 2) return;
  const delta = h - st.height;
  st.height = h;
  if (st.node.size) {
    // 增量调整：用户手动拉过的高度不被复位
    st.node.setSize([st.node.size[0], Math.max(160, st.node.size[1] + delta)]);
  }
  st.node.graph?.setDirtyCanvas(true, false);
}

function commitTiles(st) {
  const value = tilesToText(st.tiles);
  const graph = st.node.graph;
  graph?.beforeChange?.();
  st.syncing = true;
  st.native.value = value;
  st.lastSynced = value;
  try {
    st.native.callback?.(value);
  } catch (e) {
    console.error("[RandomResolutionLatent] callback failed", e);
  }
  st.syncing = false;
  graph?.afterChange?.();
  renderTiles(st);
  syncTilesHeight(st);
  st.node.graph?.setDirtyCanvas(true, false);
}

/** 从 native 文本重解析（加载旧工作流 / 撤销后调用）。只在值真的变了时动 UI。 */
function pullFromNative(st) {
  if (st.native.value === st.lastSynced) return;
  st.tiles = parseSizes(st.native.value, megapixelsOf(st.node));
  st.lastSynced = st.native.value;
  renderTiles(st);
  syncTilesHeight(st);
}

// ---------------------------------------------------------------------------
// 格子编辑器装配
// ---------------------------------------------------------------------------
function setupTiles(node) {
  if (node.__rrl) return;
  const native = node.widgets?.find((w) => w.name === "custom_resolutions");
  if (!native) return;

  const st = {
    node,
    native,
    tiles: [],
    syncing: false,
    lastSynced: null,
    height: 132,       // DOM widget 当前高度（量出来后增量修正）
    inner: null,
    gridEl: null,
    countEl: null,
    wInput: null,
    hInput: null,
    observer: null,
  };
  node.__rrl = st;

  // ---- DOM 结构 ----
  const wrap = document.createElement("div");
  wrap.className = "rrl-wrap";
  const inner = document.createElement("div");
  inner.className = "rrl-inner";
  inner.innerHTML = `
    <div class="rrl-head">
      <span class="rrl-title">自定义分辨率</span>
      <span class="rrl-count"></span>
      <button class="rrl-clear" type="button" title="移除全部格子">清空</button>
    </div>
    <div class="rrl-grid"></div>
    <div class="rrl-addrow">
      <input class="rrl-in rrl-w" type="text" spellcheck="false" placeholder="宽 / 可粘贴列表" />
      <span class="rrl-sep">×</span>
      <input class="rrl-in rrl-h" type="text" spellcheck="false" placeholder="高" />
      <button class="rrl-add" type="button">添加</button>
    </div>
    <div class="rrl-hint">支持粘贴 1024x1024, 1152x896… 或宽框里写比例 16:9（按目标像素换算）；自动对齐到 8</div>`;
  wrap.appendChild(inner);

  st.inner = inner;
  st.gridEl = inner.querySelector(".rrl-grid");
  st.countEl = inner.querySelector(".rrl-count");
  st.wInput = inner.querySelector(".rrl-w");
  st.hInput = inner.querySelector(".rrl-h");

  const domWidget = node.addDOMWidget("rrl_tiles", "自定义分辨率", wrap, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => st.height,
    getMaxHeight: () => st.height,
    getHeight: () => st.height,
  });
  lockWidgetWidth(domWidget);
  // 别让输入/点击漏给画布（否则拖滑块会连节点一起拖走、滚轮会缩放画布）
  for (const ev of ["pointerdown", "mousedown", "wheel"]) {
    wrap.addEventListener(ev, (e) => e.stopPropagation(), { passive: ev === "wheel" ? false : true });
  }
  wrap.addEventListener("contextmenu", (e) => e.stopPropagation());

  // ⚠️ DOM widget 必须留在 widgets 末尾，不能挪到 custom_resolutions 原位置：
  // 实测（前端 1.53）加载工作流时按位置给**每个** widget 无条件喂 widgets_values，
  // serialize:false 也照样消耗一个值——插在中间会让后面所有 widget 串位。
  // 放在末尾吃掉的是不存在的第 N+1 个值，无害（TagPromptEditor 同款做法）。

  // 原生文本框退居幕后：隐藏但仍序列化（唯一真相源）
  hideWidget(native);

  // ---- 添加 / 清空 ----
  function addFromInputs() {
    const wRaw = (st.wInput.value || "").trim();
    const hRaw = (st.hInput.value || "").trim();
    if (!wRaw && !hRaw) return;
    let incoming;
    if (/[x×X*,，:：]/.test(wRaw)) {
      // 宽输入框里粘了一段列表（或写了 16:9）-> 整段解析
      incoming = parseSizes(wRaw, megapixelsOf(node));
    } else {
      const w = parseInt(wRaw, 10);
      const h = parseInt(hRaw, 10);
      if (!w || !h) return;
      incoming = [normalizeSize(w, h)];
    }
    if (!incoming.length) return;
    const seen = new Set(st.tiles.map((t) => `${t.w}x${t.h}`));
    let added = 0;
    for (const s of incoming) {
      const key = `${s.w}x${s.h}`;
      if (seen.has(key)) continue;
      seen.add(key);
      st.tiles.push(s);
      added++;
    }
    st.wInput.value = "";
    st.hInput.value = "";
    if (added) commitTiles(st);
  }

  inner.querySelector(".rrl-add").addEventListener("click", addFromInputs);
  for (const el of [st.wInput, st.hInput]) {
    el.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        e.stopPropagation();
        addFromInputs();
      }
    });
  }
  inner.querySelector(".rrl-clear").addEventListener("click", () => {
    if (!st.tiles.length) return;
    st.tiles = [];
    commitTiles(st);
  });

  // ---- 外部改动（撤销 / 手写脚本改值）-> 重解析 ----
  const originCallback = native.callback;
  native.callback = function (value, ...rest) {
    const ret = originCallback?.apply(this, [value, ...rest]);
    if (!st.syncing) pullFromNative(st);
    return ret;
  };

  // ---- 高度监听 ----
  st.observer = new ResizeObserver(() => syncTilesHeight(st));
  st.observer.observe(st.gridEl);

  // ---- 初始 ----
  st.tiles = parseSizes(st.native.value, megapixelsOf(node));
  st.lastSynced = st.native.value;
  renderTiles(st);
  syncTilesHeight(st);
  // 宽度兜住（高度由 syncTilesHeight 增量修正，不动这里）
  node.setSize([Math.max(node.size[0], 430), node.size[1]]);

  // 旧工作流加载时 widgets_values 在 onConfigure 之后才到位，onConfigure 里会再补一次
}

// ---------------------------------------------------------------------------
// 节点初始化：中文 label + 隐藏 seed + 格子编辑器
// ---------------------------------------------------------------------------
function setupNode(node) {
  // 1. 中文 label
  for (const w of node.widgets || []) {
    if (LABELS[w.name]) w.label = LABELS[w.name];
  }
  // 2. seed 对用户没有意义：隐藏（序列化保留，旧工作流不串位）
  hideWidget(node.widgets?.find((w) => w.name === "seed"));
  hideWidget(node.widgets?.find((w) => w.name === "control_after_generate"));
  // 3. 格子编辑器
  setupTiles(node);
}

app.registerExtension({
  name: "RandomResolutionLatent.UI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODE_NAMES.has(nodeData?.name)) return;
    injectStyle();

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const ret = onNodeCreated?.apply(this, arguments);
      try {
        setupNode(this);
      } catch (e) {
        // 前端出错就退回原生文本框，功能降级但不影响出图
        console.error("[RandomResolutionLatent] 初始化失败，已降级", e);
      }
      return ret;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const ret = onConfigure?.apply(this, arguments);
      const node = this;
      setTimeout(() => pullFromNative(node.__rrl), 0);
      return ret;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      const st = this.__rrl;
      if (st) {
        st.observer?.disconnect();
        this.__rrl = null;
      }
      return onRemoved?.apply(this, arguments);
    };
  },
});
