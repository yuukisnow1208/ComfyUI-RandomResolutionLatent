// ComfyUI-RandomResolutionLatent · 前端扩展
//
// 做四件事（坑都是 TagPromptEditor 项目里踩平的，注释保留关键的）：
//  1. 给所有 widget 挂中文 label（canvas 绘制用 w.label ?? w.name，Python 侧不动名字）
//  2. 隐藏 seed / control_after_generate / start_index（官方 hidden 机制，序列化不受
//     影响，旧工作流 widgets_values 位置数组一个位都不动）
//  3. 自定义分辨率格子编辑器：每个分辨率一个格子，双击禁用/恢复（不删除，
//     以行首 ! 前缀持久化到底层文本，后端解析时跳过）
//  4. 宽高比列表也做成格子（添加 / 删除）
//
// 两个编辑器都由 createTilesEditor 工厂创建，共用同一条渲染/提交路径；
// 底层存储保持原文本格式（每行一个 1024x1024 / 逗号分隔比例），旧工作流 / API 完全兼容。
//
// ⚠️ DOM widget 必须留在 widgets 末尾：实测（前端 1.53）加载工作流时按位置给
// **每个** widget 无条件喂 widgets_values，serialize:false 也照样消耗一个值——
// 插在中间会让后面所有 widget 串位。放在末尾吃掉的是不存在的第 N+1 个值，无害。

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
const ASPECT_SPLIT_RE = /[\s,，;；|]+/;
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

/** 文本 -> [{w,h,disabled}]。每行一个；行首 ! = 禁用；支持 1024x1024 / 16:9（按目标像素换算），# 注释。 */
function parseSizes(text, mp) {
  const out = [];
  const seen = new Set();
  for (let line of String(text || "").split("\n")) {
    line = line.split("#")[0].trim();
    if (!line) continue;
    const disabled = line.startsWith("!");
    if (disabled) line = line.slice(1).trim();
    if (!line) continue;
    TOKEN_RE.lastIndex = 0;
    let m;
    while ((m = TOKEN_RE.exec(line))) {
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
        out.push({ w: size.w, h: size.h, disabled });
      }
    }
  }
  return out;
}

function tilesToText(tiles) {
  return tiles.map((t) => `${t.disabled ? "!" : ""}${t.w}x${t.h}`).join("\n");
}

/** 宽高比文本 -> [{label,key}]。支持 16:9 / 16/9 / 1.85，与后端 parse_aspects 对齐。 */
function parseAspects(text) {
  const out = [];
  const seen = new Set();
  for (const raw of String(text || "").split(ASPECT_SPLIT_RE)) {
    if (!raw) continue;
    let label = null;
    let key = null;
    const m = raw.match(/^(\d+(?:\.\d+)?)\s*[:：/]\s*(\d+(?:\.\d+)?)$/);
    if (m) {
      const aw = parseFloat(m[1]);
      const ah = parseFloat(m[2]);
      if (!(aw > 0) || !(ah > 0)) continue;
      label = `${m[1]}:${m[2]}`;
      key = (aw / ah).toFixed(4);
    } else {
      const v = Number(raw);
      if (!Number.isFinite(v) || v <= 0) continue;
      label = String(v);
      key = v.toFixed(4);
    }
    if (!seen.has(key)) {
      seen.add(key);
      out.push({ label, key, disabled: false });
    }
  }
  return out;
}

function aspectsToText(tiles) {
  return tiles.map((t) => t.label).join(", ");
}

// ---------------------------------------------------------------------------
// 格子渲染 / 高度 / 提交 —— 每个编辑器的状态变化只走这一条路径
// （「同一份状态被两处渲染必须走同一条更新路径」，别再拆成两份实现）
// ---------------------------------------------------------------------------

function renderTiles(st) {
  const grid = st.gridEl;
  grid.textContent = "";
  st.countEl.textContent = st.tiles.length ? `· ${st.tiles.length} 种` : "";
  if (!st.tiles.length) {
    const empty = document.createElement("span");
    empty.className = "rrl-empty";
    empty.textContent = st.cfg.emptyText;
    grid.appendChild(empty);
  }
  st.tiles.forEach((t, i) => {
    const tile = document.createElement("span");
    tile.className = "rrl-tile" + (t.disabled ? " rrl-tile-off" : "");
    tile.title = st.cfg.tileTitle(t);
    const label = document.createElement("span");
    label.className = "rrl-tile-text";
    label.textContent = st.cfg.tileLabel(t);
    const del = document.createElement("button");
    del.type = "button";
    del.textContent = "×";
    del.title = "移除";
    del.addEventListener("click", () => {
      st.tiles.splice(i, 1);
      commitTiles(st);
    });
    tile.append(label, del);
    if (st.cfg.canDisable) {
      // 双击 = 禁用 / 恢复（不删除；后端以行首 ! 识别并跳过禁用项）
      tile.addEventListener("dblclick", () => {
        t.disabled = !t.disabled;
        commitTiles(st);
      });
    }
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
  const value = st.cfg.toText(st.tiles);
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
  if (!st || st.native.value === st.lastSynced) return;
  st.tiles = st.cfg.parse(st.native.value, st.node);
  st.lastSynced = st.native.value;
  renderTiles(st);
  syncTilesHeight(st);
}

// ---------------------------------------------------------------------------
// 格子编辑器工厂：自定义分辨率 / 宽高比列表共用
// ---------------------------------------------------------------------------
function createTilesEditor(node, cfg) {
  const native = node.widgets?.find((w) => w.name === cfg.nativeName);
  if (!native) return null;

  const st = {
    node,
    native,
    cfg,
    tiles: [],
    syncing: false,
    lastSynced: null,
    height: 120,       // DOM widget 当前高度（量出来后增量修正）
    inner: null,
    gridEl: null,
    countEl: null,
    observer: null,
  };

  // ---- DOM 结构 ----
  const wrap = document.createElement("div");
  wrap.className = `rrl-wrap ${cfg.wrapClass}`;
  const inner = document.createElement("div");
  inner.className = "rrl-inner";
  inner.innerHTML = `
    <div class="rrl-head">
      <span class="rrl-title">${cfg.title}</span>
      <span class="rrl-count"></span>
      <button class="rrl-clear" type="button" title="移除全部格子">清空</button>
    </div>
    <div class="rrl-grid"></div>
    ${cfg.addRowHTML}`;
  wrap.appendChild(inner);

  st.inner = inner;
  st.gridEl = inner.querySelector(".rrl-grid");
  st.countEl = inner.querySelector(".rrl-count");

  const domWidget = node.addDOMWidget(cfg.domName, cfg.title, wrap, {
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

  // 原生文本框退居幕后：隐藏但仍序列化（唯一真相源）
  hideWidget(native);

  // ---- 添加 / 清空 ----
  function addFromInputs() {
    const added = cfg.onAdd(st);
    if (added > 0) commitTiles(st);
  }
  inner.querySelector(".rrl-add").addEventListener("click", addFromInputs);
  for (const input of inner.querySelectorAll(".rrl-addrow input")) {
    input.addEventListener("keydown", (e) => {
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
  // 必须观察 inner 整个内容区，不能只观察格子区：提示小字（.rrl-hint）会随
  // 节点宽度变化重新换行，只盯格子区时换行不会触发重测，widget 高度停留在
  // 旧值 → 和下一个 widget 重叠。观察 inner 后，格子增删 / 小字换行 /
  // 字体加载等任何内容高度变化都会触发重新同步。
  st.observer = new ResizeObserver(() => syncTilesHeight(st));
  st.observer.observe(st.inner);

  // ---- 初始 ----
  st.tiles = cfg.parse(st.native.value, node);
  st.lastSynced = st.native.value;
  renderTiles(st);
  syncTilesHeight(st);

  return st;
  // 旧工作流加载时 widgets_values 在 onConfigure 之后才到位，onConfigure 里会再补一次
}

// 两个编辑器的配置
const SIZES_CFG = {
  nativeName: "custom_resolutions",
  domName: "rrl_tiles",
  title: "自定义分辨率",
  wrapClass: "rrl-wrap-sizes",
  canDisable: true,
  emptyText: "还没有分辨率 —— 在下面输入宽高后点「添加」",
  addRowHTML: `
    <div class="rrl-addrow">
      <input class="rrl-in rrl-w" type="text" spellcheck="false" placeholder="宽 / 可粘贴列表" />
      <span class="rrl-sep">×</span>
      <input class="rrl-in rrl-h" type="text" spellcheck="false" placeholder="高" />
      <button class="rrl-add" type="button">添加</button>
    </div>`,
  parse: (text, node) => parseSizes(text, megapixelsOf(node)),
  toText: tilesToText,
  onAdd(st) {
    const node = st.node;
    const wInput = st.inner.querySelector(".rrl-w");
    const hInput = st.inner.querySelector(".rrl-h");
    const wRaw = (wInput.value || "").trim();
    const hRaw = (hInput.value || "").trim();
    if (!wRaw && !hRaw) return 0;
    let incoming;
    if (/[x×X*,，:：]/.test(wRaw)) {
      // 宽输入框里粘了一段列表（或写了 16:9）-> 整段解析
      incoming = parseSizes(wRaw, megapixelsOf(node));
    } else {
      const w = parseInt(wRaw, 10);
      const h = parseInt(hRaw, 10);
      if (!w || !h) return 0;
      incoming = [{ ...normalizeSize(w, h), disabled: false }];
    }
    if (!incoming.length) return 0;
    const seen = new Set(st.tiles.map((t) => `${t.w}x${t.h}`));
    let added = 0;
    for (const s of incoming) {
      const key = `${s.w}x${s.h}`;
      if (seen.has(key)) continue;
      seen.add(key);
      st.tiles.push(s);
      added++;
    }
    wInput.value = "";
    hInput.value = "";
    return added;
  },
  tileLabel: (t) => `${t.w}×${t.h}`,
  tileTitle: (t) =>
    t.disabled
      ? `已禁用 · 不参与调度（双击恢复）`
      : `${t.w}×${t.h} · ${((t.w * t.h) / 1e6).toFixed(2)}MP · 双击禁用`,
};

const ASPECTS_CFG = {
  nativeName: "aspect_ratios",
  domName: "rrl_aspects",
  title: "宽高比列表",
  wrapClass: "rrl-wrap-aspects",
  canDisable: false,
  emptyText: "还没有比例 —— 在下面输入后点「添加」（列表为空时按默认值兜底）",
  addRowHTML: `
    <div class="rrl-addrow">
      <input class="rrl-in rrl-a" type="text" spellcheck="false" placeholder="比例，如 3:2 / 1.85 / 可粘贴多个" />
      <button class="rrl-add" type="button">添加</button>
    </div>`,
  parse: (text) => parseAspects(text),
  toText: aspectsToText,
  onAdd(st) {
    const input = st.inner.querySelector(".rrl-a");
    const incoming = parseAspects(input.value || "");
    if (!incoming.length) return 0;
    const seen = new Set(st.tiles.map((t) => t.key));
    let added = 0;
    for (const a of incoming) {
      if (seen.has(a.key)) continue;
      seen.add(a.key);
      st.tiles.push(a);
      added++;
    }
    input.value = "";
    return added;
  },
  tileLabel: (t) => t.label,
  tileTitle: (t) => `${t.label} · 按「目标像素 (MP)」换算成具体尺寸`,
};

// ---------------------------------------------------------------------------
// 节点初始化：中文 label + 隐藏 seed / start_index + 两个格子编辑器
// ---------------------------------------------------------------------------
function setupNode(node) {
  if (node.__rrlEditors) return;
  // 1. 中文 label
  for (const w of node.widgets || []) {
    if (LABELS[w.name]) w.label = LABELS[w.name];
  }
  // 2. 对用户没意义的参数：隐藏（序列化保留原位，旧工作流不串位）
  //    start_index 功能已移除，但它在 INPUT_TYPES 中间，删掉会让旧工作流串位，
  //    所以保留占位、前端隐藏、后端忽略。
  hideWidget(node.widgets?.find((w) => w.name === "seed"));
  hideWidget(node.widgets?.find((w) => w.name === "control_after_generate"));
  hideWidget(node.widgets?.find((w) => w.name === "start_index"));
  // 3. 两个格子编辑器（都追加在 widgets 末尾，见文件头说明）
  const editors = [
    createTilesEditor(node, SIZES_CFG),
    createTilesEditor(node, ASPECTS_CFG),
  ].filter(Boolean);
  node.__rrlEditors = editors;
  // 宽度兜住（高度由 syncTilesHeight 增量修正，不动这里）
  node.setSize([Math.max(node.size[0], 430), node.size[1]]);
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
      setTimeout(() => (node.__rrlEditors || []).forEach(pullFromNative), 0);
      return ret;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      for (const st of this.__rrlEditors || []) st.observer?.disconnect();
      this.__rrlEditors = null;
      return onRemoved?.apply(this, arguments);
    };
  },
});
