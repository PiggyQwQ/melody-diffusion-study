"use strict";

const data = window.MELODY_DATA;
const music = window.MelodyMusic;
const bodyFont = getComputedStyle(document.documentElement).fontFamily;
const $ = (selector, parent = document) => parent.querySelector(selector);
const $$ = (selector, parent = document) => Array.from(parent.querySelectorAll(selector));
const times = [0, 10, 25, 50, 75, 90];
const comparisonTimes = times.slice(1);
const noisySeries = { key: "noisy", label: "加噪后", color: "#f5ac8b" };
const effectSeries = [
  noisySeries,
  { key: "model", label: "模型处理后", color: "#b4e9ab" },
];
const comparisonSeries = [
  noisySeries,
  { key: "folk", label: "中国民歌模型处理后", color: "#b4e9ab" },
  { key: "pop", label: "粤语流行模型处理后", color: "#d5b9ee" },
];
const splitNames = { train: "训练集", validation: "验证集", test: "测试集" };
const colors = { original: "#425f6c", noise: "#9f4e32", model: "#155841", pop: "#70558a" };
const pageSize = 8;
const number = new Intl.NumberFormat("zh-CN");
const comparisonCases = {
  folk: data.comparisons.filter(item => (item.corpus || "folk") === "folk"),
  pop: data.comparisons.filter(item => item.corpus === "pop"),
};

const state = {
  route: "effects",
  effect: { song: "h2_high_1", index: 3 },
  compare: { corpus: "folk", songs: { folk: "h3_prior_drift_2", pop: null }, index: 3 },
  trace: { song: data.trajectories[0].id, origin: 50, index: 0, timer: null },
  library: { corpus: "folk", song: null, fragment: 0, query: "", page: 0 },
  views: { effects: "roll", process: "roll", comparison: "roll", library: "roll" },
};
let currentTracks = new Map();
let currentTrend = null;
let activeKey = null;
let playbackProgress;
let resizeFrame;
const pendingLibraries = new Map();
const rollRanges = new WeakMap();

const player = music.createPlayer(onPlayback);

function say(message, visible = false) {
  const status = $("#app-status");
  status.className = visible ? "notice" : "sr-only";
  status.textContent = message;
}

function node(tag, className, value) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (value !== undefined) element.textContent = value;
  return element;
}

function selectOptions(select, options) {
  select.replaceChildren(...options.map(({ value, label }) => new Option(label, value)));
}

function setSelect(id, records, value) {
  const select = $(id);
  selectOptions(select, records.map(record => {
    const start = Number(record.windowId.split("@").at(-1));
    return {
      value: record.id,
      label: record.title + " " + start + "-" + (start + 7) + "小节",
    };
  }));
  select.value = value;
}

function startControls(range, ticks, values, initial, change) {
  range.max = values.length - 1;
  range.value = initial;
  const buttons = values.map((time, index) => {
    const item = node("button", "", time === 0 ? "原曲" : String(time));
    item.type = "button";
    item.setAttribute("aria-label", time === 0 ? "原旋律对照" : "第 " + time + " 档");
    item.addEventListener("click", () => {
      range.value = index;
      change(index);
    });
    return item;
  });
  ticks.replaceChildren(...buttons);
  range.addEventListener("input", () => change(Number(range.value)));
}

function paintRange(prefix, index, values) {
  const step = values[index];
  const range = $("#" + prefix + "-range");
  range.value = index;
  range.setAttribute("aria-valuetext", step === 0 ? "原旋律对照" : "第 " + step + " 档");
  $("#" + prefix + "-step").textContent = step;
  $$("#" + prefix + "-ticks button").forEach((button, i) => button.setAttribute("aria-pressed", i === index));
}

function track(key, title, tone, tokens, metric, caption = "") {
  return { key, title, tone, tokens, metric, caption };
}

function overlapLabel(reference, tokens) {
  return music.overlap(reference, tokens) + "/" + reference.length;
}

function stableRange(example, packed) {
  if (!rollRanges.has(example)) rollRanges.set(example, music.pitchRange(packed().map(music.decode)));
  return rollRanges.get(example);
}

function renderTracks(target, items, range) {
  const view = state.views[state.route];
  const sharedRange = range || music.pitchRange(items.map(item => item.tokens));
  currentTracks = new Map();
  const cards = items.map(item => {
    const card = node("article", "track-card");
    card.dataset.track = item.key;
    card.dataset.tone = item.tone;
    const head = node("div", "track-head");
    const title = node("div");
    title.append(node("h3", "", item.title));
    if (item.caption) title.append(node("p", "", item.caption));
    const metric = node("div", "track-metric");
    metric.append(node("span", "track-number", item.metric));
    if (item.metricLabel) metric.append(node("small", "", item.metricLabel));
    head.append(title, metric);
    card.append(head);

    const drawing = node("div", view === "score" ? "score" : "roll-frame");
    if (view === "roll") {
      const canvas = node("canvas");
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", item.title + "的八小节钢琴卷帘");
      drawing.append(canvas);
    } else {
      drawing.setAttribute("role", "region");
      drawing.setAttribute("aria-label", item.title + "的五线谱");
      drawing.tabIndex = 0;
    }
    const bottom = node("div", "track-bottom");
    const button = node("button", "play-button");
    button.type = "button";
    button.dataset.play = item.key;
    button.setAttribute("aria-label", "试听" + item.title);
    button.setAttribute("aria-pressed", "false");
    const mark = node("i", "play-mark");
    mark.setAttribute("aria-hidden", "true");
    const buttonText = node("span", "", "试听");
    button.append(mark, buttonText);
    bottom.append(button);
    if (view === "score") card.append(bottom, drawing);
    else card.append(drawing, bottom);

    button.addEventListener("click", async () => {
      stopTimeline();
      say("");
      try {
        await player.play(item.tokens, item.key);
      } catch {
        say("浏览器暂时无法播放音频，请使用 Chrome 或 Edge 打开。", true);
      }
    });
    currentTracks.set(item.key, { ...item, drawing, button, buttonText, range: sharedRange });
    return card;
  });
  target.replaceChildren(...cards);
  drawTracks();
}

function drawOne(item) {
  if (state.views[state.route] === "score") music.drawScore(item.drawing, item.tokens);
  drawPlayhead(item);
}

function drawPlayhead(item) {
  const progress = item.key === activeKey ? playbackProgress : undefined;
  if (state.views[state.route] === "roll") {
    music.drawRoll($("canvas", item.drawing), item.tokens, {
      range: item.range,
      color: colors[item.tone],
      labelFont: bodyFont,
      progress,
    });
  } else {
    music.setScoreProgress(item.drawing, progress);
  }
}

function drawTracks() {
  currentTracks.forEach(drawOne);
}

function onPlayback(key, progress) {
  const formerKey = activeKey;
  activeKey = key;
  playbackProgress = progress;
  if (formerKey !== key) {
    currentTracks.forEach(item => {
      const playing = item.key === key;
      item.button.setAttribute("aria-pressed", playing);
      item.button.setAttribute("aria-label", (playing ? "停止" : "试听") + item.title);
      item.buttonText.textContent = playing ? "停止" : "试听";
    });
  }
  if (formerKey && formerKey !== key && currentTracks.has(formerKey)) drawPlayhead(currentTracks.get(formerKey));
  if (key && currentTracks.has(key)) drawPlayhead(currentTracks.get(key));
}

function renderTrend(canvas, example, stages, selected, series) {
  const clean = music.decode(example.clean);
  const slots = clean.length;
  const points = stages.map(stage => {
    const level = example.levels[stage];
    const scores = series.map(({ key }) => stage === 0 ? slots : music.overlap(clean, music.decode(level[key])));
    return [stage, scores];
  });
  const details = points.map(([stage, scores]) => {
    const values = series.map(({ label }, index) => label + " " + scores[index] + "/" + slots);
    return stage + "%：" + values.join("，");
  });
  canvas.setAttribute("aria-label", "各加噪档位的重合度 · " + example.title + "。" + details.join("。") + "。");
  currentTrend = { canvas, slots, points, series, selected };
  drawTrend();
}

function drawTrend() {
  if (!currentTrend) return;
  const { canvas, slots, points, series, selected } = currentTrend;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (width < 1 || height < 1) return;
  const pixel = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = Math.round(width * pixel);
  canvas.height = Math.round(height * pixel);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(pixel, 0, 0, pixel, 0, 0);
  const pad = { left: 43, right: 17, top: 13, bottom: 23 };
  const first = points[0][0];
  const last = points.at(-1)[0];
  const x = stage => pad.left + (stage - first) / (last - first) * (width - pad.left - pad.right);
  const y = matches => pad.top + (1 - matches / slots) * (height - pad.top - pad.bottom);
  ctx.setLineDash([]);
  ctx.font = "11px " + bodyFont;
  ctx.textBaseline = "middle";
  for (const matches of [0, slots / 2, slots]) {
    ctx.strokeStyle = "#a5c7b055";
    ctx.beginPath();
    ctx.moveTo(x(first), y(matches));
    ctx.lineTo(x(last), y(matches));
    ctx.stroke();
    ctx.fillStyle = "#d3e9da";
    ctx.textAlign = "left";
    ctx.fillText(matches + "/" + slots, 0, y(matches));
  }
  for (const [stage] of points) {
    ctx.fillStyle = selected === stage ? "#fffce8" : "#bbd8c6";
    ctx.textAlign = "center";
    ctx.fillText(String(stage), x(stage), height - 9);
  }
  ctx.setLineDash([3, 4]);
  ctx.strokeStyle = "#e5f1d999";
  ctx.beginPath();
  ctx.moveTo(x(selected), y(slots));
  ctx.lineTo(x(selected), y(0));
  ctx.stroke();
  ctx.setLineDash([]);
  series.forEach(({ color }, index) => {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2.3;
    ctx.lineJoin = "round";
    ctx.beginPath();
    points.forEach(([stage, scores], i) => {
      if (i === 0) ctx.moveTo(x(stage), y(scores[index]));
      else ctx.lineTo(x(stage), y(scores[index]));
    });
    ctx.stroke();
    for (const [stage, scores] of points) {
      ctx.beginPath();
      ctx.arc(x(stage), y(scores[index]), stage === selected ? 4.8 : 2.6, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
}

function renderEffects() {
  const example = data.effects.find(item => item.id === state.effect.song);
  const step = times[state.effect.index];
  const clean = music.decode(example.clean);
  const noisy = step ? music.decode(example.levels[step].noisy) : clean;
  const model = step ? music.decode(example.levels[step].model) : clean;
  const range = stableRange(example, () => [example.clean, ...Object.values(example.levels).flatMap(Object.values)]);
  paintRange("effect", state.effect.index, times);
  renderTracks($("#effect-tracks"), [
    track("effect-original", "原旋律", "original", clean, overlapLabel(clean, clean)),
    track("effect-noisy", "加噪后旋律", "noise", noisy, overlapLabel(clean, noisy)),
    track("effect-model", "模型处理后旋律", "model", model, overlapLabel(clean, model)),
  ], range);
  $("#effect-note").hidden = step !== 0;
  renderTrend($("#effect-chart"), example, times, step, effectSeries);
}

function stopTimeline() {
  if (state.trace.timer !== null) {
    clearInterval(state.trace.timer);
    state.trace.timer = null;
  }
  const button = $("#trace-play");
  button.textContent = "播放过程";
  button.setAttribute("aria-pressed", "false");
}

function renderTrace() {
  const trace = data.trajectories.find(item => item.id === state.trace.song);
  const pivot = state.trace.origin;
  const index = state.trace.index;
  const backward = index > pivot;
  const step = backward ? pivot * 2 - index : index;
  const reverse = pivot === 50 ? trace.reverse50 : trace.reverse100;
  const tokens = music.decode(backward ? reverse[index - pivot] : trace.forward[step]);
  const clean = music.decode(trace.forward[0]);
  const finalTitle = pivot === 100 ? "新生成的旋律" : "处理后的旋律";
  const name = index === 0 ? "原旋律" : index === pivot * 2 ? finalTitle
    : index === pivot ? "反向起点" : backward ? "处理中" : "加噪中";
  const range = $("#trace-range");
  range.max = pivot * 2;
  range.value = index;
  range.setAttribute("aria-valuetext", name + "，第 " + step + " 步");
  $("#trace-step").textContent = step;
  $("#trace-phase").textContent = backward ? "反向处理" : index === pivot ? "正反向交接" : "正向加噪";
  $("#trace-midpoint").textContent = "加噪到第 " + pivot + " 步";
  $("#trace-end").textContent = finalTitle;
  const item = track("trace", name, index === 0 ? "original" : backward ? "model" : "noise",
    tokens, overlapLabel(clean, tokens), "第 " + step + " 步");
  const rollRange = stableRange(trace, () => [...trace.forward, ...trace.reverse50, ...trace.reverse100]);
  renderTracks($("#trace-tracks"), [item], rollRange);
  $("#trace-note").hidden = pivot !== 100;
}

function setTrace(index) {
  stopTimeline();
  player.stop();
  state.trace.index = index;
  renderTrace();
}

function toggleTimeline() {
  if (state.trace.timer !== null) {
    stopTimeline();
    return;
  }
  player.stop();
  const end = state.trace.origin * 2;
  if (state.trace.index === end) state.trace.index = 0;
  const button = $("#trace-play");
  button.textContent = "暂停过程";
  button.setAttribute("aria-pressed", "true");
  renderTrace();
  state.trace.timer = setInterval(() => {
    state.trace.index += 1;
    renderTrace();
    if (state.trace.index === end) stopTimeline();
  }, state.views.process === "roll" ? 95 : 190);
}

function setComparisonCorpus(corpus) {
  const cases = comparisonCases[corpus];
  state.compare.corpus = corpus;
  const example = cases.find(item => item.id === state.compare.songs[corpus]) || cases[0];
  state.compare.songs[corpus] = example.id;
  setSelect("#compare-song", cases, example.id);
  $$("[data-compare-corpus]").forEach(button => {
    button.setAttribute("aria-pressed", button.dataset.compareCorpus === corpus);
  });
}

function renderComparison() {
  const { corpus, songs } = state.compare;
  const example = comparisonCases[corpus].find(item => item.id === songs[corpus]);
  const step = comparisonTimes[state.compare.index];
  const level = example.levels[step];
  const clean = music.decode(example.clean);
  const inputs = [
    ["compare-noisy", "加噪后旋律", "noise", level.noisy],
    ["compare-folk", "匹配中国民歌模型", "model", level.folk],
    ["compare-pop", "匹配粤语流行模型", "pop", level.pop],
  ];
  const items = inputs.map(([key, title, tone, packed]) => {
    const tokens = music.decode(packed);
    return track(key, title, tone, tokens, overlapLabel(clean, tokens));
  });
  paintRange("compare", state.compare.index, comparisonTimes);
  const range = stableRange(example, () => [example.clean, ...Object.values(example.levels).flatMap(Object.values)]);
  renderTracks($("#compare-tracks"), items, range);
  renderTrend($("#compare-chart"), example, comparisonTimes, step, comparisonSeries);
}

function libraryData(corpus) {
  return window.MELODY_LIBRARIES?.[corpus];
}

function loadCorpus(corpus) {
  if (libraryData(corpus)) return Promise.resolve(libraryData(corpus));
  if (!pendingLibraries.has(corpus)) {
    const descriptor = data.libraries.find(item => item.id === corpus);
    const promise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = descriptor.file;
      script.onload = () => libraryData(corpus) ? resolve(libraryData(corpus)) : reject();
      script.onerror = reject;
      document.head.appendChild(script);
    }).catch(error => {
      pendingLibraries.delete(corpus);
      throw error;
    });
    pendingLibraries.set(corpus, promise);
  }
  return pendingLibraries.get(corpus);
}

function normalized(value) {
  return value.normalize("NFKC").toLowerCase().replace(/\s+/g, "");
}

function filterLibrarySongs(library) {
  const term = normalized(state.library.query);
  return term ? library.songs.filter(song => normalized(song.title + " " + song.detail + " " + song.id).includes(term))
    : library.songs;
}

function renderLibraryList(filtered) {
  const list = $("#library-list");
  const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const pageIndex = Math.max(0, Math.min(state.library.page, pages - 1));
  state.library.page = pageIndex;
  const visible = filtered.slice(pageIndex * pageSize, (pageIndex + 1) * pageSize);
  const results = visible.map(song => {
    const row = node("li");
    const item = node("button");
    item.type = "button";
    item.setAttribute("aria-pressed", song.id === state.library.song);
    item.append(node("strong", "", song.title),
      node("span", "", [song.detail, song.fragments.length + " 段"].filter(Boolean).join(" · ")));
    item.addEventListener("click", () => {
      player.stop();
      state.library.song = song.id;
      state.library.fragment = 0;
      renderLibrary();
      list.querySelector('button[aria-pressed="true"]')?.focus({ preventScroll: true });
    });
    row.append(item);
    return row;
  });
  list.replaceChildren(...(results.length ? results : [node("li", "list-empty", "没有找到这个曲名。")]));
  $("#library-prev").disabled = pageIndex === 0;
  $("#library-next").disabled = pageIndex + 1 >= pages;
  $("#library-page").textContent = (pageIndex + 1) + " / " + pages;
}

function turnLibraryPage(delta) {
  const library = libraryData(state.library.corpus);
  state.library.page += delta;
  renderLibraryList(filterLibrarySongs(library));
}

function renderLibrary() {
  const { corpus, query } = state.library;
  const tracks = $("#library-tracks");
  const list = $("#library-list");
  const descriptor = data.libraries.find(item => item.id === corpus);
  const library = libraryData(corpus);
  $$("[data-corpus]").forEach(button => button.setAttribute("aria-pressed", button.dataset.corpus === corpus));
  const count = number.format(descriptor.songs) + " 首\n" + number.format(descriptor.windows) + " 个片段";
  $("#library-total").textContent = count;
  if (!library) {
    $("#library-selection").hidden = true;
    list.replaceChildren(node("li", "list-empty", "正在载入曲目…"));
    tracks.replaceChildren();
    currentTracks = new Map();
    $("#library-prev").disabled = $("#library-next").disabled = true;
    $("#library-page").textContent = "";
    loadCorpus(corpus).then(() => {
      if (state.route === "library" && state.library.corpus === corpus) renderLibrary();
    }).catch(() => {
      if (state.route === "library" && state.library.corpus === corpus) {
        list.replaceChildren(node("li", "list-empty", "读取失败。请确认网页与 data 文件夹一起解压。"));
      }
    });
    return;
  }
  const filtered = filterLibrarySongs(library);
  if (normalized(query)) $("#library-total").textContent = number.format(filtered.length) + " 首符合搜索\n" + count;
  let selected = filtered.find(song => song.id === state.library.song);
  if (!selected) {
    selected = filtered[0];
    state.library.song = selected?.id || null;
    state.library.fragment = 0;
  }
  renderLibraryList(filtered);
  $("#library-selection").hidden = !selected;
  if (!selected) {
    tracks.replaceChildren();
    currentTracks = new Map();
    return;
  }
  $("#library-title").textContent = selected.title;
  $("#library-detail").textContent = [selected.detail, splitNames[selected.split]].filter(Boolean).join(" · ");
  const choice = $("#library-fragment");
  selectOptions(choice, selected.fragments.map(({ bar }, i) => ({
    value: i,
    label: "原谱第 " + bar + "–" + (bar + 7) + " 小节",
  })));
  choice.value = state.library.fragment;
  const { bar, tokens: packed } = selected.fragments[state.library.fragment];
  const tokens = music.decode(packed);
  const item = track("library", "原旋律片段", "original", tokens, String(bar), "八小节 · 2/4 拍");
  item.metricLabel = "原谱起始小节";
  renderTracks(tracks, [item]);
}

const render = {
  effects: renderEffects,
  process: renderTrace,
  comparison: renderComparison,
  library: renderLibrary,
};

function repaint() {
  say("");
  render[state.route]();
}

function switchPage() {
  const location = window.location.hash.substring(1);
  const page = Object.hasOwn(render, location) ? location : "effects";
  player.stop();
  stopTimeline();
  currentTrend = null;
  state.route = page;
  for (const route of Object.keys(render)) $("#page-" + route).hidden = route !== page;
  $$("nav a").forEach(link => {
    if (link.hash === "#" + page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  repaint();
}

const effectValues = times;
startControls($("#effect-range"), $("#effect-ticks"), effectValues, state.effect.index, index => {
  player.stop();
  state.effect.index = index;
  renderEffects();
});
startControls($("#compare-range"), $("#compare-ticks"), comparisonTimes, state.compare.index, index => {
  player.stop();
  state.compare.index = index;
  renderComparison();
});
setSelect("#effect-song", data.effects, state.effect.song);
setComparisonCorpus(state.compare.corpus);
setSelect("#trace-song", data.trajectories, state.trace.song);
$("#effect-song").addEventListener("change", event => {
  player.stop();
  state.effect.song = event.target.value;
  renderEffects();
});
$("#compare-song").addEventListener("change", event => {
  player.stop();
  state.compare.songs[state.compare.corpus] = event.target.value;
  renderComparison();
});
for (const button of $$("[data-compare-corpus]")) {
  const corpus = button.dataset.compareCorpus;
  button.disabled = comparisonCases[corpus].length === 0;
  button.addEventListener("click", () => {
    if (state.compare.corpus === corpus) return;
    player.stop();
    setComparisonCorpus(corpus);
    renderComparison();
  });
}
$("#trace-song").addEventListener("change", event => {
  state.trace.song = event.target.value;
  setTrace(0);
});
$("#trace-origin").addEventListener("change", event => {
  state.trace.origin = Number(event.target.value);
  setTrace(0);
});
$("#trace-range").addEventListener("input", event => setTrace(Number(event.target.value)));
$("#trace-start").addEventListener("click", () => setTrace(0));
$("#trace-midpoint").addEventListener("click", () => setTrace(state.trace.origin));
$("#trace-end").addEventListener("click", () => setTrace(state.trace.origin * 2));
$("#trace-play").addEventListener("click", toggleTimeline);
$("#trace-tracks").addEventListener("focusin", stopTimeline);
$(".skip-link").addEventListener("click", event => {
  event.preventDefault();
  $("#main").focus();
});

for (const button of $$("[data-view]")) {
  button.addEventListener("click", () => {
    if (state.views[state.route] === button.dataset.view) return;
    player.stop();
    stopTimeline();
    state.views[state.route] = button.dataset.view;
    $$("button", button.parentElement).forEach(choice => choice.setAttribute("aria-pressed", choice === button));
    repaint();
  });
}
for (const button of $$("[data-corpus]")) {
  button.addEventListener("click", () => {
    if (state.library.corpus === button.dataset.corpus) return;
    player.stop();
    Object.assign(state.library, { corpus: button.dataset.corpus, song: null, fragment: 0, query: "", page: 0 });
    $("#library-search").value = "";
    renderLibrary();
  });
}
$("#library-search").addEventListener("input", event => {
  player.stop();
  state.library.query = event.target.value;
  state.library.page = 0;
  renderLibrary();
});
$("#library-fragment").addEventListener("change", event => {
  player.stop();
  state.library.fragment = Number(event.target.value);
  renderLibrary();
});
$("#library-prev").addEventListener("click", () => turnLibraryPage(-1));
$("#library-next").addEventListener("click", () => turnLibraryPage(1));
$("#library-counts").replaceChildren(...data.libraries.map(item =>
  node("div", "library-counter", item.title + "曲目数：" + item.songs + "首")));

const fullscreen = $("#fullscreen");
fullscreen.hidden = !document.fullscreenEnabled;
fullscreen.addEventListener("click", async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch {
    say("请用浏览器的全屏功能进入演示。", true);
  }
});
document.addEventListener("fullscreenchange", () => {
  fullscreen.textContent = document.fullscreenElement ? "退出全屏" : "全屏演示";
});
window.addEventListener("hashchange", switchPage);
window.addEventListener("resize", () => {
  cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(() => {
    drawTrend();
    drawTracks();
  });
});
switchPage();
document.fonts.ready.then(() => {
  drawTrend();
  if (state.views[state.route] === "roll") drawTracks();
});
