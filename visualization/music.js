(() => {
  "use strict";

  const ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";
  const SLOTS_PER_BAR = 8;
  const SLOT_SECONDS = 60 / 90 / 4;
  const PITCH_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

  function decode(packed) {
    return Array.from(packed, (character) => {
      const token = ALPHABET.indexOf(character);
      if (token < 0 || token > 50) throw new Error("旋律包含无法识别的音符。");
      return token;
    });
  }

  function notes(tokens) {
    const output = [];
    let active = null;
    tokens.forEach((token, slot) => {
      if (token === 1) {
        if (active) active.length += 1;
        return;
      }
      active = token >= 2 ? { start: slot, length: 1, pitch: token + 36 } : null;
      if (active) output.push(active);
    });
    return output;
  }

  function soundingPitches(tokens) {
    const sounding = new Int16Array(tokens.length).fill(-1);
    for (const note of notes(tokens)) {
      sounding.fill(note.pitch, note.start, note.start + note.length);
    }
    return sounding;
  }

  function overlap(referenceTokens, tokens) {
    if (referenceTokens.length !== tokens.length) throw new Error("比较的两段旋律需要等长。");
    const reference = soundingPitches(referenceTokens);
    const current = soundingPitches(tokens);
    return reference.reduce((count, pitch, slot) => count + Number(pitch === current[slot]), 0);
  }

  function pitchRange(tokenLists) {
    let low = 127;
    let high = 0;
    for (const tokens of tokenLists) {
      for (const token of tokens) {
        if (token < 2) continue;
        low = Math.min(low, token + 36);
        high = Math.max(high, token + 36);
      }
    }
    if (low > high) return { low: 57, high: 71 };
    const padding = Math.max(1, Math.ceil((11 - (high - low)) / 2));
    return { low: Math.max(0, low - padding), high: Math.min(127, high + padding) };
  }

  function drawRoll(canvas, tokens, {
    range = pitchRange([tokens]), color = "#26715d", labelFont = "system-ui, sans-serif", progress,
  } = {}) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(180, Math.round(rect.width || 320));
    const height = Math.max(110, Math.round(rect.height || 152));
    const scale = Math.min(3, window.devicePixelRatio || 1);
    const pixelWidth = Math.round(width * scale);
    const pixelHeight = Math.round(height * scale);
    if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
    if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.lineWidth = 1;

    const left = 28;
    const top = 9;
    const chartWidth = width - left - 8;
    const chartHeight = height - top - 22;
    const row = chartHeight / (range.high - range.low + 1);
    const slot = chartWidth / tokens.length;
    const y = (pitch) => top + (range.high - pitch) * row;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#fafaf7";
    ctx.fillRect(left, top, chartWidth, chartHeight);

    ctx.fillStyle = "#f0f2ec";
    for (let pitch = range.low; pitch <= range.high; pitch += 1) {
      if ([1, 3, 6, 8, 10].includes(pitch % 12)) ctx.fillRect(left, y(pitch), chartWidth, row);
    }
    for (let column = 0; column <= tokens.length; column += 1) {
      const strong = column % SLOTS_PER_BAR === 0;
      if (!strong && column % 4 !== 0) continue;
      const x = Math.round(left + column * slot) + 0.5;
      ctx.strokeStyle = strong ? "#d8ded6" : "#e9ece6";
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, top + chartHeight);
      ctx.stroke();
    }

    ctx.fillStyle = "#56635c";
    ctx.font = "10px " + labelFont;
    ctx.textBaseline = "middle";
    ctx.textAlign = "right";
    for (let pitch = range.low; pitch <= range.high; pitch += 1) {
      if (pitch % 12 !== 0) continue;
      ctx.fillText("C" + (Math.floor(pitch / 12) - 1), left - 4, y(pitch) + row / 2);
    }
    ctx.textAlign = "center";
    for (let bar = 0; bar < tokens.length / SLOTS_PER_BAR; bar += 1) {
      ctx.fillText(String(bar + 1), left + (bar + 0.5) * SLOTS_PER_BAR * slot, height - 7);
    }

    ctx.fillStyle = color;
    for (const note of notes(tokens)) {
      const x = left + note.start * slot;
      const inset = Math.min(1, row * 0.16);
      const noteHeight = Math.max(1.5, row - 2 * inset);
      const noteY = y(note.pitch) + inset;
      ctx.globalAlpha = 0.40;
      ctx.fillRect(x, noteY, Math.max(1, note.length * slot - 0.6), noteHeight);
      ctx.globalAlpha = 1;
      ctx.fillRect(x, noteY, Math.max(1, slot - 0.7), noteHeight);
    }

    if (Number.isFinite(progress)) {
      const x = Math.round(left + Math.max(0, Math.min(1, progress)) * chartWidth) + 0.5;
      ctx.strokeStyle = "#253a32";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, top - 3);
      ctx.lineTo(x, top + chartHeight + 2);
      ctx.stroke();
    }
  }

  const DURATIONS = [[8, "h"], [6, "qd"], [4, "q"], [3, "8d"], [2, "8"], [1, "16"]];
  const scoreLayouts = new WeakMap();

  function segments(tokens) {
    const output = [];
    let cursor = 0;
    for (const note of notes(tokens)) {
      if (cursor < note.start) output.push({ start: cursor, length: note.start - cursor, pitch: null });
      output.push(note);
      cursor = note.start + note.length;
    }
    if (cursor < tokens.length) output.push({ start: cursor, length: tokens.length - cursor, pitch: null });
    return output;
  }

  function drawScore(container, tokens) {
    const VF = window.Vex?.Flow;
    if (!VF) throw new Error("五线谱组件未能加载。");

    const width = Math.max(400, Math.floor(container.clientWidth || 400));
    const columns = width >= 760 ? 4 : 2;
    const bars = Math.ceil(tokens.length / SLOTS_PER_BAR);
    const pitches = notes(tokens);
    const diatonic = (midi) =>
      (Math.floor(midi / 12) - 1) * 7 + "CDEFGAB".indexOf(PITCH_NAMES[midi % 12][0]);
    const low = pitches.length ? Math.min(...pitches.map((note) => diatonic(note.pitch))) : 30;
    const high = pitches.length ? Math.max(...pitches.map((note) => diatonic(note.pitch))) : 38;
    const extraTop = Math.max(0, (high - 38) * 5 - 15);
    const rowHeight = 127 + extraTop + Math.max(0, (30 - low) * 5);
    const rowWidth = width - 16;
    const barWidth = rowWidth / columns;
    const height = Math.ceil(bars / columns) * rowHeight;
    const surface = document.createElement("div");
    surface.style.position = "relative";
    const canvas = document.createElement("canvas");
    canvas.setAttribute("role", "img");
    canvas.setAttribute("aria-label", "2/4 拍五线谱，共 " + bars + " 小节");
    const playhead = document.createElement("div");
    playhead.className = "score-playhead";
    playhead.setAttribute("aria-hidden", "true");
    playhead.hidden = true;
    playhead.style.cssText = "position:absolute;width:2px;pointer-events:none;transform:translateX(-50%);"
      + "background:#24785c;border-radius:1px;box-shadow:0 0 0 1px #fff9";
    surface.append(canvas, playhead);
    container.replaceChildren(surface);
    const renderer = new VF.Renderer(canvas, VF.Renderer.Backends.CANVAS);
    renderer.resize(width, height);
    canvas.style.display = "block";
    canvas.style.width = "100%";
    canvas.style.height = "auto";
    const ctx = renderer.getContext();
    ctx.setFillStyle("#26352e");
    ctx.setStrokeStyle("#26352e");

    const perBar = Array.from({ length: bars }, () => []);
    const ties = [];
    for (const event of segments(tokens)) {
      let at = event.start;
      const end = event.start + event.length;
      let previous = null;
      while (at < end) {
        const bar = Math.floor(at / SLOTS_PER_BAR);
        const available = Math.min(end - at, SLOTS_PER_BAR - (at % SLOTS_PER_BAR));
        const [length, duration] = DURATIONS.find(([durationSlots]) => durationSlots <= available);
        const isRest = event.pitch === null;
        const pitch = isRest ? 71 : event.pitch;
        const key = PITCH_NAMES[pitch % 12].toLowerCase().replace("♯", "#") + "/" + (Math.floor(pitch / 12) - 1);
        const note = new VF.StaveNote({
          keys: [key],
          duration: duration + (isRest ? "r" : ""),
          auto_stem: true,
          align_center: isRest && length === SLOTS_PER_BAR,
        });
        if (duration.endsWith("d")) VF.Dot.buildAndAttach([note], { all: true });
        perBar[bar].push({ note, slot: at % SLOTS_PER_BAR, fullRest: isRest && length === SLOTS_PER_BAR });
        if (!isRest) {
          const system = Math.floor(bar / columns);
          if (previous) ties.push([previous.note, note, previous.system === system]);
          previous = { note, system };
        }
        at += length;
      }
    }

    const systems = [];
    const timing = perBar.map((entries, index) => {
      const column = index % columns;
      const systemIndex = Math.floor(index / columns);
      const stave = new VF.Stave(8 + column * barWidth, systemIndex * rowHeight + extraTop, barWidth);
      if (column === 0) stave.addClef("treble");
      if (index === 0) stave.addTimeSignature("2/4");
      stave.setContext(ctx).draw();
      const barNotes = entries.map(({ note }) => note);
      const voice = new VF.Voice({ num_beats: 2, beat_value: 4 });
      voice.addTickables(barNotes);
      VF.Accidental.applyAccidentals([voice], "C");
      const beams = VF.Beam.generateBeams(barNotes, { groups: [new VF.Fraction(1, 4)] });
      new VF.Formatter().joinVoices([voice]).formatToStave([voice], stave);
      voice.draw(ctx, stave);
      for (const beam of beams) beam.setContext(ctx).draw();

      const system = systems[systemIndex] ??= { top: stave.getYForLine(0), bottom: stave.getYForLine(4) };
      for (const note of barNotes) {
        system.top = Math.min(system.top, ...note.getYs());
        system.bottom = Math.max(system.bottom, ...note.getYs());
      }
      // Align timed events with the engraved noteheads. A centered whole-bar
      // rest represents the entire measure, so its timeline starts at the left.
      const stops = entries.map(({ note, slot, fullRest }) => ({
        slot,
        x: fullRest ? stave.getNoteStartX() : (note.getNoteHeadBeginX() + note.getNoteHeadEndX()) / 2,
      }));
      stops.push({ slot: SLOTS_PER_BAR, x: stave.getNoteEndX() });
      return { systemIndex, system, stops };
    });

    function drawTie(first, last) {
      new VF.StaveTie({
        first_note: first, last_note: last, first_indices: [0], last_indices: [0],
      }).setContext(ctx).draw();
    }
    for (const [first, last, sameSystem] of ties) {
      if (sameSystem) drawTie(first, last);
      else {
        drawTie(first, null);
        drawTie(null, last);
      }
    }
    scoreLayouts.set(container, { playhead, timing, width, height, slots: tokens.length, activeSystem: -1 });
  }

  function setScoreProgress(container, progress) {
    const score = scoreLayouts.get(container);
    if (!score) return;
    const { playhead, timing, slots, width, height } = score;
    if (!Number.isFinite(progress) || !timing.length) {
      playhead.hidden = true;
      score.activeSystem = -1;
      return;
    }

    const slot = Math.max(0, Math.min(1, progress)) * slots;
    const barIndex = Math.min(timing.length - 1, Math.floor(slot / SLOTS_PER_BAR));
    const offset = slot - barIndex * SLOTS_PER_BAR;
    const { systemIndex, system, stops } = timing[barIndex];
    const nextIndex = stops.findIndex((stop, index) => index > 0 && stop.slot >= offset);
    const previous = stops[nextIndex - 1];
    const next = stops[nextIndex];
    const x = previous.x + (next.x - previous.x) * (offset - previous.slot) / (next.slot - previous.slot);
    playhead.style.left = (x / width * 100) + "%";
    playhead.hidden = false;
    if (score.activeSystem !== systemIndex) {
      score.activeSystem = systemIndex;
      const top = Math.max(0, system.top - 12);
      const bottom = Math.min(height, system.bottom + 12);
      playhead.style.top = (top / height * 100) + "%";
      playhead.style.height = ((bottom - top) / height * 100) + "%";
      if (container.scrollHeight > container.clientHeight + 2) {
        if (playhead.offsetTop < container.scrollTop) {
          container.scrollTop = playhead.offsetTop;
        } else if (playhead.offsetTop + playhead.offsetHeight > container.scrollTop + container.clientHeight) {
          container.scrollTop = playhead.offsetTop + playhead.offsetHeight - container.clientHeight;
        }
      }
    }
  }

  const PARTIALS = [
    [1, 0.49, 1.7], [2, 0.22, 3.4], [3, 0.12, 5.8],
    [4, 0.065, 8.3], [5, 0.038, 12], [6, 0.022, 17],
  ];

  function createPlayer(onUpdate) {
    let context;
    let session = null;
    let frame = 0;

    function stop() {
      const playing = session;
      session = null;
      window.cancelAnimationFrame(frame);
      if (playing?.output) {
        const now = context.currentTime;
        playing.output.gain.cancelScheduledValues(now);
        playing.output.gain.setTargetAtTime(0, now, 0.003);
        for (const oscillator of playing.oscillators) oscillator.stop(now + 0.016);
        window.setTimeout(() => playing.output.disconnect(), 24);
      }
      onUpdate(null, 0);
    }

    function addNote(playing, note, origin) {
      const start = origin + note.start * SLOT_SECONDS;
      const duration = note.length * SLOT_SECONDS;
      const release = start + duration;
      const attack = 0.004;
      const fundamental = 440 * Math.pow(2, (note.pitch - 69) / 12);
      for (const [multiple, volume, decay] of PARTIALS) {
        const oscillator = context.createOscillator();
        const envelope = context.createGain();
        const frequency = fundamental * multiple * Math.sqrt(1 + multiple * multiple * 0.00028);
        oscillator.frequency.value = frequency;
        envelope.gain.setValueAtTime(0.00001, start);
        envelope.gain.exponentialRampToValueAtTime(volume, start + attack);
        const releaseLevel = Math.max(0.00001, volume * Math.exp(-decay * duration));
        envelope.gain.exponentialRampToValueAtTime(releaseLevel, release);
        envelope.gain.exponentialRampToValueAtTime(0.00001, release + 0.045);
        oscillator.connect(envelope);
        envelope.connect(playing.output);
        oscillator.start(start);
        oscillator.stop(release + 0.055);
        playing.oscillators.push(oscillator);
      }
    }

    async function play(tokens, key) {
      const toggleOff = session?.key === key;
      stop();
      if (toggleOff) return false;

      const Audio = window.AudioContext || window.webkitAudioContext;
      if (!Audio) throw new Error("当前浏览器无法播放合成音频。");
      context ??= new Audio();
      const playing = { key, output: null, oscillators: [] };
      session = playing;
      try {
        if (context.state !== "running") await context.resume();
        if (session !== playing) return false;

        playing.output = context.createGain();
        playing.output.gain.value = 0.42;
        playing.output.connect(context.destination);
        const start = context.currentTime + 0.025;
        const duration = tokens.length * SLOT_SECONDS;
        for (const note of notes(tokens)) addNote(playing, note, start);

        function advance() {
          if (session !== playing) return;
          const elapsed = context.currentTime - start;
          if (elapsed >= duration + 0.05) {
            stop();
          } else {
            onUpdate(key, Math.max(0, Math.min(1, elapsed / duration)));
            frame = window.requestAnimationFrame(advance);
          }
        }
        advance();
        return true;
      } catch (error) {
        if (session === playing) stop();
        throw error;
      }
    }

    return { play, stop };
  }

  window.MelodyMusic = { decode, notes, overlap, pitchRange, drawRoll, drawScore, setScoreProgress, createPlayer };
})();
