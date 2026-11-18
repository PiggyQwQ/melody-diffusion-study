"""Select diverse windows and export verified MIDI, MusicXML and staff notation."""

from collections import Counter
from contextlib import ExitStack
from html import escape
import json
from pathlib import Path
import random
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from screen_folk_scores import BAR_SLOTS, METADATA_ARCHIVE, WINDOW_BARS, read_measures, write_csv


def sample_features(window):
    spans = window.spans
    features = set()
    if any(pitch is None for _, pitch, _ in spans):
        features.add("rest")
    if spans[0][1] is None:
        features.add("leading_rest")
    if any(pitch is None and duration > 1 for _, pitch, duration in spans):
        features.add("sustained_rest")
    if any(a[1] == b[1] is None for a, b in zip(spans, spans[1:])):
        features.add("adjacent_rests")
    if any(a[1] is not None and a[1] == b[1] for a, b in zip(spans, spans[1:])):
        features.add("same_pitch_reattack")
    if any(pitch is not None and start // BAR_SLOTS != (start + duration - 1) // BAR_SLOTS
           for start, pitch, duration in spans):
        features.add("cross_bar_tie")
    return features


def select_samples(windows, songs, groups, memberships, excluded, seed):
    pool = [(w, sample_features(w)) for w in windows if w.window_id not in excluded]
    available_features = set().union(*(features for _, features in pool))
    random.Random(seed).shuffle(pool)
    selected = []
    used_groups = set()
    feature_counts = Counter()
    pattern_counts = Counter()
    split_counts = Counter()
    volumes = sorted({s["volume"] for s in songs.values()})

    def pick(volume=None):
        options = ((w, f) for w, f in pool if groups[w.song_id] not in used_groups
                   and (volume is None or songs[w.song_id]["volume"] == volume))
        window, features = max(options, key=lambda wf: (
            len(wf[1] - feature_counts.keys()),
            -pattern_counts[frozenset(wf[1])],
            -split_counts[memberships[wf[0].song_id]],
        ))
        selected.append((window, sorted(features)))
        used_groups.add(groups[window.song_id])
        feature_counts.update(features)
        pattern_counts[frozenset(features)] += 1
        split_counts[memberships[window.song_id]] += 1

    for volume in volumes:
        pick(volume)
    for _ in range(len(selected), 20):
        pick()
    if not available_features <= feature_counts.keys() or set(split_counts) != set(memberships.values()):
        raise ValueError("Samples do not cover the required features and splits")
    return selected


def add_element(parent, tag, text):
    ET.SubElement(parent, tag).text = str(text)


NOTE_TYPES = {1: ("16th", False), 2: ("eighth", False), 3: ("eighth", True),
              4: ("quarter", False), 6: ("quarter", True), 8: ("half", False)}
PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def write_musicxml(path, spans):
    score = ET.Element("score-partwise", version="4.0")
    part_list = ET.SubElement(score, "part-list")
    part_info = ET.SubElement(part_list, "score-part", id="P1")
    add_element(part_info, "part-name", "Melody")
    part = ET.SubElement(score, "part", id="P1")
    bars = [ET.SubElement(part, "measure", number=str(i + 1)) for i in range(WINDOW_BARS)]
    attrs = ET.SubElement(bars[0], "attributes")
    add_element(attrs, "divisions", 4)
    time = ET.SubElement(attrs, "time")
    add_element(time, "beats", 2)
    add_element(time, "beat-type", 4)
    clef = ET.SubElement(attrs, "clef")
    add_element(clef, "sign", "G")
    add_element(clef, "line", 2)
    ET.SubElement(bars[4], "print", {"new-system": "yes"})

    for start, pitch, duration in spans:
        end = start + duration
        position = start
        while position < end:
            bar, offset = divmod(position, BAR_SLOTS)
            length = min(BAR_SLOTS - offset, end - position)
            if length not in NOTE_TYPES:  # 5 or 7 ticks in a 2/4 bar.
                length = 4
            note = ET.SubElement(bars[bar], "note")
            ties = []
            if pitch is None:
                ET.SubElement(note, "rest")
            else:
                node = ET.SubElement(note, "pitch")
                name = PITCH_NAMES[pitch % 12]
                add_element(node, "step", name[0])
                if len(name) == 2:
                    add_element(node, "alter", 1)
                add_element(node, "octave", pitch // 12 - 1)
                ties = (["stop"] if position > start else []) + (["start"] if position + length < end else [])
            add_element(note, "duration", length)
            for tie in ties:
                ET.SubElement(note, "tie", type=tie)
            kind, dotted = NOTE_TYPES[length]
            add_element(note, "type", kind)
            if dotted:
                ET.SubElement(note, "dot")
            if ties:
                notations = ET.SubElement(note, "notations")
                for tie in ties:
                    ET.SubElement(notations, "tied", type=tie)
            position += length
    ET.indent(score)
    ET.ElementTree(score).write(path, encoding="utf-8", xml_declaration=True)


def export_midi(path, spans):
    import mido

    ticks = 120  # 480 PPQ / four sixteenth-note slots per quarter.
    midi = mido.MidiFile(type=0, ticks_per_beat=4 * ticks)
    track = midi.add_track()
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(90)))
    track.append(mido.MetaMessage("time_signature", numerator=2, denominator=4))
    track.append(mido.Message("program_change", program=0))
    position = 0
    expected = [(start, pitch, duration) for start, pitch, duration in spans if pitch is not None]
    for start, pitch, duration in expected:
        track.append(mido.Message("note_on", note=pitch, velocity=76, time=(start - position) * ticks))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=duration * ticks))
        position = start + duration
    track.append(mido.MetaMessage("end_of_track", time=(WINDOW_BARS * BAR_SLOTS - position) * ticks))
    midi.save(path)

    # Recover absolute attacks and note lengths from the actual MIDI bytes.
    readback = mido.MidiFile(path)
    position = 0
    active = {}
    observed = []
    for message in readback.tracks[0]:
        position += message.time
        if message.type == "note_on":
            active[message.note] = position
        elif message.type == "note_off":
            start = active.pop(message.note)
            observed.append((start, message.note, position - start))
    if (active or readback.ticks_per_beat != 4 * ticks or position != WINDOW_BARS * BAR_SLOTS * ticks
            or observed != [(s * ticks, p, d * ticks) for s, p, d in expected]):
        raise ValueError(f"MIDI readback mismatch: {path.name}")


def event_tuples(measure):
    if measure.issues:
        raise ValueError(f"Invalid sampled measure {measure.number}: {sorted(measure.issues)}")
    return [[e.pitch, e.slots, e.tie_stop, e.tie_start] for e in measure.events]


def export_score(output, stem, spans, toolkit, source_spans):
    xml = output / f"{stem}.musicxml"
    write_musicxml(xml, spans)
    reconstructed, _, _ = read_measures(ET.parse(xml).getroot().find("part"))
    readback = source_spans(e for m in reconstructed for e in event_tuples(m))
    same_notes = [s for s in readback if s[1] is not None] == [s for s in spans if s[1] is not None]
    if len(reconstructed) != WINDOW_BARS or not same_notes:
        raise ValueError(f"MusicXML export differs for {stem}")
    midi = output / f"{stem}.mid"
    export_midi(midi, spans)
    toolkit.resetXmlIdSeed(1)
    if not toolkit.loadData(xml.read_text(encoding="utf-8")) or toolkit.getPageCount() != 1:
        raise ValueError(f"Could not engrave one page for {stem}")
    engraving = toolkit.renderToSVG(1)
    if engraving.count('class="measure"') != WINDOW_BARS:
        raise ValueError(f"Incomplete engraving for {stem}")
    svg = output / f"{stem}.svg"
    svg.write_text(engraving, encoding="utf-8", newline="\n")
    return dict(midi=midi.name, musicxml=xml.name, svg=svg.name)


def export_samples(output, archive_dir, selections, songs, source_spans, synthetic):
    import verovio

    output.mkdir()
    toolkit = verovio.toolkit(False)
    if not toolkit.setResourcePath(str(Path(verovio.__file__).with_name("data"))):
        raise ValueError("Could not load Verovio resources; install it to an ASCII-only path on Windows")
    toolkit.setOptions({"breaks": "encoded", "adjustPageHeight": True, "pageWidth": 2000,
                        "scale": 45, "header": "none", "footer": "none"})
    with ZipFile(archive_dir / METADATA_ARCHIVE) as metadata:
        release = json.loads(metadata.read("release-assets.json"))
    filenames = {vol: pkg["filename"] for pkg in release["packages"] for vol in pkg["volumes"]}
    rows = []
    examples = []
    with ExitStack() as stack:
        archives = {}
        for slot, (window, tokens, spans, features, metadata) in enumerate(selections, start=1):
            song = songs[window.song_id]
            filename = filenames[song["volume"]]
            if filename not in archives:
                archives[filename] = stack.enter_context(ZipFile(archive_dir / filename))
            raw = ET.fromstring(archives[filename].read(song["musicxml"]))
            measures, _, _ = read_measures(raw.find("part"))
            raw_bars = measures[window.start_bar - 1:window.start_bar - 1 + WINDOW_BARS]
            bars = [{"source_number": m.number, "events": event_tuples(m)} for m in raw_bars]
            if tuple(bars) != window.bars or source_spans(e for bar in bars for e in bar["events"]) != spans:
                raise ValueError(f"Source MusicXML differs for {window.window_id}")

            stem = f"{slot:02d}-{window.song_id.replace(':', '-')}-bar{window.start_bar}"
            files = export_score(output, stem, spans, toolkit, source_spans)
            row = dict(**metadata, title=song["title"], pitch_reference=song["pitch_reference"],
                       score_archive=filename, source_musicxml=song["musicxml"],
                       source_bar_numbers="|".join(m.number for m in raw_bars), features="|".join(features),
                       **files)
            rows.append(row)
            examples.append(dict(**row, tokens=tokens, source_bars=bars, decoded_spans=spans))
    write_csv(output / "index.csv", tuple(rows[0]), rows)
    (output / "round_trip.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8", newline="\n")
    cards = []
    for row in rows:
        title = escape(row["title"])
        caption = escape(f"{row['window_id']} · {row['split']} · 简谱参照 {row['pitch_reference'] or '未填'}")
        cards.append(f'<section><h2>{title}</h2><p>{caption}</p>'
                     f'<p><a href="{row["midi"]}">MIDI</a> · <a href="{row["musicxml"]}">MusicXML</a></p>'
                     f'<img src="{row["svg"]}" alt="{title}，八小节五线谱" loading="lazy"></section>')
    if synthetic:
        fixture_dir = output / "synthetic_tie"
        fixture_dir.mkdir()
        files = export_score(fixture_dir, "cross_bar_tie", synthetic["decoded_spans"], toolkit, source_spans)
        (fixture_dir / "round_trip.json").write_text(json.dumps(synthetic, indent=2) + "\n", encoding="utf-8")
        cards.append('<section><h2>跨小节延音测试</h2><p>合成测试片段，独立于 ACFS 数据。</p>'
                     f'<p><a href="synthetic_tie/{files["midi"]}">MIDI</a> · '
                     f'<a href="synthetic_tie/{files["musicxml"]}">MusicXML</a></p>'
                     f'<img src="synthetic_tie/{files["svg"]}" alt="跨小节延音五线谱"></section>')
    html = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>八小节编码回放抽查</title>'
            '<style>body{max-width:960px;margin:32px auto;padding:0 16px;font:16px/1.5 system-ui}'
            'section{border-top:1px solid #ccc;margin:2rem 0}img{width:100%;height:auto}</style>'
            '<h1>八小节编码回放抽查</h1><p>MIDI 采用固定每分钟 90 拍和钢琴音色。</p>'
            + "\n".join(cards) + '</html>\n')
    (output / "index.html").write_text(html, encoding="utf-8", newline="\n")
    return rows
