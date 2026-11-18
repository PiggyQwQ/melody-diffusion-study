import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from zipfile import ZipFile


def read_fixture(bars, *, opening=("=1",), closing="==", header="**kern\t**text"):
    from read_cantopop_scores import read_cantopop

    lines = [header, "*M2/4\t*", "*C:\t*"]
    for token in opening:
        lines.append(f"{token}\t{token}" if token.startswith("=") else f"{token}\t.")
    for index, bar in enumerate(bars):
        if index:
            # Deliberately skip the printed measure number 2.
            lines.append(f"={index + 2}\t={index + 2}")
        lines.extend(f"{token}\t*" if token.startswith("*") else f"{token}\t."
                     for token in bar)
    if closing:
        lines.append(f"{closing}\t{closing}")
    lines.append("*-\t*-")

    with TemporaryDirectory() as directory:
        archive_path = Path(directory) / "scores.zip"
        with ZipFile(archive_path, "w") as archive:
            archive.writestr("corpus/Cantopop corpus list_final.csv",
                             "ID,Song Title,Year,Singer,Composer\nC0001,測試,2000,歌手,作曲家\n")
            archive.writestr("corpus/Humdrum-files/C0001.krn", "\n".join(lines))
        songs, windows, audit = read_cantopop(archive_path, 38, 86)
    json.dumps(audit)
    return songs["cantopop:C0001"], windows, audit


class CantopopParserTests(unittest.TestCase):
    def test_pickup_and_printed_numbers_have_distinct_provenance(self):
        song, windows, audit = read_fixture([["2c"]] * 8, opening=["8c", "=1"])
        self.assertEqual(song["id"], "cantopop:C0001")
        self.assertEqual(song["title"], "測試")
        self.assertEqual(song["artist"], "歌手")
        self.assertEqual(song["year"], "2000")
        self.assertEqual(song["source_kern"], "corpus/Humdrum-files/C0001.krn")
        self.assertEqual(song["source_bars"], 9)
        self.assertEqual(song["candidate_windows"], 1)
        self.assertEqual(windows[0].start_bar, 2)
        self.assertEqual([bar["source_number"] for bar in windows[0].bars],
                         ["1", "3", "4", "5", "6", "7", "8", "9"])
        self.assertEqual(windows[0].bars[0]["source_line"], 5)
        self.assertEqual(audit["stable_key_windows"], 1)

    def test_explicit_start_and_end_barlines_required(self):
        for boundaries in ({"opening": ()}, {"closing": None}):
            with self.subTest(boundaries=boundaries):
                song, windows, _ = read_fixture([["2c"]] * 8, **boundaries)
                self.assertEqual(song["source_bars"], 8)
                self.assertEqual(windows, [])

    def test_cross_bar_tie_and_same_pitch_reattack(self):
        bars = [["4r", "[4f#"], ["4f#]", "4f#"]] + [["2r"]] * 6
        _, windows, _ = read_fixture(bars)
        self.assertEqual(windows[0].spans[:3], ((0, None, 4), (4, 66, 8), (12, 66, 4)))
        self.assertEqual(windows[0].bars[1]["events"][0], [66, 4, True, False])

    def test_windows_cannot_cross_unmatched_or_cut_ties(self):
        untied = [["2c"]]
        tie_cases = ([["2c]"]] + untied * 7,
                     untied * 7 + [["[2c"]],
                     [["[2c"], ["2d]"]] + untied * 6)
        for bars in tie_cases:
            with self.subTest(bars=bars):
                _, windows, audit = read_fixture(bars)
                self.assertEqual(windows, [])
                self.assertEqual(audit["grid_windows"], 1)
                self.assertEqual(audit["uncut_sounding_windows"], 0)

    def test_tie_continue_and_null_record_do_not_add_an_onset(self):
        bars = [["[2c"], ["2c_", "."], ["2c]"]] + [["2r"]] * 5
        _, windows, _ = read_fixture(bars)
        self.assertEqual(windows[0].spans[0], (0, 60, 24))
        self.assertEqual(windows[0].bars[1]["events"], [[60, 8, True, True]])

    def test_exact_dotted_and_rational_values_and_absolute_octaves(self):
        bars = [["4.B--", "8r"], ["8e-.", "16%3cc#", "8fL"]] + [["2r"]] * 6
        _, windows, _ = read_fixture(bars)
        self.assertEqual(windows[0].bars[0]["events"],
                         [[57, 6, False, False], [None, 2, False, False]])
        self.assertEqual(windows[0].bars[1]["events"],
                         [[63, 3, False, False], [73, 3, False, False], [65, 2, False, False]])

    def test_grid_rejects_grace_tuplets_chords_and_incomplete_bars(self):
        bad_bars = (["8dq", "2c"], ["12c"] * 6, ["2c 2e"], ["4c"], ["[4c", "4c]]"])
        for bad_bar in bad_bars:
            with self.subTest(bad_bar=bad_bar):
                _, windows, audit = read_fixture([["2c"]] * 3 + [bad_bar] + [["2c"]] * 4)
                self.assertEqual(windows, [])
                self.assertEqual(audit["meter_windows"], 1)
                self.assertEqual(audit["grid_windows"], 0)

    def test_midbar_meter_change_invalidates_bar(self):
        bars = [["2c"]] * 7 + [["4c", "*M3/4", "4d"]]
        _, windows, audit = read_fixture(bars)
        self.assertEqual(windows, [])
        self.assertEqual(audit["meter_windows"], 0)

    def test_meter_change_at_a_barline_starts_a_new_run(self):
        bars = [["*M3/4", "2.c"], ["*M2/4", "2c"]] + [["2c"]] * 7
        _, windows, _ = read_fixture(bars)
        self.assertEqual([w.start_bar for w in windows], [2])

    def test_midbar_key_change_leaves_that_bar_unlabeled(self):
        bars = [["4c", "*a:", "4e"]] + [["2a"]] * 7
        _, windows, audit = read_fixture(bars)
        self.assertIsNone(windows[0].bars[0]["key"])
        self.assertEqual(windows[0].bars[1]["key"], "*a:")
        self.assertEqual(audit["stable_key_windows"], 0)

    def test_no_silent_or_out_of_vocabulary_windows(self):
        for pitch, stage_count in (("r", 0), ("GGGG", 1)):
            with self.subTest(pitch=pitch):
                _, windows, audit = read_fixture([[f"2{pitch}"]] * 8)
                self.assertEqual(windows, [])
                self.assertEqual(audit["uncut_sounding_windows"], stage_count)
                self.assertEqual(audit["in_vocab_windows"], 0)

    def test_additional_kern_spine_does_not_yield_melody(self):
        song, windows, audit = read_fixture([["2c"]] * 8, header="**kern\t**kern")
        self.assertEqual(song["candidate_windows"], 0)
        self.assertEqual(windows, [])
        self.assertEqual(audit["single_kern_songs"], 0)


if __name__ == "__main__":
    unittest.main()
