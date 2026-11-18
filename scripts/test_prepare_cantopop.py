import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


class CantopopPreparationTest(unittest.TestCase):
    def test_cross_corpus_exact_matching_ignores_transposition(self):
        from encode_folk_windows import Window
        from prepare_cantopop import exact_cross_corpus

        notes = tuple((i, 60 + (i % 5), 1) for i in range(64))
        folk = Window("folk", 10, (), notes)
        shifted = tuple((i, pitch + 3, length) for i, pitch, length in notes)
        pop = Window("pop", 2, (), shifted)
        other = Window("other", 2, (), shifted[:-1] + ((63, 72, 1),))
        matches = exact_cross_corpus([folk], [pop, other])
        self.assertEqual([(a.window_id, b.window_id) for a, b in matches], [("folk@10", "pop@2")])

    def test_training_description_uses_one_best_template_per_song(self):
        from prepare_cantopop import describe_training

        row = {"song_id": "a", "tokens": [2, 4, 6, 9, 11, 0] + [1] * 58}
        stats = describe_training([row], 60, 86)
        self.assertEqual(stats["pitch_class_onsets"], [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0])
        self.assertEqual(sum(stats["pitch_class_duration_slots"]), 5)
        self.assertEqual(stats["best_pentatonic_fit"]["pooled_onset_coverage"], 1)
        self.assertEqual(stats["best_pentatonic_fit"]["pooled_duration_coverage"], 1)

    def test_sparse_exact_windows_are_not_evidence_of_a_shared_work(self):
        from prepare_cantopop import informative_melody

        self.assertFalse(informative_melody(((0, 0, 8),)))
        self.assertFalse(informative_melody(tuple((i, i % 2, 1) for i in range(16))))
        self.assertTrue(informative_melody(tuple((i, i % 3, 1) for i in range(12))))

    def test_choose_one_representative_for_each_training_group(self):
        from prepare_cantopop import one_song_per_group

        rows = [{"group_id": "g1", "song_id": "a"}, {"group_id": "g1", "song_id": "b"},
                {"group_id": "g1", "song_id": "b"}, {"group_id": "g2", "song_id": "c"}]
        self.assertEqual(one_song_per_group(rows), rows[1:])

    def test_shared_vocabulary_rejects_wrong_meter_or_resolution(self):
        from prepare_cantopop import read_folk

        with TemporaryDirectory() as directory:
            vocab = dict(id_to_token=["REST", "HOLD", "MIDI_60"], window_slots=64, size=3,
                         midi_min=60, midi_max=60, meter="2/4", slots_per_quarter=4)
            for extra in (dict(meter="4/4"), dict(slots_per_quarter=8)):
                with self.subTest(extra=extra):
                    path = Path(directory)
                    (path / "vocab.json").write_text(json.dumps({**vocab, **extra}), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "2/4"):
                        read_folk(path)


if __name__ == "__main__":
    unittest.main()
