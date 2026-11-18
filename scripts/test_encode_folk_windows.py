import unittest


class CodecTest(unittest.TestCase):
    def test_rest_tie_and_rearticulation(self):
        from encode_folk_windows import decode, encode_events, source_spans

        events = [(None, 2, False, False), (60, 6, False, True),
                  (60, 2, True, False), (60, 2, False, False),
                  (None, 1, False, False), (None, 3, False, False)]
        encoded = encode_events(events, 60, 72)
        self.assertEqual(encoded, [0, 1, 2] + [1] * 7 + [2, 1, 0, 0, 1, 1])
        expected = [(0, None, 2), (2, 60, 8), (10, 60, 2), (12, None, 1), (13, None, 3)]
        self.assertEqual(decode(encoded, 60, 72), expected)
        self.assertEqual(source_spans(events), expected)

    def test_cut_tie_and_out_of_vocabulary_fail(self):
        from encode_folk_windows import decode, encode_events, source_spans

        with self.assertRaises(ValueError):
            source_spans([(60, 8, True, False)])
        with self.assertRaises(ValueError):
            source_spans([(60, 4, False, True), (62, 4, True, False)])
        with self.assertRaises(ValueError):
            source_spans([(60, 8, False, True)])
        with self.assertRaises(ValueError):
            encode_events([(59, 8, False, False)], 60, 72)
        with self.assertRaises(ValueError):
            decode([1, 2], 60, 72)

    def test_only_declared_one_based_starts(self):
        from encode_folk_windows import song_windows

        bars = [{"source_number": str(i), "events": [(i + 30, 8, False, False)]}
                for i in range(10, 20)]
        song = {"id": "song", "runs": [{"first_bar": 10, "bars": bars, "window_starts": [11]}]}
        (window,) = list(song_windows(song))
        self.assertEqual((window.start_bar, window.spans[0], window.spans[-1]),
                         (11, (0, 41, 8), (56, 48, 8)))

    def test_exact_fingerprint_keeps_rearticulation_and_ignores_rest_segmentation(self):
        from folk_window_groups import relative_notes

        first = [(0, None, 2), (2, 60, 2), (4, 60, 4)]
        second = [(0, None, 1), (1, None, 1), (2, 63, 2), (4, 63, 4)]
        sustained = [(0, None, 2), (2, 63, 6)]
        self.assertEqual(relative_notes(first), relative_notes(second))
        self.assertNotEqual(relative_notes(first), relative_notes(sustained))

    def test_near_match_across_transposition(self):
        from encode_folk_windows import Window
        from folk_window_groups import near_matches

        notes = tuple((i * 4, 60 + i % 5, 4) for i in range(16))
        shifted = tuple((start, pitch + 3 + (start == 0), length) for start, pitch, length in notes)
        matched = list(near_matches([Window("a", 1, (), notes), Window("b", 1, (), shifted)]))
        self.assertEqual([(result.shift, result.matches) for _, _, result in matched], [(3, 15)])

    def test_near_match_with_one_changed_onset_in_each_block(self):
        from encode_folk_windows import Window
        from folk_window_groups import near_matches

        notes = tuple((i, 60 + i % 5, 1) for i in range(64))
        shifted = tuple((i, pitch + 2 + (i % 16 == 0), length) for i, pitch, length in notes)
        matched = list(near_matches([Window("a", 1, (), notes), Window("b", 1, (), shifted)]))
        self.assertEqual([(result.shift, result.matches) for _, _, result in matched], [(2, 60)])


if __name__ == "__main__":
    unittest.main()
