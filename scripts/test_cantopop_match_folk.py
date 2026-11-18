import unittest


def window(song, group, bar, pitches, rest_slots=0, split="train"):
    tokens = [0] + [1] * (rest_slots - 1) if rest_slots else []
    tokens.extend(pitch - 38 + 2 for pitch in pitches)
    tokens.extend([1] * (64 - len(tokens)))
    return dict(window_id=f"{song}@{bar}", song_id=song, group_id=group,
                volume="fixture", split=split, start_bar=bar, tokens=tokens)


class MatchFolkTrainTest(unittest.TestCase):
    def test_matches_budget_and_preserves_original_rows(self):
        from cantopop_match_folk import match_folk_train

        pop = [window("small", "p1", 1, [62, 65]),
               window("large", "p2", 1, [60, 63], 4),
               window("large", "p2", 2, [60, 63], 4)]
        folk = [window("a", "shared", 1, [60, 63], 4),
                window("a", "shared", 2, [60, 63], 4),
                window("variant", "shared", 1, [62, 65]),
                window("b", "other", 1, [62, 65]),
                window("b", "other", 2, [78, 80], 12)]
        selected, pairs, stats = match_folk_train(folk, pop, 38, 86, seed=12)
        self.assertEqual({r["window_id"] for r in selected}, {"a@1", "a@2", "b@1"})
        self.assertTrue(all(any(row is original for original in folk) for row in selected))
        self.assertEqual({p["pop_song_id"]: (p["folk_group_id"], p["windows"]) for p in pairs},
                         {"large": ("shared", 2), "small": ("other", 1)})
        self.assertEqual(stats["matched_folk_train"]["tokens"], 192)
        self.assertEqual(stats["matched_folk_train"]["mean"]["rest_slots"], 8 / 3)
        self.assertEqual(list(stats["mean_differences_folk_minus_pop"].values()), [0] * 4)
        self.assertEqual(match_folk_train(folk[::-1], pop[::-1], 38, 86, seed=12),
                         (selected, pairs, stats))

    def test_does_not_reuse_group_when_capacity_is_insufficient(self):
        from cantopop_match_folk import match_folk_train

        folk = [window("a", "shared", 1, [60]), window("b", "shared", 1, [60])]
        pop = [window("a", "p1", 1, [60]), window("b", "p2", 1, [60])]
        with self.assertRaisesRegex(ValueError, "distinct ACFS training groups"):
            match_folk_train(folk, pop, 38, 86, seed=12)

    def test_rejects_held_out_folk_data(self):
        from cantopop_match_folk import match_folk_train

        pop = [window("p", "p", 1, [60])]
        folk = [window("f", "f", 1, [60], split="validation")]
        with self.assertRaisesRegex(ValueError, "train windows"):
            match_folk_train(folk, pop, 38, 86, seed=12)


if __name__ == "__main__":
    unittest.main()
