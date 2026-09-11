import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from hubert_phonemes import build_phoneme_plan, collapse_phoneme_intervals


class UnresolvedTokenFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.dictionary = Path(self.temp_dir.name) / "dict.txt"
        self.dictionary.write_text(
            "ring\tr ih ng\n"
            "ringring\tr ih ng r ih ng\n"
            "freezing\tf r iy z ih ng\n"
            "yeah\ty ae\n"
            "cafe\tk ae f ey\n"
            "world\tw er l d\n"
            "ion\tay ax n\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_hyphenated_oov_uses_split_pronunciation_but_stays_one_term(self):
        plan = build_phoneme_plan(
            "ring-ring",
            dictionary_path=self.dictionary,
            include_singing_variants=False,
        )

        self.assertEqual(len(plan.words), 1)
        self.assertEqual(plan.words[0].text, "ring-ring")
        self.assertEqual(plan.words[0].candidates[0].source, "separator-split")
        self.assertEqual(plan.phones, ["r", "ih", "ng", "r", "ih", "ng"])

        intervals = [(phone, index / 10, (index + 1) / 10) for index, phone in enumerate(plan.phones)]
        words = collapse_phoneme_intervals(plan, intervals)
        self.assertEqual(len(words), 1)
        self.assertEqual(words[0].text, "ring-ring")
        self.assertEqual(words[0].start, 0)
        self.assertEqual(words[0].end, 0.6)

    def test_unresolved_punctuation_retries_compact_spelling(self):
        plan = build_phoneme_plan(
            "ring!ring",
            dictionary_path=self.dictionary,
            include_singing_variants=False,
        )

        self.assertEqual(len(plan.words), 1)
        self.assertEqual(plan.words[0].text, "ring!ring")
        self.assertEqual(plan.words[0].candidates[0].source, "alphanumeric-compact")
        self.assertEqual(plan.phones, ["r", "ih", "ng", "r", "ih", "ng"])

    def test_dropped_g_spelling_retries_ing_but_preserves_lyrics(self):
        for lyric in ("freezin", "freezin'"):
            with self.subTest(lyric=lyric):
                plan = build_phoneme_plan(
                    lyric,
                    dictionary_path=self.dictionary,
                    include_singing_variants=False,
                )

                self.assertEqual(len(plan.words), 1)
                self.assertEqual(plan.words[0].text, lyric)
                self.assertEqual(plan.words[0].candidates[0].source, "colloquial-ing")
                self.assertEqual(plan.phones, ["f", "r", "iy", "z", "ih", "ng"])

    def test_expressive_repetition_and_accents_resolve_without_rewriting(self):
        plan = build_phoneme_plan(
            "yeaaah café",
            dictionary_path=self.dictionary,
            include_singing_variants=False,
        )

        self.assertEqual([word.text for word in plan.words], ["yeaaah", "café"])
        self.assertEqual(plan.words[0].candidates[0].source, "expressive-repeat")
        self.assertEqual(plan.words[1].candidates[0].source, "hubert-dictionary")

    def test_curated_stylized_alias_beats_initialism(self):
        plan = build_phoneme_plan(
            "WRLD",
            dictionary_path=self.dictionary,
            include_singing_variants=False,
        )

        self.assertEqual(plan.words[0].text, "WRLD")
        self.assertEqual(plan.words[0].candidates[0].source, "lyric-alias")
        self.assertEqual(plan.phones, ["w", "er", "l", "d"])

    def test_ambiguous_dictionary_word_keeps_lyric_alias_for_audio_scoring(self):
        plan = build_phoneme_plan(
            "ion",
            dictionary_path=self.dictionary,
            include_singing_variants=False,
        )

        sources = [candidate.source for candidate in plan.words[0].candidates]
        self.assertEqual(sources[0], "hubert-dictionary")
        self.assertIn("lyric-alias", sources)

    def test_token_local_g2p_works_without_sentence_pos_tagger(self):
        plan = build_phoneme_plan(
            "xanny foo-bar 24",
            dictionary_path=self.dictionary,
            include_singing_variants=False,
        )

        self.assertEqual([word.text for word in plan.words], ["xanny", "foo-bar", "24"])
        self.assertTrue(all(word.phones for word in plan.words))
        self.assertEqual(
            [word.candidates[0].source for word in plan.words],
            ["g2p-token", "separator-split", "g2p-token"],
        )


if __name__ == "__main__":
    unittest.main()
