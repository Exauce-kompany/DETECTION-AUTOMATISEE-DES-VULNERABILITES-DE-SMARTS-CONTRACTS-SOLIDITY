import copy
import unittest

import numpy as np

from src.build_dataset_v3 import assign_groups, audit_splits, normalize_sample, quarantine_conflicts_and_deduplicate, split_records
from src.preprocessing_v3 import as_windows, build_vocabulary, encode, load_config, prepare_code, tokenize


class PreprocessingTests(unittest.TestCase):
    def test_comments_are_removed_and_quoted_urls_are_preserved(self):
        code = 'contract C { string s = "https://host/*text*/"; /* @vulnerable_at_lines: 9 */ uint n; // SWC-107\n}'
        tokens = tokenize(code, normalize_identifiers=False)
        self.assertIn('"https://host/*text*/"', tokens)
        self.assertNotIn("SWC", " ".join(tokens))
        self.assertNotIn("vulnerable_at_lines", " ".join(tokens))

    def test_renaming_identifiers_removes_annotation_names_consistently(self):
        self.assertEqual(tokenize('function bug_reentrancy(uint bug) { bug += 1; }'), tokenize('function routine(uint amount) { amount += 1; }'))
        self.assertIn("+=", tokenize("x += 1;"))

    def test_all_tokens_including_tail_are_encoded(self):
        tokens = tokenize("uint balance; " * 300 + "selfdestruct(msg.sender);")
        vocab = build_vocabulary([tokens], 512)
        windows, info = prepare_code("uint balance; " * 300 + "selfdestruct(msg.sender);", vocab, 256)
        self.assertGreater(len(windows), 1)
        np.testing.assert_array_equal(windows.ravel()[:len(tokens)], encode(tokens, vocab))
        self.assertEqual(info["tokens_used"], len(tokens))
        self.assertFalse(info["truncated"])
        self.assertIn(vocab["selfdestruct"], windows[-1])

    def test_empty_code_after_comments_is_rejected(self):
        with self.assertRaises(ValueError):
            prepare_code("/* annotation only */", {"<PAD>": 0, "<UNK>": 1}, 256)

    def test_vocabulary_is_fitted_on_supplied_training_tokens_only(self):
        vocab = build_vocabulary([["train_only"]], 512)
        self.assertNotIn("held_out_only", vocab)
        self.assertEqual(encode(["held_out_only"], vocab)[0], 1)
        self.assertFalse(np.array_equal(encode(["unknown_a"], vocab), encode(["unknown_b"], vocab)))


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def record(self, code, label, index, **extra):
        return normalize_sample({"context": code, "has_vulnerability": label, "source_dataset": "synthetic", **extra}, "train", index, self.config)

    def test_opposite_labels_are_quarantined_without_majority_relabeling(self):
        records = [self.record("contract A { uint x; }", label, i) for i, label in enumerate([0, 0, 1])]
        kept, quarantine, duplicates = quarantine_conflicts_and_deduplicate(records)
        self.assertEqual(len(kept), 0)
        self.assertEqual(len(quarantine), 3)
        self.assertEqual(duplicates, [])

    def test_consistent_duplicates_retain_provenance(self):
        records = [self.record("contract A { uint x; }", 1, i) for i in range(2)]
        kept, quarantine, duplicates = quarantine_conflicts_and_deduplicate(records)
        self.assertEqual((len(kept), len(quarantine), len(duplicates)), (1, 0, 1))
        self.assertEqual(len(kept[0]["source_refs"]), 2)

    def test_partial_cgt_negatives_and_contextless_functions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "incomplete_target_coverage"):
            self.record("contract A {}", 0, 0, source_dataset="CGT", metadata={"assessed_types": "arithmetic|reentrancy"})
        with self.assertRaisesRegex(ValueError, "parent_contract_context"):
            self.record("function f() {}", 1, 0, granularity="function")

    def test_renamed_variants_are_grouped(self):
        rows = [self.record("contract A { uint x = 10; }", 0, 0), self.record("contract B { uint y = 11; }", 1, 1)]
        assign_groups(rows)
        self.assertEqual(rows[0]["group_id"], rows[1]["group_id"])

    def test_audit_rejects_shared_effective_representations(self):
        row = self.record("contract A { uint x; }", 0, 0)
        assign_groups([row])
        vocab = build_vocabulary([row["tokens"]], 512)
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit_splits({"train": [row], "test": [copy.deepcopy(row)]}, vocab)

    def test_source_holdout_includes_the_entire_related_group(self):
        rows = []
        for i in range(100):
            row = self.record("contract A { " + "uint x; " * (i + 1) + "}", i % 2, i)
            rows.append(row)
        rows[0]["sources"].append("CGT")
        assign_groups(rows)
        splits = split_records(rows, self.config)
        vocab = build_vocabulary((r["tokens"] for r in splits["train"]), 512)
        self.assertTrue(audit_splits(splits, vocab)["passed"])
        self.assertIn(rows[0], splits["source_holdout"])
        self.assertEqual(sum(len(r) for r in splits.values()), len(rows))


if __name__ == "__main__":
    unittest.main()
