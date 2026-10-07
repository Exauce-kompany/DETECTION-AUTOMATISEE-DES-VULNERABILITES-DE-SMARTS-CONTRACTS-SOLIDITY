"""Regressions for lexical matching and probability units in static analysis."""

import unittest

from src.analysis.common import mask_solidity_non_code
from src.analysis.performance_analyzer import (
    analyze_contract_performance,
    extract_metrics,
    strip_comments,
)
from src.analysis.risk_analyzer import (
    analyze_contract_risks,
    normalize_probability,
    strip_comments_preserve_lines,
)


class SolidityLexicalTests(unittest.TestCase):
    def test_comments_preserve_positions_whitespace_and_token_boundaries(self):
        code = "function/* description\r\n\tcontinues */run() {} // trailing\r\n"
        clean = strip_comments(code)

        self.assertEqual(len(clean), len(code))
        self.assertEqual(clean.index("run"), code.index("run"))
        self.assertEqual(clean.splitlines(), strip_comments_preserve_lines(code))
        for position, character in enumerate(code):
            if character.isspace():
                self.assertEqual(clean[position], character)
        self.assertEqual(extract_metrics(code)["functions"], 1)
        self.assertEqual(analyze_contract_risks(code)["metrics"]["functions"], 1)

    def test_comment_markers_inside_literals_do_not_hide_following_code(self):
        code = """contract C {
    function run() external {
        string memory url = "https://example.org"; require(tx.origin == owner);
        string memory text = '/* documentation */'; target.call("");
    }
}"""
        risks = analyze_contract_risks(code)["static_analysis"]["findings"]
        risk_locations = {(finding["id"], finding["line"]) for finding in risks}

        self.assertIn(("TX_ORIGIN", 3), risk_locations)
        self.assertIn(("UNCHECKED_CALL", 4), risk_locations)
        performance = analyze_contract_performance(code)
        self.assertEqual(performance["metrics"]["external_calls"], 1)
        finding = next(
            finding for finding in performance["findings"] if finding["id"] == "PERF-CALL-002"
        )
        self.assertEqual(finding["line"], 4)
        self.assertIn('"https://example.org"', strip_comments(code))

    def test_literals_and_comments_do_not_create_executable_findings(self):
        code = """contract C {
    string constant a = "tx.origin; target.call(); for (;;) {}";
    string constant b = 'selfdestruct(owner); while (true) {}';
    // tx.origin; target.send(1); assembly {}
    /* delegatecall(); block.timestamp; for (;;) {} */
}"""
        risk = analyze_contract_risks(code)
        self.assertEqual(risk["static_analysis"]["findings"], [])
        performance = analyze_contract_performance(code)
        for metric in ("loops", "external_calls", "assembly_blocks", "functions"):
            self.assertEqual(performance["metrics"][metric], 0, metric)

    def test_escaped_quotes_keep_comment_markers_inside_literals(self):
        code = r"""string memory a = "\" // tx.origin /* "; require(tx.origin == owner);
string memory b = '\' // selfdestruct(owner) /* '; target.call("");"""
        clean = mask_solidity_non_code(code)
        self.assertNotIn("selfdestruct", clean)
        self.assertEqual(clean.count("tx.origin"), 1)
        self.assertEqual(extract_metrics(code)["external_calls"], 1)
        risk = analyze_contract_risks(code)
        self.assertEqual(
            {finding["id"] for finding in risk["static_analysis"]["findings"]},
            {"UNCHECKED_CALL", "TX_ORIGIN"},
        )

    def test_multiline_comments_keep_finding_line_numbers(self):
        code = "/* first\r\nsecond */\r\n// third\r\nrequire(tx.origin == owner);\r\n"
        risk = analyze_contract_risks(code)
        self.assertEqual(risk["metrics"]["lines"], 4)
        self.assertEqual(risk["static_analysis"]["findings"][0]["line"], 4)

    def test_equality_checks_are_not_counted_as_state_writes(self):
        metrics = extract_metrics("require(balance == threshold); balance += 1;")
        self.assertEqual(metrics["estimated_state_writes"], 1)


class FindingEvidenceTests(unittest.TestCase):
    def test_delegatecall_evidence_keeps_the_original_signature(self):
        statement = (
            'implementation.delegatecall(abi.encodeWithSignature("initialize(address)", owner));'
        )
        findings = analyze_contract_risks(statement)["static_analysis"]["findings"]
        finding = next(item for item in findings if item["id"] == "DELEGATECALL")
        self.assertEqual(finding["evidence"], statement)

    def test_reentrancy_evidence_keeps_call_and_mutation_literals(self):
        code = 'target.call(abi.encodeWithSignature("withdraw()"));\nstatus = "pending";'
        findings = analyze_contract_risks(code)["static_analysis"]["findings"]
        finding = next(item for item in findings if item["id"] == "POTENTIAL_REENTRANCY")
        self.assertIn('"withdraw()"', finding["evidence"])
        self.assertIn('status = "pending";', finding["evidence"])
        self.assertEqual(finding["line"], 1)

    def test_loop_call_evidence_keeps_original_literal(self):
        code = 'for (;;) {\n target.call("payload  with spaces");\n}'
        findings = analyze_contract_risks(code)["static_analysis"]["findings"]
        finding = next(item for item in findings if item["id"] == "EXTERNAL_CALL_IN_LOOP")
        self.assertEqual(finding["evidence"], 'target.call("payload  with spaces");')


class LoopPerformanceTests(unittest.TestCase):
    def loop_finding(self, code):
        return next(
            (
                item
                for item in analyze_contract_performance(code)["findings"]
                if item["id"] == "PERF-LOOP-CALL-001"
            ),
            None,
        )

    def test_long_comment_does_not_hide_call_inside_loop(self):
        template = "for (uint i = 0; i < n; i++) {\n%s\nrecipients[i].transfer(1);\n}"
        for padding in ("", "/*" + "documentation " * 200 + "*/", " " * 2000):
            with self.subTest(padding_length=len(padding)):
                finding = self.loop_finding(template % padding)
                self.assertIsNotNone(finding)
                self.assertEqual(finding["line"], 3)

    def test_call_after_loop_is_not_reported_as_inside_it(self):
        code = "for (uint i = 0; i < n; i++) { total += i; }\nrecipient.transfer(1);"
        self.assertIsNone(self.loop_finding(code))

    def test_nested_parentheses_and_blocks_stay_inside_loop(self):
        code = 'while (hasNext(cursor)) { if (ready) {\n target.call("");\n} }'
        self.assertIsNotNone(self.loop_finding(code))

    def test_literals_and_comments_cannot_fake_a_loop_body_call(self):
        code = 'for (;;) { string memory text = "} target.call(0); {"; /* target.send(1); */ }'
        self.assertIsNone(self.loop_finding(code))


class ProbabilityUnitsTests(unittest.TestCase):
    def test_low_percentages_are_not_rescaled(self):
        for probability in (0, 0.5, 1, 50, 100):
            with self.subTest(probability=probability):
                result = analyze_contract_risks(
                    "contract C {}",
                    probability_vulnerable=probability,
                )
                self.assertEqual(
                    result["combined_analysis"]["model_probability_vulnerable"],
                    probability,
                )
                self.assertFalse(result["combined_analysis"]["available"])

    def test_absent_probability_stays_absent(self):
        result = analyze_contract_risks("contract C {}")
        self.assertIsNone(result["combined_analysis"]["model_probability_vulnerable"])

    def test_invalid_probability_is_rejected_instead_of_clamped(self):
        for value in (
            -0.01,
            100.01,
            float("nan"),
            float("inf"),
            -float("inf"),
            "invalid",
            True,
            False,
            object(),
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    analyze_contract_risks("contract C {}", probability_vulnerable=value)

    def test_numeric_percentage_text_keeps_its_unit(self):
        self.assertEqual(normalize_probability("0.5"), 0.5)


if __name__ == "__main__":
    unittest.main()
