# -*- coding: utf-8 -*-
"""data_audit 的单元测试（临时小样本，不读取研究数据）。"""
import json
import pathlib
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from data_audit import _type_name, scan_dataset, table_missing


class TestTypeName(unittest.TestCase):
    def test_scalar_and_container_types(self):
        self.assertEqual(_type_name(None), "null")
        self.assertEqual(_type_name(True), "bool")
        self.assertEqual(_type_name(1), "number")
        self.assertEqual(_type_name(1.5), "number")
        self.assertEqual(_type_name("x"), "str")
        self.assertEqual(_type_name([]), "list")
        self.assertEqual(_type_name({}), "dict")


class TestTableMissing(unittest.TestCase):
    def test_only_reports_gaps(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
        got = table_missing(df)
        self.assertEqual(got, {"a": {"missing": 1, "share": round(1 / 3, 6)}})


class TestScanDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "sample.jsonl"
        records = [
            {"note_id": "a", "desc": "开心" * 30, "local_time": "2023010112",
             "liked_count": 5, "domain": "美食", "note_title": "t"},
            {"note_id": "a", "desc": "开心" * 30, "local_time": "2023010112",
             "liked_count": 5, "domain": "美食", "note_title": "t"},
            {"note_id": "b", "note_content": "难过" * 30,
             "time": "2023020110", "likes": 0, "domain": "未知"},
            {"note_id": "c", "desc": "", "local_time": "bad",
             "liked_count": -1, "domain": None},
        ]
        self.path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_quality_counters_and_arrays(self):
        scan = scan_dataset("sample", str(self.path))
        self.assertEqual(scan["n"], 4)
        q = scan["quality"]
        self.assertEqual(q["empty_body"], 1)
        self.assertEqual(q["short_body"], 1)
        self.assertEqual(q["duplicate_note_rows"], 1)
        self.assertEqual(q["duplicate_content_rows"], 1)
        self.assertEqual(q["missing_domain"], 1)
        self.assertEqual(q["unknown_domain"], 1)
        self.assertEqual(q["missing_or_invalid_time"], 1)
        self.assertEqual(q["negative_engagement"], 1)
        self.assertEqual(int((scan["body_len"] > 0).sum()), 3)
        self.assertEqual(scan["month"][:2], ["202301", "202301"])

    def test_schema_coverage(self):
        scan = scan_dataset("sample", str(self.path))
        self.assertEqual(scan["schema"]["note_id"]["present"], 4)
        self.assertEqual(scan["schema"]["note_content"]["present"], 1)
        self.assertEqual(scan["schema"]["desc"]["present"], 3)


if __name__ == "__main__":
    unittest.main()
