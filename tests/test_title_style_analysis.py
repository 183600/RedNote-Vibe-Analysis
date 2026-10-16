# -*- coding: utf-8 -*-
"""title_style_analysis 纯函数单元测试（不触碰研究数据文件）。"""
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from title_style_analysis import half_of


class TestHalfOf(unittest.TestCase):
    def test_standard_months(self):
        self.assertEqual(half_of("202301"), "2023H1")
        self.assertEqual(half_of("202306"), "2023H1")
        self.assertEqual(half_of("202307"), "2023H2")
        self.assertEqual(half_of("202412"), "2024H2")

    def test_invalid_input(self):
        self.assertIsNone(half_of(""))
        self.assertIsNone(half_of("2023-01"))
        self.assertIsNone(half_of("20231a"))
        self.assertIsNone(half_of("2023"))


if __name__ == "__main__":
    unittest.main()
