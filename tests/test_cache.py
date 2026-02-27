"""Tests for the data cache."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from sports_predict.utils.cache import DataCache


class TestDataCache:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = DataCache(cache_dir=self.tmpdir, ttl_hours=1)

    def test_set_and_get_df(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        self.cache.set_df("test_key", df)
        result = self.cache.get_df("test_key")
        assert result is not None
        pd.testing.assert_frame_equal(df, result)

    def test_get_missing_df(self):
        result = self.cache.get_df("nonexistent")
        assert result is None

    def test_set_and_get_json(self):
        data = {"key": "value", "number": 42}
        self.cache.set_json("test_json", data)
        result = self.cache.get_json("test_json")
        assert result == data

    def test_clear(self):
        self.cache.set_json("a", {"x": 1})
        self.cache.set_json("b", {"x": 2})
        count = self.cache.clear()
        assert count == 2
        assert self.cache.get_json("a") is None

    def test_different_keys_different_files(self):
        self.cache.set_json("key1", {"a": 1})
        self.cache.set_json("key2", {"b": 2})
        assert self.cache.get_json("key1") == {"a": 1}
        assert self.cache.get_json("key2") == {"b": 2}
