import os
import time
from json import dump, loads

from pytest import mark, raises

from starfish import data
from starfish.util.config import Config


simple_str = '{"a": 1}'
simple_map = loads(simple_str)

deep_str = '{"a": {"b": {"c": [1, 2, 3]}}}'
deep_map = loads(deep_str)

def test_simple_config_value_str():
    config = Config(simple_str)
    assert config.data["a"] == 1


def test_simple_config_value_map():
    config = Config(simple_map)
    assert config.data["a"] == 1


def test_simple_config_value_default_key(monkeypatch):
    monkeypatch.setenv("STARFISH_CONFIG", simple_str)
    config = Config()
    assert config.data["a"] == 1


def test_simple_config_value_file(tmpdir):
    f = tmpdir.join("config.json")
    f.write(simple_str)
    config = Config(f"@{f}")
    assert config.data["a"] == 1


def test_lookup_dne():
    config = Config(simple_str)
    with raises(KeyError):
        config.lookup(["foo"])
    assert config.lookup(["foo"], 1) == 1
    assert config.lookup(["foo", "bar"], 2) == 2


def test_lookup_deep():
    config = Config(deep_str)
    assert config.lookup(["a"]) == {"b": {"c": [1, 2, 3]}}
    assert config.lookup(["a", "b"]) == {"c": [1, 2, 3]}
    assert config.lookup(["a", "b", "c"]) == [1, 2, 3]
    with raises(AttributeError):
        config.lookup(["a", "b", "c", "d"])
    assert config.lookup(["a", "b", "c", "d"], "x") == "x"


def test_cache_config():
    config = Config("""{
        "caching": {
             "enabled": true,
             "size_limit": 5e9,
             "directory": "/tmp"
         }
    }""")
    cache_config = config.lookup(("caching",), {})
    assert cache_config["enabled"]
    assert cache_config["size_limit"] == 5 * 10 ** 9


@mark.parametrize("name,config", (
    ("enabled", {
        "expected": 69688,
        "validation": {"strict": True},
        "backend": {
            "caching": {
                "directory": "REPLACEME",
            }}}),
    ("disabled", {
        "expected": 0,
        "validation": {"strict": True},
        "backend": {
            "caching": {
                # No directory, ergo disabled.
            }}}),
    ("limited", {
        "expected": 0,
        "validation": {"strict": True},
        "backend": {
            "caching": {
                "directory": "REPLACEME",
                "size_limit": 9e5,
            }}}),
))
def test_cache_merfish(tmpdir, name, config, monkeypatch):

    if "directory" in config["backend"]["caching"]:
        # If not present, then this cache is disabled.
        config["backend"]["caching"]["directory"] = str(tmpdir / "caching")

    config_file = tmpdir / "config"
    with open(config_file, "w") as o:
        dump(config, o)
    monkeypatch.setitem(os.environ, "STARFISH_CONFIG", f"@{config_file}")

    start_1 = time.time()
    fish = data.MERFISH(use_test_data=True)
    stop_1 = time.time()
    cache_size = get_size(tmpdir / "caching")
    assert config["expected"] == cache_size
    start_2 = time.time()
    fish = data.MERFISH(use_test_data=True)
    stop_2 = time.time()
    assert config["expected"] == cache_size
    assert (stop_2-start_2) < (stop_1-start_1)


def get_size(start_path = '.'):
    """helper method for listing file sizes in a directory"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size
