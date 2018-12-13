import os
import time
from json import dump, loads

from diskcache import Cache
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
             "enabled": True,
             "size_limit": 5e9,
             "directory": "/tmp"
         }
    }""")
    cache_config = config.lookup(("caching",), {})
    assert cache_config["enabled"]
    assert cache_config["size_limit"] == 5 * 10 ** 9


@mark.parametrize("name,config", (
    ("enabled", {
        "expected": (3e6, 4e6),
        "validation": {"strict": True},
        "backend": {
            "caching": {
                "directory": "REPLACEME",
            }}}),
    ("disabled", {
        "expected": (0, 0),
        "validation": {"strict": True},
        "backend": {
            "caching": {
                # No directory, ergo disabled.
            }}}),
    ("limited", {
        "expected": (1e5, 3e6),
        "validation": {"strict": True},
        "backend": {
            "caching": {
                "directory": "REPLACEME",
                "size_limit": 1e5,
            }}}),
))
def test_cache_merfish(tmpdir, name, config, monkeypatch):

    cache_enabled = "directory" in config["backend"]["caching"]
    if cache_enabled:
        config["backend"]["caching"]["directory"] = str(tmpdir / "caching")

    config_file = tmpdir / "config"
    with open(config_file, "w") as o:
        dump(config, o)
    monkeypatch.setitem(os.environ, "STARFISH_CONFIG", f"@{config_file}")

    # Run 1
    data.MERFISH(use_test_data=True).fov()["primary"]

    # Run 2
    if cache_enabled:
        data.MERFISH(use_test_data=True).fov()["primary"]

    # Check constraints
    if cache_enabled:
        # Enforce smallest size
        cache = Cache(str(tmpdir / "caching"))
        cache.cull()

    cache_size = get_size(tmpdir / "caching")
    min, max = config["expected"]
    assert (min <= cache_size) and (cache_size <= max)


def get_size(start_path = '.'):
    """helper method for listing file sizes in a directory"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size
