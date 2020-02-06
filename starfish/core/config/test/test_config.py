import os
import tempfile
import warnings
from json import dump, loads

from diskcache import Cache
from pytest import mark, raises

from starfish import data
from starfish.core.util.config import Config, NestedDict
from .. import environ, StarfishConfig


def test_nested_dict():
    rd = dict()
    nd = NestedDict()
    rd[1] = {2: None}
    nd[3][4][5][6] = 7
    nd.update(rd)
    nd[1][2] = 9
    nd[1][8] = 10


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


def test_simple_config_false():
    # config says the value is False.  Do we return False?
    config = Config("{\"a\": false}")
    assert config.lookup(("a",), None) is False

    # config says the value is not known.  Do we return the default valueS?
    config = Config("{}")
    assert config.lookup(("a",), None) is None
    assert config.lookup(("a",), False) is False


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


def test_cache_remove_step_wise():
    config = Config("""{
        "a": {
             "b": {
                "c": true,
                "d": true
             },
             "e": 1
         },
         "f": 2
    }""")
    config.lookup(("a", "b", "c"), {}, remove=True)
    assert "c" not in config.data["a"]["b"]
    config.lookup(("a", "b"), {}, remove=True)
    assert "b" not in config.data["a"]
    config.lookup(("a",), {}, remove=True)
    assert "a" not in config.data


def test_cache_remove_all():
    config = Config("""{
        "a": {
             "b": {
                "c": true
             }
         }
    }""")
    config.lookup(("a", "b", "c"), {}, remove=True)
    assert not config.data


def test_starfish_config_value_default_key(monkeypatch):
    with monkeypatch.context() as mc:
        mc.setenv("STARFISH_CONFIG", simple_str)
        config = StarfishConfig()._config_obj
        assert config.data["a"] == 1


@mark.parametrize("name,expected,config", (
    ("enabled", (2658848, 4e6), {
        "validation": {"strict": True},
        "slicedimage": {
            "caching": {
                "directory": "REPLACEME",
            }}}),
    ("disabled", (0, 0), {
        "validation": {"strict": True},
        "slicedimage": {
            "caching": {
                "size_limit": 0,
            }}}),
    ("limited", (1e5, 3e6), {
        "validation": {"strict": True},
        "slicedimage": {
            "caching": {
                "directory": "REPLACEME",
                "size_limit": 1e5,
            }}}),
))
def test_cache_merfish(tmpdir, name, expected, config, monkeypatch):

    cache_enabled = (0 != config["slicedimage"]["caching"].get("size_limit", None))
    if cache_enabled:
        config["slicedimage"]["caching"]["directory"] = str(tmpdir / "caching")

    with monkeypatch.context() as mc:
        setup_config(config, tmpdir, mc)

        # Run 1
        data.MERFISH(use_test_data=True).fov().get_image("primary")

        # Run 2
        if cache_enabled:
            data.MERFISH(use_test_data=True).fov().get_image("primary")

        # Check constraints
        if cache_enabled:
            # Enforce smallest size
            cache = Cache(str(tmpdir / "caching"))
            cache.cull()

        cache_size = get_size(tmpdir / "caching")
        min, max = expected
        assert (min <= cache_size) and (cache_size <= max)


def test_starfish_config(tmpdir, monkeypatch):
    with monkeypatch.context() as mc:
        config = {"slicedimage": {"caching": {"size_limit": 0}}}
        setup_config(config, tmpdir, mc, STARFISH_SLICEDIMAGE_CACHING_SIZE_LIMIT="1")
        assert 1 == StarfishConfig().slicedimage["caching"]["size_limit"]

    with monkeypatch.context() as mc:
        config = {"slicedimage": {"caching": {"size_limit": 0}}}
        setup_config(config, tmpdir, mc, SLICEDIMAGE_CACHING_SIZE_LIMIT="1")
        assert 1 == StarfishConfig().slicedimage["caching"]["size_limit"]

    with monkeypatch.context() as mc:
        config = {"validation": {"strict": True}}
        setup_config(config, tmpdir, mc)
        assert StarfishConfig().strict

    with monkeypatch.context() as mc:
        config = {"validation": {"strict": False}}
        setup_config(config, tmpdir, mc, STARFISH_VALIDATION_STRICT=None)  # Disable from travis
        assert not StarfishConfig().strict

    with monkeypatch.context() as mc:
        setup_config({}, tmpdir, mc, STARFISH_VALIDATION_STRICT="true")
        assert StarfishConfig().strict

    with monkeypatch.context() as mc:
        setup_config({}, tmpdir, mc, STARFISH_VALIDATION_STRICT="false")
        assert not StarfishConfig().strict


def test_starfish_warn(tmpdir, monkeypatch):
    config = {"unknown": True}
    with monkeypatch.context() as mc:
        setup_config(config, tmpdir, mc, STARFISH_SLICEDIMAGE_CACHING_SIZE_LIMIT="1")
        with warnings.catch_warnings(record=True) as warnings_:
            StarfishConfig()
            assert len(warnings_) == 1  # type: ignore


def test_starfish_environ(monkeypatch):
    with monkeypatch.context() as mc:
        mc.setenv("STARFISH_CONFIG", "{}")
        assert not StarfishConfig().strict
        with environ(VALIDATION_STRICT="true"):
            assert StarfishConfig().strict


def test_starfish_environ_overrides(monkeypatch, tmp_path):
    """Test that configuration set by specific environment variable overrides the config file or
    global config."""

    # config is provided by the STARFISH_CONFIG path
    with monkeypatch.context() as mc:
        mc.setenv("STARFISH_CONFIG", "{\"validation\": {\"strict\": false}}")
        assert not StarfishConfig().strict
        with environ(VALIDATION_STRICT="true"):
            assert StarfishConfig().strict

    with monkeypatch.context() as mc:
        mc.setenv("STARFISH_CONFIG", "{\"validation\": {\"strict\": true}}")
        assert StarfishConfig().strict
        with environ(VALIDATION_STRICT="false"):
            assert not StarfishConfig().strict

    with tempfile.NamedTemporaryFile(mode="w+", dir=tmp_path, delete=False) as config_tempfile:
        config_tempfile.write("{\"validation\": {\"strict\": false}}")

    with monkeypatch.context() as mc:
        mc.setenv("STARFISH_CONFIG", f"@{config_tempfile.name}")
        assert not StarfishConfig().strict
        with environ(VALIDATION_STRICT="true"):
            assert StarfishConfig().strict


def test_starfish_environ_warn(tmpdir, monkeypatch):
    with monkeypatch.context() as mc:
        setup_config({}, tmpdir, mc)
        with environ(UNKNOWN="true"):
            with warnings.catch_warnings(record=True) as warnings_:
                StarfishConfig()
                assert len(warnings_) == 1  # type: ignore


def test_starfish_environ_warn_on_duplicate(tmpdir, monkeypatch):
    with monkeypatch.context() as mc:
        setup_config({}, tmpdir, mc,
                     SLICEDIMAGE_CACHING_DIRECTORY=None,
                     STARFISH_SLICEDIMAGE_CACHING_DIRECTORY=None)
        assert "directory" not in StarfishConfig().slicedimage
        with environ(SLICEDIMAGE_CACHING_DIRECTORY="foo",
                     STARFISH_SLICEDIMAGE_CACHING_DIRECTORY="bar"):
            with warnings.catch_warnings(record=True) as warnings_:
                assert "bar" == StarfishConfig().slicedimage["caching"]["directory"]
                assert len(warnings_) == 1  # type: ignore

#
# HELPERS
#


def get_size(start_path='.'):
    """helper method for listing file sizes in a directory"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size


def setup_config(config, tmpdir, monkeypatch, **environment_variables):
    config_file = tmpdir / "config"
    with open(config_file, "w") as o:
        dump(config, o)
    monkeypatch.setitem(os.environ, "STARFISH_CONFIG", f"@{config_file}")
    for k, v in environment_variables.items():
        if v is None:
            monkeypatch.delitem(os.environ, k, raising=False)
        else:
            monkeypatch.setitem(os.environ, k, v)
