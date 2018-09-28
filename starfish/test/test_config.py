from starfish.util.config import Config
from json import loads


simple_str = '{"a": 1}'
simple_map = loads(simple_str)

def test_simple_config_value_str():
    config = Config(simple_str)
    assert config["a"] == 1


def test_simple_config_value_map():
    config = Config(simple_map)
    assert config["a"] == 1


def test_simple_config_value_default_key(monkeypatch):
    monkeypatch.setenv("STARFISH_CONFIG", simple_str)
    config = Config()
    assert config["a"] == 1


def test_simple_config_value_file(tmpdir):
    f = tmpdir.join("config.json")
    f.write(simple_str)
    config = Config(f"@{f}")
    assert config["a"] == 1
