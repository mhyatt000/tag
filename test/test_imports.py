import importlib


def test_package_importable():
    tag = importlib.import_module("tag")
    assert hasattr(tag, "__path__")


def test_subpackages_importable():
    for name in ["tag.brax", "tag.gen", "tag.gym", "tag.gym.base"]:
        module = importlib.import_module(name)
        assert module.__name__ == name
