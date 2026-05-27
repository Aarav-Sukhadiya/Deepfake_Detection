from pathlib import Path

from dfdetect.config import InputConfig
from dfdetect.input.loader import discover
from dfdetect.input.validator import classify
from dfdetect.types import MediaKind


def _write_png(path: Path) -> None:
    # 1×1 PNG (smallest valid). Header check should classify as IMAGE.
    png = bytes.fromhex(
        "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
        "0000000A49444154789C6300010000000500010D0A2DB40000000049454E44AE426082"
    )
    path.write_bytes(png)


def test_classify_image_by_header(tmp_path: Path):
    p = tmp_path / "x.png"
    _write_png(p)
    assert classify(p, InputConfig()) == MediaKind.IMAGE


def test_classify_unknown_returns_none(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    assert classify(p, InputConfig()) is None


def test_discover_walks_directory(tmp_path: Path):
    img = tmp_path / "a.png"
    _write_png(img)
    (tmp_path / "ignore.txt").write_text("nope")
    items = discover([tmp_path], InputConfig())
    assert [i.path.name for i in items] == ["a.png"]
    assert items[0].kind == MediaKind.IMAGE


def test_discover_dedup(tmp_path: Path):
    img = tmp_path / "a.png"
    _write_png(img)
    items = discover([img, img, tmp_path], InputConfig())
    assert len(items) == 1
