"""Check HEVC and AV1 HEIF exports without importing image codecs."""

from test_tf_producer_cases import CORPUS, _events

from tboardfs import TensorBoardFS
from tboardfs.classify import detect_extension


def test_heic_and_avif_image_and_tensor_exports() -> None:
    fs = TensorBoardFS(CORPUS / "heic")
    values = {value.tag: value for e in _events("heic") for value in e.summary.value}
    for suffix, item, config in (
        ("heic", b"hvc1", b"hvcC"),
        ("avif", b"av01", b"av1C"),
    ):
        image = values[f"{suffix}-image"].image
        assert (image.width, image.height, image.colorspace) == (32, 16, 3)
        blob = image.encoded_image_string
        assert blob[4:12] == b"ftyp" + suffix.encode()
        assert item in blob and config in blob
        assert list(values[f"{suffix}-tensor"].tensor.string_val) == [
            b"32",
            b"16",
            blob,
        ]
        for tag in (f"{suffix}-image", f"{suffix}-tensor"):
            assert fs.read(f"/images/{tag}/000001.{suffix}", len(blob) + 1, 0) == blob
        generic_major = blob[:8] + b"mif1" + blob[12:]
        assert detect_extension(generic_major) == suffix


def _ftyp(major: bytes, compatible: bytes, minor: bytes = bytes(4)) -> bytes:
    return (
        (16 + len(compatible)).to_bytes(4, "big") + b"ftyp" + major + minor + compatible
    )


def test_heif_explicit_codec_brands() -> None:
    for brand in (
        b"heic",
        b"heix",
        b"hevc",
        b"hevx",
        b"heim",
        b"heis",
        b"hevm",
        b"hevs",
    ):
        assert detect_extension(_ftyp(brand, b"mif1")) == "heic"
        assert detect_extension(_ftyp(b"mif1", brand)) == "heic"
    for brand in (b"avif", b"avis"):
        assert detect_extension(_ftyp(brand, b"mif1")) == "avif"
        assert detect_extension(_ftyp(b"mif1", brand)) == "avif"


def test_heif_generic_ambiguous_and_out_of_box_brands() -> None:
    for blob in (
        _ftyp(b"mif1", b"miaf"),
        _ftyp(b"msf1", b""),
        _ftyp(b"mif1", b"mif1", minor=b"heic"),
        _ftyp(b"mif1", b"mif1") + b"avifheic",
        _ftyp(b"mif1", b"heicavif"),
        _ftyp(b"heic", b"avif"),
        _ftyp(b"avif", b"heic"),
    ):
        assert detect_extension(blob) not in {"avif", "heic"}


def test_heif_file_type_box_size_variants() -> None:
    body = b"mif1" + bytes(4) + b"heic"
    extended = b"\x00\x00\x00\x01ftyp" + (16 + len(body)).to_bytes(8, "big") + body
    assert detect_extension(extended) == "heic"
    assert detect_extension(bytes(4) + b"ftyp" + body) == "heic"
    assert detect_extension(extended[:-1]) == "bin"
    assert detect_extension(_ftyp(b"avif", b"mif1" * 1021)) == "bin"
