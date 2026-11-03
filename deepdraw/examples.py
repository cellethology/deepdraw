"""Create a local copy of the bundled dummy example."""

from importlib.resources import files
from pathlib import Path

_EXAMPLE_FILES = (
    "design_pool.csv",
    "embeddings.npz",
    "measurements.csv",
    "README.md",
    "make_dummy_inputs.py",
)


def write_dummy_example(output_dir: str | Path) -> Path:
    source = files("deepdraw").joinpath("_example_data")
    if not source.is_dir():
        source = Path(__file__).resolve().parents[1] / "examples" / "deepdraw_dummy"
    contents = {name: source.joinpath(name).read_bytes() for name in _EXAMPLE_FILES}
    destination = Path(output_dir)
    destination.mkdir(parents=True)
    for name, content in contents.items():
        with (destination / name).open("xb") as handle:
            handle.write(content)
    return destination
