from dataclasses import dataclass
import pathlib
import re
from typing import List, Tuple


Point = Tuple[float, float]


@dataclass
class ElementGeometry:
    element_id: int
    name: str
    points: List[Point]
    source_path: str
    boundary_marker: str


@dataclass
class MultiElementGeometry:
    elements: List[ElementGeometry]



def parse_selig_dat(dat_path: pathlib.Path) -> Tuple[str, List[Point]]:
    lines = dat_path.read_text(encoding="utf-8").splitlines()
    name = dat_path.stem
    coords: List[Point] = []
    seen_coords = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        fields = [part for part in re.split(r"[\s,]+", line) if part]
        if len(fields) < 2:
            if not seen_coords:
                name = line
            continue

        try:
            x = float(fields[0])
            y = float(fields[1])
        except ValueError:
            if not seen_coords:
                name = line
            continue

        coords.append((x, y))
        seen_coords = True

    if len(coords) < 3:
        raise ValueError(f"Selig .dat file must contain at least 3 coordinate pairs: {dat_path}")

    return name, coords



def load_multielement_geometry(dat_paths: List[pathlib.Path]) -> MultiElementGeometry:
    elements: List[ElementGeometry] = []
    for idx, path in enumerate(dat_paths, start=1):
        name, pts = parse_selig_dat(path)
        elements.append(
            ElementGeometry(
                element_id=idx,
                name=name,
                points=pts,
                source_path=str(path),
                boundary_marker=f"wall_e{idx}",
            )
        )
    if not elements:
        raise ValueError("At least one element path is required")
    return MultiElementGeometry(elements=elements)
