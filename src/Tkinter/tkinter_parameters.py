from dataclasses import dataclass
from pathlib import Path


@dataclass
class Window:
    title: str
    geometry: str
    label_main: str

window = Window(title="Object detector",
                geometry="900x750+300+10",
                label_main="Welcome to the Object Detector")


labelImg_path = Path('./labelImg')