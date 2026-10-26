from .scan import scan
from .cure import cure
from .pipeline import make_pipeline
from .train import train
from .diagnose import diagnose
from .explain import explain
from .autopilot import autopilot
from .serve import serve
from .tracking import launch_dashboard as dashboard, launch_dashboard as ui

__all__ = ["scan", "cure", "make_pipeline", "train", "diagnose", "explain", "autopilot", "serve", "dashboard", "ui"]
__version__ = "1.3.0"
__author__ = "Vishal More"