import os
from .hyperlpr import LPR
import sys
name = "hyperlpr_python_pkg"

PR = LPR("./plate_rec/models")


def HyperLPR_plate_recognition(Input_BGR, box, minSize=30, charSelectionDeskew=True, region="CH"):
    return PR.plate_recognition(Input_BGR, box, minSize, charSelectionDeskew)
