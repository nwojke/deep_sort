# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from .detection import *
from .kalman_filter import *
from .nn_matching import *
from .track import *
from .tracker import *

def create_box_encoder(batch_size=32, loss_mode="cosine", model_filename=None):
    if model_filename is None:
        import os
        model_filename = os.path.join(
            os.path.dirname(__file__), "..", "resources", "networks",
            "mars-small128.ckpt-68577")
    import generate_detections
    encoder = generate_detections.create_box_encoder(
        model_filename, batch_size, loss_mode)
    return encoder
