# This Python file uses the following encoding: utf-8

import os

import yaml

from _Babel_RingTx.Babel_RingTx import RingTx
from Utils.paths import resource_path


class R15473(RingTx):
    def load_ui(self):
        super(R15473, self).load_ui()
        self.Widget.labelTPORange.setText('Range steering (mm)')
        self.Widget.labelTPODistance.setText('Steering from outplane (mm)')


    def DefaultConfig(self):
        # Specific parameters for the R15473 - to be configured later via a yaml
        with open(os.path.join(resource_path(__file__), 'default.yaml'), 'r') as file:
            config = yaml.safe_load(file)

        self.Config=config
