# This Python file uses the following encoding: utf-8

import os

import yaml

from _Babel_RingTx.Babel_RingTx import RingTx
from Utils.paths import resource_path


class CTX500(RingTx):
    def DefaultConfig(self):
        #Specific parameters for the CTX500 - to be configured later via a yaml
        with open(os.path.join(resource_path(__file__), 'default.yaml'), 'r') as file:
            config = yaml.safe_load(file)

        self.Config=config

