# This Python file uses the following encoding: utf-8
import os

import yaml

from Babel_H317.Babel_H317 import H317


class H301(H317):
    def DefaultConfig(self):
        #Specific parameters for the H301 - configured via a yaml
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'default.yaml'), 'r') as file:
            config = yaml.safe_load(file)
        print("H301 configuration:")
        print(config)
        self.Config=config
