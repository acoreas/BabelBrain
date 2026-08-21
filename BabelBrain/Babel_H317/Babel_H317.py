# This Python file uses the following encoding: utf-8
import os

import yaml

from _BabelBasePhasedArray import BabelBasePhaseArray
from Utils.paths import resource_path


class H317(BabelBasePhaseArray):
    def __init__(self,parent=None,MainApp=None):
        super().__init__(parent=parent, MainApp=MainApp, formtype=os.path.join(resource_path(__file__), "."))

    def DefaultConfig(self):
        #Specific parameters for the H317 -  configured later via a yaml

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),'default.yaml'), 'r') as file:
            config = yaml.safe_load(file)
        print("H317 configuration:")
        print(config)

        self.Config=config
