import logging
logger = logging.getLogger(__name__)
import os
import re

import numpy as np
import yaml

# CONSTANTS
TX_GEOMETRIES = {
    "simple_focused": {
        "annular": False,
        "coordinate_system": "cartesian",
        "flat": True,
        "spherical": False,
        "steering_axes": None,
    },
    "flat_annular_array": {
        "annular": True,
        "coordinate_system": "spherical",
        "flat": True,
        "spherical": False,
        "steering_axes": None,
    },
    "focused_annular_array": {
        "annular": True,
        "coordinate_system": ("cartesian", "spherical"),
        "flat": False,
        "spherical": True,
        "steering_axes": {"z"},
    },
    "flat_array_2D": {
        "annular": False,
        "coordinate_system": "cartesian",
        "flat": True,
        "spherical": False,
        "steering_axes": {"x", "y", "z"},
    },
    "focused_array": {
        "annular": False,
        "coordinate_system": ("cartesian", "spherical"),
        "flat": True,
        "spherical": False,
        "steering_axes": {"x", "y", "z"},
    },
}
VALID_FREQUENCIES = range(200000,1005000,5000)
VARS_CARTESIAN = ('x', 'y', 'z')
VARS_SPHERICAL = ('r', 'theta', 'phi')

class CustomTransducer():

    def __init__(self,transducer_yaml):
        
        # Intial Values
        self.aperture_size = None
        self.coordinate_system = None
        self.coordinate_vars = []
        self.distance_outplane = None
        self.elements = None
        self.frequencies = []
        self.focal_length = None
        self.geometry_type = None
        self.is_annular = False
        self.is_spherical = False
        self.is_steerable = False
        self.name = None
        self.num_elements = None
        self.PlanTUS = None
        self.rings = None
        self.steering_axes = None
        self.xsteering_limits = None
        self.ysteering_limits = None
        self.zsteering_limits = None
        
        # Load/Validate transducer details
        tx_params = self.load_custom_tx_config_file(transducer_yaml)
        self._validate_custom_tx_params(tx_params)
        
        logger.info("Custom transducer file loading and validation complete")
        
    def load_custom_tx_config_file(self,tx_yaml):

        logger.info("Loading custom transducer file")
        
        # Read yaml then save.return tx_params dict
        if not os.path.isfile(tx_yaml):
            raise ValueError(f'{tx_yaml} does not exist')
        
        with open(tx_yaml, 'r') as file:
            custom_tx_params = yaml.safe_load(file)  

        return custom_tx_params
    
    def _validate_custom_tx_params(self,tx_params):
        raise NotImplementedError("_validate_custom_tx_params not yet implemented")                                                                  # sets: self.PlanTUS
    
    def create_tx_files(self):
        raise NotImplementedError("create_tx_files not yet implemented")   
    
    def validate_custom_tx(self):
        raise NotImplementedError("validate_custom_tx not yet implemented")   

    def update_tx_list(self):
        raise NotImplementedError("update_tx_list not yet implemented")   