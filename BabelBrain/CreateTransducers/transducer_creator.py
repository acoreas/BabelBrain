import logging
logger = logging.getLogger(__name__)
import os
import re

import numpy as np
import yaml

# CONSTANTS
COORD_VARS = {'cartesian': ('x', 'y', 'z'), 'spherical': ('r', 'theta', 'phi')}
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
        "flat": False,
        "spherical": True,
        "steering_axes": {"x", "y", "z"},
    },
}
VALID_FREQUENCIES = range(200000,1005000,5000)

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
        logger.info("Validating custom transducer file")
        
        self._validate_name(tx_params)                                                                          # sets: self.name
        self._validate_geometry(tx_params)                                                                      # sets: self.geometry_type, self.is_annular, ...
        self._validate_frequencies(tx_params)                                                                   # sets: self.frequencies
        self._validate_positive_param('aperture_size', (int, float), tx_params, unit="m")                       # sets: self.aperture_size
        self._validate_positive_param('focal_length',  (int, float), tx_params, unit="m")                       # sets: self.focal_length
        self._validate_positive_param('distance_outplane', (int, float), tx_params, allow_zero=True, unit="m")  # sets: self.distance_outplane
        self._validate_positive_param('num_elements',  int, tx_params)                                          # sets: self.num_elements
        self._validate_coordinate_system(tx_params)                                                             # sets: self.coordinate_system, self.coordinate_vars
    
    def create_tx_files(self):
        raise NotImplementedError("create_tx_files not yet implemented")
    
    def validate_custom_tx(self):
        raise NotImplementedError("validate_custom_tx not yet implemented")

    def update_tx_list(self):
        raise NotImplementedError("update_tx_list not yet implemented")
    
    def _get_param(self, key, expected_type, param_dict, optional=False):
        """
        Helper function to ensure key exists in dict and the value is the correct type
        
        Args:
            key (str): key to be checked in param_dict.
            expected_type (type): expected type of param_dict[key].
            param_dict (dict): dict containing values.
            optional (bool): Ignores missing key error if True.
        
        Returns:
            val: value of param_dict[<key>]
        
        Raises:
            ValueError: If key does not exist in param_dict or it's value type does not match expected_type
        """
        
        # Check key exists
        if key not in param_dict.keys():
            if not optional:
                raise ValueError(f"The following parameter is missing from the custom transducer yaml: {key}")
            else:
                return
        
        # Check value type
        val = param_dict[key]
        if not isinstance(val, expected_type):
            type_name = expected_type.__name__ if isinstance(expected_type, type) else " or ".join(t.__name__ for t in expected_type)
            raise ValueError(f"{key} was not specified as {type_name} in custom transducer yaml file")
        
        # Return value
        if isinstance(val, (list, dict)):
            return val.copy() # return copy for mutable values
        else:
            return val
    
    def _validate_positive_param(self, key, expected_type, tx_params, allow_zero=False, unit=""):
        """
        Validates that a transducer parameter exists, is the correct type, and is positive.

        Args:
            key (str): Parameter name to look up in tx_params.
            expected_type (type or tuple of types): Expected type(s) for the parameter value.
            tx_params (dict): Raw transducer parameters loaded from yaml file.
            allow_zero (bool): If True, accepts values >= 0. If False, requires values > 0. Defaults to False.
            unit (str): Unit of parameter

        Raises:
            ValueError: If the parameter is missing, not the expected type, or fails the
                        positivity check.

        Sets:
            self.<key> (expected_type): Validated parameter value, converted to float if
                                        expected_type is not int.
        """
        val = self._get_param(key, expected_type, tx_params)
        
        # Check value is positive
        if allow_zero and val < 0:
                raise ValueError(f"{key} ({val} {unit}) must be >= 0 {unit}")
        elif not allow_zero and val <= 0:
                raise ValueError(f"{key} ({val} {unit}) must be > 0")
        
        # Ensure value is float if that is expected type
        result = float(val) if expected_type is not int else val
        
        # Assign to self
        setattr(self,key,result) # Equivalent to self.<key> = result
        logger.info(f"Transducer {key}: {result} {unit}")
    
    def _validate_name(self, tx_params: dict) -> None:
        """
        Validates the transducer name parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If name is missing, not valid type, contains spaces, 
                        special characters, or does not begin with a letter.
        
        Sets:
            self.name (str): Validated transducer name.
        """
        tx_name = self._get_param('name', str, tx_params)

        if not re.match(r'^[a-zA-Z]', tx_name):
            raise ValueError("Transducer name must begin with a letter")
        
        if re.search(r'\s', tx_name):
            raise ValueError("Transducer name cannot contain spaces")
        
        special_chars = set(re.findall(r'[^a-zA-Z0-9_-]', tx_name))
        if special_chars:
            raise ValueError(f"Transducer name cannot contain special characters ({', '.join(special_chars)})")
        
        self.name = tx_name
        logger.info(f"Transducer Name: {tx_name}")
    
    def _validate_geometry(self, tx_params: dict) -> None:
        """
        Validates the transducer geometry_type parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If geometry_type is missing, not valid type, or isn't valid choice
        
        Sets:
            self.geometry_type (str): Validated transducer geometry_type
            self.is_annular (bool): Boolean indicating geometry is of annular variation
            self.is_flat (bool): Boolean indicating geometry is flat
            self.is_spherical (bool): Boolean indicating geometry is spherical
            self.is_steerable (bool): Boolean indicating if geometry type allows electronic steering of focus
            self.steering_axes (tuple): Tuple indicating axes which have steering capabilities
        """
        
        # Geometry type validation
        tx_geometry_type = self._get_param('geometry_type', str, tx_params)
        if tx_geometry_type not in TX_GEOMETRIES.keys():
            valid_geoms_str = ", ".join(TX_GEOMETRIES.keys())
            raise ValueError(f"{tx_geometry_type} is not a valid geometry choice\n Expecting one of the following: {valid_geoms_str}")
        self.geometry_type = tx_geometry_type
        logger.info(f"Transducer Geometry: {tx_geometry_type}")
        
        # Property assignments
        tx_steering_axes = TX_GEOMETRIES[tx_geometry_type]['steering_axes']
        if tx_steering_axes is not None:
            self.is_steerable = True
            self.steering_axes = tx_steering_axes
            
        self.is_annular = TX_GEOMETRIES[tx_geometry_type]['annular']
        self.is_flat = TX_GEOMETRIES[tx_geometry_type]['flat']
        self.is_spherical = TX_GEOMETRIES[tx_geometry_type]['spherical']
    
    def _validate_frequencies(self, tx_params: dict) -> None:
        """
        Validates the transducer frequencies parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If frequencies is missing, not valid type (int or float), 
            not in valid range (200-1000kHz), or isn't at valid frequency step (5kHz)
        
        Sets:
            self.frequencies (list): Validated transducer frequencies
        """
        
        tx_frequencies = self._get_param('frequencies', list, tx_params)
        logger.info("Transducer Frequencies:")
        for freq in tx_frequencies:
            
            # Ensure no decimal points in frequency
            if not isinstance(freq,(int,float)):
                raise ValueError(f"frequency entry ({freq}) was not specified as an int or float in custom transducer yaml file")
            if not freq.is_integer():
                raise ValueError(f"Invalid specified frequency ({freq} Hz), frequency must be an integer value")
            
            # Ensure frequency is at valid step in frequency range
            int_freq = int(freq)
            if int_freq not in VALID_FREQUENCIES:
                raise ValueError(f"Invalid specified frequency ({int_freq} Hz), frequency must be at a 5kHz interval value within the 200-1000 kHz range")

            # Add valid frequency to list
            self.frequencies.append(int_freq)
            logger.info(f"   {int_freq} Hz")
        
        # Check for duplicate frequencies 
        if len(tx_frequencies) != len(set(tx_frequencies)):
            raise ValueError("frequencies list contains duplicate entries")

    def _validate_coordinate_system(self, tx_params: dict) -> None:
        """
        Validates the transducer element_coordinate_system parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If element_coordinate_system is missing, not valid type, or isn't valid choice
        
        Sets:
            self.coordinate_system (str): Validated transducer element coordinate system
            self.coordinate_vars (list): List of dimension variable names (x,y,z for cartesian or r,theta,phi for spherical)
        """
        # Non-spherical geometries have a fixed coordinate system — no user input needed
        if not self.is_spherical:
            self.coordinate_system = TX_GEOMETRIES[self.geometry_type]['coordinate_system']
            self.coordinate_vars = COORD_VARS[self.coordinate_system]
            logger.info(f"Transducer Coordinate System: {self.coordinate_system}")
            logger.info(f"Transducer Coordinate Variables: {self.coordinate_vars}")
            return
        
        # Spherical geometries allow user to choose coordinate system
        tx_coordinate_system = self._get_param('element_coordinate_system', str, tx_params)
        
        # Validate user specified coordinate system
        valid_tx_coordinate_systems = TX_GEOMETRIES[self.geometry_type]['coordinate_system']
        if tx_coordinate_system not in valid_tx_coordinate_systems:
            valid_coord_systems_str = ", ".join(valid_tx_coordinate_systems)
            raise ValueError(f"{tx_coordinate_system} is not a valid coordinate system choice\n Expecting one of the following: {valid_coord_systems_str}")
        
        # Assign properties
        self.coordinate_system = tx_coordinate_system
        self.coordinate_vars = COORD_VARS[tx_coordinate_system]
        logger.info(f"Transducer Coordinate System: {tx_coordinate_system}")
        logger.info(f"Transducer Coordinate Variables: {self.coordinate_vars}")