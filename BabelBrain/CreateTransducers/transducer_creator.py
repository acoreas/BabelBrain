import logging
import os
import re

import numpy as np
import yaml

logger = logging.getLogger(__name__)

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

class CustomTransducer:

    def __init__(self, transducer_yaml: str) -> None:
        
        # Intial Values
        self.aperture_size: float | None = None
        self.coordinate_system: str | None = None
        self.coordinate_vars: list[str] = []
        self.distance_outplane: float | None = None
        self.elements: dict | None = None
        self.frequencies: list[int] = []
        self.focal_length: float | None = None
        self.geometry_type: str | None = None
        self.is_annular: bool = False
        self.is_spherical: bool = False
        self.is_steerable: bool = False
        self.name: str | None = None
        self.num_elements: int | None = None
        self.PlanTUS: dict | None = None
        self.rings: dict | None = None
        self.steering_axes: set | None = None
        self.xsteering_limits: list | None = None
        self.ysteering_limits: list | None = None
        self.zsteering_limits: list | None = None
        
        # Load/Validate transducer details
        tx_params = self.load_custom_tx_config_file(transducer_yaml)
        self._validate_custom_tx_params(tx_params)
        
        logger.info("Custom transducer file loading and validation complete")
        
    def load_custom_tx_config_file(self, tx_yaml: str) -> dict:

        logger.info("Loading custom transducer file")
        
        # Read yaml then save.return tx_params dict
        if not os.path.isfile(tx_yaml):
            raise ValueError(f'{tx_yaml} does not exist')
        
        with open(tx_yaml, 'r') as file:
            custom_tx_params = yaml.safe_load(file)  

        return custom_tx_params
    
    def _validate_custom_tx_params(self, tx_params: dict) -> None:
        logger.info("Validating custom transducer file")
        
        self._validate_name(tx_params)                                                                          # sets: self.name
        self._validate_geometry(tx_params)                                                                      # sets: self.geometry_type, self.is_annular, ...
        self._validate_frequencies(tx_params)                                                                   # sets: self.frequencies
        self._validate_positive_param('aperture_size', (int, float), tx_params, unit="m")                       # sets: self.aperture_size
        self._validate_positive_param('focal_length',  (int, float), tx_params, unit="m")                       # sets: self.focal_length
        self._validate_positive_param('distance_outplane', (int, float), tx_params, allow_zero=True, unit="m")  # sets: self.distance_outplane
        self._validate_positive_param('num_elements',  int, tx_params)                                          # sets: self.num_elements
        self._validate_coordinate_system(tx_params)                                                             # sets: self.coordinate_system, self.coordinate_vars
        self._validate_elements(tx_params)                                                                      # sets: self.elements
        self._validate_annular(tx_params)                                                                       # sets: self.rings
        self._validate_steering(tx_params)                                                                      # sets: self.xsteering_limits, self.ysteering_limits, self.zsteering_limits
        self._validate_PlanTUS(tx_params)                                                                       # sets: self.PlanTUS
    
    def create_tx_files(self) -> None:
        raise NotImplementedError("create_tx_files not yet implemented")
    
    def validate_custom_tx(self) -> None:
        raise NotImplementedError("validate_custom_tx not yet implemented")

    def update_tx_list(self) -> None:
        raise NotImplementedError("update_tx_list not yet implemented")
    
    def _get_param(self, key: str, expected_type: type | tuple[type, ...], param_dict: dict, optional: bool = False) -> object:
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
    
    def _validate_positive_param(self, key: str, expected_type: type | tuple[type, ...], tx_params: dict, allow_zero: bool = False, unit: str = "") -> None:
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
    
    def _validate_numeric_list_dict( self, param_dict: dict, num_elements: int | None = None, context_name: str = "parameter", allow_negative: bool = True) -> None:
        """
        Validates that all entries in a dict of lists are numeric and match the expected length.

        Iterates over each key-value pair in param_dict, confirming that every list has
        exactly num_elements entries and that each entry is an int or float. All invalid
        entries are collected before raising, so the error message reports every problem
        at once rather than stopping at the first.

        Args:
            param_dict (dict): Dictionary mapping parameter names to lists of values.
            num_elements (int): Expected length of each list, typically self.num_elements.
            context_name (str): Human-readable label for param_dict used in error messages
                                (e.g. 'elements', 'annular').
            allow_negative (bool): Set to False if element values should be positive

        Raises:
            ValueError: If any list length does not match num_elements, or if any entry
                        is not an int or float. Length mismatches are raised immediately
                        on the offending key; type errors are collected and raised together
                        after all lists are checked.
        """
    
        bad_entries = []
        for key, values in param_dict.items():
            if num_elements is not None and len(values) != num_elements:
                raise ValueError(f"Number of entries in {key} ({len(values)}) does not match num_elements ({num_elements})")
            
            for i, val in enumerate(values):
                if not isinstance(val, (int, float)):
                    bad_entries.append(f"   {key}[{i}]: {val!r} (expected numeric)")
                elif not allow_negative and val < 0:
                    bad_entries.append(f"   {key}[{i}]: {val!r} (negative values not allowed)")
        
        if bad_entries:
            raise ValueError(f"{context_name} contains invalid entries:\n" + "\n".join(bad_entries))
    
    def _validate_limits(self, limits: list, context_name: str = "") -> None:
        """
        Validate limits supplied in list
        
        Args:
            limits (list): [min_value max_value]
            context_name (str): Optional string to provide more detail to error message
        
        Raises:
            ValueError: If max_value is less than min_value
        """
        if len(limits) != 2:
            raise ValueError(f"{context_name} limits must have exactly 2 entries [min, max], got {len(limits)}")

        min_limit = limits[0]
        max_limit = limits[1]
        if min_limit > max_limit:
            raise ValueError(f"{context_name}: Min value ({min_limit}) must be less than max value ({max_limit})")
    
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
         
    def _validate_elements(self, tx_params: dict) -> None:
        """
        Validates the transducer elements parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If elements or any of its subcomponents are missing or not valid type. If there is a mismatch in
                        number of sub elements and the num_element parameter. If element position do not make sense physically.
        
        Sets:
            self.elements (dict): Validated transducer elements.
        """
        # Element coordinates do not need to be specified for non-spherical transducers
        if not self.is_spherical:
            return
        
        tx_elements = self._get_param('elements', dict, tx_params)
        for dim_var in self.coordinate_vars:
            _ = self._get_param(dim_var, list, tx_elements)
        self._validate_numeric_list_dict(tx_elements,self.num_elements,'elements')
        
        self.elements = tx_elements
        for dim_key,dim_values in tx_elements.items():
            logger.debug(f"Transducer Element {dim_key} Values:\n{dim_values}")
    
    def _validate_annular(self, tx_params: dict) -> None:
        """
        Validates the transducer annular parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If annular or any of its subcomponents are missing or not valid type. If there is a mismatch in
                        number of rings and the num_element parameter. If ring diameters do not make sense physically.
        
        Sets:
            self.rings (dict): Validated transducer ring diameters.
        """
        if not self.is_annular:
            return
        
        tx_rings = self._get_param('annular', dict, tx_params)
        tx_rings_new = {}
        inner_diameters = self._get_param('inner_ring_diameters', list, tx_rings)
        outer_diameters = self._get_param('outer_ring_diameters', list, tx_rings)
        self._validate_numeric_list_dict(tx_rings,self.num_elements,'annular',allow_negative=False)
        
        # Check outer ring is always bigger than inner ring
        bad_entries = []
        for i, (inner, outer) in enumerate(zip(inner_diameters, outer_diameters)):
            if outer <= inner:
                bad_entries.append(f"inner_ring_diameters[{i}] ({inner}) > outer_ring_diameters[{i}] ({outer})")
        if bad_entries:
            raise ValueError(f"inner_ring_diameters cannot be bigger than corresponding outer_ring_diameter:\n{bad_entries}")
            
        # Rename keys
        tx_rings_new["inner_diameters"] = inner_diameters
        tx_rings_new["outer_diameters"] = outer_diameters
        self.rings = tx_rings_new
        
        inner_diams_mm = [d * 1e3 for d in inner_diameters]
        outer_diams_mm = [d * 1e3 for d in outer_diameters]
        logger.info(f"Transducer Inner Ring Diameters (mm): {inner_diams_mm}")
        logger.info(f"Transducer Outer Ring Diameters (mm): {outer_diams_mm}")

    def _validate_steering(self, tx_params: dict) -> None:
        """
        Validates the transducer steering parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If steering or any of its subcomponents are missing or not valid type.
                        If steering limits do not make sense physically.
        
        Sets:
            self.xsteering_limits (list): [min_steering_limit, max_steering_limit]
            self.ysteering_limits (list): [min_steering_limit, max_steering_limit]
            self.zsteering_limits (list): [min_steering_limit, max_steering_limit]
        """
        
        if not self.is_steerable:
            return
        
        tx_steering = self._get_param('steering', dict, tx_params)
        tx_xsteering = tx_ysteering = tx_zsteering = None
        
        if 'x' in self.steering_axes:
            tx_xsteering = self._get_param('x', list, tx_steering)
            self._validate_limits(tx_xsteering,"X Steering")
            logger.info(f"Transducer X Steering Limits (m): {tx_xsteering}")
        if 'y' in self.steering_axes:
            tx_ysteering = self._get_param('y', list, tx_steering)
            self._validate_limits(tx_ysteering,"Y Steering")
            logger.info(f"Transducer Y Steering Limits (m): {tx_ysteering}")
        if 'z' in self.steering_axes:
            tx_zsteering = self._get_param('z', list, tx_steering)
            self._validate_limits(tx_zsteering,"Z Steering")
            logger.info(f"Transducer Z Steering Limits (m): {tx_zsteering}")
        
        self._validate_numeric_list_dict(tx_steering,2,'steering')
        
        # Check negative z steering does not exceed focal length
        if 'z' in self.steering_axes:
            abs_zsteering_min = abs(tx_zsteering[0])
            if abs_zsteering_min > self.focal_length:
                raise ValueError(f"Z minimum steering limit ({abs_zsteering_min}) exceeds focal length distance ({self.focal_length})")
        
        self.xsteering_limits = tx_xsteering
        self.ysteering_limits = tx_ysteering
        self.zsteering_limits = tx_zsteering
        
    def _validate_PlanTUS(self, tx_params: dict) -> None:
        """
        Validates the transducer PlanTUS parameter.
        
        Args:
            tx_params (dict): Raw transducer parameters loaded from yaml file.
        
        Raises:
            ValueError: If listed frequencies do not match transducer frequencies. If FocalDistanceList or 
                        FHMLList are missing from file or are wrong type. If the number of elements between
                        FocalDistanceList and FHMLList do not match.
        
        Sets:
            self.PlanTUS (dict): Validated PlanTUS dict
        """
        
        tx_planTUS = self._get_param('PlanTUS', dict, tx_params, optional=True)
        tx_planTUS_new = {}
        
        if tx_planTUS is not None:
            logger.info("PlanTUS Parameters")
            tx_freqs = self.frequencies.copy()

            for planTUS_key, planTUS_value in tx_planTUS.items():
                
                # Validate PlanTUS frequency
                if not isinstance(planTUS_key,(int,float)):
                    raise ValueError(f"Frequencies under PlanTUS should be int or float, you put {planTUS_key} ({type(planTUS_key)})")
                
                planTUS_freq = int(planTUS_key)
                logger.info(f"    {planTUS_freq} Hz")
                
                if planTUS_freq not in tx_freqs:
                    raise ValueError(f"PlanTUS frequency ({planTUS_freq} Hz) is not listed as one of the transducer frequencies")
                
                # Validate elements in PlanTUS frequency
                if planTUS_value is None:
                    raise ValueError(f"FocalDistanceList and FHMLList are missing from {planTUS_key} parameter under PlanTUS parameter")
                tx_planTUS_focal_dists = self._get_param('FocalDistanceList', list, planTUS_value)
                tx_planTUS_focal_FHMLs = self._get_param('FHMLList', list, planTUS_value)
                self._validate_numeric_list_dict(planTUS_value,None,f"PlanTUS {planTUS_key}",allow_negative=False)
                
                # Element Num check
                if len(tx_planTUS_focal_dists) != len(tx_planTUS_focal_FHMLs):
                    raise ValueError(f"Number of elements in FocalDistanceList ({len(tx_planTUS_focal_dists)}) does not match number in FHMLList ({len(tx_planTUS_focal_FHMLs)})")
                                
                # Rename keys
                logger.info("        focal_distances (m):")
                logger.info(f"           {tx_planTUS_focal_dists}")
                logger.info("        FHMLs:")
                logger.info(f"           {tx_planTUS_focal_FHMLs}")
                tx_planTUS_new[planTUS_freq] = {'focal_distances': tx_planTUS_focal_dists, 
                                                'FHMLs': tx_planTUS_focal_FHMLs}
                
                # Remove current PlanTUS freq from check
                tx_freqs.remove(planTUS_freq)
            
            if len(tx_freqs) > 0:
                missing_details = ", ".join(f"{freq} Hz" for freq in tx_freqs)
                raise ValueError(f"PlanTUS parameter is missing details for following frequencies: {missing_details}")
        
            self.PlanTUS = tx_planTUS_new