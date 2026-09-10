"""
Built-in transducer registry for BabelBrain.

Each entry is a dict that is stored as QComboBox item data, so any field
can be retrieved at runtime with combo.itemData(i) / combo.currentData().

Fields
------
name     : str  — identifier used throughout the codebase (must match Babel_<name> folders)
steering : bool — True when the transducer supports multipoint / electronic steering
transducer_type : bool — one of 5 transducer geometries (simple_focused,flat_annular_array,focused_annular_array,flat_array_2D,focused_array)
"""

TRANSDUCER_LIST = [
    {'name': 'Single',      'steering': False,      'transducer_type': 'simple_focused'},
    {'name': 'CTX_500',     'steering': False,      'transducer_type': 'focused_annular_array'},
    {'name': 'CTX_250',     'steering': False,      'transducer_type': 'focused_annular_array'},
    {'name': 'CTX_250_2ch', 'steering': False,      'transducer_type': 'focused_annular_array'},
    {'name': 'DPX_500',     'steering': False,      'transducer_type': 'focused_annular_array'},
    {'name': 'DPXPC_300',   'steering': False,      'transducer_type': 'focused_annular_array'},
    {'name': 'H317',        'steering': True,       'transducer_type': 'focused_array'},
    {'name': 'H246',        'steering': False,      'transducer_type': 'flat_annular_array'},
    {'name': 'BSonix',      'steering': False,      'transducer_type': 'simple_focused'},
    {'name': 'REMOPD',      'steering': False,      'transducer_type': 'flat_array_2D'},
    {'name': 'I12378',      'steering': True,       'transducer_type': 'focused_array'},
    {'name': 'ATAC',        'steering': True,       'transducer_type': 'focused_array'},
    {'name': 'R15148',      'steering': True,       'transducer_type': 'focused_array'},
    {'name': 'R15287',      'steering': False,      'transducer_type': 'focused_annular_array'},
    {'name': 'R15473',      'steering': False,      'transducer_type': 'focused_annular_array'},
    {'name': 'R15646',      'steering': True,       'transducer_type': 'focused_array'},
    {'name': 'IGT64_500',   'steering': True,       'transducer_type': 'focused_array'},
    {'name': 'H301',        'steering': True,       'transducer_type': 'focused_array'},
    {'name': 'DomeTx',      'steering': True,       'transducer_type': 'focused_array'},
]
