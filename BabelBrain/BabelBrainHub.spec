# -*- mode: python ; coding: utf-8 -*-
'''
PyInstaller spec for the **BabelBrain Version Selector** (hub.py).

Run from the BabelBrain/ directory:

    pyinstaller BabelBrainHub.spec --noconfirm --clean

Builds BabelBrain-Version-Selector.app (macOS) / BabelBrain-Version-Selector.exe
(Windows): the picker that lets users choose, download, and switch versions.
It is intentionally small (PySide6 + PyYAML + certifi + the Hub package); it does
NOT embed any BabelBrain version. The actual frozen versions live in the shared /
per-user versions store, into which the installer seeds a default version.
'''
import platform

from PyInstaller.utils.hooks import collect_all, collect_submodules

is_mac = "Darwin" in platform.system()

# The launcher apps carry their own version, independent of any BabelBrain version.
hub_version = "1.0.0"

datas = []
binaries = []

# hub.py imports the Hub package dynamically (Hub.cli -> ui/installer/...).
hiddenimports = collect_submodules("Hub") + [
    "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
    "yaml",
]

# Bundle certifi's CA bundle so HTTPS (manifest fetch + version downloads) can
# verify certificates. Without this a frozen app has no CA path and every HTTPS
# request fails verification (see Hub/netutil.py).
_cf_datas, _cf_bins, _cf_hidden = collect_all("certifi")
datas += _cf_datas
binaries += _cf_bins
hiddenimports += _cf_hidden + ["certifi"]

block_cipher = None

a = Analysis(
    ["hub.py"],
    pathex=["./"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Keep it light: exclude the heavy scientific stack only the real BabelBrain
    # versions need.
    excludes=[
        "SimpleITK", "itk", "vtk", "vtkmodules", "nibabel", "trimesh",
        "BabelViscoFDTD", "cupy", "pyopencl", "mlx", "scipy", "skimage",
        "matplotlib", "pandas",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="BabelBrain-Version-Selector",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=not is_mac,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    entitlements_file=None,
    icon=None if is_mac else ["Proteus-Alciato-logo.ico"],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=is_mac,
    upx_exclude=[],
    name="BabelBrain-Version-Selector",
)

if is_mac:
    app = BUNDLE(
        coll,
        name="BabelBrain-Version-Selector.app",
        bundle_identifier="com.ucalgary.babelbrain.selector",
        version=hub_version,
        icon="./Proteus-Alciato-logo.png",
    )
