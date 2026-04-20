"""
nifti_viewer.py
===============
PySide6 + VTK NIfTI viewer — multi-volume with overlay transparency.

Features
--------
  • Load any number of NIfTI files.  The first file sets the reference
    geometry (affine orientation, camera planes, slice count).
  • Each subsequent file is an overlay rendered on top, with its own
    opacity slider (0–100 %) and colour-map selector (Grey / Hot / Cool /
    Green / Red).
  • Individual eye-icon toggle to show/hide each overlay.
  • Both AFFINE mode (native voxel axes of file 1) and MEDICAL mode
    (RAS world axes: Axial / Coronal / Sagittal) are supported.
  • Radiological flip (L↔R) available in medical mode.
  • Crosshair lines synchronised across all three views on scroll.

Architecture
------------
  VolumeRecord  – bundles (vtk_idx, vtk_xform, vtk_property) for one file.
  SliceViewport – owns a vtkRenderer with one vtkImageSlice per loaded volume,
                  all sharing the same camera (driven by volume 0 geometry).
  NiftiViewer   – manages the list of VolumeRecords and coordinates viewports.
  LayerPanel    – sidebar widget; one LayerRow per volume with opacity/cmap/eye.

Camera / geometry
-----------------
  Always derived from volume 0's affine (affine mode) or from the RAS
  bounding box of volume 0 (medical mode).  Overlay volumes are rendered
  with their own affine UserTransform so VTK cuts them at the same world-
  space plane as the base volume — even if they have different voxel grids.

Requirements
------------
    pip install PySide6 vtk nibabel numpy

Usage
-----
    python nifti_viewer.py [file1.nii.gz [file2.nii.gz ...]]
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPalette, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QSlider,
    QSizePolicy, QFileDialog, QPushButton, QFrame, QSplitter, QStatusBar,
    QMainWindow, QToolBar, QButtonGroup, QRadioButton, QCheckBox, QGroupBox,
    QScrollArea, QComboBox, QToolButton,
)

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util import numpy_support

try:
    import nibabel as nib
except ImportError:
    nib = None


# ── Palette ────────────────────────────────────────────────────────────────
BG_DARK   = "#1a1a1e"
BG_PANEL  = "#222228"
BG_VP     = "#0d0d0f"
BG_LAYER  = "#28282f"
ACCENT    = "#00c8ff"
COLORS    = ["#ff6b6b", "#6bffb8", "#ffda6b"]
TEXT      = "#d4d4d8"
TEXT_DIM  = "#71717a"

# Per-volume accent colours for the layer panel rows
VOL_COLORS = ["#ffffff", "#ff9f43", "#48dbfb", "#ff6b81",
              "#a29bfe", "#00d2d3", "#ffd32a", "#0abde3"]

AFFINE_NAMES  = ["Slice k  (axial-like)", "Slice i  (sagittal-like)", "Slice j  (coronal-like)"]
MEDICAL_NAMES = ["Axial", "Coronal", "Sagittal"]

CMAPS = {
    "Grey":  None,          # built below: black→white
    "Hot":   "hot",
    "Cool":  "cool",
    "Green": "green",
    "Red":   "red",
}


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def _np4(m: np.ndarray) -> vtk.vtkMatrix4x4:
    mat = vtk.vtkMatrix4x4()
    for r in range(4):
        for c in range(4):
            mat.SetElement(r, c, float(m[r, c]))
    return mat


def _make_lut(name: str | None, lo: float, hi: float) -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetRange(lo, hi)
    lut.SetNumberOfColors(256)

    if name is None or name == "grey":          # greyscale
        lut.SetSaturationRange(0, 0)
        lut.SetHueRange(0, 0)
        lut.SetValueRange(0, 1)
    elif name == "hot":                          # black→red→yellow→white
        lut.SetHueRange(0.0, 0.1667)
        lut.SetSaturationRange(1, 0)
        lut.SetValueRange(0.5, 1)
    elif name == "cool":                         # cyan→magenta
        lut.SetHueRange(0.5, 0.833)
        lut.SetSaturationRange(1, 1)
        lut.SetValueRange(1, 1)
    elif name == "green":
        lut.SetHueRange(0.333, 0.333)
        lut.SetSaturationRange(0, 1)
        lut.SetValueRange(0, 1)
    elif name == "red":
        lut.SetHueRange(0.0, 0.0)
        lut.SetSaturationRange(0, 1)
        lut.SetValueRange(0, 1)
    else:
        lut.SetSaturationRange(0, 0)
        lut.SetHueRange(0, 0)
        lut.SetValueRange(0, 1)

    lut.Build()
    return lut


# ── PlaneGeometry ──────────────────────────────────────────────────────────
@dataclass
class PlaneGeometry:
    normal:  np.ndarray
    right:   np.ndarray
    up:      np.ndarray
    centre:  np.ndarray
    step:    float
    n:       int
    ps:      float
    flip_lr: bool = False   # True → place camera on opposite side (radiological flip)


# ── VolumeRecord ───────────────────────────────────────────────────────────
@dataclass
class VolumeRecord:
    """Everything VTK needs to render one loaded NIfTI file."""
    name:      str
    vtk_idx:   vtk.vtkImageData       # data in index/physical space
    vtk_xform: vtk.vtkTransform       # VTK physical → world (RAS)
    lo:        float
    hi:        float
    opacity:   float = 1.0
    visible:   bool  = True
    cmap:      str   = "Grey"         # key into CMAPS


# ── Affine / geometry helpers ──────────────────────────────────────────────

def _decompose(affine: np.ndarray):
    M  = affine[:3, :3]
    sp = np.linalg.norm(M, axis=0)
    return sp, M / sp[np.newaxis, :]


def _ras_bbox(affine: np.ndarray, shape: tuple):
    corners = np.array([
        (affine @ np.array([i, j, k, 1.]))[:3]
        for i in (0, shape[0]-1)
        for j in (0, shape[1]-1)
        for k in (0, shape[2]-1)
    ])
    return corners.min(axis=0), corners.max(axis=0)


def _make_vtk_transform(affine: np.ndarray, spacing: np.ndarray) -> vtk.vtkTransform:
    S_inv = np.diag([1./spacing[0], 1./spacing[1], 1./spacing[2], 1.])
    xf = vtk.vtkTransform()
    xf.SetMatrix(_np4(affine @ S_inv))
    return xf


def numpy_to_vtk_index(data: np.ndarray, spacing: np.ndarray) -> vtk.vtkImageData:
    img = vtk.vtkImageData()
    img.SetDimensions(int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
    img.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
    img.SetOrigin(0., 0., 0.)
    flat = data.ravel(order="F")
    arr  = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    img.GetPointData().SetScalars(arr)
    return img


def load_volume_record(path: str) -> tuple[VolumeRecord, tuple, tuple, str]:
    """Load a NIfTI file and return a VolumeRecord + metadata."""
    if nib is None:
        raise ImportError("nibabel is required: pip install nibabel")
    img  = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D, got shape {data.shape}")

    affine = img.affine.astype(np.float64)
    sp, _  = _decompose(affine)
    lo, hi = float(data.min()), float(data.max())

    rec = VolumeRecord(
        name      = os.path.basename(path),
        vtk_idx   = numpy_to_vtk_index(data, sp),
        vtk_xform = _make_vtk_transform(affine, sp),
        lo        = lo,
        hi        = hi,
    )
    zooms = img.header.get_zooms()[:3]
    code  = "".join(nib.aff2axcodes(affine))
    return rec, data.shape, tuple(float(z) for z in zooms), code, affine


# ── Plane geometry factories ───────────────────────────────────────────────

def affine_plane_geoms(affine: np.ndarray, shape: tuple) -> list[PlaneGeometry]:
    sp, R = _decompose(affine)
    di, dj, dk = R[:,0], R[:,1], R[:,2]
    si, sj, sk = sp
    ni, nj, nk = shape
    mid = np.array([(ni-1)/2., (nj-1)/2., (nk-1)/2., 1.])
    ctr = (affine @ mid)[:3]
    def ps(a, b): return max(a, b) / 2.
    return [
        PlaneGeometry(normal=dk, right=di, up=dj,  centre=ctr.copy(), step=sk, n=nk, ps=ps(ni*si, nj*sj)),
        PlaneGeometry(normal=di, right=dj, up=dk,  centre=ctr.copy(), step=si, n=ni, ps=ps(nj*sj, nk*sk)),
        PlaneGeometry(normal=dj, right=di, up=dk,  centre=ctr.copy(), step=sj, n=nj, ps=ps(ni*si, nk*sk)),
    ]


def medical_plane_geoms(ras_min, ras_max, iso, radiological) -> list[PlaneGeometry]:
    """
    Three PlaneGeometry objects for the three RAS-axis planes.

    Radiological L/R flip is achieved by setting flip_lr=True, which makes
    _init_camera place the camera on the opposite side of the focal plane
    (position = focal - normal*DIST instead of + normal*DIST).  VTK then
    computes screen-right = up × viewPlaneNormal where viewPlaneNormal = -normal,
    giving -(up × normal) — effectively mirroring left and right while keeping
    up unchanged.
    """
    ctr  = (ras_min + ras_max) / 2.
    flip = radiological

    ax_x  = np.array([1.,  0., 0.])   # right (always +X in world; flip is in camera)
    ax_y  = np.array([0.,  1., 0.])
    ax_z  = np.array([0.,  0., 1.])
    sa_r  = np.array([0., -1., 0.])   # sagittal screen-right = -Y (ant on screen-right)

    def ns(axis): return max(2, int(np.ceil((ras_max[axis]-ras_min[axis])/iso))+1)
    def ps(a, b): return max(a, b) / 2.
    ex = ras_max[0]-ras_min[0]; ey = ras_max[1]-ras_min[1]; ez = ras_max[2]-ras_min[2]

    return [
        PlaneGeometry(normal=ax_z,  right=ax_x, up=ax_y, centre=ctr.copy(), step=iso, n=ns(2), ps=ps(ex,ey), flip_lr=flip),
        PlaneGeometry(normal=ax_y,  right=ax_x, up=ax_z, centre=ctr.copy(), step=iso, n=ns(1), ps=ps(ex,ez), flip_lr=flip),
        PlaneGeometry(normal=ax_x,  right=sa_r, up=ax_z, centre=ctr.copy(), step=iso, n=ns(0), ps=ps(ey,ez), flip_lr=flip),
    ]


# ── Line actor helpers ─────────────────────────────────────────────────────

def _make_line_actor(color_hex):
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(0, 0, 0); pts.InsertNextPoint(1, 0, 0)
    poly = vtk.vtkPolyData(); poly.SetPoints(pts)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(2); cells.InsertCellPoint(0); cells.InsertCellPoint(1)
    poly.SetLines(cells)
    m = vtk.vtkPolyDataMapper(); m.SetInputData(poly)
    a = vtk.vtkActor(); a.SetMapper(m)
    a.GetProperty().SetColor(*_hex_rgb(color_hex))
    a.GetProperty().SetLineWidth(1.2)
    a.VisibilityOff()
    return a


def _set_line(actor, p1, p2):
    poly = actor.GetMapper().GetInput()
    pts  = poly.GetPoints()
    pts.SetPoint(0, *p1.tolist()); pts.SetPoint(1, *p2.tolist())
    pts.Modified(); poly.Modified()


# ── SliceViewport ──────────────────────────────────────────────────────────

class SliceViewport(QFrame):
    """
    One orthogonal view.  Maintains a list of (vtkImageSlice, vtkImageProperty)
    pairs, one per VolumeRecord.  All share the same vtkRenderer and camera.

    The base volume (index 0) drives the camera geometry.
    Overlays are stacked on top in insertion order with their own properties.
    """

    slice_changed = Signal(int, int)

    def __init__(self, plane_idx: int, parent=None):
        super().__init__(parent)
        self.plane_idx = plane_idx
        self._pg: PlaneGeometry | None = None
        self._current_slice = 0
        # Per-volume actor/property pairs
        self._layers: list[tuple[vtk.vtkImageSlice, vtk.vtkImageProperty]] = []
        self._build_ui()
        self._build_pipeline()

    # ── UI ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        c = COLORS[self.plane_idx]
        self.setStyleSheet(
            f"SliceViewport {{ border:2px solid {c}; border-radius:4px; background:{BG_VP}; }}")
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(0)

        hdr = QWidget(); hdr.setFixedHeight(28)
        hdr.setStyleSheet(f"background:{BG_PANEL}; border-bottom:1px solid {c};")
        hl = QHBoxLayout(hdr); hl.setContentsMargins(8,0,8,0)
        self._lbl_title = QLabel("—")
        self._lbl_title.setStyleSheet(f"color:{c}; font-size:11px; font-weight:bold; letter-spacing:2px;")
        self._lbl_slice = QLabel("—")
        self._lbl_slice.setStyleSheet(f"color:{TEXT_DIM}; font-size:11px;")
        hl.addWidget(self._lbl_title); hl.addStretch(); hl.addWidget(self._lbl_slice)
        lay.addWidget(hdr)

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lay.addWidget(self.vtk_widget)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{ height:4px; background:#333340; }}
            QSlider::handle:horizontal {{
                background:{c}; width:14px; height:14px; margin:-5px 0; border-radius:7px; }}
            QSlider::sub-page:horizontal {{ background:{c}; }}""")
        self._slider.setFixedHeight(24)
        self._slider.valueChanged.connect(self._on_slider)
        lay.addWidget(self._slider)

    def set_title(self, t):
        self._lbl_title.setText(t.upper())

    # ── VTK pipeline ──────────────────────────────────────────────────────
    def _build_pipeline(self):
        rw = self.vtk_widget.GetRenderWindow()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.05, 0.06)
        rw.AddRenderer(self.renderer)

        self._cross_h = _make_line_actor(ACCENT)
        self._cross_v = _make_line_actor(ACCENT)
        self.renderer.AddActor(self._cross_h)
        self.renderer.AddActor(self._cross_v)

        self.renderer.GetActiveCamera().ParallelProjectionOn()
        rw.GetInteractor().SetInteractorStyle(vtk.vtkInteractorStyleImage())

    def _make_slice_actor(self) -> tuple[vtk.vtkImageSlice, vtk.vtkImageProperty]:
        mapper = vtk.vtkImageResliceMapper()
        mapper.SliceFacesCameraOn()
        mapper.SliceAtFocalPointOn()
        mapper.BorderOff()
        mapper.SetSlabThickness(0)

        prop = vtk.vtkImageProperty()
        prop.SetInterpolationTypeToLinear()

        actor = vtk.vtkImageSlice()
        actor.SetMapper(mapper)
        actor.SetProperty(prop)
        return actor, prop

    # ── Public API ────────────────────────────────────────────────────────

    def configure_base(self, rec: VolumeRecord, pg: PlaneGeometry) -> None:
        """
        Set up the base (first) volume and camera geometry.
        Clears any previously loaded volumes.
        """
        # Remove old actors
        for actor, _ in self._layers:
            self.renderer.RemoveActor(actor)
        self._layers.clear()

        self._pg = pg

        # Create actor for base volume
        actor, prop = self._make_slice_actor()
        actor.GetMapper().SetInputData(rec.vtk_idx)
        actor.SetUserTransform(rec.vtk_xform)
        self._apply_volume_property(prop, rec)
        self.renderer.AddActor(actor)
        self._layers.append((actor, prop))

        # Slider
        mid = pg.n // 2
        self._slider.blockSignals(True)
        self._slider.setMinimum(0)
        self._slider.setMaximum(pg.n - 1)
        self._slider.setValue(mid)
        self._slider.blockSignals(False)
        self._current_slice = mid
        self._lbl_slice.setText(f"{mid+1} / {pg.n}")

        # Re-add crosshairs above image actors
        self.renderer.RemoveActor(self._cross_h)
        self.renderer.RemoveActor(self._cross_v)
        self.renderer.AddActor(self._cross_h)
        self.renderer.AddActor(self._cross_v)

        self._init_camera(mid)
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

    def add_overlay(self, rec: VolumeRecord) -> int:
        """Add an overlay volume on top.  Returns the layer index."""
        actor, prop = self._make_slice_actor()
        actor.GetMapper().SetInputData(rec.vtk_idx)
        actor.SetUserTransform(rec.vtk_xform)
        self._apply_volume_property(prop, rec)

        # Insert before crosshairs (which are the last two actors)
        self.renderer.RemoveActor(self._cross_h)
        self.renderer.RemoveActor(self._cross_v)
        self.renderer.AddActor(actor)
        self.renderer.AddActor(self._cross_h)
        self.renderer.AddActor(self._cross_v)

        self._layers.append((actor, prop))
        self.vtk_widget.GetRenderWindow().Render()
        return len(self._layers) - 1

    def remove_overlay(self, layer_idx: int) -> None:
        """Remove an overlay by layer index (0 = base, cannot be removed here)."""
        if layer_idx == 0 or layer_idx >= len(self._layers):
            return
        actor, _ = self._layers.pop(layer_idx)
        self.renderer.RemoveActor(actor)
        self.vtk_widget.GetRenderWindow().Render()

    def update_layer_property(self, layer_idx: int, rec: VolumeRecord) -> None:
        """Update opacity / visibility / cmap for a layer."""
        if layer_idx >= len(self._layers):
            return
        actor, prop = self._layers[layer_idx]
        self._apply_volume_property(prop, rec)
        actor.SetVisibility(rec.visible)
        self.vtk_widget.GetRenderWindow().Render()

    def set_slice(self, index: int) -> None:
        if self._pg is None:
            return
        index = max(0, min(index, self._pg.n - 1))
        self._slider.blockSignals(True)
        self._slider.setValue(index)
        self._slider.blockSignals(False)
        self._move_to_slice(index)

    def current_slice(self) -> int:
        return self._current_slice

    def world_position(self) -> np.ndarray:
        return self._focal_for(self._current_slice)

    def set_crosshair(self, world_pt: np.ndarray, half_len: float) -> None:
        if self._pg is None:
            return
        _set_line(self._cross_h, world_pt - self._pg.right * half_len, world_pt + self._pg.right * half_len)
        _set_line(self._cross_v, world_pt - self._pg.up    * half_len, world_pt + self._pg.up    * half_len)
        self._cross_h.VisibilityOn(); self._cross_v.VisibilityOn()
        self.vtk_widget.GetRenderWindow().Render()

    # ── Internals ─────────────────────────────────────────────────────────

    def _apply_volume_property(self, prop: vtk.vtkImageProperty, rec: VolumeRecord):
        prop.SetOpacity(rec.opacity if rec.visible else 0.0)
        prop.SetColorWindow(rec.hi - rec.lo or 1.)
        prop.SetColorLevel((rec.hi + rec.lo) / 2.)
        lut = _make_lut(CMAPS.get(rec.cmap), rec.lo, rec.hi)
        prop.SetLookupTable(lut)

    def _focal_for(self, index: int) -> np.ndarray:
        mid = self._pg.n // 2
        return self._pg.centre + (index - mid) * self._pg.step * self._pg.normal

    def _init_camera(self, index: int) -> None:
        focal  = self._focal_for(index)
        DIST   = 2000.
        # Radiological flip: place camera on the opposite side of the focal plane.
        # VTK screen-right = up × viewPlaneNormal, where viewPlaneNormal points
        # from scene toward camera.  Flipping the camera side negates viewPlaneNormal,
        # which negates screen-right (L↔R swap) while keeping up unchanged.
        side   = -1. if self._pg.flip_lr else 1.
        cam    = self.renderer.GetActiveCamera()
        cam.ParallelProjectionOn()
        cam.SetFocalPoint(*focal.tolist())
        cam.SetPosition(*(focal + self._pg.normal * DIST * side).tolist())
        cam.SetViewUp(*self._pg.up.tolist())
        cam.SetParallelScale(self._pg.ps)
        cam.SetClippingRange(1., DIST * 2.)

    def _move_to_slice(self, index: int) -> None:
        if self._pg is None:
            return
        old_fp = np.array(self.renderer.GetActiveCamera().GetFocalPoint())
        new_fp = self._focal_for(index)
        delta  = new_fp - old_fp
        cam    = self.renderer.GetActiveCamera()
        cam.SetFocalPoint(*(old_fp + delta).tolist())
        cam.SetPosition(*(np.array(cam.GetPosition()) + delta).tolist())
        self._current_slice = index
        self._lbl_slice.setText(f"{index+1} / {self._pg.n}")
        self.vtk_widget.GetRenderWindow().Render()

    def _on_slider(self, value: int) -> None:
        self._move_to_slice(value)
        self.slice_changed.emit(self.plane_idx, value)


# ── LayerRow ───────────────────────────────────────────────────────────────

class LayerRow(QWidget):
    """
    One row in the layer panel representing a single loaded volume.
    Emits signals when opacity / cmap / visibility change.
    """
    opacity_changed    = Signal(int, float)    # (vol_idx, opacity 0..1)
    cmap_changed       = Signal(int, str)      # (vol_idx, cmap_name)
    visibility_changed = Signal(int, bool)     # (vol_idx, visible)
    remove_requested   = Signal(int)           # (vol_idx)

    def __init__(self, vol_idx: int, rec: VolumeRecord, parent=None):
        super().__init__(parent)
        self._vol_idx = vol_idx
        self._is_base = (vol_idx == 0)
        self._build_ui(rec)

    def _build_ui(self, rec: VolumeRecord):
        color = VOL_COLORS[self._vol_idx % len(VOL_COLORS)]
        self.setStyleSheet(f"""
            LayerRow {{
                background:{BG_LAYER};
                border-left: 3px solid {color};
                border-radius: 3px;
                margin: 2px 0;
            }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)

        # ── Header row: eye | name | [×] ──────────────────────────────
        hrow = QHBoxLayout()
        hrow.setSpacing(6)

        # Eye / visibility toggle
        self._eye_btn = QToolButton()
        self._eye_btn.setCheckable(True)
        self._eye_btn.setChecked(True)
        self._eye_btn.setFixedSize(22, 22)
        self._eye_btn.setStyleSheet(f"""
            QToolButton {{ border:none; background:transparent; color:{TEXT}; font-size:14px; }}
            QToolButton:checked {{ color:{color}; }}
        """)
        self._eye_btn.setText("👁")
        self._eye_btn.toggled.connect(
            lambda checked: self.visibility_changed.emit(self._vol_idx, checked))
        hrow.addWidget(self._eye_btn)

        lbl = QLabel(rec.name)
        lbl.setStyleSheet(f"color:{color}; font-size:11px; font-weight:bold;")
        lbl.setToolTip(rec.name)
        hrow.addWidget(lbl, 1)

        if not self._is_base:
            rm_btn = QToolButton()
            rm_btn.setText("✕")
            rm_btn.setFixedSize(20, 20)
            rm_btn.setStyleSheet(f"""
                QToolButton {{ border:none; background:transparent;
                               color:{TEXT_DIM}; font-size:11px; }}
                QToolButton:hover {{ color:#ff6b6b; }}
            """)
            rm_btn.clicked.connect(lambda: self.remove_requested.emit(self._vol_idx))
            hrow.addWidget(rm_btn)

        lay.addLayout(hrow)

        # ── Opacity row (overlays only) ────────────────────────────────
        if not self._is_base:
            orow = QHBoxLayout(); orow.setSpacing(6)
            olk = QLabel("Opacity")
            olk.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
            orow.addWidget(olk)

            self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
            self._opacity_slider.setRange(0, 100)
            self._opacity_slider.setValue(100)
            self._opacity_slider.setFixedHeight(16)
            self._opacity_slider.setStyleSheet(f"""
                QSlider::groove:horizontal {{ height:3px; background:#333340; }}
                QSlider::handle:horizontal {{
                    background:{color}; width:12px; height:12px;
                    margin:-5px 0; border-radius:6px; }}
                QSlider::sub-page:horizontal {{ background:{color}; }}
            """)
            self._opacity_slider.valueChanged.connect(
                lambda v: self.opacity_changed.emit(self._vol_idx, v / 100.0))
            orow.addWidget(self._opacity_slider, 1)

            self._pct_lbl = QLabel("100%")
            self._pct_lbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
            self._pct_lbl.setFixedWidth(32)
            self._opacity_slider.valueChanged.connect(
                lambda v: self._pct_lbl.setText(f"{v}%"))
            orow.addWidget(self._pct_lbl)
            lay.addLayout(orow)

        # ── Colourmap row ──────────────────────────────────────────────
        crow = QHBoxLayout(); crow.setSpacing(6)
        clbl = QLabel("Colourmap")
        clbl.setStyleSheet(f"color:{TEXT_DIM}; font-size:10px;")
        crow.addWidget(clbl)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(list(CMAPS.keys()))
        self._cmap_combo.setCurrentText("Grey")
        self._cmap_combo.setFixedHeight(22)
        self._cmap_combo.setStyleSheet(f"""
            QComboBox {{
                background:#1e1e25; color:{TEXT}; border:1px solid #444455;
                border-radius:3px; padding:1px 6px; font-size:11px;
            }}
            QComboBox::drop-down {{ border:none; }}
            QComboBox QAbstractItemView {{
                background:#1e1e25; color:{TEXT}; selection-background-color:#333345;
            }}
        """)
        self._cmap_combo.currentTextChanged.connect(
            lambda t: self.cmap_changed.emit(self._vol_idx, t))
        crow.addWidget(self._cmap_combo, 1)
        lay.addLayout(crow)


# ── LayerPanel ─────────────────────────────────────────────────────────────

class LayerPanel(QWidget):
    """
    Scrollable sidebar listing all loaded volumes as LayerRows.
    Emits the same signals as LayerRow, forwarding vol_idx.
    """
    opacity_changed    = Signal(int, float)
    cmap_changed       = Signal(int, str)
    visibility_changed = Signal(int, bool)
    remove_requested   = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setStyleSheet(f"background:{BG_PANEL};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        hdr = QLabel("  LAYERS")
        hdr.setFixedHeight(32)
        hdr.setStyleSheet(
            f"background:{BG_PANEL}; color:{TEXT_DIM}; font-size:10px; "
            f"font-weight:bold; letter-spacing:2px; "
            f"border-bottom:1px solid #333340;")
        outer.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border:none; background:transparent; }")
        outer.addWidget(scroll, 1)

        self._container = QWidget()
        self._container.setStyleSheet(f"background:{BG_PANEL};")
        self._vlay = QVBoxLayout(self._container)
        self._vlay.setContentsMargins(6, 6, 6, 6)
        self._vlay.setSpacing(4)
        self._vlay.addStretch(1)
        scroll.setWidget(self._container)

        self._rows: list[LayerRow] = []

    def add_row(self, vol_idx: int, rec: VolumeRecord) -> None:
        row = LayerRow(vol_idx, rec, self._container)
        row.opacity_changed.connect(self.opacity_changed)
        row.cmap_changed.connect(self.cmap_changed)
        row.visibility_changed.connect(self.visibility_changed)
        row.remove_requested.connect(self.remove_requested)
        # Insert before the stretch
        self._vlay.insertWidget(self._vlay.count() - 1, row)
        self._rows.append(row)

    def remove_row(self, vol_idx: int) -> None:
        for row in self._rows:
            if row._vol_idx == vol_idx:
                self._vlay.removeWidget(row)
                row.deleteLater()
                self._rows.remove(row)
                break
        # Re-number remaining rows' internal vol_idx
        for i, row in enumerate(self._rows):
            row._vol_idx = i


# ── NiftiViewer ────────────────────────────────────────────────────────────

class NiftiViewer(QWidget):
    """
    Three-view viewer that can display multiple NIfTI volumes simultaneously.

    Volume 0 (base) drives camera geometry in both AFFINE and MEDICAL modes.
    Overlay volumes (indices 1+) are rendered with their own affine transforms
    so that different-grid overlays are still correctly co-registered.
    """

    MODE_AFFINE  = "affine"
    MODE_MEDICAL = "medical"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._volumes:     list[VolumeRecord] = []
        self._base_affine: np.ndarray | None = None
        self._base_shape:  tuple | None = None
        self._ras_min:     np.ndarray | None = None
        self._ras_max:     np.ndarray | None = None
        self._mode         = self.MODE_AFFINE
        self._radiological = False
        self._geoms:       list[PlaneGeometry] = []
        self._half_len:    float = 100.
        self._build_ui()

    def _build_ui(self):
        self.setStyleSheet(f"background:{BG_DARK};")
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Three viewports
        vp_container = QWidget()
        vp_lay = QVBoxLayout(vp_container)
        vp_lay.setContentsMargins(8, 8, 4, 8)
        vp_lay.setSpacing(6)

        spl = QSplitter(Qt.Orientation.Horizontal)
        spl.setStyleSheet("QSplitter::handle{background:#333340;width:4px;}")
        self._vps: list[SliceViewport] = []
        for i in range(3):
            vp = SliceViewport(i, self)
            vp.slice_changed.connect(self._on_slice_changed)
            self._vps.append(vp)
            spl.addWidget(vp)
        spl.setSizes([450, 450, 450])
        vp_lay.addWidget(spl)
        root.addWidget(vp_container, 1)

        # Layer panel
        self._layer_panel = LayerPanel(self)
        self._layer_panel.opacity_changed.connect(self._on_opacity_changed)
        self._layer_panel.cmap_changed.connect(self._on_cmap_changed)
        self._layer_panel.visibility_changed.connect(self._on_visibility_changed)
        self._layer_panel.remove_requested.connect(self._on_remove_requested)
        root.addWidget(self._layer_panel)

    # ── Public API ────────────────────────────────────────────────────────

    def load_base(self, path: str) -> tuple:
        """Load the first (base) volume.  Clears all existing volumes."""
        rec, shape, zooms, code, affine = load_volume_record(path)
        self._volumes.clear()
        self._volumes.append(rec)
        self._base_affine = affine
        self._base_shape  = shape
        self._ras_min, self._ras_max = _ras_bbox(affine, shape)
        self._half_len = np.linalg.norm(self._ras_max - self._ras_min) * 0.65

        # Rebuild layer panel
        for row in list(self._layer_panel._rows):
            self._layer_panel._vlay.removeWidget(row)
            row.deleteLater()
        self._layer_panel._rows.clear()
        self._layer_panel.add_row(0, rec)

        self._refresh()
        return shape, zooms, code

    def add_overlay(self, path: str) -> tuple:
        """Add an overlay volume.  Requires at least one base volume loaded."""
        if not self._volumes:
            raise RuntimeError("Load a base volume first.")
        rec, shape, zooms, code, _ = load_volume_record(path)
        self._volumes.append(rec)
        vol_idx = len(self._volumes) - 1
        self._layer_panel.add_row(vol_idx, rec)

        for vp in self._vps:
            vp.add_overlay(rec)

        return shape, zooms, code

    def set_mode(self, mode: str, radiological: bool = False) -> None:
        self._mode = mode
        self._radiological = radiological
        if self._volumes:
            self._refresh()

    def set_radiological(self, enabled: bool) -> None:
        if self._radiological == enabled:
            return
        self._radiological = enabled
        if self._mode == self.MODE_MEDICAL and self._volumes:
            self._refresh()

    # ── Layer signal handlers ─────────────────────────────────────────────

    def _on_opacity_changed(self, vol_idx: int, opacity: float) -> None:
        if vol_idx >= len(self._volumes):
            return
        self._volumes[vol_idx].opacity = opacity
        for vp in self._vps:
            vp.update_layer_property(vol_idx, self._volumes[vol_idx])

    def _on_cmap_changed(self, vol_idx: int, cmap: str) -> None:
        if vol_idx >= len(self._volumes):
            return
        self._volumes[vol_idx].cmap = cmap
        for vp in self._vps:
            vp.update_layer_property(vol_idx, self._volumes[vol_idx])

    def _on_visibility_changed(self, vol_idx: int, visible: bool) -> None:
        if vol_idx >= len(self._volumes):
            return
        self._volumes[vol_idx].visible = visible
        for vp in self._vps:
            vp.update_layer_property(vol_idx, self._volumes[vol_idx])

    def _on_remove_requested(self, vol_idx: int) -> None:
        if vol_idx == 0 or vol_idx >= len(self._volumes):
            return
        self._volumes.pop(vol_idx)
        for vp in self._vps:
            vp.remove_overlay(vol_idx)
        self._layer_panel.remove_row(vol_idx)

    # ── Internal refresh ──────────────────────────────────────────────────

    def _refresh(self):
        """Rebuild camera geometry and reconfigure all viewports."""
        if not self._volumes:
            return

        if self._mode == self.MODE_AFFINE:
            self._geoms = affine_plane_geoms(self._base_affine, self._base_shape)
        else:
            sp, _ = _decompose(self._base_affine)
            iso   = float(np.min(sp))
            self._geoms = medical_plane_geoms(
                self._ras_min, self._ras_max, iso, self._radiological)

        names = AFFINE_NAMES if self._mode == self.MODE_AFFINE else MEDICAL_NAMES

        for i, vp in enumerate(self._vps):
            vp.set_title(names[i])
            # Reconfigure base volume (resets camera + slider)
            vp.configure_base(self._volumes[0], self._geoms[i])
            # Re-add all overlays
            for rec in self._volumes[1:]:
                vp.add_overlay(rec)

    # ── Crosshair sync ────────────────────────────────────────────────────

    def _on_slice_changed(self, _plane_idx: int, _slice_idx: int) -> None:
        if not self._geoms:
            return
        pts     = [vp.world_position() for vp in self._vps]
        normals = [g.normal for g in self._geoms]
        A = np.vstack(normals)
        b = np.array([np.dot(normals[i], pts[i]) for i in range(3)])
        try:
            world_pt = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            world_pt = self._geoms[0].centre.copy()
        for vp in self._vps:
            vp.set_crosshair(world_pt, self._half_len)


# ── Main window ────────────────────────────────────────────────────────────

TB_STYLE = f"""
QToolBar {{
    background:{BG_PANEL}; border-bottom:1px solid #333340;
    spacing:6px; padding:4px 8px;
}}
QPushButton {{
    background:#2d2d38; color:{TEXT};
    border:1px solid #444455; border-radius:4px;
    padding:4px 12px; font-size:12px;
}}
QPushButton:hover {{ background:#3d3d50; border-color:{ACCENT}; color:{ACCENT}; }}
QPushButton:disabled {{ color:#555566; border-color:#333340; }}
QRadioButton, QCheckBox {{ color:{TEXT}; font-size:12px; }}
QGroupBox {{
    color:{TEXT_DIM}; font-size:11px;
    border:1px solid #444455; border-radius:4px;
    margin-top:6px; padding:2px 8px;
}}
QGroupBox::title {{ subcontrol-origin:margin; left:6px; }}
"""


class NiftiViewerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NIfTI Viewer — Multi-Volume")
        self.resize(1580, 600)
        self._apply_palette()
        self.viewer = NiftiViewer()
        self.setCentralWidget(self.viewer)
        self._build_toolbar()
        self._status = QStatusBar()
        self._status.setStyleSheet(
            f"background:{BG_PANEL}; color:{TEXT_DIM}; font-size:11px;")
        self.setStatusBar(self._status)
        self._status.showMessage("Open a NIfTI file to begin.  "
                                 "Add overlays with 'Add overlay…'")

    def _build_toolbar(self):
        tb = QToolBar("Controls")
        tb.setMovable(False)
        tb.setStyleSheet(TB_STYLE)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        # Open base
        self._btn_open = QPushButton("  Open NIfTI…")
        self._btn_open.clicked.connect(self._open_base)
        tb.addWidget(self._btn_open)

        # Add overlay
        self._btn_overlay = QPushButton("  Add overlay…")
        self._btn_overlay.setEnabled(False)
        self._btn_overlay.clicked.connect(self._add_overlay)
        tb.addWidget(self._btn_overlay)

        tb.addSeparator()

        # Display mode
        mode_box = QGroupBox("Display mode")
        ml = QHBoxLayout(mode_box); ml.setContentsMargins(6,2,6,2); ml.setSpacing(12)
        self._rb_affine  = QRadioButton("Affine (native axes)")
        self._rb_medical = QRadioButton("Medical (RAS)")
        self._rb_affine.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self._rb_affine,  0)
        grp.addButton(self._rb_medical, 1)
        grp.idClicked.connect(self._on_mode)
        ml.addWidget(self._rb_affine); ml.addWidget(self._rb_medical)
        tb.addWidget(mode_box)

        tb.addSeparator()

        # Convention
        conv_box = QGroupBox("Convention")
        cl = QHBoxLayout(conv_box); cl.setContentsMargins(6,2,6,2)
        self._cb_radio = QCheckBox("Radiological (flip L↔R)")
        self._cb_radio.setEnabled(False)
        self._cb_radio.stateChanged.connect(
            lambda s: self.viewer.set_radiological(bool(s)))
        cl.addWidget(self._cb_radio)
        tb.addWidget(conv_box)

    def _on_mode(self, btn_id: int):
        mode = NiftiViewer.MODE_AFFINE if btn_id == 0 else NiftiViewer.MODE_MEDICAL
        self._cb_radio.setEnabled(btn_id == 1)
        self.viewer.set_mode(mode, self._cb_radio.isChecked())

    def _open_base(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Base NIfTI", "",
            "NIfTI Files (*.nii *.nii.gz);;All Files (*)")
        if not path:
            return
        try:
            shape, sp, code = self.viewer.load_base(path)
            name = os.path.basename(path)
            self._status.showMessage(
                f"[base] {name}  |  {shape[0]}×{shape[1]}×{shape[2]}"
                f"  |  {sp[0]:.3f}×{sp[1]:.3f}×{sp[2]:.3f} mm  |  {code}")
            self._btn_overlay.setEnabled(True)
        except Exception as exc:
            self._status.showMessage(f"Error: {exc}")
            raise

    def _add_overlay(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Add Overlay NIfTI", "",
            "NIfTI Files (*.nii *.nii.gz);;All Files (*)")
        if not path:
            return
        try:
            shape, sp, code = self.viewer.add_overlay(path)
            name = os.path.basename(path)
            n = len(self.viewer._volumes) - 1
            self._status.showMessage(
                f"[overlay {n}] {name}  |  {shape[0]}×{shape[1]}×{shape[2]}"
                f"  |  {sp[0]:.3f}×{sp[1]:.3f}×{sp[2]:.3f} mm  |  {code}")
        except Exception as exc:
            self._status.showMessage(f"Error: {exc}")
            raise

    def _apply_palette(self):
        pal = QPalette()
        pal.setColor(QPalette.ColorRole.Window,     QColor(BG_DARK))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
        pal.setColor(QPalette.ColorRole.Base,       QColor(BG_PANEL))
        pal.setColor(QPalette.ColorRole.Text,       QColor(TEXT))
        QApplication.setPalette(pal)


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("NIfTI Viewer")
    win = NiftiViewerWindow()
    win.show()
    # First positional arg = base, rest = overlays
    args = sys.argv[1:]
    if args:
        try:
            win.viewer.load_base(args[0])
            win._btn_overlay.setEnabled(True)
            for path in args[1:]:
                win.viewer.add_overlay(path)
        except Exception as exc:
            win._status.showMessage(f"Error loading from command line: {exc}")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()