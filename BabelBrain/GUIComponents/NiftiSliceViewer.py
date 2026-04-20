"""NiftiSliceViewer
Three-panel orthogonal VTK slice viewer for NIfTI data.

Replaces the Matplotlib imshow panels in BabelBrain.UpdateMask.

Key design principle
--------------------
Every vtkImageData is built with the *full affine* from the nibabel header
(origin + spacing + direction-cosine matrix).  This puts T1W and mask in a
shared world coordinate system, so the vtkResliceCursor can slice both
coherently without any pre-reorientation step.

Orientation modes
-----------------
'raw'     – cursor axes = image voxel axes (i, j, k directions in world space,
            i.e. the column vectors of the affine rotation matrix).
            Matches the previous Matplotlib behaviour.
'medical' – cursor axes = world RAS axes ([1,0,0], [0,1,0], [0,0,1]),
            giving standard Coronal / Sagittal / Axial views.

Switching modes only updates the cursor axes and re-renders — the VTK
pipeline (images, reslice filters, actors) is NOT rebuilt.

Target VTK: 9.5.x
"""

import numpy as np
import nibabel
import nibabel.affines

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PySide6.QtCore import Qt

import vtkmodules.all as vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import numpy_to_vtk


# ---------------------------------------------------------------------------
# Tissue colour table  (matches existing BabelBrain colour definitions)
# mask integer value → (R, G, B, A) in 0-255
# ---------------------------------------------------------------------------
_TISSUE_RGBA: dict[int, tuple[int, int, int, int]] = {
    0: (  0,   0,   0,   0),   # background  – fully transparent
    1: (  0,  77, 255, 160),   # scalp
    2: (  0, 128, 255, 160),   # cortical bone
    3: ( 21, 255, 225, 160),   # trabecular bone
    4: (124, 255, 121, 160),   # brain (non-segmented)
    5: (255, 255,   0, 220),   # focal-point voxel  (bright yellow)
    6: (255, 148,   0, 160),   # white matter
    7: (255,  29,   0, 160),   # grey matter
    8: (127,   0,   0, 160),   # CSF
}

_N_LABELS = max(_TISSUE_RGBA) + 1


# ---------------------------------------------------------------------------
# Affine → VTK helpers
# ---------------------------------------------------------------------------

def _affine_decompose(affine: np.ndarray):
    """Decompose a 4×4 nibabel affine into (spacing, origin, direction).

    Returns
    -------
    spacing   : ndarray shape (3,)  – voxel sizes in mm (always positive)
    origin    : ndarray shape (3,)  – world coords of voxel (0,0,0)
    direction : ndarray shape (3,3) – unit column vectors = direction cosines
                direction[:,j] is the world-space direction of image axis j.
                The matrix determinant may be −1 (left-handed / flipped image).
    """
    linear   = affine[:3, :3]
    spacing  = np.sqrt((linear ** 2).sum(axis=0))          # column norms
    origin   = affine[:3, 3].copy()
    direction = linear / spacing[np.newaxis, :]             # unit columns
    return spacing, origin, direction


def _nibabel_to_vtk(nib_img, *, as_uint8: bool = False) -> vtk.vtkImageData:
    """Convert a nibabel spatial image to a vtkImageData.

    The full affine (origin, spacing, direction cosines) is encoded so that
    both T1W and mask share the same VTK world coordinate system.
    VTK 9.x SetDirectionMatrix is used to handle oblique / flipped images.
    """
    data    = nib_img.get_fdata()
    affine  = nib_img.affine

    spacing, origin, direction = _affine_decompose(affine)

    if as_uint8:
        arr   = data.astype(np.uint8)
        vtype = vtk.VTK_UNSIGNED_CHAR
    else:
        arr   = data.astype(np.float32)
        vtype = vtk.VTK_FLOAT

    # VTK stores arrays in Fortran (column-major) order
    vtk_arr = numpy_to_vtk(arr.ravel(order='F'), deep=True, array_type=vtype)

    img = vtk.vtkImageData()
    img.SetDimensions(int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]))
    img.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
    img.SetOrigin(float(origin[0]),  float(origin[1]),  float(origin[2]))

    # Direction cosines – column j of direction is world direction of image axis j.
    # vtkMatrix3x3 / SetDirectionMatrix use row-major order: element (i, j).
    mat3 = vtk.vtkMatrix3x3()
    for i in range(3):
        for j in range(3):
            mat3.SetElement(i, j, float(direction[i, j]))
    img.SetDirectionMatrix(mat3)

    img.GetPointData().SetScalars(vtk_arr)
    return img


# ---------------------------------------------------------------------------
# Lookup-table builders
# ---------------------------------------------------------------------------

def _build_tissue_lut() -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(_N_LABELS)
    lut.SetTableRange(0, _N_LABELS - 1)
    for v in range(_N_LABELS):
        r, g, b, a = _TISSUE_RGBA.get(v, (0, 0, 0, 0))
        lut.SetTableValue(v, r / 255.0, g / 255.0, b / 255.0, a / 255.0)
    lut.Build()
    return lut


def _build_grayscale_lut(vmin: float, vmax: float, alpha: float) -> vtk.vtkLookupTable:
    lut = vtk.vtkLookupTable()
    lut.SetRange(vmin, vmax)
    lut.SetHueRange(0.0, 0.0)
    lut.SetSaturationRange(0.0, 0.0)
    lut.SetValueRange(0.0, 1.0)
    lut.SetAlphaRange(alpha, alpha)
    lut.SetRampToLinear()
    lut.Build()
    return lut


# ---------------------------------------------------------------------------
# Per-panel helper
# ---------------------------------------------------------------------------

class _SlicePanel:
    """One VTK slice panel.

    Owns:
    • vtkRenderer
    • vtkResliceCursorWidget  + vtkResliceCursorLineRepresentation  (T1W)
    • vtkImageReslice  →  vtkImageMapToColors  →  vtkImageActor     (mask)
    """

    def __init__(self,
                 interactor:      QVTKRenderWindowInteractor,
                 reslice_cursor:  vtk.vtkResliceCursor,
                 plane_normal:    int,          # 0=X-normal  1=Y-normal  2=Z-normal
                 mask_vtk:        vtk.vtkImageData,
                 tissue_lut:      vtk.vtkLookupTable,
                 bg_color:        tuple = (0.12, 0.12, 0.12)):

        self.interactor = interactor
        ren_win = interactor.GetRenderWindow()

        # ── Renderer ──────────────────────────────────────────────────────
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*bg_color)
        ren_win.AddRenderer(self.renderer)

        # ── T1W layer via vtkResliceCursorWidget ──────────────────────────
        self.rep = vtk.vtkResliceCursorLineRepresentation()
        algo = self.rep.GetResliceCursorActor().GetCursorAlgorithm()
        algo.SetResliceCursor(reslice_cursor)
        algo.SetReslicePlaneNormal(plane_normal)

        self.widget = vtk.vtkResliceCursorWidget()
        self.widget.SetDefaultRenderer(self.renderer)
        self.widget.SetInteractor(interactor)
        self.widget.SetRepresentation(self.rep)
        self.widget.SetEnabled(1)

        # ── Mask overlay ──────────────────────────────────────────────────
        # vtkImageReslice slices the mask at the same plane as the T1W cursor.
        # Axes are synced via _sync_mask() which copies the cursor's internal
        # reslice-axes matrix.
        self.mask_reslice = vtk.vtkImageReslice()
        self.mask_reslice.SetInputData(mask_vtk)
        self.mask_reslice.SetOutputDimensionality(2)
        self.mask_reslice.SetInterpolationModeToNearestNeighbor()
        self.mask_reslice.SetBackgroundLevel(0)

        self.mask_map = vtk.vtkImageMapToColors()
        self.mask_map.SetInputConnection(self.mask_reslice.GetOutputPort())
        self.mask_map.SetLookupTable(tissue_lut)
        self.mask_map.SetOutputFormatToRGBA()

        self.mask_actor = vtk.vtkImageActor()
        self.mask_actor.GetMapper().SetInputConnection(self.mask_map.GetOutputPort())
        self.mask_actor.SetOpacity(1.0)   # per-label alpha is in the LUT
        self.renderer.AddActor(self.mask_actor)

    # ── Mask sync ─────────────────────────────────────────────────────────

    def _sync_mask(self):
        """Copy the cursor's current reslice-axes matrix to the mask filter.

        rep.GetReslice() exposes the internal vtkImageReslice of the
        vtkResliceCursorRepresentation.  Its ResliceAxes matrix is a 4×4
        describing the current slice plane in world space.  Applying the same
        matrix to mask_reslice ensures both layers cut the same plane.
        """
        internal = self.rep.GetReslice()
        if internal is None:
            return
        axes = internal.GetResliceAxes()
        if axes is None:
            return
        mat = vtk.vtkMatrix4x4()
        mat.DeepCopy(axes)
        self.mask_reslice.SetResliceAxes(mat)
        self.mask_reslice.Modified()

    def on_interaction(self):
        self._sync_mask()
        self.mask_map.Update()

    # ── Visibility / camera ───────────────────────────────────────────────

    def set_marks_visible(self, visible: bool):
        self.rep.GetResliceCursorActor().SetVisibility(int(visible))

    def reset_camera(self):
        self.renderer.ResetCamera()


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class NiftiSliceViewer(QWidget):
    """Three-panel orthogonal NIfTI viewer (VTK 9.5.x).

    Usage
    -----
    viewer = NiftiSliceViewer(parent)
    layout.addWidget(viewer)

    viewer.set_volumes(t1w_nib, mask_nib, focal_voxel, alpha=0.5)

    # Slots wired to existing form controls:
    viewer.set_transparency(alpha)    # TransparencyScrollBar
    viewer.set_marks_visible(vis)     # HideMarkscheckBox
    """

    RAW     = 'raw'
    MEDICAL = 'medical'

    # Panel → cursor plane-normal index
    # 0 = normal along cursor-X → YZ slice (sagittal in medical mode)
    # 1 = normal along cursor-Y → XZ slice (coronal  in medical mode)
    # 2 = normal along cursor-Z → XY slice (axial    in medical mode)
    _PLANE_NORMALS = (1, 0, 2)   # panels: coronal, sagittal, axial

    _LABELS_RAW     = ('XZ (coronal-like)', 'YZ (sagittal-like)', 'XY (axial-like)')
    _LABELS_MEDICAL = ('Coronal', 'Sagittal', 'Axial')

    def __init__(self, parent=None):
        super().__init__(parent)

        self._mode   = self.RAW
        self._alpha  = 0.5
        self._panels: list[_SlicePanel]               = []
        self._interactors: list[QVTKRenderWindowInteractor] = []
        self._panel_labels: list[QLabel]              = []
        self._tissue_lut  = _build_tissue_lut()
        self._t1w_lut     = None          # built in _rebuild()
        self._cursor      = None          # vtkResliceCursor, built in _rebuild()
        self._t1w_direction = None        # 3×3 ndarray, image direction cosines

        # nibabel originals set by set_volumes()
        self._t1w_orig  = None
        self._mask_orig = None
        self._focal_orig: np.ndarray | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # Orientation toolbar
        tb = QHBoxLayout()
        tb.setSpacing(8)
        tb.addWidget(QLabel('View:'))
        self._combo = QComboBox()
        self._combo.addItems([
            'Raw data axes',
            'Medical convention  (Coronal / Sagittal / Axial)',
        ])
        self._combo.currentIndexChanged.connect(self._on_mode_changed)
        tb.addWidget(self._combo)
        tb.addStretch()
        root.addLayout(tb)

        # Three VTK panels side by side
        panel_row = QHBoxLayout()
        panel_row.setSpacing(4)
        for i in range(3):
            col = QVBoxLayout()
            col.setSpacing(2)
            lbl = QLabel(self._LABELS_RAW[i])
            lbl.setAlignment(Qt.AlignCenter)
            self._panel_labels.append(lbl)
            col.addWidget(lbl)
            interactor = QVTKRenderWindowInteractor(self)
            self._interactors.append(interactor)
            col.addWidget(interactor, stretch=1)
            panel_row.addLayout(col)
        root.addLayout(panel_row, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_volumes(self,
                    t1w_nib,
                    mask_nib,
                    focal_voxel,
                    alpha: float = 0.5):
        """Load NIfTI volumes and render all three panels.

        Parameters
        ----------
        t1w_nib     : nibabel spatial image (T1W, resampled to mask resolution)
        mask_nib    : nibabel spatial image (integer segmentation labels)
        focal_voxel : (i, j, k) voxel indices of the focal point in t1w_nib space
        alpha       : float [0,1]  T1W transparency (0 = opaque, 1 = transparent)
        """
        self._t1w_orig   = t1w_nib
        self._mask_orig  = mask_nib
        self._focal_orig = np.asarray(focal_voxel, dtype=float)
        self._alpha      = alpha
        self._rebuild()

    def set_transparency(self, alpha: float):
        """Update T1W opacity (slot for TransparencyScrollBar)."""
        self._alpha = float(alpha)
        if self._t1w_lut is not None:
            self._t1w_lut.SetAlphaRange(1.0 - alpha, 1.0 - alpha)
            self._t1w_lut.Modified()
            for panel in self._panels:
                panel.rep.GetColorMap().Modified()
            self._render_all()

    def set_marks_visible(self, visible: bool):
        """Show / hide crosshair cursor lines (slot for HideMarkscheckBox)."""
        for panel in self._panels:
            panel.set_marks_visible(visible)
        self._render_all()

    # ------------------------------------------------------------------
    # Internal – pipeline build
    # ------------------------------------------------------------------
    def _rebuild(self):
        """Build (or rebuild) the full VTK pipeline for the current volumes."""
        # Disable previous cursor widgets
        for p in self._panels:
            p.widget.SetEnabled(0)
        self._panels.clear()

        t1w_nib  = self._t1w_orig
        mask_nib = self._mask_orig

        # Build VTK images with full affine encoded

        t1w_vtk  = _nibabel_to_vtk(t1w_nib)
        mask_vtk = _nibabel_to_vtk(mask_nib, as_uint8=True)

        # reader = vtk.vtkNIFTIImageReader()
        # reader.SetFileName(t1w_nib)
        # reader.Update()
        # t1w_vtk = reader.GetOutput()

        # reader = vtk.vtkNIFTIImageReader()
        # reader.SetFileName(mask_nib)
        # reader.SetDataScalarTypeToUnsignedChar()
        # reader.Update()
        # mask_vtk = reader.GetOutput()

        # Focal point in world (mm) coordinates via the affine
        affine=nibabel.load(t1w_nib).affine
        focal_world = nibabel.affines.apply_affine(
            affine, self._focal_orig
        )

        # Store direction cosines for mode switching (no rebuild needed)
        _, _, self._t1w_direction = _affine_decompose(affine)

        # Shared reslice cursor – image is set here, not on the representation
        cursor = vtk.vtkResliceCursor()
        cursor.SetCenter(*focal_world.tolist())
        cursor.SetImage(t1w_vtk)
        cursor.SetThickMode(False)
        self._cursor = cursor

        # Set cursor axes for the current orientation mode
        self._apply_cursor_axes()

        # T1W grayscale LUT (alpha inverted vs slider)
        t1w_range = t1w_vtk.GetScalarRange()
        self._t1w_lut = _build_grayscale_lut(
            t1w_range[0], t1w_range[1],
            alpha=1.0 - self._alpha,
        )

        for i, interactor in enumerate(self._interactors):
            # Remove stale renderers (collect first, remove after)
            ren_win = interactor.GetRenderWindow()
            old = []
            col = ren_win.GetRenderers()
            col.InitTraversal()
            while (r := col.GetNextItem()) is not None:
                old.append(r)
            for r in old:
                ren_win.RemoveRenderer(r)

            panel = _SlicePanel(
                interactor    = interactor,
                reslice_cursor= cursor,
                plane_normal  = self._PLANE_NORMALS[i],
                mask_vtk      = mask_vtk,
                tissue_lut    = self._tissue_lut,
            )

            # Wire T1W grayscale LUT into the representation's color-map filter
            panel.rep.GetColorMap().SetLookupTable(self._t1w_lut)
            panel.rep.GetColorMap().SetOutputFormatToRGBA()

            panel.widget.AddObserver('InteractionEvent', self._on_interaction)
            panel.widget.AddObserver('WindowLevelEvent',  self._on_interaction)

            panel.reset_camera()
            if not interactor.GetInitialized():
                interactor.Initialize()

            self._panels.append(panel)

        self._render_all()

    def _apply_cursor_axes(self):
        """Set the cursor X/Y/Z axes for the current orientation mode.

        raw mode     – axes aligned with the image voxel directions (column
                       vectors of the affine rotation matrix).
        medical mode – axes aligned with world RAS ([1,0,0],[0,1,0],[0,0,1]).

        The vtkResliceCursorAlgorithm.SetReslicePlaneNormal(n) selects which
        cursor axis is perpendicular to each panel's view plane:
          n=0 → normal = cursor-X → views the YZ plane
          n=1 → normal = cursor-Y → views the XZ plane
          n=2 → normal = cursor-Z → views the XY plane
        """
        if self._cursor is None or self._t1w_direction is None:
            return

        d = self._t1w_direction  # shape (3,3), columns = image axis directions

        if self._mode == self.MEDICAL:
            # World RAS axes
            self._cursor.SetXAxis(1.0, 0.0, 0.0)
            self._cursor.SetYAxis(0.0, 1.0, 0.0)
            self._cursor.SetZAxis(0.0, 0.0, 1.0)
        else:
            # Image voxel axes expressed in world coordinates
            self._cursor.SetXAxis(float(d[0, 0]), float(d[1, 0]), float(d[2, 0]))
            self._cursor.SetYAxis(float(d[0, 1]), float(d[1, 1]), float(d[2, 1]))
            self._cursor.SetZAxis(float(d[0, 2]), float(d[1, 2]), float(d[2, 2]))

        self._cursor.Modified()

    # ------------------------------------------------------------------
    # Interaction callbacks
    # ------------------------------------------------------------------
    def _on_interaction(self, caller, event):
        """Sync mask overlays in all panels when the cursor moves."""
        for panel in self._panels:
            panel.on_interaction()
        self._render_all()

    def _on_mode_changed(self, idx: int):
        self._mode = self.RAW if idx == 0 else self.MEDICAL
        labels = self._LABELS_RAW if self._mode == self.RAW else self._LABELS_MEDICAL
        for lbl, text in zip(self._panel_labels, labels):
            lbl.setText(text)
        # Only update cursor axes — no VTK pipeline rebuild required
        if self._cursor is not None:
            self._apply_cursor_axes()
            self._render_all()

    def _render_all(self):
        for interactor in self._interactors:
            interactor.GetRenderWindow().Render()

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        for interactor in self._interactors:
            interactor.Finalize()
        super().closeEvent(event)
