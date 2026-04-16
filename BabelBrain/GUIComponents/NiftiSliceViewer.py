"""NiftiSliceViewer
Three-panel orthogonal VTK slice viewer for NIfTI data.

Replaces the Matplotlib imshow panels in BabelBrain.UpdateMask.

Orientation modes
-----------------
'raw'     – slices through the raw data array axes (i, j, k), identical to the
            previous Matplotlib behaviour.
'medical' – data reoriented to nearest-RAS canonical orientation via
            nibabel.as_closest_canonical(), giving labelled
            Coronal / Sagittal / Axial panels.

The shared vtkResliceCursor gives FSLeyes-style synchronized crosshairs: dragging
the crosshair in one panel scrolls the other two panels to that position.
The mask overlay is kept in sync via the cursor representation's internal reslice
axes matrix, so both layers slice the same plane automatically.
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
# Helpers
# ---------------------------------------------------------------------------

def _nibabel_to_vtk(nib_img, *, as_uint8: bool = False) -> vtk.vtkImageData:
    """Convert a nibabel spatial image to a vtkImageData (float32 or uint8).

    VTK stores arrays in Fortran (column-major) order: the first axis changes
    fastest in memory.  numpy's default is C order, so we explicitly flatten
    with order='F' before handing the buffer to VTK.
    """
    data = nib_img.get_fdata()
    zooms = np.array(nib_img.header.get_zooms()[:3], dtype=float)

    if as_uint8:
        data = data.astype(np.uint8)
        vtype = vtk.VTK_UNSIGNED_CHAR
    else:
        data = data.astype(np.float32)
        vtype = vtk.VTK_FLOAT

    vtk_arr = numpy_to_vtk(data.ravel(order='F'), deep=True, array_type=vtype)

    img = vtk.vtkImageData()
    img.SetDimensions(int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
    img.SetSpacing(float(zooms[0]), float(zooms[1]), float(zooms[2]))
    img.SetOrigin(0.0, 0.0, 0.0)
    img.GetPointData().SetScalars(vtk_arr)
    return img


def _build_tissue_lut() -> vtk.vtkLookupTable:
    """Return a vtkLookupTable mapping integer mask labels → RGBA colours."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(_N_LABELS)
    lut.SetTableRange(0, _N_LABELS - 1)
    for v in range(_N_LABELS):
        r, g, b, a = _TISSUE_RGBA.get(v, (0, 0, 0, 0))
        lut.SetTableValue(v, r / 255.0, g / 255.0, b / 255.0, a / 255.0)
    lut.Build()
    return lut


def _build_grayscale_lut(vmin: float, vmax: float, alpha: float) -> vtk.vtkLookupTable:
    """Return a grayscale (hue=0, sat=0) LUT with the given opacity."""
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
    """Wraps one QVTKRenderWindowInteractor panel.

    Owns:
      - vtkRenderer
      - vtkResliceCursorWidget  (T1W slice + crosshair navigation)
      - vtkResliceCursorLineRepresentation
      - vtkImageReslice         (mask, kept in sync with cursor axes)
      - vtkImageActor           (mask overlay)
    """

    def __init__(self,
                 interactor: QVTKRenderWindowInteractor,
                 reslice_cursor: vtk.vtkResliceCursor,
                 plane_normal: int,       # 0=YZ, 1=XZ, 2=XY
                 mask_vtk: vtk.vtkImageData,
                 tissue_lut: vtk.vtkLookupTable,
                 bg_color: tuple = (0.12, 0.12, 0.12)):

        self.interactor = interactor
        ren_win = interactor.GetRenderWindow()

        # Renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(*bg_color)
        ren_win.AddRenderer(self.renderer)

        # ------------------------------------------------------------------
        # T1W layer: vtkResliceCursorWidget does the slicing + displays it
        # ------------------------------------------------------------------
        self.rep = vtk.vtkResliceCursorLineRepresentation()
        cursor_algo = self.rep.GetResliceCursorActor().GetCursorAlgorithm()
        cursor_algo.SetResliceCursor(reslice_cursor)
        cursor_algo.SetReslicePlaneNormal(plane_normal)

        self.widget = vtk.vtkResliceCursorWidget()
        self.widget.SetDefaultRenderer(self.renderer)
        self.widget.SetInteractor(interactor)
        self.widget.SetRepresentation(self.rep)
        self.widget.SetEnabled(1)

        # ------------------------------------------------------------------
        # Mask layer: vtkImageReslice driven by the cursor's reslice matrix
        # ------------------------------------------------------------------
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
        # Tissue LUT already encodes per-label alpha; actor opacity stays at 1.
        self.mask_actor.SetOpacity(1.0)
        self.renderer.AddActor(self.mask_actor)

        # Sync the mask reslice axes with the T1W cursor axes now and on each
        # interaction.  The cursor representation exposes its internal
        # vtkImageReslice via GetReslice(); we copy its axes matrix.
        self._sync_mask()

    def _sync_mask(self):
        """Copy the cursor's current reslice-axes matrix to the mask filter."""
        internal = self.rep.GetReslice()
        if internal is None:
            return
        axes = internal.GetResliceAxes()
        if axes is None:
            return
        # Clone the matrix so we own our own copy
        mat = vtk.vtkMatrix4x4()
        mat.DeepCopy(axes)
        self.mask_reslice.SetResliceAxes(mat)
        self.mask_reslice.Modified()

    def on_interaction(self):
        """Call after any cursor interaction to keep mask in sync."""
        self._sync_mask()
        self.mask_map.Update()

    def set_marks_visible(self, visible: bool):
        self.rep.GetResliceCursorActor().SetVisibility(int(visible))

    def reset_camera(self):
        self.renderer.ResetCamera()


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class NiftiSliceViewer(QWidget):
    """Three-panel orthogonal NIfTI viewer (VTK-based).

    Usage
    -----
    viewer = NiftiSliceViewer(parent)
    # plug it into a layout that currently holds the Matplotlib canvas:
    layout.addWidget(viewer)

    # Load data (called from UpdateMask):
    viewer.set_volumes(t1w_nib, mask_nib, focal_voxel, alpha=0.5)

    # Slots wired to existing form controls:
    viewer.set_transparency(alpha)   # connected to TransparencyScrollBar
    viewer.set_marks_visible(vis)    # connected to HideMarkscheckBox
    """

    RAW     = 'raw'
    MEDICAL = 'medical'

    # Panel index 0 → coronal-like, 1 → sagittal-like, 2 → axial-like
    # vtkResliceCursor plane normals: 0=YZ, 1=XZ, 2=XY
    _PLANE_NORMALS = (1, 0, 2)

    _LABELS_RAW     = ('XZ (coronal-like)', 'YZ (sagittal-like)', 'XY (axial-like)')
    _LABELS_MEDICAL = ('Coronal', 'Sagittal', 'Axial')

    def __init__(self, parent=None):
        super().__init__(parent)

        self._mode   = self.RAW
        self._alpha  = 0.5
        self._panels: list[_SlicePanel] = []
        self._interactors: list[QVTKRenderWindowInteractor] = []
        self._panel_labels: list[QLabel] = []
        self._tissue_lut = _build_tissue_lut()

        # nibabel images set by set_volumes()
        self._t1w_orig  = None
        self._mask_orig = None
        self._focal_orig: np.ndarray | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(2)

        # --- Orientation toolbar ---
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

        # --- Three VTK panels side by side ---
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
        t1w_nib : nibabel spatial image
            T1-weighted anatomical (resampled, matching mask resolution).
        mask_nib : nibabel spatial image
            Integer segmentation mask (labels per _TISSUE_RGBA).
        focal_voxel : array-like, shape (3,)
            (i, j, k) voxel coordinates of the focal point **in t1w_nib space**.
        alpha : float
            Initial T1W transparency (0 = opaque, 1 = fully transparent).
        """
        self._t1w_orig   = t1w_nib
        self._mask_orig  = mask_nib
        self._focal_orig = np.asarray(focal_voxel, dtype=float)
        self._alpha      = alpha
        self._rebuild()

    def set_transparency(self, alpha: float):
        """Update T1W opacity (slot for TransparencyScrollBar).

        Parameters
        ----------
        alpha : float  in [0, 1]  (0 = fully opaque, 1 = fully transparent,
                                    matching the current scrollbar convention)
        """
        self._alpha = float(alpha)
        if self._t1w_lut is not None:
            self._t1w_lut.SetAlphaRange(1.0 - alpha, 1.0 - alpha)
            self._t1w_lut.Modified()
            for panel in self._panels:
                panel.rep.GetColorMap().Modified()
            self._render_all()

    def set_marks_visible(self, visible: bool):
        """Show / hide the crosshair cursor lines (slot for HideMarkscheckBox)."""
        for panel in self._panels:
            panel.set_marks_visible(visible)
        self._render_all()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _get_oriented(self):
        """Return (t1w_nib, mask_nib, focal_voxel) in the active orientation."""
        if self._mode == self.MEDICAL:
            t1w  = nibabel.as_closest_canonical(self._t1w_orig)
            mask = nibabel.as_closest_canonical(self._mask_orig)
            # Remap the focal voxel through the reorientation affines
            focal_mm  = nibabel.affines.apply_affine(self._t1w_orig.affine,
                                                      self._focal_orig)
            focal_vox = nibabel.affines.apply_affine(np.linalg.inv(t1w.affine),
                                                     focal_mm)
        else:
            t1w       = self._t1w_orig
            mask      = self._mask_orig
            focal_vox = self._focal_orig.copy()
        return t1w, mask, focal_vox

    def _rebuild(self):
        """Tear down and recreate the VTK pipeline for the current orientation."""
        # Disable old cursor widgets before destroying panels
        for p in self._panels:
            p.widget.SetEnabled(0)
        self._panels.clear()

        t1w_nib, mask_nib, focal_vox = self._get_oriented()

        t1w_vtk  = _nibabel_to_vtk(t1w_nib)
        mask_vtk = _nibabel_to_vtk(mask_nib, as_uint8=True)

        # Focal point in VTK world coordinates (origin=0, so just vox * spacing)
        sp = t1w_vtk.GetSpacing()
        focal_world = tuple(float(focal_vox[i]) * sp[i] for i in range(3))

        # Shared reslice cursor (drives crosshairs and slice planes in all panels)
        cursor = vtk.vtkResliceCursor()
        cursor.SetCenter(*focal_world)
        cursor.SetImage(t1w_vtk)
        cursor.SetThickMode(False)
        self._cursor = cursor

        # T1W grayscale LUT (alpha inverted: slider at 0 → opaque T1W)
        t1w_range = t1w_vtk.GetScalarRange()
        self._t1w_lut = _build_grayscale_lut(
            t1w_range[0], t1w_range[1],
            alpha=1.0 - self._alpha,
        )

        for i, interactor in enumerate(self._interactors):
            # Collect existing renderers first, then remove them.
            # Removing inside the GetNextItem loop modifies the collection mid-iteration.
            ren_win = interactor.GetRenderWindow()
            old_renderers = []
            coll = ren_win.GetRenderers()
            coll.InitTraversal()
            while (r := coll.GetNextItem()) is not None:
                old_renderers.append(r)
            for r in old_renderers:
                ren_win.RemoveRenderer(r)

            panel = _SlicePanel(
                interactor    = interactor,
                reslice_cursor= cursor,
                plane_normal  = self._PLANE_NORMALS[i],
                mask_vtk      = mask_vtk,
                tissue_lut    = self._tissue_lut,
            )

            # The image was already given to the cursor via cursor.SetImage().
            # In VTK 9.x the LUT is set on the representation's internal color-map
            # filter (vtkImageMapToColors), not on the widget or representation directly.
            panel.rep.GetColorMap().SetLookupTable(self._t1w_lut)
            panel.rep.GetColorMap().SetOutputFormatToRGBA()

            # Observe cursor interactions to keep mask reslice in sync
            panel.widget.AddObserver('InteractionEvent', self._on_interaction)
            panel.widget.AddObserver('WindowLevelEvent',  self._on_interaction)

            panel.reset_camera()
            if not interactor.GetInitialized():
                interactor.Initialize()

            self._panels.append(panel)

        self._render_all()

    def _on_interaction(self, caller, event):
        """Sync mask overlays across all panels when the cursor moves."""
        for panel in self._panels:
            panel.on_interaction()
        self._render_all()

    def _render_all(self):
        for interactor in self._interactors:
            interactor.GetRenderWindow().Render()

    def _on_mode_changed(self, idx: int):
        self._mode = self.RAW if idx == 0 else self.MEDICAL
        labels = self._LABELS_RAW if self._mode == self.RAW else self._LABELS_MEDICAL
        for lbl, text in zip(self._panel_labels, labels):
            lbl.setText(text)
        if self._t1w_orig is not None:
            self._rebuild()

    def closeEvent(self, event):
        for interactor in self._interactors:
            interactor.Finalize()
        super().closeEvent(event)
