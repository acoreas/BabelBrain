import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from GUIComponents.AppStyle import style_nav_toolbar


class PyVistaPlotWidget(QWidget):
    def __init__(
        self,
        tx,
        grid_info,
        acoustic_data,
        show_sub_elements=False,
        parent=None,
    ):
        super().__init__(parent)

        self.setMinimumSize(400, 350)
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            False,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.setWindowModality(Qt.WindowModality.WindowModal)

        self.plotter = QtInteractor(
            parent=self,
            off_screen=False,
            auto_update=False,
        )

        layout.addWidget(self.plotter.interactor)

        self.plotter.interactor.setEnabled(True)
        self.plotter.interactor.setFocusPolicy(
            Qt.FocusPolicy.StrongFocus
        )
        self.plotter.interactor.setMouseTracking(True)

        # --- Control row: view buttons, export image, toggle acoustic slice ---
        controls_layout = QHBoxLayout()
        controls_layout.setContentsMargins(4, 4, 4, 4)
        controls_layout.setSpacing(6)

        xz_view_button = QPushButton("XZ View")
        xz_view_button.clicked.connect(self._set_xz_view)
        controls_layout.addWidget(xz_view_button)

        iso_view_button = QPushButton("Iso View")
        iso_view_button.clicked.connect(self._set_iso_view)
        controls_layout.addWidget(iso_view_button)

        export_image_button = QPushButton("Export Image")
        export_image_button.clicked.connect(self._export_image)
        controls_layout.addWidget(export_image_button)

        self.acoustic_slice_checkbox = QCheckBox("Show Acoustic Slice")
        self.acoustic_slice_checkbox.setChecked(False)
        self.acoustic_slice_checkbox.toggled.connect(
            self._toggle_acoustic_slice
        )
        controls_layout.addWidget(self.acoustic_slice_checkbox)

        controls_layout.addStretch(1)

        layout.addLayout(controls_layout)
        # ----------------------------------------------------------------

        spatial_step = grid_info["spatial_step"]

        xfmin = grid_info["xfmin"]
        xfmax = grid_info["xfmax"]

        yfmin = grid_info["yfmin"]
        yfmax = grid_info["yfmax"]
        y_middle = (yfmin + yfmax) / 2

        zfmin = grid_info["zfmin"]
        zfmax = grid_info["zfmax"]

        self._full_grid_bounds = (
            xfmin * 1e3,
            xfmax * 1e3,
            yfmin * 1e3,
            yfmax * 1e3,
            zfmin * 1e3,
            zfmax * 1e3,
        )

        xfield = np.linspace(
            xfmin,
            xfmax,
            int(np.ceil((xfmax - xfmin) / spatial_step)) + 1,
        )

        yfield = np.array([y_middle])

        zfield = np.linspace(
            zfmin,
            zfmax,
            int(np.ceil((zfmax - zfmin) / spatial_step)) + 1,
        )

        xp, yp, zp = np.meshgrid(
            xfield * 1e3,
            yfield * 1e3,
            zfield * 1e3,
            indexing="ij",
        )

        grid = pv.StructuredGrid(xp, yp, zp)

        scalar_values = np.abs(acoustic_data.T).ravel(order="F")

        if scalar_values.size != grid.n_points:
            raise ValueError(
                "The number of pressure values does not match the "
                f"structured grid: {scalar_values.size} values for "
                f"{grid.n_points} points."
            )

        self.acoustic_actor = self.plotter.add_mesh(
            grid,
            scalars=scalar_values,
            show_scalar_bar=False,
            opacity=0.9,
            cmap=plt.cm.jet,
        )
        self.acoustic_actor.SetVisibility(False)

        # Rename variables for annular txs
        if "RingFaceDisplay" in tx:
            tx["FaceDisplay"],tx["VertDisplay"] = self._combine_ring_displays(tx)
            
        faces = np.hstack((
            np.full((tx["FaceDisplay"].shape[0], 1),4,dtype=np.int64,),
            np.asarray(tx["FaceDisplay"],dtype=np.int64,),
        ))

        mesh = pv.PolyData(np.asarray(tx["VertDisplay"] * 1e3),faces,)

        self._tx_bounds = mesh.bounds

        display_sub_elems = {}
        if show_sub_elements:
            display_sub_elems['show_edges']=True
            display_sub_elems['edge_color']="black"
            display_sub_elems['line_width']=1
        
        self.plotter.add_mesh(
            mesh,
            color="venetian_red",
            label="Transducer Elements",
            **display_sub_elems,
        )

        self.plotter.add_points(
            np.array([[0.0, 0.0, 0.0]]),
            color="k",
            point_size=15,
            render_points_as_spheres=True,
            label="Origin",
        )

        self.plotter.add_legend(
            size=(0.2, 0.08),
            face="rectangle",
            border=True,
            loc="upper left",
        )

        self._set_xz_view()

        self.plotter.render()

    @staticmethod
    def _combine_ring_displays(tx):
        combined_face_display = None
        prev_vert_index = 0
        for index in range(len(tx["RingFaceDisplay"])):
            if index == 0:
                combined_face_display = tx["RingFaceDisplay"][index]
            else:
                temp_arr = tx["RingFaceDisplay"][index] + prev_vert_index
                combined_face_display = np.vstack([combined_face_display,temp_arr])
            
            prev_vert_index = int(np.prod(combined_face_display.shape))
        
        combined_vert_display = np.vstack(tx['RingVertDisplay'])
        
        return combined_face_display, combined_vert_display
    
    # --- Nice-bounds rounding helpers ---------------------------------

    @staticmethod
    def _nice_tick_step(axis_range, target_ticks=5):
        """Pick a human-friendly tick spacing (1/2/5 x power of 10) for
        a given axis range, similar to matplotlib's default locator."""
        if axis_range <= 0:
            return 1.0

        raw_step = axis_range / target_ticks
        magnitude = 10 ** np.floor(np.log10(raw_step))
        residual = raw_step / magnitude

        if residual < 1.5:
            nice_residual = 1
        elif residual < 3:
            nice_residual = 2
        elif residual < 7:
            nice_residual = 5
        else:
            nice_residual = 10

        return nice_residual * magnitude

    def _round_bounds_to_nice(self, bounds, target_ticks=5):
        """Expand each axis of `bounds` outward (mins floor, maxs ceil)
        to the nearest nice step, so both the box edges and the tick
        values land on the same clean numbers. Never shrinks/clips."""
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

        x_step = self._nice_tick_step(xmax - xmin, target_ticks)
        y_step = self._nice_tick_step(ymax - ymin, target_ticks)
        z_step = self._nice_tick_step(zmax - zmin, target_ticks)

        rounded_bounds = (
            np.floor(xmin / x_step) * x_step,
            np.ceil(xmax / x_step) * x_step,
            np.floor(ymin / y_step) * y_step,
            np.ceil(ymax / y_step) * y_step,
            np.floor(zmin / z_step) * z_step,
            np.ceil(zmax / z_step) * z_step,
        )

        return rounded_bounds, (x_step, y_step, z_step)

    def _current_bounds_and_steps(self, target_ticks=5):
        """Raw (pre-rounding) bounds for the current display state --
        full grid if the acoustic slice is visible, otherwise the
        transducer's own bounds with z matched to max(x_range, y_range)
        -- then expanded to nice tick-aligned bounds."""
        if self.acoustic_slice_checkbox.isChecked():
            bounds = self._full_grid_bounds
        else:
            xmin, xmax, ymin, ymax, zmin, zmax = self._tx_bounds
            x_range = xmax - xmin
            y_range = ymax - ymin

            z_range_target = max(x_range, y_range)
            z_center = (zmin + zmax) / 2
            half = z_range_target / 2

            bounds = (xmin, xmax, ymin, ymax, z_center - half, z_center + half)

        return self._round_bounds_to_nice(bounds, target_ticks)

    # -------------------------------------------------------------------

    def _refresh_bounds_and_camera(self):
        bounds, (x_step, y_step, z_step) = self._current_bounds_and_steps()

        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        n_xlabels = max(2, round((xmax - xmin) / x_step) + 1)
        n_ylabels = max(2, round((ymax - ymin) / y_step) + 1)
        n_zlabels = max(2, round((zmax - zmin) / z_step) + 1)

        self.plotter.remove_bounds_axes()
        self.plotter.show_bounds(
            bounds=bounds,
            grid="back",
            all_edges=True,
            location="outer",
            xtitle=f"X (mm)",
            ytitle=f"Y (mm)",
            ztitle=f"Z (mm)",
            fmt="%.0f",
            n_xlabels=n_xlabels,
            n_ylabels=n_ylabels,
            n_zlabels=n_zlabels,
        )
        self.plotter.reset_camera(bounds=bounds)

    def _set_xz_view(self):
        """Flat xz projection, matching the original default orientation."""
        self.plotter.view_xz()
        self.plotter.camera.up = (0, 0, -1)
        self._refresh_bounds_and_camera()
        self.plotter.render()

    def _set_iso_view(self):
        """Isometric orientation, viewed from the opposite side of
        pyvista's default (rotated 180° in azimuth)."""
        self.plotter.view_isometric()
        self.plotter.camera.up = (0, 0, -1)
        self.plotter.camera.elevation += 90
        self._refresh_bounds_and_camera()
        self.plotter.render()

    def _export_image(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Transducer Geometry Image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;All Files (*)",
        )
        if not file_path:
            return
        self.plotter.screenshot(file_path)

    def _toggle_acoustic_slice(self, checked):
        self.acoustic_actor.SetVisibility(checked)
        self._refresh_bounds_and_camera()
        self.plotter.render()

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)


class PlotWidget(QWidget):
    """Matplotlib canvas with a toolbar docked inside the bottom edge."""

    def __init__(self, canvas, parent=None):
        super().__init__(parent)

        self.canvas = canvas
        self.toolbar = style_nav_toolbar(NavigationToolbar2QT(canvas, self))

        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.toolbar)

        self.setMinimumSize(400, 350)


class TransducerVerificationDialog(QDialog):
    def __init__(
        self,
        tx_data,
        acoustic_data,
        grid_info,
        parent=None,
    ):
        super().__init__(parent)

        acoustic_slice = acoustic_data[:, acoustic_data.shape[1] // 2, :]  # Get xz slice at middle y

        self.setWindowTitle(
            "Please verify transducer geometry/performance"
        )

        self.resize(1200, 700)
        self.setMinimumSize(1000, 600)

        main_layout = QVBoxLayout(self)
        plots_layout = QHBoxLayout()

        self.geometry_plot = PyVistaPlotWidget(
            tx=tx_data,
            grid_info=grid_info,
            acoustic_data=acoustic_slice,
        )
        self.image_canvas = self._create_image_plot(acoustic_slice, grid_info)

        geometry_container = self._wrap_widget(self.geometry_plot, "Transducer Geometry")

        (image_container, self.image_plot_widget) = self._wrap_plot(self.image_canvas, "Acoustics Water Sim")

        plots_layout.addWidget(geometry_container, stretch=1)
        plots_layout.addWidget(image_container, stretch=1)

        main_layout.addLayout(plots_layout, stretch=1)
        main_layout.addSpacing(10)

        continue_label = QLabel(
            "If transducer geometry and performance look accurate, "
            "press continue to complete transducer creation."
        )
        continue_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        continue_label.setWordWrap(True)

        main_layout.addWidget(continue_label)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )

        continue_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Ok
        )
        continue_button.setText("Continue")

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        main_layout.addWidget(self.button_box)

        self.ensurePolished()
        main_layout.activate()

        self.image_canvas.draw()

    def _wrap_plot(self, canvas, title):
        container = QWidget()

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        font = title_label.font()
        font.setBold(True)
        title_label.setFont(font)

        plot_widget = PlotWidget(canvas)

        layout.addWidget(title_label)
        layout.addWidget(plot_widget, stretch=1)

        return container, plot_widget

    def _wrap_widget(self, widget, title):
        container = QWidget()

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        font = title_label.font()
        font.setBold(True)
        title_label.setFont(font)

        layout.addWidget(title_label)
        layout.addWidget(widget, stretch=1)

        return container

    @staticmethod
    def _create_scatter_plot(xyz_data, scatter_step=1):
        xyz_data = np.asarray(xyz_data)

        x_data = xyz_data[::scatter_step, 0]
        y_data = xyz_data[::scatter_step, 1]
        z_data = xyz_data[::scatter_step, 2]

        figure = Figure(
            figsize=(7, 4),
            tight_layout=True,
        )
        canvas = FigureCanvasQTAgg(figure)

        axes = figure.add_subplot(
            111,
            projection="3d",
        )

        axes.scatter(
            0,
            0,
            0,
            marker="x",
            color="red",
            s=100,
            label="Origin",
        )

        axes.set_xlabel(f"X (mm)")
        axes.set_ylabel(f"Y (mm)")
        axes.set_zlabel(f"Z (mm)")
        axes.legend()

        # Keep all axes visually proportional.
        x_range = np.ptp(x_data)
        y_range = np.ptp(y_data)
        z_range = np.ptp(z_data)

        max_range = max(
            x_range,
            y_range,
            z_range,
        )

        if max_range > 0:
            x_center = (
                np.max(x_data) + np.min(x_data)
            ) / 2
            y_center = (
                np.max(y_data) + np.min(y_data)
            ) / 2
            z_center = (
                np.max(z_data) + np.min(z_data)
            ) / 2

            half_range = max_range / 2

            axes.set_xlim(
                x_center - half_range,
                x_center + half_range,
            )
            axes.set_ylim(
                y_center - half_range,
                y_center + half_range,
            )
            axes.set_zlim(
                z_center - half_range,
                z_center + half_range,
            )

        axes.invert_zaxis()
        axes.set_box_aspect((1, 1, 1))

        return canvas

    @staticmethod
    def _create_image_plot(image_data, grid_info):
        figure = Figure(
            figsize=(2, 4),
            tight_layout=True,
        )
        canvas = FigureCanvasQTAgg(figure)

        axes = figure.add_subplot(111)

        extents = (
            grid_info['xfmin'] * 1e3,
            grid_info['xfmax'] * 1e3,
            grid_info['zfmin'] * 1e3,
            grid_info['zfmax'] * 1e3,
        )
        image = axes.imshow(
            image_data,
            extent=extents,
            cmap=plt.cm.jet,
        )

        axes.set_xlabel(f"X (mm)")
        axes.set_ylabel(f"Z (mm)")

        figure.colorbar(
            image,
            ax=axes,
            label="Amplitude",
        )

        return canvas