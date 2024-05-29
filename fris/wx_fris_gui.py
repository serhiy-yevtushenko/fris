"""Fris GUI.

A simple graphical frontend for FRiS Algorithms mainly intended for didactic purposes. You can create data points by
point and click and visualize the decision region induced by different kernels and parameter settings.

To create positive examples click the left mouse button; to create negative examples click the right button.
"""

import argparse
import logging
import string
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence
from functools import partial
from typing import Any
from typing import cast
from typing import Final
from typing import Literal
from typing import Optional
from typing import Union

import matplotlib
import numpy as np
import wx.grid
from matplotlib import cm
from matplotlib.backend_bases import Event
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import Colormap
from matplotlib.contour import ContourSet
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from wx.grid import GridCellAttr
from wx.html2 import WebView

from fris.fris_datastructures import FrisStolpWithDistanceCaching
from fris.fris_datastructures import FrisStolpWithDistanceCachingCorrected
from fris.fris_functions import average
from fris.fris_functions import fris_function
from fris.fris_functions import geometric_average
from fris.fris_types import ClassType
from fris.fris_types import DataPointArray
from fris.fris_types import LabelsArray
from fris.nn_data_generator import checker_predictor
from fris.nn_data_generator import generate_circles_points
from fris.nn_data_generator import generate_classifier_data
from fris.nn_data_generator import generate_moon_points
from fris.nn_data_generator import generate_separable_data
from fris.nn_data_generator import generate_separable_data_with_boundary
from fris.nn_data_generator import narrow_saw_predictor
from fris.nn_data_generator import stripes_predictor
from fris.nn_data_generator import wide_saw_predictor
from fris.nn_utils import compactness_profile
from fris.nn_utils import compute_ccvs

REPRESENTATIVE_FRIS_CLASS: Final[str] = "Representative"

TYPICAL_FRIS_CLASS: Final[str] = "Reliably classified"

BOUNDARY_FRIS_CLASS: Final[str] = "Boundary"

ERROR_PRONE_FRIS_CLASS: Final[str] = "Error(s)"

OUTLIER_FRIS_CLASS: Final[str] = "Outlier(s)"

X_COORD_INDEX: Final[int] = 0
Y_COORD_INDEX: Final[int] = 1
LABEL_INDEX: Final[int] = 2

EXAMPLES_LOADED_EVENT: Final[str] = "examples_loaded"
EXAMPLE_ADDED_EVENT: Final[str] = "example_added"
CLEAR_DATA_EVENT: Final[str] = "clear"
SURFACE_EVENT: Final[str] = "surface"

print(__doc__)
print("Matplotlib version", matplotlib.__version__)

y_min, y_max = -50, 50
x_min, x_max = -50, 50


class Knob:
    """
    Knob - simple class with a "setKnob" method.

    A Knob instance is attached to a Param instance, e.g., param.attach(knob)
    A base class is for documentation purposes.
    """

    @abstractmethod
    def setKnob(self, value: Optional[float]) -> None: ...


class Param:
    """A parameter (value) that is controllable via attached UI controls.

    The idea of the "Param" class is that some parameter in the GUI may have several knobs that both control it and
    reflect the parameter's state, e.g. a slider, text, and dragging can all change the value of the frequency in the
    waveform of this example. The class allows a cleaner way to update/"feedback" to the other knobs when one is being
    changed.  Also, this class handles min/max constraints for all the knobs.

    Idea - knob list - in "set" method, knob object is passed as well
      - the other knobs in the knob list have a "set" method which gets
        called for the others.
    """

    def __init__(self, initial_value: float, minimum: float = 0.0, maximum: float = 1.0) -> None:
        self.minimum = minimum
        self.maximum = maximum
        if self.minimum > self.maximum:
            raise ValueError(f"Minimum {self.minimum} should be not greater then maximum {self.maximum}")
        if initial_value != self.constrain(initial_value):
            raise ValueError("illegal initial value")
        self.value = initial_value
        self.knobs: list[Knob] = []

    def attach(self, knob: Knob) -> None:
        self.knobs += [knob]

    def set(self, value: float, knob: Optional[Knob] = None) -> float:
        self.value = self.constrain(value)
        for feedbackKnob in self.knobs:
            if feedbackKnob != knob:
                feedbackKnob.setKnob(self.value)
        return self.value

    def constrain(self, value: float) -> float:
        if value <= self.minimum:
            value = self.minimum
        if value >= self.maximum:
            value = self.maximum
        return value


class SliderGroup(Knob):
    """A knob implementation that represents slider together with attached text label."""

    def __init__(self, parent: wx.Window, label: str, param: Param) -> None:
        self.sliderLabel = wx.StaticText(parent, label=label)
        self.sliderText = wx.TextCtrl(parent, -1, style=wx.TE_PROCESS_ENTER)
        self.slider = wx.Slider(parent, -1)
        self.slider.SetRange(int(param.minimum * 1000), int(param.maximum * 1000))
        self.setKnob(param.value)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.sliderLabel, 0, wx.EXPAND | wx.ALL, border=2)
        sizer.Add(self.sliderText, 0, wx.EXPAND | wx.ALL, border=2)
        sizer.Add(self.slider, 1, wx.EXPAND)
        self.sizer = sizer

        self.slider.Bind(wx.EVT_SLIDER, self.sliderHandler)
        self.sliderText.Bind(wx.EVT_TEXT_ENTER, self.sliderTextHandler)

        self.param = param
        self.param.attach(self)

    def sliderHandler(self, evt: wx.CommandEvent) -> None:
        value = evt.GetInt() / 1000.0
        self.param.set(value)

    def sliderTextHandler(self, _evt: wx.Event) -> None:
        value = float(self.sliderText.GetValue())
        self.param.set(value)

    def setKnob(self, value: Optional[float]) -> None:
        if value is not None:
            self.sliderText.SetValue("%g" % value)
            self.slider.SetValue(int(value * 1000))


class CharValidator(wx.Validator):
    """Validates data as it is entered into the text controls."""

    def __init__(self, mode: Literal["no-alpha", "no-digit"]) -> None:
        wx.Validator.__init__(self)
        self.mode = mode
        self.Bind(wx.EVT_CHAR, self.OnChar)

    def Clone(self) -> "CharValidator":
        """Clone object.

        Required Validator method.
        """
        return CharValidator(self.mode)

    def Validate(self, win: wx.Window) -> bool:
        return True

    def TransferToWindow(self) -> bool:
        return True

    def TransferFromWindow(self) -> bool:
        return True

    def OnChar(self, event: wx.KeyEvent) -> None:
        keycode = int(event.GetKeyCode())
        if keycode < 256:
            key = chr(keycode)
            if self.mode == "no-alpha" and key in string.ascii_letters:
                return
            if self.mode == "no-digit" and key in string.digits:
                return
        event.Skip()


class ModelObserver:
    """Base class for model observers."""

    def update(self, event: str, model: "Model") -> None:
        pass


SurfaceType = tuple[np.ndarray, np.ndarray, np.ndarray]


class Model:
    """The Model which holds the data.

    It implements the observable in the observer pattern and notifies the registered observers on change event.
    """

    def __init__(self) -> None:
        self.observers: list[ModelObserver] = []
        self.surface: Optional[SurfaceType] = None
        self.data: list[tuple[float, float, int]] = []
        self.clf: Optional[FrisStolpWithDistanceCaching] = None
        self.f_0 = 0.0
        self._lambda = 0.5
        self.do_second_round = False
        self.allocate_to_covering_cluster = False

    def is_fitted(self) -> bool:
        return self.clf is not None

    def is_stolp(self, index: int) -> bool:
        if self.clf is not None:
            return index in self.clf.support_x_indices_
        return False

    def changed(self, event: str) -> None:
        """Notify the observers."""
        for observer in self.observers:
            observer.update(event, self)

    def add_observer(self, observer: ModelObserver) -> None:
        """Register an observer."""
        self.observers.append(observer)

    def set_surface(self, surface: Optional[SurfaceType]) -> None:
        self.surface = surface

    def get_point_defensibility(self, row: int) -> Optional[float]:
        if self.clf is not None and row < len(self.clf.element_defensibility):
            return cast(float, self.clf.element_defensibility[row])
        return None

    def get_point_tolerance(self, row: int) -> Optional[float]:
        if self.clf is not None and row < len(self.clf.element_tolerances):
            return cast(float, self.clf.element_tolerances[row])
        return None

    def get_point_efficiency(self, row: int) -> Optional[float]:
        if self.clf is not None and row < len(self.clf.element_efficiences):
            return cast(float, self.clf.element_efficiences[row])
        return None

    def get_point_stolp_index(self, row: int) -> int:
        if self.clf is not None:
            return self.clf.get_stolp_index_for_point_index(row)
        return row

    def get_nearest_inclass_neighbour_index(self, row: int) -> int:
        if self.clf is not None:
            if row in self.clf.all_points_neighbours:
                index = self.clf.all_points_neighbours[row].nearest_inclass_index
                if index is not None:
                    return index
        return row

    def get_nearest_inclass_neighbour_distance(self, row: int) -> float:
        nearest_inclass_neighbour_index = self.get_nearest_inclass_neighbour_index(row)
        if nearest_inclass_neighbour_index >= 0:
            if self.clf is not None:
                return self.clf.distance(self.get_data_point(row), self.get_data_point(nearest_inclass_neighbour_index))
        return 0.0

    def get_data_point(self, row: int) -> np.ndarray:
        return np.array(self.data[row])[0:2]

    def get_X(self) -> np.ndarray:
        return np.array(self.data)[:, 0:2]

    def get_y(self) -> list[ClassType]:
        return cast(list[ClassType], self.get_labels().tolist())

    def get_labels(self) -> np.ndarray:
        return np.array(self.data)[:, 2]

    def get_classes(self) -> list[ClassType]:
        if len(self.data) > 0:
            return cast(list[ClassType], np.unique(self.get_labels()).tolist())
        return []

    def get_class_points_count(self, a_class: ClassType) -> int:
        return cast(int, (self.get_labels() == a_class).sum())

    def get_class_stolp_count(self, a_class: ClassType) -> int:
        if self.clf is None:
            return 0
        return cast(int, (np.array(self.clf.support_y_) == a_class).sum())

    def get_nearest_competitor_neighbour_index(self, row: int) -> int:
        if self.clf is not None:
            if row in self.clf.all_points_neighbours:
                index = self.clf.all_points_neighbours[row].nearest_competitor_index
                if index is not None:
                    return index
        return -1

    def get_nearest_competitor_neighbour_distance(self, row: int) -> float:
        nearest_competitor_neighbour_index = self.get_nearest_competitor_neighbour_index(row)
        if nearest_competitor_neighbour_index >= 0:
            if self.clf is not None:
                return self.clf.distance(
                    self.get_data_point(row), self.get_data_point(nearest_competitor_neighbour_index)
                )
        return 0.0

    def get_fris_value(self, row: int) -> Optional[float]:
        if self.clf is not None:
            neighbour_distance = self.get_nearest_inclass_neighbour_distance(row)
            competitor_distance = self.get_nearest_competitor_neighbour_distance(row)
            return fris_function(neighbour_distance, competitor_distance)
        return None

    def has_data(self) -> bool:
        return len(self.data) > 0

    def get_data_value(self, row: int, col: int) -> Union[int, float]:
        return self.data[row][col]

    def get_cluster_size(self, row: int) -> int:
        if self.clf is not None:
            return self.clf.get_point_cluster_size(row)
        return 1

    def dump_svmlight_file(self, file: str) -> None:
        dump_svmlight_file(self.get_X(), self.get_y(), file)

    def set_data(self, X: np.ndarray, y: np.ndarray) -> None:
        self.clear()
        merged_data = np.concatenate((X, y[:, None]), axis=1)
        self.data = []
        for i in merged_data:
            self.data.append((i[0], i[1], i[2]))
        self.changed(EXAMPLES_LOADED_EVENT)

    def clear(self) -> None:
        self.data = []
        self.clf = None
        self.surface = None
        self.changed(CLEAR_DATA_EVENT)

    def load_svmlight_file(self, file: str) -> None:
        data = load_svmlight_file(file)
        X = data[0].toarray()
        y = data[1]
        self.set_data(X, y)

    def compute_class_compactness_map(self) -> dict[ClassType, float]:
        if self.clf is None:
            return {}
        return self.clf.get_class_compactness_map()

    def form_point_description(self, row: int) -> str:
        """Form description of point, located at `row`."""
        is_representative = self.is_stolp(row)
        fris_value = self.get_fris_value(row)

        tip = f"Representative: {'yes' if is_representative else 'no'}\n"
        if fris_value is not None:
            tip += f"Outlier {'yes' if fris_value < 0 else 'no'}\n"
        if is_representative:
            tip += f"Cluster Size: {self.get_cluster_size(row)}\n"
        tip += f"Defensibility {safe_format_float(self.get_point_defensibility(row))}\n"
        tip += f"Tolerance {safe_format_float(self.get_point_tolerance(row))}\n"
        tip += f"Efficiency {safe_format_float(self.get_point_efficiency(row))}\n"
        tip += f"Inclass NN Index {to_table_index(self.get_nearest_inclass_neighbour_index(row))}\n"
        tip += f"Competitor NN Index {to_table_index(self.get_nearest_competitor_neighbour_index(row))}\n"
        tip += f"Inclass NN Distance {self.get_nearest_inclass_neighbour_distance(row):0.3f}\n"
        tip += f"Competitor NN Distance {self.get_nearest_competitor_neighbour_distance(row):0.3f}\n"
        tip += f"FRiS {fris_value :0.3f}\n"
        return tip

    def form_model_description(self) -> str:
        classes = self.get_classes()
        description = """
         <html>
         <head>
         <style>
         html *
         {
            font: 12px Verdana, Arial, sans-serif !important;
            color: #000 !important;
            font-family: Arial !important;
            font-weight: bold !important;
         }
         tbody td {
           text-align: right;
         }
         </style>
         </head>
         """
        description += "<table><th><td>Model</td>"
        for i in range(len(classes)):
            description += f"<td>{i + 1} class </td>"
        description += "</th>"
        description += f"<tr><td>Data Points</td><td>{len(self.data)}</td>"
        for i, a_class in enumerate(classes):
            description += f"<td>{self.get_class_points_count(a_class)}</td>"
        description += "</tr>"
        if self.clf is None:
            description += "</table><br>"
            description += "Model is not fitted."
        else:
            description += f"<tr><td>Representatives</td><td>{len(self.clf.support_y_)}</td>"
            for i, a_class in enumerate(classes):
                description += f"<td>{self.get_class_stolp_count(a_class)}</td>"
            description += "</tr>"
            description += f"<tr><td>Reduction</td><td>{1 - (len(self.clf.support_y_) / len(self.data)):0.2f}</td>"
            for i, a_class in enumerate(classes):
                description += (
                    f"<td>{1 - (self.get_class_stolp_count(a_class) / self.get_class_points_count(a_class)):0.2f}</td>"
                )
            description += "</tr>"

            compactness_map = self.compute_class_compactness_map()
            description += "<tr><td>Compactness:</td><td></td>"
            for i, a_class in enumerate(classes):
                description += f"<td>{compactness_map[a_class]:0.3f}</td>"
            description += "</tr>"
            description += "</table><br>"
            compactness_for_classes = compactness_map.values()
            description += f"Model <b>Average compactness</b> {average(compactness_for_classes): 0.3f}"
            description += (
                f"&nbsp;<b>Geometric compactness</b> {geometric_average(compactness_for_classes): 0.3f}&nbsp;"
            )
            description += f"<b>Accuracy</b> {self.clf.score(self.get_X(), self.get_y()) * 100: 0.2f}%<br>"
        return description


def show_error(message: str) -> int:
    return cast(int, wx.MessageBox(message, "ERROR", wx.ICON_ERROR))


class Controller:
    """An UI Controller."""

    def __init__(self, model: Model) -> None:
        self.model = model
        # Whether a model has been fitted
        self.fitted = False

    def fit(self) -> None:
        if not self.model.has_data():
            show_error("Please add some points or generate sample points from model datasets.")
            return
        X = self.model.get_X().copy()
        y = self.model.get_y().copy()

        if len(np.unique(y)) == 1:
            show_error("Not able to build classification on the base of one class")
            self.model.clf = None
            return
        try:
            clf = FrisStolpWithDistanceCachingCorrected(
                f_0=self.model.f_0,
                _lambda=self.model._lambda,
                do_second_round=self.model.do_second_round,
                allocate_points_to_nearest_stolp=not self.model.allocate_to_covering_cluster,
            )
            clf.fit(X, y)
            X1, X2, Z = self.decision_surface(clf)
            self.model.clf = clf
            self.model.set_surface((X1, X2, Z))
            self.fitted = True
            self.model.changed(SURFACE_EVENT)
        except ValueError as e:
            show_error(f"Not able to build classifier: {e}")
            self.model.clf = None
            self.fitted = False
            return

    def decision_surface(self, clf: FrisStolpWithDistanceCaching) -> SurfaceType:
        delta = 1
        x = np.arange(x_min, x_max + delta, delta)
        y = np.arange(y_min, y_max + delta, delta)
        X1, X2 = np.meshgrid(x, y)
        points = np.c_[X1.ravel(), X2.ravel()]
        Z = clf.decision_function(points)
        Z = Z[:, 0]
        Z = Z.reshape(X1.shape)
        return X1, X2, Z

    def clear_data(self) -> None:
        self.model.clear()
        self.fitted = False

    def add_example(self, x: float, y: float, label: int) -> None:
        self.model.data.append((x, y, label))
        # update decision surface if already fitted.
        self.refit()
        self.model.changed(EXAMPLE_ADDED_EVENT)

    def refit(self) -> None:
        """Refit the model if already fitted."""
        if self.fitted:
            self.fit()


INTERBOX_BORDER: Final[int] = 10


def to_table_index(index: int) -> int:
    """Convert python 0-based index to human-understandable 1-based index.

    Used for producing human-readable output.
    """
    return index + 1


def ask(parent: wx.Window = None, message: str = "", default_value: str = "") -> str:
    """Get input from user using message box.

    Return:
         - Value entered by user - if he presses OK
         - Empty String - if user presses Cancel.
    """
    with wx.TextEntryDialog(parent, message, value=default_value) as dlg:
        if wx.ID_OK == dlg.ShowModal():
            return cast(str, dlg.GetValue())
    return ""


def safe_format_float(value: Optional[float], default_value: str = "N/A") -> str:
    if value is not None:
        return f"{value: 0.3f}"
    return default_value


COLORMAP: Final[Colormap] = cm.coolwarm


class ModelPanel(wx.Panel, ModelObserver, Knob):
    """Panel which displays data points and decision surfaces, and contains controls for changing model parameters."""

    def __init__(self, model: Model, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.model = model
        model.add_observer(self)

        self.create_canvas(model, self)
        self.create_buttons(self)
        self.create_sliders(self)
        self.create_static_text(self)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, proportion=1, flag=wx.EXPAND | wx.RIGHT | wx.LEFT)
        self.sizer.Add(self.data_buttons_sizer, 0, flag=wx.EXPAND)
        self.sizer.Add(self.buttons_sizer, 0, flag=wx.EXPAND)
        self.sizer.Add(self.f_0_slider_group.sizer, 0, wx.EXPAND | wx.ALL, border=5)
        self.sizer.Add(self.lambda_slider_group.sizer, 0, wx.EXPAND | wx.ALL, border=5)
        self.sizer.Add(self.status_text, 0, wx.EXPAND | wx.ALL, border=5)
        self.SetSizer(self.sizer)

    def create_static_text(self, parent: wx.Window) -> None:
        self.status_text = WebView.New(parent, size=(-1, 150))

    def create_canvas(self, model: Model, parent: wx.Window) -> None:
        self.controller = Controller(model)
        self.contours: list[Union[ContourSet, list]] = []

        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        divider = make_axes_locatable(self.ax)
        self.colorbar_ax = divider.append_axes("right", size="5%", pad=0.05)
        self.figure.colorbar(
            ScalarMappable(norm=BoundaryNorm(list(-1 + 0.1 * i for i in range(21)), ncolors=COLORMAP.N), cmap=COLORMAP),
            self.colorbar_ax,
        )

        self.tooltip = wx.ToolTip(tip="tip with a long %s line and a newline\n" % (" " * 100))

        self.canvas = FigureCanvas(parent, -1, self.figure)
        self.canvas.SetToolTip(self.tooltip)
        self.tooltip.Enable(False)
        self.tooltip.SetDelay(0)

        self.canvas.callbacks.connect("button_press_event", self.on_mouse_click)
        self.canvas.mpl_connect("motion_notify_event", self.handle_tooltips)
        self.f_0_param = Param(0, minimum=-1.0, maximum=1.0)
        self._lambda_param = Param(0.5, minimum=0.0, maximum=1.0)

        self.f_0_param.attach(self)
        self._lambda_param.attach(self)

        self.do_second_round = False
        self.allocate_to_covering_cluster = model.allocate_to_covering_cluster

    def on_mouse_click(self, event: MouseEvent) -> None:
        if event.xdata and event.ydata:
            if event.button == 1:
                self.controller.add_example(event.xdata, event.ydata, 1)
            elif event.button == 3:
                self.controller.add_example(event.xdata, event.ydata, -1)

    def handle_tooltips(self, event: Event) -> None:
        mouse_event = cast(MouseEvent, event)
        collisionFound = False
        if mouse_event.inaxes:
            if mouse_event.xdata is not None and mouse_event.ydata is not None:  # mouse is inside the axes
                x, y = mouse_event.xdata, mouse_event.ydata
                for i in range(len(self.model.data)):
                    radius = 1
                    data_point_x = self.model.get_data_value(i, X_COORD_INDEX)
                    data_point_y = self.model.get_data_value(i, Y_COORD_INDEX)
                    if abs(x - data_point_x) < radius and abs(y - data_point_y) < radius:
                        label = cast(ClassType, self.model.get_data_value(i, LABEL_INDEX))
                        tip = self.form_tooltip_text(data_point_x, data_point_y, i, label)
                        self.tooltip.SetTip(tip)
                        self.tooltip.Enable(True)
                        collisionFound = True
                        break
        if not collisionFound:
            self.tooltip.Enable(False)

    def form_tooltip_text(self, data_point_x: float, data_point_y: float, row: int, label: ClassType) -> str:
        tip = f"id={to_table_index(row)}\nx={data_point_x:0.3f}\ny={data_point_y:0.3f}\nlabel={label}\n"
        if self.model.is_fitted():
            tip += self.model.form_point_description(row)
        return tip

    def create_sliders(self, panel: wx.Panel) -> None:
        self.f_0_slider_group = SliderGroup(panel, label="f_0:", param=self.f_0_param)
        self.lambda_slider_group = SliderGroup(panel, label="Lambda", param=self._lambda_param)

    def on_fit(self, _event: wx.Event) -> None:
        self.update_parameters()
        self.controller.fit()

    def on_clear(self, _event: wx.Event) -> None:
        self.controller.clear_data()

    def on_reset_parameters(self, _event: wx.Event) -> None:
        self.f_0_param.set(0.0)
        self._lambda_param.set(0.5)

    def on_save_dataset(self, _event: wx.Event) -> None:
        file_name = ask(self, "Please give the file name to save", "dataset.ds")
        if len(file_name.strip()) != "":
            self.model.dump_svmlight_file(file_name.strip())

    def on_check_covering_clusters(self, event: wx.Event) -> None:
        cb = event.GetEventObject()
        self.allocate_to_covering_cluster = cb.GetValue()
        # Force update
        self.setKnob(None)

    def on_check_second_round(self, event: wx.Event) -> None:
        cb = event.GetEventObject()
        self.do_second_round = cb.GetValue()
        # Force update
        self.setKnob(None)

    def update_example(self, model: Model, index: int) -> None:
        x, y, label = model.data[index]
        color = self.label_to_color(label)
        self.ax.plot([x], [y], "%so" % color, scalex=False, scaley=False)
        self.canvas.Refresh()

    def update_model_description(self, model: Model) -> None:
        self.status_text.SetPage(html=model.form_model_description(), baseUrl="")

    def update(self, event: str, model: Model) -> None:
        if event == EXAMPLES_LOADED_EVENT:
            self.clear()
            for i in range(len(model.data)):
                self.update_example(model, i)
            self.update_fit_button()
        elif event == EXAMPLE_ADDED_EVENT:
            self.update_example(model, -1)
            self.update_fit_button()
        elif event == CLEAR_DATA_EVENT:
            self.clear()
            self.update_fit_button()
        elif event == SURFACE_EVENT:
            self.redraw_surface(self.model)

        self.update_model_description(model)
        self.canvas.draw()

    def update_fit_button(self) -> None:
        self.fit_button.Enable(self.model.has_data())

    def plot_class_references(
        self,
        support_x: np.ndarray,
        support_x_points_: list[np.ndarray],
        support_y_: list[ClassType],
        data: list[tuple[float, float, int]],
        model: Model,
    ) -> None:
        """Plot the class representative points (stolps) by placing circles over the corresponding data points.

        The added circles are appended to the internal contours list.
        """

        for cluster_center, class_members, label in zip(support_x, support_x_points_, support_y_):
            col = self.label_to_color(label)

            for x in class_members:
                self.contours.append(self.ax.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col))

        for index, (x, y, label) in enumerate(data):
            color = self.label_to_color(label)
            fris_value = model.get_fris_value(index)
            if fris_value is not None:
                edge_color = get_color_for_fris_value(fris_value)
            else:
                edge_color = color
            if model.is_stolp(index):
                self.contours.append(
                    self.ax.plot(
                        [x],
                        [y],
                        "o",
                        markerfacecolor=color,
                        markeredgecolor=edge_color,
                        markersize=10,
                        markeredgewidth=2,
                    )
                )
            else:
                self.contours.append(
                    self.ax.plot([x], [y], "o", markerfacecolor=color, markeredgecolor=edge_color, markeredgewidth=2)
                )

    def label_to_color(self, label_class: Union[int, str]) -> str:
        if label_class == 1:
            color = "r"
        elif label_class == -1:
            color = "b"
        return color

    def plot_decision_surface(self, surface: SurfaceType) -> None:
        X1, X2, Z = surface
        contour_1 = self.ax.contourf(
            X1,
            X2,
            Z,
            20,
            cmap=COLORMAP,
            origin="lower",
            alpha=0.85,
        )
        self.contours.append(contour_1)
        self.contours.append(self.ax.contour(X1, X2, Z, [0.0], colors="k", linestyles=["solid"]))

    def get_count_of_sampled_points(self) -> int:
        return int(self.points_count.GetValue())

    def generate_model_dataset(self, generator_function: Callable[[int], tuple[DataPointArray, LabelsArray]]) -> None:
        points = self.get_count_of_sampled_points()
        if points < 2:
            show_error("There should be at least two points in the generated dataset")
            return
        X, y = generator_function(points)
        self.model.set_data(X, y)

    def on_compact_dataset(self, _event: wx.Event) -> None:
        self.generate_model_dataset(generate_separable_data)

    def on_separable_with_boundary_dataset(self, _event: wx.Event) -> None:
        self.generate_model_dataset(generate_separable_data_with_boundary)

    def on_stripes_dataset(self, _event: wx.Event) -> None:
        self.generate_model_dataset(partial(generate_classifier_data, predictor=stripes_predictor))

    def on_saw_dataset(self, _event: wx.Event) -> None:
        self.generate_model_dataset(partial(generate_classifier_data, predictor=wide_saw_predictor))

    def on_narrow_saw_dataset(self, _event: wx.Event) -> None:
        self.generate_model_dataset(partial(generate_classifier_data, predictor=narrow_saw_predictor))

    def on_checkers_dataset(self, _event: wx.Event) -> None:
        self.generate_model_dataset(partial(generate_classifier_data, predictor=checker_predictor))

    def on_circles_dataset(self, _event: wx.Event) -> None:
        self.generate_model_dataset(generate_circles_points)

    def on_moons_dataset(self, _event: wx.Event) -> None:
        self.generate_model_dataset(generate_moon_points)

    def create_buttons(self, panel: wx.Panel) -> None:
        self.fit_button = wx.Button(panel, wx.ID_ANY, "Fit")
        panel.Bind(wx.EVT_BUTTON, self.on_fit, self.fit_button)
        self.fit_button.Enable(False)

        clear_button = wx.Button(panel, wx.ID_ANY, "Clear")
        panel.Bind(wx.EVT_BUTTON, self.on_clear, clear_button)
        reset_button = wx.Button(panel, wx.ID_ANY, "Reset Parameters")
        panel.Bind(wx.EVT_BUTTON, self.on_reset_parameters, reset_button)

        second_round_checkbox = wx.CheckBox(panel, wx.ID_ANY, "Do second round")
        second_round_checkbox.SetValue(self.do_second_round)
        panel.Bind(wx.EVT_CHECKBOX, self.on_check_second_round, second_round_checkbox)

        show_covering_cluster_checkbox = wx.CheckBox(panel, wx.ID_ANY, "Allocate points to covering stolps")
        show_covering_cluster_checkbox.SetValue(self.allocate_to_covering_cluster)
        panel.Bind(wx.EVT_CHECKBOX, self.on_check_covering_clusters, show_covering_cluster_checkbox)

        save_button = wx.Button(panel, wx.ID_ANY, "Save Dataset")
        panel.Bind(wx.EVT_BUTTON, self.on_save_dataset, save_button)

        buttons = [
            self.fit_button,
            clear_button,
            reset_button,
            second_round_checkbox,
            show_covering_cluster_checkbox,
            save_button,
        ]

        self.points_count = wx.TextCtrl(panel, validator=CharValidator("no-alpha"))
        self.points_count.SetValue("200")

        data_points_label = wx.StaticText(panel, label="Point Count:")

        compact_data_button = wx.Button(panel, wx.ID_ANY, "Compact")
        panel.Bind(wx.EVT_BUTTON, self.on_compact_dataset, compact_data_button)

        separable_data_button = wx.Button(panel, wx.ID_ANY, "Separable")
        panel.Bind(wx.EVT_BUTTON, self.on_separable_with_boundary_dataset, separable_data_button)
        saw_data_button = wx.Button(panel, wx.ID_ANY, "Wide saw")
        panel.Bind(wx.EVT_BUTTON, self.on_saw_dataset, saw_data_button)

        narrow_saw_data_button = wx.Button(panel, wx.ID_ANY, "Narrow saw")
        panel.Bind(wx.EVT_BUTTON, self.on_narrow_saw_dataset, narrow_saw_data_button)

        checkers_data_button = wx.Button(panel, wx.ID_ANY, "Checkers")
        panel.Bind(wx.EVT_BUTTON, self.on_checkers_dataset, checkers_data_button)

        stripes_data_button = wx.Button(panel, wx.ID_ANY, "Stripes")
        panel.Bind(wx.EVT_BUTTON, self.on_stripes_dataset, stripes_data_button)

        circles_data_button = wx.Button(panel, wx.ID_ANY, "Circles")
        panel.Bind(wx.EVT_BUTTON, self.on_circles_dataset, circles_data_button)

        moons_data_button = wx.Button(panel, wx.ID_ANY, "Moons")
        panel.Bind(wx.EVT_BUTTON, self.on_moons_dataset, moons_data_button)

        data_buttons = [
            compact_data_button,
            separable_data_button,
            saw_data_button,
            narrow_saw_data_button,
            checkers_data_button,
            stripes_data_button,
            circles_data_button,
            moons_data_button,
        ]
        data_horizontal_box_sizer = wx.StaticBoxSizer(orient=wx.HORIZONTAL, parent=panel, label="Generate Data Set")
        # data_horizontal_box_sizer.Add((0, 0), proportion=1)

        horizontal_box_sizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        for button in buttons:
            horizontal_box_sizer.Add(button, border=INTERBOX_BORDER // 2, proportion=1, flag=wx.ALL | wx.EXPAND)

        self.data_buttons_sizer = data_horizontal_box_sizer
        inputBoxSizer = wx.BoxSizer(wx.HORIZONTAL)
        inputBoxSizer.Add(data_points_label, flag=wx.ALL)
        inputBoxSizer.AddSpacer(10)
        inputBoxSizer.Add(self.points_count, flag=wx.ALL)
        data_horizontal_box_sizer.Add(
            inputBoxSizer, proportion=1, border=INTERBOX_BORDER // 2, flag=wx.LEFT | wx.TOP | wx.BOTTOM | wx.EXPAND
        )
        for button in data_buttons:
            data_horizontal_box_sizer.Add(
                button, border=INTERBOX_BORDER // 2, proportion=1, flag=wx.LEFT | wx.TOP | wx.BOTTOM | wx.EXPAND
            )

        self.buttons_sizer = horizontal_box_sizer

    def remove_surface(self) -> None:
        """Remove old decision surface."""
        if len(self.contours) > 0:
            for contour in reversed(self.contours):
                if isinstance(contour, list):
                    for line in contour:
                        line.remove()
                else:
                    contour.remove()

            self.contours = []
            # self.colorbar_ax = None

    def clear(self) -> None:
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))
        self.remove_surface()

        self.Refresh()

    def redraw_surface(self, model: Model) -> None:
        self.remove_surface()
        if model.clf is not None:
            self.plot_class_references(
                model.clf.support_x_, model.clf.support_x_points_, model.clf.support_y_, model.data, model
            )
            if model.surface is not None:
                self.plot_decision_surface(model.surface)

    def setKnob(self, _value: Optional[float]) -> None:
        # Note, we ignore value arg here and just go by state of the params
        self.update_parameters()
        self.controller.refit()

    def update_parameters(self) -> None:
        self.controller.model.f_0 = float(self.f_0_param.value)
        self.controller.model._lambda = float(self._lambda_param.value)
        self.controller.model.do_second_round = self.do_second_round
        self.controller.model.allocate_to_covering_cluster = self.allocate_to_covering_cluster


def make_grid_cell_attribute(color: Union[str, wx.Colour], classified: bool = False) -> wx.grid.GridCellAttr:
    result = wx.grid.GridCellAttr()
    result.SetBackgroundColour(color)
    fontweight = wx.FONTWEIGHT_NORMAL if classified else wx.FONTWEIGHT_BOLD
    result.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, fontweight))
    return result


class FrisStolpDataTable(wx.grid.GridTableBase):
    """Table implementation for displaying data points, on which model is build."""

    def __init__(self, model: Model) -> None:
        wx.grid.GridTableBase.__init__(self)
        self.model = model

        self.even_row_style = make_grid_cell_attribute("white")
        self.even_row_odd_col_style = make_grid_cell_attribute(wx.Colour(192, 200, 216, 255))

        self.odd_row_style = make_grid_cell_attribute("Light Blue")

        self.odd_row_odd_col_style = make_grid_cell_attribute(wx.Colour(216, 216, 216, 255))
        self.not_classified_attributes = (
            (self.even_row_style, self.even_row_odd_col_style),
            (self.odd_row_style, self.odd_row_odd_col_style),
        )
        self.classified_attributes = (
            (make_grid_cell_attribute("white", True), make_grid_cell_attribute(wx.Colour(192, 200, 216, 255), True)),
            (
                make_grid_cell_attribute("Light Blue", True),
                make_grid_cell_attribute(wx.Colour(216, 216, 216, 255), True),
            ),
        )

        self.COLUMNS_VALUES: dict[str, Callable[[int], int | Optional[float] | bool]] = {
            "X": lambda row: self.model.get_data_value(row, X_COORD_INDEX),
            "Y": lambda row: self.model.get_data_value(row, Y_COORD_INDEX),
            "Label": lambda row: self.model.get_data_value(row, LABEL_INDEX),
            "Is Stolp": self.model.is_stolp,
            "Cluster Size": self.model.get_cluster_size,
            "Stolp Index": lambda row: to_table_index(self.model.get_point_stolp_index(row)),
            "Defensibility": self.model.get_point_defensibility,
            "Tolerance": self.model.get_point_tolerance,
            "Efficiency": self.model.get_point_efficiency,
            "NN index from class": lambda row: to_table_index(self.model.get_nearest_inclass_neighbour_index(row)),
            "Distance to inclass NN": self.model.get_nearest_inclass_neighbour_distance,
            "NN index from competitor": lambda row: to_table_index(
                self.model.get_nearest_competitor_neighbour_index(row)
            ),
            "Distance to competitor NN": self.model.get_nearest_competitor_neighbour_distance,
            "FRiS": self.model.get_fris_value,
        }

        self.COLUMN_NAMES = list(self.COLUMNS_VALUES.keys())

    def GetNumberRows(self) -> int:
        return len(self.model.data)

    def GetNumberCols(self) -> int:
        if self.model.clf is None:
            return 3  # self.dataset.column_count()
        return len(self.COLUMN_NAMES)  # (data, is stolp, stolp index, defensibility, tolerance, efficiency)

    def IsEmptyCell(self, row: int, col: int) -> bool:
        return self.GetValue(row, col) == ""

    def GetValue(self, row: int, col: int) -> Union[str, float, int]:
        value = self.COLUMNS_VALUES[self.COLUMN_NAMES[col]](row)

        if value is not None:
            if isinstance(value, (float, np.float64)):
                return f"{value:0.3f}"
            else:
                return value
        else:
            return ""

    def SetValue(self, row: int, col: int, value: Any) -> None:
        # write is not supported
        pass

    def GetAttr(self, row: int, col: int, kind: GridCellAttr.AttrKind) -> GridCellAttr:
        attr_selector = self.classified_attributes if self.model.is_stolp(row) else self.not_classified_attributes

        attr = attr_selector[row % 2][col % 2]
        attr.IncRef()
        return attr

    def GetColLabelValue(self, col: int) -> str:
        return self.COLUMN_NAMES[col]


class ModelGraphPanel(wx.Panel, ModelObserver):
    """Base class for the panel, which contains model-dependent graph."""

    def __init__(self, model: Model, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.model = model
        model.add_observer(self)
        self.create_figure_and_canvas()
        self.sizer = self.create_sizer()
        self.SetSizer(self.sizer)

    def create_figure_and_canvas(self) -> None:
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.init_figure_and_axes()
        self.canvas = FigureCanvas(self, -1, self.figure)

    def create_sizer(self) -> wx.BoxSizer:
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, proportion=1, flag=wx.EXPAND | wx.RIGHT | wx.LEFT)
        return sizer

    def init_figure_and_axes(self) -> None:
        pass

    def redraw_figure(self) -> None:
        pass

    def update(self, event: str, model: Model) -> None:
        # print(f"{self.__class__.__name__}.update {event}")
        if event in [EXAMPLES_LOADED_EVENT, EXAMPLE_ADDED_EVENT, SURFACE_EVENT]:
            self.redraw_figure()
        if event == CLEAR_DATA_EVENT:
            self.redraw_figure()
        self.canvas.draw()


def fill_one_before(where_array: np.ndarray) -> Sequence[bool]:
    # print(type(where_array))
    itemindex = np.where(where_array == True)  # noqa:E712
    if len(itemindex[0]) > 0:
        index = itemindex[0][0]
        if index > 0:
            where_array[index - 1] = True
    return cast(Sequence[bool], where_array)


class CompactnessProfilePanel(ModelGraphPanel):
    """Panel for displaying compactness profile of the dataset and expected complete cross validation error."""

    def __init__(self, model: Model, *args: Any, **kw: Any) -> None:
        super().__init__(model, *args, **kw)

    def create_figure_and_canvas(self) -> None:
        self.figure = Figure()
        self.ax = self.figure.add_subplot(211)
        self.ax_ccv = self.figure.add_subplot(212)
        self.canvas = FigureCanvas(self, -1, self.figure)

    def init_figure_and_axes(self) -> None:
        self.ax.set_xticks([])
        self.ax.set_yticks([0.1 * i for i in range(11)])
        self.ax.set_ylim((-0.1, 1.1))
        self.ax.set_xlabel("k")
        self.ax.set_ylabel("% of object of other classes \n among k ordered neighbours")

        self.ax_ccv.set_xticks([])
        self.ax_ccv.set_yticks([0.1 * i for i in range(11)])
        self.ax_ccv.set_ylim((-0.1, 1.1))
        self.ax_ccv.set_xlabel("training set size")
        self.ax_ccv.set_ylabel("complete cross validation error")

    def redraw_figure(self) -> None:
        if self.model.has_data():
            profile = compactness_profile(self.model.get_X(), self.model.get_y())
            ccv_graph = compute_ccvs(profile, min(100, len(profile)))
        else:
            profile = []
            ccv_graph = {}

        self.ax.clear()
        self.ax.set_xlim((1, max(len(profile), 2)))
        self.ax.plot([x + 1 for x in range(len(profile))], profile, "b")
        self.ax.set_ylim((-0.1, 1.1))
        self.ax.set_xlabel("k")
        self.ax.set_ylabel("% of object of other classes \n among k ordered neighbours")
        self.ax.grid()

        self.ax_ccv.clear()
        self.ax_ccv.set_xlim((1, max(len(profile), 2)))
        self.ax_ccv.plot([x for x in ccv_graph.keys()], [v for v in ccv_graph.values()], "b")
        self.ax_ccv.set_ylim((-0.1, 1.1))
        self.ax_ccv.set_xlabel("training set size")
        self.ax_ccv.set_ylabel("complete cross validation error")
        self.ax_ccv.grid()


def classify_fris_value(value: float) -> str:
    assert -1 <= value <= 1.0
    if value <= -0.5:
        return OUTLIER_FRIS_CLASS
    if value <= -0.15:
        return ERROR_PRONE_FRIS_CLASS
    if value <= 0.15:
        return BOUNDARY_FRIS_CLASS
    if value <= 0.7:
        return TYPICAL_FRIS_CLASS
    return REPRESENTATIVE_FRIS_CLASS


FRIS_CLASS_COLOR_MAP: dict[str, str] = {
    OUTLIER_FRIS_CLASS: "red",
    ERROR_PRONE_FRIS_CLASS: "lightsalmon",
    BOUNDARY_FRIS_CLASS: "gold",
    TYPICAL_FRIS_CLASS: "lightgreen",
    REPRESENTATIVE_FRIS_CLASS: "green",
}

FRIS_CLASS_VALUE_RANGES_MAP: dict[str, str] = {
    OUTLIER_FRIS_CLASS: "-1.00 < V <= -0.50",
    ERROR_PRONE_FRIS_CLASS: "-0.50 < V <= -0.15",
    BOUNDARY_FRIS_CLASS: "-0.15 < V <= 0.15",
    TYPICAL_FRIS_CLASS: "0.15 < V <= 0.70",
    REPRESENTATIVE_FRIS_CLASS: "0.70 < V <= 1.00",
}


def get_color_for_fris_class(a_class: str) -> str:
    return FRIS_CLASS_COLOR_MAP[a_class]


def get_ranges_for_fris_class(a_class: str) -> str:
    return FRIS_CLASS_VALUE_RANGES_MAP[a_class]


def get_color_for_fris_value(value: float) -> str:
    return get_color_for_fris_class(classify_fris_value(value))


class FrisDistributionPanel(ModelGraphPanel):
    """A panel displaying the distribution of FRiS function values for the current dataset."""

    def __init__(self, model: Model, *args: Any, **kw: Any) -> None:
        super().__init__(model, *args, **kw)

    def create_static_text(self, parent: wx.Window) -> None:
        self.status_text = WebView.New(parent, size=(-1, 150))

    def create_figure_and_canvas(self) -> None:
        super().create_figure_and_canvas()
        self.create_static_text(self)

    def create_sizer(self) -> wx.BoxSizer:
        sizer = super().create_sizer()
        sizer.Add(self.status_text, 0, wx.EXPAND | wx.ALL, border=5)
        return sizer

    def init_figure_and_axes(self) -> None:
        self.ax.set_xticks([])
        self.ax.set_yticks([0.1 * i for i in range(-11, 11)])
        self.ax.set_ylim((-1.1, 1.1))
        self.ax.set_xlabel("object count")
        self.ax.set_ylabel("FRiS value distribution")

    def redraw_figure(self) -> None:
        self.ax.clear()

        if not self.model.is_fitted():
            return
        data_size = len(self.model.data)
        fris_values = np.array(sorted([cast(float, self.model.get_fris_value(i)) for i in range(data_size)]))
        # print(fris_values)
        self.ax.set_xlim((1, data_size))
        x = [v + 1 for v in range(data_size)]
        self.ax.plot(x, fris_values, "b")
        self.ax.set_ylim((-1.1, 1.1))
        self.ax.fill_between(x, 0, fris_values, where=cast(Sequence[bool], (fris_values <= -0.5)), facecolor="r")
        self.ax.fill_between(
            x,
            0,
            fris_values,
            where=fill_one_before((-0.5 <= fris_values) & (fris_values <= -0.15)),
            facecolor="lightsalmon",
        )
        self.ax.fill_between(
            x, 0, fris_values, where=fill_one_before((-0.15 < fris_values) & (fris_values <= 0.15)), facecolor="gold"
        )
        self.ax.fill_between(
            x,
            0,
            fris_values,
            where=fill_one_before((0.15 < fris_values) & (fris_values <= 0.7)),
            facecolor="lightgreen",
        )
        self.ax.fill_between(x, 0, fris_values, where=fill_one_before(0.7 < fris_values), facecolor="g")
        self.ax.grid()
        self.ax.set_xlabel("objects count")
        self.ax.set_ylabel("values of FRiS function")

    def update(self, event: str, model: Model) -> None:
        self.update_model_description(model)
        super().update(event, model)

    def update_model_description(self, model: Model) -> None:
        if not model.is_fitted():
            description = ""
        else:
            description = """
            <html>
            <head>
            <style>
            html *
            {
               font: 12px Verdana, Arial, sans-serif !important;
               color: #000 !important;
               font-family: Arial !important;
               font-weight: bold !important;
            }
            tbody td {
              text-align: right;
            }
            </style>
            </head>
            """
            fris_values_counter: dict[str, int] = defaultdict(int)
            total_count = len(model.data)
            for i in range(total_count):
                value = model.get_fris_value(i)
                if value is not None:
                    fris_values_counter[classify_fris_value(value)] += 1
            description += "<center>"
            description += "<table><tr><td>Point Type</td><td>Count</td><td>Percent</td><td>Value Range</td></tr>"

            for a_class in FRIS_CLASS_COLOR_MAP.keys():
                if a_class in fris_values_counter:
                    class_count = fris_values_counter[a_class]
                    description += f"<tr><td bgcolor='{get_color_for_fris_class(a_class)}'>"
                    description += f"{a_class}</td><td>{class_count}</td>"
                    description += f"<td>{class_count/total_count*100.:0.1f}%</td>"
                    description += f"<td>{get_ranges_for_fris_class(a_class)}</td></tr>"
            description += "</table>"
            description += "</center>"
            description += "</html>"

        self.status_text.SetPage(html=description, baseUrl="")


class CanvasFrame(wx.Frame, ModelObserver):
    """Main application data frame."""

    def __init__(self) -> None:
        wx.Frame.__init__(self, None, -1, title="Fris Stolp Demo Gui", size=(1024, 768))
        self.notebook = self.create_notebook(self)
        sizer = wx.BoxSizer()
        sizer.Add(self.notebook, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.Maximize(True)
        self.Fit()

    def create_cases_list(self, parent: wx.Window) -> wx.grid.Grid:
        return wx.grid.Grid(parent, -1)

    def update_grid(self) -> None:
        self.grid.SetTable(FrisStolpDataTable(self.model), True)
        self.grid.AutoSize()
        self.grid.AdjustScrollbars()
        self.grid.ForceRefresh()
        self.grid.GetParent().Layout()

    def create_notebook(self, parent: wx.Window) -> wx.Notebook:
        notebook = wx.Notebook(parent, wx.ID_ANY, size=(1024, 768))
        self.model = Model()
        self.panel = ModelPanel(self.model, notebook)
        notebook.AddPage(self.panel, "Drawing")

        panel = wx.Panel(notebook)
        self.grid = self.create_cases_list(panel)
        self.model.add_observer(self)

        self.update_grid()
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.grid, proportion=1, flag=wx.EXPAND | wx.RIGHT | wx.LEFT)
        panel.SetSizer(sizer)
        notebook.AddPage(panel, "Data")

        compactness_profile_panel = CompactnessProfilePanel(self.model, notebook)
        notebook.AddPage(compactness_profile_panel, "Compactness Profile")

        fris_values_panel = FrisDistributionPanel(self.model, notebook)
        notebook.AddPage(fris_values_panel, "FRiS Values Distribution")
        return notebook

    def update(self, event: str, _model: Model) -> None:
        if event in [EXAMPLES_LOADED_EVENT, EXAMPLE_ADDED_EVENT, CLEAR_DATA_EVENT, SURFACE_EVENT]:
            self.update_grid()
        else:
            logging.warning(f"{self.__class__.__name__} unhandled event: {event}")


class App(wx.App):
    """Main application class."""

    def OnInit(self) -> bool:
        """Create the main window and insert the custom frame."""
        self.frame = CanvasFrame()
        self.frame.Show()
        return True


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input", action="store", type=str, dest="input", help="Path where to load data from")
    argument_parser.add_argument("--output", action="store", type=str, dest="output", help="Path where to dump data.")
    args = argument_parser.parse_args()

    app = App(0)
    if args.input:
        app.frame.model.load_svmlight_file(args.input)
        print(f"Model data was loaded from {args.input}")
    app.MainLoop()
    if args.output:
        app.frame.model.dump_svmlight_file(args.output)
        print(f"Model stored to file {args.output}")


if __name__ == "__main__":
    main()
