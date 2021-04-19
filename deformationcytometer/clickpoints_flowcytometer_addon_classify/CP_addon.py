import numpy as np
import clickpoints
from deformationcytometer.evaluation.helper_functions import load_all_data_new
from qtpy import QtCore, QtGui, QtWidgets


class Addon(clickpoints.Addon):
    signal_update_plot = QtCore.Signal()
    signal_plot_finished = QtCore.Signal()
    disp_text_existing = "displaying existing data"
    disp_text_new = "displaying new data"

    def __init__(self, *args, **kwargs):
        clickpoints.Addon.__init__(self, *args, **kwargs)

        self.layout = QtWidgets.QVBoxLayout(self)

        # Setting up marker Types
        self.marker_type_cell1 = self.db.setMarkerType("cell", "#0a2eff", self.db.TYPE_Ellipse)
        self.marker_type_cell2 = self.db.setMarkerType("cell include", "#25c638", self.db.TYPE_Ellipse)
        self.marker_type_cell3 = self.db.setMarkerType("cell exclude", "#ff0000", self.db.TYPE_Ellipse)
        self.cp.reloadTypes()

        # set the title and layout
        self.setWindowTitle("DeformationCytometer - ClickPoints")

        self.filename = self.db.getImage(0).get_full_filename()[:-4]+"_evaluated_new.csv"
        self.data, self.config = load_all_data_new(self.db.getImage(0).get_full_filename())
        if "manual_exclude" not in self.data:
            self.data["manual_exclude"] = np.nan

        print("loading finished", self.db.getImage(0).get_full_filename())

        self.label = QtWidgets.QLabel("Cell")
        self.layout.addWidget(self.label)
        self.progressbar = QtWidgets.QProgressBar()
        self.layout.addWidget(self.progressbar)

        pixel_size = self.config["pixel_size"]
        self.db.deleteEllipses(type=self.marker_type_cell1)
        self.db.deleteEllipses(type=self.marker_type_cell2)
        self.db.deleteEllipses(type=self.marker_type_cell3)
        self.markers = []

        if 0:
            self.markers = self.db.setEllipses(
                           frame=list(np.array(self.data.frames).astype(np.int)),
                           x=np.array(self.data.x),
                           y=np.array(self.data.y),
                           width=np.array(self.data.long_axis / pixel_size),
                           height=np.array(self.data.short_axis / pixel_size),
                           angle=np.array(self.data.angle),
                           type=self.marker_type_cell1
            )
        print(self.markers)
        self.show()
        if 1:
            self.progressbar.setRange(0, len(self.data)-1)
            for i, d in self.data.iterrows():
                 self.progressbar.setValue(i)
                 self.label.setText(f"Cell {i}")
                 self.cp.window.app.processEvents()
                 type_ = self.marker_type_cell3 if d.manual_exclude == 1 else self.marker_type_cell2 if d.manual_exclude == 0 else self.marker_type_cell1
                 ell = self.db.setEllipse(frame=int(d.frames), x=d.x, y=d.y,
                              width=d.long_axis / pixel_size, height=d.short_axis / pixel_size,
                              text=f"{i} {d.cell_id}",
                              angle=d.angle, type=type_)
                 self.markers.append(ell)
        self.index = 0
        self.focusOnCell()

    # not sure what this is for ^^
    def buttonPressedEvent(self):
        self.show()

    def keyPressEvent(self, event):
        keys = [QtCore.Qt.Key_2, QtCore.Qt.Key_3, QtCore.Qt.Key_4, QtCore.Qt.Key_9, QtCore.Qt.Key_0]
        types = {
            QtCore.Qt.Key_2: self.marker_type_cell1,
            QtCore.Qt.Key_3: self.marker_type_cell2,
            QtCore.Qt.Key_4: self.marker_type_cell3,
        }
        types2 = {
            QtCore.Qt.Key_2: np.nan,
            QtCore.Qt.Key_3: 0,
            QtCore.Qt.Key_4: 1,
        }
        if event.key() in keys:
            if self.index >= 0 and event.key() in types:
                self.markers[self.index].changeType(types[event.key()])
                self.cp.reloadMarker()
                self.cp.window.app.processEvents()
                self.data.at[self.index, "manual_exclude"] = types2[event.key()]

                self.data.to_csv(self.filename, index=False)
            if event.key() != QtCore.Qt.Key_9:
                self.index += 1
            else:
                if self.index > 0:
                    self.index -= 1
            self.focusOnCell()

    def focusOnCell(self):
            self.label.setText(f"Cell {self.index}")
            d = self.data.iloc[self.index]
            print(self.index)
            # jump to the frame in time
            self.cp.jumpToFrame(d.frames)
            # and to the xy position
            self.cp.centerOn(d.x, d.y)

