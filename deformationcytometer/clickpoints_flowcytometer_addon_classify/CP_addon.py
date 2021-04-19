import numpy as np
import clickpoints
from qimage2ndarray import array2qimage
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
        self.pen_neutral = QtGui.QPen(QtGui.QColor("#0a2eff"), 3)
        self.marker_type_cell2 = self.db.setMarkerType("cell include", "#25c638", self.db.TYPE_Ellipse)
        self.pen_include = QtGui.QPen(QtGui.QColor("#25c638"), 3)
        self.marker_type_cell3 = self.db.setMarkerType("cell exclude", "#ff0000", self.db.TYPE_Ellipse)
        self.pen_exclude = QtGui.QPen(QtGui.QColor("#ff0000"), 3)
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
        self.progressbar.setRange(0, len(self.data) - 1)
        if 1:
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


        from clickpoints.includes import QExtendedGraphicsView
        self.view = QExtendedGraphicsView()
        self.layout.addWidget(self.view)
        #self.view.zoomEvent = self.zoomEvent
        #self.view.panEvent = self.panEvent
        self.local_scene = self.view.scene
        self.origin = self.view.origin

        self.cell_rects = {}
        self.addCellRect(0, 0)
        self.addCellRect(1, 300)
        self.addCellRect(2, 600)

        w = 600
        h = 400
        self.focus_rect = QtWidgets.QGraphicsRectItem(-w/2, -200, w, h*2, self.origin)
        self.focus_rect.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 3))
        self.focus_rect.setZValue(10)

        self.view.setExtend(w * 5, h)
        self.view.fitInView()

        self.start_pos = 0
        self.current_pos = self.index-0.1
        self.timer_percent = 0
        self.transition_timer = QtCore.QTimer()
        self.transition_timer.setTimerType(100)
        self.transition_timer.timeout.connect(self.timerCall)
        self.transition_timer.start()

        self.view.keyPressEvent = self.keyPressEvent2

        layout = QtWidgets.QHBoxLayout()
        self.layout.addLayout(layout)
        self.auto_save = QtWidgets.QCheckBox()
        layout.addWidget(self.auto_save)
        self.auto_save_label = QtWidgets.QLabel("autosave")
        layout.addWidget(self.auto_save_label)
        self.button_save = QtWidgets.QPushButton("save")
        self.button_save.clicked.connect(self.save)
        layout.addWidget(self.button_save)
        layout.addStretch()

        self.focusOnCell()

    def addCellRect(self, index, offset):
        if index in self.cell_rects:
            return
        w = 600
        h = 400
        s = 2
        d = self.data.iloc[index]
        self.rect_parent = QtWidgets.QGraphicsRectItem(0, -h, w, h*3, self.origin)
        #self.rect.setFlag(QtWidgets.QGraphicsItem.ItemClipsChildrenToShape, True)
        self.rect_parent.setX(index*w)
        self.rect_parent.setBrush(QtGui.QColor("black"))

        self.rect = QtWidgets.QGraphicsRectItem(0, 0, w, h, self.rect_parent)
        self.rect.setFlag(QtWidgets.QGraphicsItem.ItemClipsChildrenToShape, True)
        #self.rect.setX(index*w)
        self.rect.setBrush(QtGui.QColor("gray"))

        self.pix_origin = QtWidgets.QGraphicsPixmapItem(self.rect)

        self.im = self.db.getImage(d.frames).data
        self.pixmap = QtWidgets.QGraphicsPixmapItem(self.rect)
        self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(self.im)))
        self.pixmap.setScale(s)
        self.pixmap.setOffset((-d.x)+(w/2)/s, (-d.y)+(h/2)/s)
        pixel_size = self.config["pixel_size"]
        self.ellipse = QtWidgets.QGraphicsEllipseItem(- d.long_axis / pixel_size / 2, - d.short_axis / pixel_size / 2, d.long_axis / pixel_size, d.short_axis / pixel_size, self.pixmap)
        if d.manual_exclude == 0:
            self.ellipse.setPen(self.pen_include)
            self.rect_parent.setY(-200)
        elif d.manual_exclude == 1:
            self.ellipse.setPen(self.pen_exclude)
            self.rect_parent.setY(200)
        else:
            self.ellipse.setPen(self.pen_neutral)

        self.ellipse.setRotation(d.angle)
        self.ellipse.setPos((w/2)/s, (h/2)/s)
        self.rect_parent.ellipse = self.ellipse
        #self.pixmap.setScale(0.5)
        self.cell_rects[index] = self.rect_parent

    def timerCall(self):
        w = 600
        h = 400
        if self.index != self.current_pos:
            if self.timer_percent < 100:
                self.timer_percent += 10
                self.current_pos = self.start_pos * (100-self.timer_percent) / 100 + self.index * self.timer_percent / 100
            if 0:
                if self.index > self.current_pos:
                    self.current_pos = min(self.current_pos + 0.01, self.index)
                elif self.index < self.current_pos:
                    self.current_pos = max(self.current_pos - 0.01, self.index)

            self.focus_rect.setX(w * (self.current_pos + 0.5))
            self.view.centerOn(w * (self.current_pos + 0.5), h / 2)

    # not sure what this is for ^^
    def buttonPressedEvent(self):
        self.show()

    def save(self):
        print("save")
        self.data.to_csv(self.filename, index=False)

    def setCellCategory(self, index, value):
        self.data.at[index, "manual_exclude"] = value
        if value == 1:
            self.cell_rects[self.index].ellipse.setPen(self.pen_include)
            self.cell_rects[self.index].setY(-200)
            self.markers[self.index].changeType(self.marker_type_cell2)
        elif value == 0:
            self.cell_rects[self.index].ellipse.setPen(self.pen_exclude)
            self.cell_rects[self.index].setY(200)
            self.markers[self.index].changeType(self.marker_type_cell3)
        else:
            self.cell_rects[self.index].ellipse.setPen(self.pen_neutral)
            self.cell_rects[self.index].setY(0)
            self.markers[self.index].changeType(self.marker_type_cell1)

        self.cp.reloadMarker()
        self.cp.window.app.processEvents()

        if self.auto_save.isChecked():
            print("save")
            self.data.to_csv(self.filename, index=False)

    def keyPressEvent2(self, event):
        old_index = self.index
        if event.key() == QtCore.Qt.Key_Left:
            if self.index > 0:
                self.index -= 1
        if event.key() == QtCore.Qt.Key_Right:
            if self.index < len(self.data)-1:
                self.index += 1
        if event.key() == QtCore.Qt.Key_Up:
            self.setCellCategory(self.index, 0)
            if self.index < len(self.data) - 1:
                self.index += 1
        if event.key() == QtCore.Qt.Key_Down:
            self.setCellCategory(self.index, 1)
            if self.index < len(self.data) - 1:
                self.index += 1

        if self.index != old_index:
            self.focusOnCell()

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
                self.setCellCategory(self.index, types2[event.key()])
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

            self.progressbar.setValue(self.index)
            for i in range(self.index - 2, self.index + 3):
                self.start_pos = self.current_pos
                self.timer_percent = 0
                if i >= 0:
                    self.addCellRect(i, 0)
            for key in list(self.cell_rects.keys()):
                if np.abs(key - self.index) > 10:
                    self.cell_rects[key].scene().removeItem(self.cell_rects[key])
                    del self.cell_rects[key]

