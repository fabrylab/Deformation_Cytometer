import numpy as np
import clickpoints
from qimage2ndarray import array2qimage
from deformationcytometer.evaluation.helper_functions import load_all_data_new
from qtpy import QtCore, QtGui, QtWidgets
from threading import Thread
from clickpoints.includes import QExtendedGraphicsView


def AnimationChange(target, setter, start=0, end=1, duration=200, fps=36, endcall=None, transition="ease"):
    # stop old animation if present
    old_timer = getattr(target, "animation_timer", None)
    if old_timer is not None:
        old_timer.stop()
    # create a new timer
    timer = QtCore.QTimer()
    # convert ms to s
    duration /= 1e3

    # define the timer callback
    def timerEvent():
        # advance the time
        timer.animation_time += 1. / (fps * duration)
        # it the animation is finished, directly set the end value and stop the timer
        if timer.animation_time >= 1:
            setter(end)
            timer.stop()
            if endcall:
                endcall()
            return
        # calculate the transition function
        x = timer.animation_time
        k = 3
        if transition == "ease" or transition == "ease-in-out":
            y = 0.5 * (x * 2) ** k * (x < 0.5) + (1 - 0.5 * ((1 - x) * 2) ** k) * (x >= 0.5)
        elif transition == "ease-out":
            y = 1 - 0.5 * (1 - x) ** k
        elif transition == "ease-in":
            y = 0.5 * x ** k
        elif transition == "linear":
            y = x
        else:
            raise ValueError("Unknown transition")
        # set the value
        setter(y * (end - start) + start)

    timer.timeout.connect(timerEvent)
    timer.animation_time = 0
    # set the start value
    setter(start)
    # attach the timer to the target object
    # This is important to prevent the garbage collector from removing the timer
    # also it is needed to stop the timer in case a second timer gets attached to this object
    target.animation_timer = timer
    # start the animation
    timer.start(1e3 / fps)


class Addon(clickpoints.Addon):
    image_loaded = QtCore.Signal(int, QtGui.QPixmap)
    w = 500
    h = 300
    s = 2

    def __init__(self, *args, **kwargs):
        clickpoints.Addon.__init__(self, *args, **kwargs)

        # open the window
        self.layout = QtWidgets.QVBoxLayout(self)
        self.resize(1300, 400)
        self.show()

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

        # load the data
        self.filename = self.db.getImage(0).get_full_filename()[:-4] + "_evaluated_new.csv"
        self.data, self.config = load_all_data_new(self.db.getImage(0).get_full_filename().replace(".tif", "_evaluated_new.csv"), do_group=False, do_excude=False)
        if "manual_exclude" not in self.data:
            self.data["manual_exclude"] = np.nan

        print("loading finished", self.db.getImage(0).get_full_filename())

        # add a label and the progressbar
        self.label = QtWidgets.QLabel("Cell")
        self.layout.addWidget(self.label)
        self.progressbar = QtWidgets.QProgressBar()
        self.progressbar.setRange(0, len(self.data) - 1)
        self.layout.addWidget(self.progressbar)

        pixel_size = self.config["pixel_size"]
        # remove previous markers of the time in case there are any (e.g. when the addon is activated twice)
        self.db.deleteEllipses(type=self.marker_type_cell1)
        self.db.deleteEllipses(type=self.marker_type_cell2)
        self.db.deleteEllipses(type=self.marker_type_cell3)
        self.markers = []

        if 0:
            # add all markers at once
            self.markers = self.db.setEllipses(
                frame=list(np.array(self.data.frames).astype(np.int)),
                x=np.array(self.data.x),
                y=np.array(self.data.y),
                width=np.array(self.data.long_axis / pixel_size),
                height=np.array(self.data.short_axis / pixel_size),
                angle=np.array(self.data.angle),
                type=self.marker_type_cell1
            )
        if 0:
            # iteratively create the markers
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

        self.view = QExtendedGraphicsView()
        self.layout.addWidget(self.view)
        # self.view.zoomEvent = self.zoomEvent
        # self.view.panEvent = self.panEvent
        self.local_scene = self.view.scene
        self.origin = self.view.origin

        self.cell_rects = {}

        self.focus_rect = QtWidgets.QGraphicsRectItem(-self.w / 2, -self.h / 2, self.w, self.h * 2, self.origin)
        self.focus_rect.setPen(QtGui.QPen(QtGui.QColor(255, 255, 0), 8))
        self.focus_rect.setZValue(10)

        self.view.setExtend(self.w * 5, self.h*2)
        self.view.fitInView()

        self.start_pos = 0
        self.current_pos = self.index - 0.1
        self.timer_percent = 0

        self.view.keyPressEvent = self.keyPressEvent2

        self.slider = QtWidgets.QSlider()
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.layout.addWidget(self.slider)
        self.slider.setRange(0, len(self.data)-1)
        def setIndex(x):
            if x != self.index:
                self.index = x
                self.focusOnCell()
        self.slider.valueChanged.connect(setIndex)

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

        self.image_loaded.connect(self.addImage)

        self.focusOnCell()

    def loadImage(self, index):
        print("load image", index)
        d = self.data.iloc[index]
        im = self.db.getImage(d.frames).data
        pixmap = QtGui.QPixmap(array2qimage(im))
        self.image_loaded.emit(index, pixmap)

    def addImage(self, index, pixmap):
        print("loaded image", index)
        if index in self.cell_rects:
            self.cell_rects[index].pixmap.setPixmap(pixmap)

    def addCellRect(self, index):
        if index in self.cell_rects or not (0 <= index < len(self.data)):
            return

        d = self.data.iloc[index]
        rect_parent = QtWidgets.QGraphicsRectItem(0, -self.h, self.w, self.h * 3, self.origin)
        rect_parent.setX(index * self.w)
        rect_parent.setBrush(QtGui.QColor("black"))

        rect_parent.rect = QtWidgets.QGraphicsRectItem(0, 0, self.w, self.h, rect_parent)
        rect_parent.rect.setFlag(QtWidgets.QGraphicsItem.ItemClipsChildrenToShape, True)
        rect_parent.rect.setBrush(QtGui.QColor("gray"))

        rect_parent.pixmap = QtWidgets.QGraphicsPixmapItem(rect_parent.rect)
        rect_parent.pixmap.setScale(self.s)
        rect_parent.pixmap.setOffset((-d.x)+(self.w/2)/self.s, (-d.y)+(self.h/2)/self.s)
        #rect_parent.pixmap.setOffset((-d.x), (-d.y))

        if 1:
            rect_parent.load_thread = Thread(target=self.loadImage, args=(index,))
            rect_parent.load_thread.start()
        else:
            im = self.db.getImage(d.frames).data
            pixmap = QtGui.QPixmap(array2qimage(im))
            rect_parent.pixmap.setPixmap(pixmap)

        pixel_size = self.config["pixel_size"]
        ellipse = QtWidgets.QGraphicsEllipseItem(- d.long_axis / pixel_size / 2, - d.short_axis / pixel_size / 2,
                                                 d.long_axis / pixel_size, d.short_axis / pixel_size,
                                                 rect_parent.pixmap)
        if d.manual_exclude == 0:
            ellipse.setPen(self.pen_include)
            rect_parent.setY(-self.h / 2)
        elif d.manual_exclude == 1:
            ellipse.setPen(self.pen_exclude)
            rect_parent.setY(self.h / 2)
        else:
            ellipse.setPen(self.pen_neutral)

        ellipse.setRotation(d.angle)
        ellipse.setPos((self.w / 2) / self.s, (self.h / 2) / self.s)
        rect_parent.ellipse = ellipse
        self.cell_rects[index] = rect_parent

    def setFocusPos(self, index):
        self.current_pos = index
        self.focus_rect.setX(self.w * (self.current_pos + 0.5))
        self.view.centerOn(self.w * (self.current_pos + 0.5), self.h / 2)

    # not sure what this is for ^^
    def buttonPressedEvent(self):
        self.show()

    def save(self):
        print("save")
        self.data.to_csv(self.filename, index=False)

    def setCellCategory(self, index, value):
        self.data.at[index, "manual_exclude"] = value
        if value == 0:
            self.cell_rects[self.index].ellipse.setPen(self.pen_include)
            AnimationChange(self.cell_rects[self.index], self.cell_rects[self.index].setY,
                            start=self.cell_rects[self.index].y(), end=-self.h/2)
            # self.cell_rects[self.index].setY(-200)
            if self.index in self.markers:
                self.markers[self.index].changeType(self.marker_type_cell2)
        elif value == 1:
            self.cell_rects[self.index].ellipse.setPen(self.pen_exclude)
            AnimationChange(self.cell_rects[self.index], self.cell_rects[self.index].setY,
                            start=self.cell_rects[self.index].y(), end=self.h/2)
            # self.cell_rects[self.index].setY(200)
            if self.index in self.markers:
                self.markers[self.index].changeType(self.marker_type_cell3)
        else:
            self.cell_rects[self.index].ellipse.setPen(self.pen_neutral)
            self.cell_rects[self.index].setY(0)
            self.markers[self.index].changeType(self.marker_type_cell1)

        # self.cp.reloadMarker()
        # self.cp.window.app.processEvents()

        if self.auto_save.isChecked():
            print("save")
            self.data.to_csv(self.filename, index=False)

    def keyPressEvent2(self, event):
        old_index = self.index
        if event.key() == QtCore.Qt.Key_Left:
            if self.index > 0:
                self.index -= 1
        if event.key() == QtCore.Qt.Key_Right:
            if self.index < len(self.data) - 1:
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
        # self.cp.jumpToFrame(d.frames)
        # and to the xy position
        # self.cp.centerOn(d.x, d.y)

        self.progressbar.setValue(self.index)
        self.slider.setValue(self.index)
        for i in range(self.index - 4, self.index + 4 + 1):
            self.start_pos = self.current_pos
            self.timer_percent = 0
            if i >= 0:
                self.addCellRect(i)
        for key in list(self.cell_rects.keys()):
            if np.abs(key - self.index) > 10:
                self.cell_rects[key].scene().removeItem(self.cell_rects[key])
                del self.cell_rects[key]
        AnimationChange(self, self.setFocusPos,
                        start=self.current_pos, end=self.index, transition="ease-out")
