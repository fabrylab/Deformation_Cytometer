#!/usr/bin/env python3

"""
Display Tool for mmap based camera interfaces.
Based on QtExtendedGraphicsView by R. Gerum & S. Richter
https://bitbucket.org/fabry_biophysics/qextendedgraphicsview
S. Richter 2019, sebastian.sr.richter@gmail.com
"""

from __future__ import division, print_function

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
import time
import re

from qtpy import QtGui, QtCore, QtWidgets

# conditional import
from skimage.filters import sobel
from skimage.color import rgb2gray

from deformationcytometer.recording.includes.QExtendedGraphicsView import QExtendedGraphicsView
from deformationcytometer.recording.includes.Tools import GraphicsItemEventFilter, array2qimage
from deformationcytometer.recording.includes.MemMap import MemMap
import configparser

config = configparser.ConfigParser()
config.read("config.txt")

def help():
    print("===================================================================")
    print("LiveDisplay.py")
    print("Display Tool for mmap based camera interfaces.")
    print("Usage:")
    print("     ./LiveDisplay.py  CAM  OPTIONS   ")
    print("         CAM             xml definition or camera shorthand\n")
    print("     Options:")
    print("         --draw          enable Drawing Mode (or F4)")
    print("         --draw_id       for multi camera systems")
    print("         --fps=N         set maximum frame rate [float]")
    print("         --rotation=N    set CW rotation in degree [float] ")
    print("         --help          show this help\n")
    print("     Hotkeys:")
    print("         F3              toggle Focus Mode")
    print("         F4              toggle Drawing Mode")
    print("         R               rotate in 90° steps cw")
    print("         F               fit to view")
    print("===================================================================")

# camera class to retrieve newest image
class GigECam():
    def __init__(self,mmap_xml):
        self.mmap = MemMap(mmap_xml)
        print("connect to ", mmap_xml)
        self.counter_last = -1

        # percentil calc paramter
        self.pct_counter = 0
        self.pct_min = None
        self.pct_max = None

    def getNewestImage(self, return16bit=False):
        # get newest counter
        counters = [ slot.counter for slot in self.mmap.rbf]

        counter_max = np.max(counters)
        counter_max_idx = np.argmax(counters)

        # return if there is no new one
        if counter_max == self.counter_last:
            # print("not new!")
            return None

        image = self.mmap.rbf[counter_max_idx].image
        # print(image.shape, image.dtype)


        try:
            im_channels = image.shape[2]
        except:
            im_channels = 1

        im_rsize = self.mmap.rbf[counter_max_idx].width * self.mmap.rbf[counter_max_idx].height * im_channels

        # check for ROI
        #if not (im_rsize == len(image.flatten())):
        #    im_roi = image.flatten()[0: im_rsize ]

        #    image = im_roi.reshape([self.mmap.rbf[counter_max_idx].height,self.mmap.rbf[counter_max_idx].width,im_channels])

        # check for 16bit data
        if image.dtype == 'uint16' and not return16bit:

            if self.pct_min == None or self.pct_counter % 100 == 0:
                # calculate new min max values
                self.pct_min = np.percentile(image,1)
                self.pct_max = np.percentile(image,99)
                # print(self.pct_min,self.pct_max)

            image = ((image - self.pct_min) / (self.pct_max - self.pct_min) * 255)
            image [image < 0] = 0
            image [image > 255] = 255
            image = image.astype('uint8')
            self.pct_counter += 1

        self.counter_last = counter_max
        # print("img type:", image.dtype)
        return image

# module TileHelper
class ModuleTileHelper():
    def __init__(self, _parent,  _config='cfg/ws_config.yml'):
        self.parent = _parent       # the parent window containing the graphics view
        self.cfg_file = _config

        self.detector_count = None  # number of detector configs to cycle through
        self.loadCFG()

        self.draw_elemets = []      # to keep track of drawn elements

        # color cycle for multiscale windows
        self.colors = [[0,    255,  125],
                       [255,    0,  125],
                       [225,    50,  255]]

        # id of the detector to draw (cycle using hot key)
        # expands to "detector_$id" entry in yaml
        self.id = 0

        self.image_shape = None


    def loadCFG(self):
        """ load yaml config, reload during runtime to update """
        try:
            import yaml
            self.cfg = yaml.safe_load(open(self.cfg_file, 'r'))

            self.detector_count = np.sum([k.startswith('detector') for k in self.cfg.keys()])
        except Exception as e:
            print("Failed loading TileHelper Module")
            print(e)


    def drawTiles(self, img_shape):
        """ print tile layout for current selector """
        id = self.id
        for layer,tiles in enumerate(self.cfg['detector_%d' % id]['tiles']):
            color = self.colors[layer]

            # draw border frames
            xo1,xo2,yo1,yo2 = tiles['limits']  # offset of the border
            h,w,c = img_shape

            # DEBUG
            # print(id, layer, tiles['limits'], w,h,c)

            self.draw_elemets.append(
                self._draw_rect(0 + xo1,
                                0 + yo1,
                                (w - xo2) - xo1,
                                (h - yo2) - yo1,
                                text="  tiles %d" % layer,
                                text_pos='tl',
                                color=color,
                                lw=1)
            )

            # draw tile samples
            tw,th = tiles['size']
            self.draw_elemets.append(
                self._draw_rect(0 + xo1,
                                0 + yo1,
                                tw,
                                th,
                                text="  %dx%d" % (tw,th),
                                text_pos='bl',
                                color=color,
                                lw=1)
            )

            # draw offset tile sample
            tw,th = tiles['size']
            txo,tyo = tiles['stride']
            self.draw_elemets.append(
                self._draw_rect(0 + xo1 + txo,
                                0 + yo1 + tyo,
                                tw,
                                th,
                                color=[*color,125],
                                lw=1)
            )

    def clearTiles(self):
        """ remove all drawings from scene """
        for obj in self.draw_elemets:
            for item in obj:
                item.scene().removeItem(item)
        self.draw_elemets = []

    def _draw_rect(self, x, y, w, h, text="", color = [255,0,125], lw=2, text_pos = 'tl'):

        # set pen
        pen = QtGui.QPen(QtGui.QColor(*color))
        pen.setWidth(lw)
        pen.setCosmetic(True)

        tmp = []
        # draw 4 lines for a rectangle
        tmp.append(QtWidgets.QGraphicsLineItem(
            QtCore.QLineF(QtCore.QPointF(*[x, y]),
                          QtCore.QPointF(*[x + w, y]))))
        tmp.append(QtWidgets.QGraphicsLineItem(
            QtCore.QLineF(QtCore.QPointF(*[x, y]),
                          QtCore.QPointF(*[x, y + h]))))
        tmp.append(QtWidgets.QGraphicsLineItem(
            QtCore.QLineF(QtCore.QPointF(*[x + w, y]),
                          QtCore.QPointF(*[x + w, y + h]))))
        tmp.append(QtWidgets.QGraphicsLineItem(
            QtCore.QLineF(QtCore.QPointF(*[x, y + h]),
                          QtCore.QPointF(*[x + w, y + h]))))

        # set Pen, ZValue, and Display Parent
        _ = [line.setPen(pen) for line in tmp]
        _ = [line.setZValue(10) for line in tmp]
        _ = [line.setParentItem(self.parent) for line in tmp]


        # display focus value on screen
        font = QtGui.QFont()
        font.setPointSize(5)

        texte = QtWidgets.QGraphicsSimpleTextItem(self.parent)
        texte.setFont(font)

        if text_pos == 'bl':
            texte.setPos(x, y + h + 1)
        if text_pos == 'tl':
            texte.setPos(x, y - 10)
        texte.setZValue(10)
        texte.setBrush(QtGui.QBrush(QtGui.QColor(*color)))
        texte.setText(text)

        tmp.append(texte)


        # return draw elements
        return tmp

# module FocusHelper
class ModuleFocus():
    def __init__(self, _parent,  _camera, _frame=[ 0.4, 0.4, 0.6, 0.6],):
        self.parent = _parent       # the parent window containing the graphics view
        self.cam = _camera          # mmap based camera input
        self.focus_frame_position = _frame  # percentage of image dimesnions [ x1 y1 x2 y2] used for calculation
        self.draw_frame = []        # list of the frame elements

        self.anker = None
        self.text = None


    def showFrame(self, img):
        # initial frame draw and text position
        # draw frame in which the focus is calculated
        frame_x1 = img.shape[1] * self.focus_frame_position[0]
        frame_y1 = img.shape[0] * self.focus_frame_position[1]
        frame_x2 = img.shape[1] * self.focus_frame_position[2]
        frame_y2 = img.shape[0] * self.focus_frame_position[3]

        self.focus_frame_position_px = np.round([frame_x1, frame_y1, frame_x2, frame_y2]).astype(np.int)

        # set pen
        pen = QtGui.QPen(QtGui.QColor("#ff5f00"))
        pen.setWidth(2)
        pen.setCosmetic(True)

        # draw closing vertices
        self.draw_frame.append(QtWidgets.QGraphicsLineItem(
            QtCore.QLineF(QtCore.QPointF(*[frame_x1, frame_y1]), QtCore.QPointF(*[frame_x2, frame_y1]))))
        self.draw_frame.append(QtWidgets.QGraphicsLineItem(
            QtCore.QLineF(QtCore.QPointF(*[frame_x1, frame_y1]), QtCore.QPointF(*[frame_x1, frame_y2]))))
        self.draw_frame.append(QtWidgets.QGraphicsLineItem(
            QtCore.QLineF(QtCore.QPointF(*[frame_x2, frame_y1]), QtCore.QPointF(*[frame_x2, frame_y2]))))
        self.draw_frame.append(QtWidgets.QGraphicsLineItem(
            QtCore.QLineF(QtCore.QPointF(*[frame_x1, frame_y2]), QtCore.QPointF(*[frame_x2, frame_y2]))))

        # set Pen, ZValue, and Display Parent
        _ = [line.setPen(pen) for line in self.draw_frame]
        _ = [line.setZValue(10000) for line in self.draw_frame]
        _ = [line.setParentItem(self.parent) for line in self.draw_frame]

        # display focus value on screen
        self.font = QtGui.QFont()
        self.font.setPointSize(16)
        if not self.anker:
            self.anker = QtWidgets.QGraphicsPathItem(self.parent)
            self.anker.setPos(np.round(self.focus_frame_position_px[0] +
                                       (self.focus_frame_position_px[2] - self.focus_frame_position_px[0]) / 2),  # x middle
                              self.focus_frame_position_px[3])  # y lower
            self.anker.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations)
        if not self.text:
            self.text = QtWidgets.QGraphicsSimpleTextItem(self.anker)
            self.text.setFont(self.font)
            self.text.setPos(-30, -25)
            self.text.setZValue(10)
            self.text.setBrush(QtGui.QBrush(QtGui.QColor("#ff5f00")))
            self.text.setText("N/A")

    def hideFrame(self):
        # remove frame
        for line in self.draw_frame:
            line.scene().removeItem(line)
        # hide text
        self.text.setText('')

    def calcFocus(self, img):
        focus_img = img[self.focus_frame_position_px[0]:self.focus_frame_position_px[2],
                    self.focus_frame_position_px[1]:self.focus_frame_position_px[3]].copy()
        if len(focus_img.squeeze().shape) > 2:
            # a color image
            sobel_img = sobel(rgb2gray(focus_img))
        else:
            # a grayscale iamge
            sobel_img = sobel(focus_img.squeeze())

        focus_value = np.std(sobel_img)
        # print("Focus Value:  %.6f" % focus_value)

        self.text.setText("%.3f" % (focus_value * 10 ** 4))

# module Drawing
class ModuleDraw():
    def __init__(self,_parent, draw_id = None):

        if not draw_id is None:
            drawmem_xml = r"cfg/ws_draw_%d.xml" % (draw_id + 1)
        else:
            drawmem_xml = r"cfg/ws_draw.xml"

        self.mem = MemMap(drawmem_xml)
        self.parent = _parent

        self.draw_list = []  # chache all elements for faster drawing
        self.last_frame = -1 # to check if this is acutally new data

    def clearAsyncElements(self):
        for element in self.mem.rbf[200:220]:
            try:
                for item in element.draw:
                    item.scene().removeItem(item)
                draw = []
            except AttributeError:
                pass

    def updateAsyncElements(self):
        # clear all elements
        self.clearAsyncElements()

        # draw elements
        for element in self.mem.rbf[200:220]:
            if not element.id == 0:
                element.draw = self._draw_rect(element.x,
                                               element.y,
                                               element.w,
                                               element.h,
                                               lw = 3,
                                               color=np.array(element.color),
                                               text="ID_%d" % element.score)

    def clearElements(self):
        # iterate over elements
        # delete all content
        for element in self.mem.rbf[0:200]:
            try:
                for item in element.draw:
                    item.scene().removeItem(item)
                draw = []
            except AttributeError:
                pass

    def updateElements(self):
        if not self.mem.frame_nr == self.last_frame:
            # clear all elements
            self.clearElements()

            # draw elements
            for element in self.mem.rbf[0:200]:
                if not element.id == 0:

                    if element.activity > 0:
                        txt = "%d\n%.2f" % (element.score, element.activity)
                    else:
                        txt = "%d" % element.score

                    element.draw =  self._draw_rect(element.x,
                                                    element.y,
                                                    element.w,
                                                    element.h,
                                                    lw=1,
                                                    color=np.array(element.color),
                                                    text=txt)




    def _draw_rect(self, x, y, w, h, text="", color = [255,0,125], lw=2):

        # set pen
        pen = QtGui.QPen(QtGui.QColor(*color))
        pen.setWidth(lw)
        pen.setCosmetic(True)

        tmp = []
        # draw 4 lines for a rectangle
        tmp.append(QtWidgets.QGraphicsLineItem(
            QtCore.QLineF(QtCore.QPointF(*[x, y]),
                          QtCore.QPointF(*[x + w, y]))))
        tmp.append(QtWidgets.QGraphicsLineItem(
            QtCore.QLineF(QtCore.QPointF(*[x, y]),
                          QtCore.QPointF(*[x, y + h]))))
        tmp.append(QtWidgets.QGraphicsLineItem(
            QtCore.QLineF(QtCore.QPointF(*[x + w, y]),
                          QtCore.QPointF(*[x + w, y + h]))))
        tmp.append(QtWidgets.QGraphicsLineItem(
            QtCore.QLineF(QtCore.QPointF(*[x, y + h]),
                          QtCore.QPointF(*[x + w, y + h]))))

        # set Pen, ZValue, and Display Parent
        _ = [line.setPen(pen) for line in tmp]
        _ = [line.setZValue(10) for line in tmp]
        _ = [line.setParentItem(self.parent) for line in tmp]


        # display focus value on screen
        font = QtGui.QFont()
        font.setPointSize(5)

        texte = QtWidgets.QGraphicsSimpleTextItem(self.parent)
        texte.setFont(font)
        texte.setPos(x, y + h + 1)
        texte.setZValue(10)
        texte.setBrush(QtGui.QBrush(QtGui.QColor(*color)))
        texte.setText(text)

        tmp.append(texte)


        # return draw elements
        return tmp



    # viewer class

class Window(QtWidgets.QWidget):
    def __init__(self, parent = None):
        QtWidgets.QWidget.__init__(self, parent)

        # default
        mmap_xml = config["camera"]["output_mmap"]
        self.counter = 0
        self.image_shape = None   # track last image shape

        # view
        self.rotation = 0
        self.framerate = 10

        # module
        self.do_calcFocus = False
        self.do_draw = False
        self.draw_id = None
        self.do_tiles = False # doubles as detector selector


        # parse comand line arguments
        if len(sys.argv):
            print(sys.argv)
            # use an implemented shorthand
            for arg in sys.argv[1::]:
                if arg == "GE4000":
                    mmap_xml = "cfg/camera_GE4000.xml"
                elif arg == "GE4000_raw":
                    mmap_xml = "cfg/camera_GE4000_raw.xml"
                elif arg == "GX6600":
                    mmap_xml = "cfg/camera_GX6600.xml"
                elif arg == "GX6600_raw":
                    mmap_xml = "cfg/camera_GX6600_raw.xml"
                elif arg == "AX5":
                    mmap_xml = "cfg/camera_FLIRAX5.xml"
                elif arg == "GT2460_raw":
                    mmap_xml = "cfg/camera_GT2460_raw.xml"
                elif arg == "GT2460":
                    mmap_xml = "cfg/camera_GT2460.xml"
                # or simply assume its the path to the xml
                elif arg.endswith('.xml'):
                    mmap_xml = arg
                elif arg == "--draw" or arg == "-d":
                    self.do_draw = True
                elif "--drawid=" in arg:
                    try:
                        self.draw_id = int(arg.replace("-drawid=",''))
                    except Exception as e:
                        print(e)
                        print("Invalid --drawid= argument", arg)
                elif "-did=" in arg:
                    try:
                        self.draw_id = int(arg.replace("-did=",''))
                    except Exception as e:
                        print(e)
                        print("Invalid -did= argument", arg)
                elif "--rotation=" in arg:
                    try:
                        self.rotation = float(arg.replace("--rotation=",''))
                        print("Rotation set to %.2f" % self.rotation)
                    except Exception as e:
                        print(e)
                        print("Invalid --rotation argument", arg)
                elif "--fps=" in arg:
                        try:
                            self.framerate = float(arg.replace("--fps=", ''))
                        except Exception as e:
                            print(e)
                            print("Invalid frame rate parameter:", arg)
                else:
                    print("Unknown argument: ",  arg)

        print("wsLiveDisplay")
        print("using drawid: ", self.draw_id)

        # setup cam
        self.cam = GigECam(mmap_xml=mmap_xml)

        # setup gui
        self.setWindowTitle('wsViewer')
        self.move(1240,  0)
        self.layout = QtWidgets.QHBoxLayout(self)

        self.view = QExtendedGraphicsView()
        self.layout.addWidget(self.view)
        self.view.rotate(self.rotation)

        self.pixmap = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap(1344,1024), self.view.origin)
        self.scene_event_filter = GraphicsItemEventFilter(self.view.origin, self)
        self.pixmap.installSceneEventFilter(self.scene_event_filter)
        
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.updateview)
        self.timer.start(1000/ self.framerate)

        # ModuleFocus
        self.module_focus = ModuleFocus(self.view.origin,
                                        self.cam,
                                        [0.4, 0.4, 0.6, 0.6],  # percentage of image dimesnions [ x1 y1 x2 y2]
                                        )
        # # ModuleDraw
        # self.module_draw = ModuleDraw(self.view.origin, self.draw_id)
        #
        # # ModuleTileHelper
        # self.module_tiles = ModuleTileHelper(self.view.origin, 'cfg/ws_config.yml')

    def updateview(self):
        t_start = time.perf_counter()
        img = self.cam.getNewestImage()

        if not img is None:
            self.im = img  # .copy()  # do we really need a .copy() for display purpose ?
            self.image_shape = self.im.shape
            self.pixmap.setPixmap(QtGui.QPixmap(array2qimage(self.im)))
            self.view.setExtend(self.im.shape[1], self.im.shape[0])

            # if focus calculation is enabled calculate the std of the sobel filtered focus region
            if self.do_calcFocus:
                self.module_focus.calcFocus(img)

        # NOTE: I removed all utility modules in this version!

        #    if self.do_draw:
                # print("Draw: \tEnabled: %d\n\tFrame: %d\n" %
                #       (self.module_draw.mem.enable, self.module_draw.mem.frame_nr))

                # self.module_draw.updateElements()



            # output info
            if self.counter % 10 == 0:
                dt = time.perf_counter() - t_start
                print("View update in %.6fs\t" % (dt))
            self.counter += 1

        # update async elements anyway
        if self.do_draw:
            self.module_draw.updateAsyncElements()




    def sceneEventFilter(self, event):
        if event.type() == QtCore.QEvent.GraphicsSceneMousePress and event.button() == QtCore.Qt.LeftButton:
            print(event.pos().x(), event.pos().y())
            return True
        return False

    def keyPressEvent(self, event):
        # image rotation
        if event.key() == QtCore.Qt.Key_R:
            print("## R Rotating screen CW 90°")
            self.view.rotate(90)

        # fit image to screen
        if event.key() == QtCore.Qt.Key_F:
            self.view.fitInView()


        # focus module
        if event.key() == QtCore.Qt.Key_F3:
            self.do_calcFocus = not self.do_calcFocus
            print("## Toggle focus module to", self.do_calcFocus)
            if self.do_calcFocus:
                self.module_focus.showFrame(self.im)
            else:
                self.module_focus.hideFrame()

        # draw module
        # if event.key() == QtCore.Qt.Key_F4:
        #     self.do_draw = not self.do_draw
        #     print("## Toggle draw module to", self.do_draw)
        #     if not self.do_draw:
        #         self.module_draw.clearElements()
        #         self.module_draw.clearAsyncElements()

        # tile module
        # if event.key() == QtCore.Qt.Key_F5:
        #     self.module_tiles.loadCFG()
        #     if not self.image_shape is None:
        #         if self.do_tiles < self.module_tiles.detector_count:
        #             self.do_tiles += 1
        #             print("## Toggle tiles module to \"detector_%d\"" % (self.do_tiles-1))
        #             self.module_tiles.clearTiles()
        #             self.module_tiles.id = self.do_tiles - 1
        #             self.module_tiles.drawTiles(self.image_shape)
        #
        #
        #         else:
        #             print("## Toggle tiles module to OFF")
        #             self.module_tiles.clearTiles()
        #             self.do_tiles = 0

if __name__ == '__main__':

    test = True
    test = False

    if test:
        cam = GigECam(config["output_mmap"])
    else:
        # start the Qt application
        app = QtWidgets.QApplication(sys.argv)

        window = Window()
        window.show()
        sys.exit(app.exec_())
