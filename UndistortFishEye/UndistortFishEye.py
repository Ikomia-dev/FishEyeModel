import sys
import numpy
import PyCore
import PyDataProcess
import UndistortFishEye_process as processMod
import UndistortFishEye_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class UndistortFishEye(PyDataProcess.CPluginProcessInterface):

    def __init__(self):
        PyDataProcess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.UndistortFishEyeProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.UndistortFishEyeWidgetFactory()
