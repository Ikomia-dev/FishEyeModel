import PyCore
import PyDataProcess
import QtConversion
import UndistortFishEye_process as processMod

#PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CProtocolTaskWidget from Ikomia API
# --------------------
class UndistortFishEyeWidget(PyCore.CProtocolTaskWidget):

    def __init__(self, param, parent):
        PyCore.CProtocolTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = processMod.UndistortFishEyeProcessParam()
        else:
            self.parameters = param

        label_calib_file = QLabel("Calibration file")
        self.edit_calib_file = QLineEdit(self.parameters.calib_file_path)
        browse_btn = QPushButton("...")
        browse_btn.setToolTip("Select calibration file")

        # Mandatory : apply button to launch the process
        applyButton = QPushButton("Apply")

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()
        self.gridLayout.addWidget(label_calib_file, 0, 0)
        self.gridLayout.addWidget(self.edit_calib_file, 0, 1)
        self.gridLayout.addWidget(browse_btn, 0, 2)
        self.gridLayout.addWidget(applyButton, 1, 0, 1, 3)

        # Set clicked signal connection
        browse_btn.clicked.connect(self.on_browse)
        applyButton.clicked.connect(self.on_apply)

        # PyQt -> Qt wrapping
        layoutPtr = QtConversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.setLayout(layoutPtr)

    def on_browse(self):
        filter = "Calibration file(*.txt)"
        calibration_file = self.get_selected_file(filter)
        self.edit_calib_file.setText(calibration_file);

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        self.parameters.calib_file_path = self.edit_calib_file.text()

        # Send signal to launch the process
        self.emitApply(self.parameters)

    def get_selected_file(self, filter):
        file_path = str()
        file_dlg = QFileDialog()
        file_dlg.setFileMode(QFileDialog.ExistingFile)
        file_dlg.setViewMode(QFileDialog.Detail)
        file_dlg.setNameFilter(filter)

        if file_dlg.exec():
            path_list = file_dlg.selectedFiles()
            file_path = path_list[0]

        return file_path


#--------------------
#- Factory class to build process widget object
#- Inherits PyDataProcess.CWidgetFactory from Ikomia API
#--------------------
class UndistortFishEyeWidgetFactory(PyDataProcess.CWidgetFactory):

    def __init__(self):
        PyDataProcess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "UndistortFishEye"

    def create(self, param):
        # Create widget object
        return UndistortFishEyeWidget(param, None)
