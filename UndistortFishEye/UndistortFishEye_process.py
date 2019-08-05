import PyCore
import PyDataProcess
import copy
import fisheye_module


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CProtocolTaskParam from Ikomia API
# --------------------
class UndistortFishEyeProcessParam(PyCore.CProtocolTaskParam):

    def __init__(self):
        PyCore.CProtocolTaskParam.__init__(self)
        # Place default value initialization here
        self.calib_file_path = str()

    def setParamMap(self, paramMap):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.calib_file_path = paramMap['calibrationFile']

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        paramMap = PyCore.ParamMap()
        paramMap['calibrationFile'] = self.calib_file_path
        return paramMap


# --------------------
# - Class which implements the process
# - Inherits PyCore.CProtocolTask or derived from Ikomia API
# --------------------
class UndistortFishEyeProcess(PyDataProcess.CImageProcess2d):

    def __init__(self, name, param):
        PyDataProcess.CImageProcess2d.__init__(self, name)

        #Create parameters class
        if param is None:
            self.setParam(UndistortFishEyeProcessParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 3

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        param = self.getParam()
        k, d, dims = fisheye_module.load_calibration(param.calib_file_path)

        # Step progress bar:
        self.emitStepProgress()

        img_input = self.getInput(0)
        src_img = img_input.getImage()
        dst_img = fisheye_module.undistort(src_img, k, d, dims)

        # Step progress bar:
        self.emitStepProgress()

        output = self.getOutput(0)
        output.setImage(dst_img)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CProcessFactory from Ikomia API
# --------------------
class UndistortFishEyeProcessFactory(PyDataProcess.CProcessFactory):

    def __init__(self):
        PyDataProcess.CProcessFactory.__init__(self)
        # Set process information as string here
        self.info.name = "UndistortFishEye"
        self.info.description = "This process undistorts images acquired by FishEye camera. " \
                                "It requires a calibration file that models extrinsic parameters of the camera. " \
                                "This file can be generated with the given Python script calibrate.py"
        self.info.authors = "Ikomia"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/FishEye"
        # self.info.iconPath = "your path to a specific icon"
        self.info.keywords = "distortion,fisheye,camera,calibration"

    def create(self, param=None):
        # Create process object
        return UndistortFishEyeProcess(self.info.name, param)
