import cv2
import numpy as np
import os
import traceback

from CallbacksExt import CallbacksExt
from TDStoreTools import StorageManager
TDF = op.TDModules.mod.TDFunctions

class TDStructuredLight(CallbacksExt):
    """
    TDStructuredLight is an implementation of cv2 structured light
    """
    def __init__(self, ownerComp):
        # the component to which this extension is attached
        self.ownerComp = ownerComp
        # the component to which data is stored
        self.dataComp = ownerComp.op('data')

        # set up structured light grey code patterns
        self.structured_light_gcp = None
        self.patterns_gcp = []

        self.pattern_img = np.random.randint(
            0,
            high=1,
            size=(
                self.ownerComp.par.Resolutionh,
                self.ownerComp.par.Resolutionw,
                4
            ),
            dtype='uint8'
        )

        self.pattern_top = ownerComp.op('pattern')

        self.camera_top = ownerComp.op('camera')

        # init callbacks
        self.callbackDat = self.ownerComp.par.Callbackdat.eval()
        try:
            CallbacksExt.__init__(self, ownerComp)
        except:
            self.ownerComp.addScriptError(traceback.format_exc() + \
                    "Error in CallbacksExt __init__. See textport.")
            print()
            print("Error initializing callbacks - " + self.ownerComp.path)
            print(traceback.format_exc())
        # run onInit callback
        try:
            self.DoCallback('onInit', {'exampleExt':self})
        except:
            self.ownerComp.addScriptError(traceback.format_exc() + \
                    "Error in custom onInit callback. See textport.")
            print(traceback.format_exc())


        # stored items (persistent across saves and re-initialization):
        storedItems = [
            # Only 'name' is required...
            {'name': 'Data', 'default': None, 'readOnly': False,
                                     'property': True, 'dependable': True},
        ]
        self.stored = StorageManager(self, self.dataComp, storedItems)

    # do something in the future
    def _future(self, attrib, args=(), group_name=None, delayMilliSeconds=0, delayFrames=0):
        if group_name == None:
            group_name = self.ownerComp.name
        self.ownerComp.op('future').run(attrib, args, group=group_name, delayFrames=delayFrames, delayMilliSeconds=delayMilliSeconds)

    # kill all runs with group name
    def _killRuns(self, group_name):
        for r in runs:
            if r.group == group_name:
                r.kill()

    
    # structured light stuff


    def create(self, width, height):
        self.structured_light_gcp = cv2.structured_light_GrayCodePattern.create(int(width), int(height))

    def generate_patterns(self):
        success, patterns = self.structured_light_gcp.generate()
        if not success:
            raise ValueError("Pattern generation failed")
        self.patterns_gcp = patterns

    def output_pattern(self, patterns, index):

        # Validate index
        if index < 0 or index >= len(patterns):
            raise ValueError(f"Index out of range. Must be between 0 and {len(patterns)-1}")

        # Assuming patterns are a list of numpy arrays
        pattern_image = patterns[index]

        # Ensure the image is 8-bit before converting
        if pattern_image.dtype != np.uint8:
            pattern_image = np.uint8(pattern_image * 255)

        # Convert the pattern to a format suitable for TouchDesigner
        pattern_td = cv2.cvtColor(pattern_image, cv2.COLOR_GRAY2BGRA)

        self.pattern_img = pattern_td
        self.pattern_top.cook(force=True)

    def capture(self, index):
        capture_folder = project.folder + '/' + self.ownerComp.par.Captures.eval()
        filepath = capture_folder + '/capture_' + str(index) + '.tiff'
        self.camera_top.save(filepath)
        #print(capture_folder)
        pass

    def load_captured_images(self):
        captures_folder = project.folder + '/' + self.ownerComp.par.Captures.eval()
        captured_images = []

        # Check if the folder exists
        if not os.path.exists(captures_folder):
            raise FileNotFoundError(f"The folder {captures_folder} does not exist")

        # Loop through all files in the folder
        for file_name in sorted(os.listdir(captures_folder)):
            if file_name.startswith('capture_') and file_name.endswith('.tiff'):
                # Construct the full file path
                file_path = os.path.join(captures_folder, file_name)

                # Load the image
                image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if image is not None:
                    captured_images.append(image)
                else:
                    print(f"Failed to load image: {file_name}")

        return captured_images

    def decode_images(self):
        captured_images = self.load_captured_images()
        # Ensure there is a sufficient number of images
        if len(captured_images) < self.structured_light_gcp.getNumberOfPatternImages():
            raise ValueError("Not enough captured images for decoding.")

        # Convert the captured images to the format expected by OpenCV
        # OpenCV often expects images in a list of lists where each inner list contains one channel of the image.
        images_list = [ [img] for img in captured_images ]

        # Prepare containers for decoded information
        width, height = captured_images[0].shape[:2]
        decoded_cols = np.zeros((height, width), np.uint16)
        decoded_rows = np.zeros((height, width), np.uint16)

        # Perform the decoding
        # The decode function returns a mask indicating valid decoded pixels.
        mask, decoded_cols, decoded_rows = self.structured_light_gcp.decode(images_list, decoded_cols, decoded_rows)

        # Convert the decoded rows and columns into a depth map or point cloud as needed
        # This conversion is specific to your setup and may require calibration data
        depth_map = self.convert_to_depth(decoded_cols, decoded_rows, mask)

        return depth_map

    def convert_to_depth(self, decoded_cols, decoded_rows, mask):
        # This function should convert the decoded rows and columns into depth information.
        # The implementation will vary greatly depending on the specifics of your setup,
        # including the geometry of the scene, the calibration of the camera and projector, etc.
        # For simplicity, this example returns the raw decoded data.
        return decoded_cols, decoded_rows, mask


    # example method with callback
    def Start(self):
        self.DoCallback('onStart', {'data': 'start'})
        self._future('done', delayFrames=1)

    # example method with callback
    def done(self):
        self.DoCallback('onDone', {'data': 'done'})

    # example pulse parameter handler
    def pulse_Editextension(self):
        self.ownerComp.op('TDStructuredLight').par.edit.pulse()

    def pulse_Create(self):
        self.create(self.ownerComp.par.Resolutionw.eval(), self.ownerComp.par.Resolutionh.eval())

    def pulse_Generatepatterns(self):
        self.generate_patterns()

    def pulse_Outputpattern(self):
        index = self.ownerComp.par.Patternindex.eval()
        self.output_pattern(self.patterns_gcp, int(index))

    def pulse_Capture(self):
        index = self.ownerComp.par.Patternindex.eval()
        self.capture(index)