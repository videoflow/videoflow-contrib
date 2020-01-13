# videoflow-contrib: Videoflow community contributions

[![Build Status](https://travis-ci.org/videoflow/videoflow-contrib.svg?branch=master)](https://travis-ci.org/videoflow/videoflow-contrib)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/videoflow/videoflow-contrib/blob/master/LICENSE)

This library is the official extension repository for the Videoflow library. 
It contains additional consumers, producers, processors, subflows, etc. which are not yet available within Videoflow itself. 
All of these additional modules can be used in conjunction with core Videoflow flows.
This is done in the interest of keeping Videoflow succinct, clean, and simple, with as minimal dependencies to third-party
libraries as necessaries.

This contribution repository is both the proving ground for new functionality, and the archive for functionality that (while useful) may not fit well into the Videoflow paradigm.

## Independent sub-packages
Each folder in the repository corresponds to an individual sub-package that follows the [native namespace package](https://packaging.python.org/guides/packaging-namespace-packages/#native-namespace-packages) Python 3 standard.  The project follows that structure to facilitate per subpackage independent licensing and installation.

See the [Tensorflow Object detection](detector_tf) sub-package for an example of how to structure ``videoflow_contrib`` sub-packages.  Each sub-package should have a ``setup.py`` file and a ``Dockerfile`` that describes the environment needed to use it.

## Example Usage
Consumers, producers and processors from the Videoflow-contrib library are used
in the same way as the components within Videoflow itself.

```
import videoflow
import videoflow.core.flow as flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow_contrib.detector_tf import TensorflowObjectDetector
from videoflow.processors.vision.annotators import BoundingBoxAnnotator
from videoflow.utils.downloader import get_file

BASE_URL_EXAMPLES = "https://github.com/videoflow/videoflow/releases/download/examples/"
VIDEO_NAME = 'intersection.mp4'
URL_VIDEO = BASE_URL_EXAMPLES + VIDEO_NAME

class FrameIndexSplitter(videoflow.core.node.ProcessorNode):
    def __init__(self):
        super(FrameIndexSplitter, self).__init__()
    
    def process(self, data):
        index, frame = data
        return frame

def main():
    input_file = get_file(
        VIDEO_NAME, 
        URL_VIDEO)
    output_file = "output.avi"
    reader = VideofileReader(input_file)
    frame = FrameIndexSplitter()(reader)
    detector = TensorflowObjectDetector()(frame)
    annotator = BoundingBoxAnnotator()(frame, detector)
    writer = VideofileWriter(output_file, fps = 30)(annotator)
    fl = flow.Flow([reader], [writer], flow_type = BATCH)
    fl.run()
    fl.join()

if __name__ == "__main__":
    main()

```


