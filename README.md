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

Videoflow now runs every node as its own worker process wired together through a
message broker (locally via `LocalProcessEngine`, in production on Kubernetes). A
flow is described by a `build_flow()` function that returns a `Flow` built from its
consumer (sink) nodes — producers are discovered automatically by walking the graph
backwards. Give each node a stable `name=` so it can be addressed across processes,
and store/forward `**kwargs` on any custom node so it can be reconstructed inside a
worker.

```python
import videoflow
from videoflow.core import Flow
from videoflow.core.constants import BATCH
from videoflow.consumers import VideofileWriter
from videoflow.producers import VideofileReader
from videoflow.processors.vision.annotators import BoundingBoxAnnotator
from videoflow.utils.downloader import get_file

BASE_URL_EXAMPLES = "https://github.com/videoflow/videoflow/releases/download/examples/"
VIDEO_NAME = 'intersection.mp4'
URL_VIDEO = BASE_URL_EXAMPLES + VIDEO_NAME

class FrameIndexSplitter(videoflow.core.node.ProcessorNode):
    def __init__(self, **kwargs):
        super(FrameIndexSplitter, self).__init__(**kwargs)

    def process(self, data):
        index, frame = data
        return frame

def build_flow():
    from videoflow_contrib.detector_tf import TensorflowObjectDetector
    input_file = get_file(VIDEO_NAME, URL_VIDEO)
    output_file = "output.avi"
    reader = VideofileReader(input_file, name = 'reader')
    frame = FrameIndexSplitter(name = 'frame')(reader)
    detector = TensorflowObjectDetector(name = 'detector')(frame)
    annotator = BoundingBoxAnnotator(name = 'annotator')(frame, detector)
    writer = VideofileWriter(output_file, fps = 30, name = 'writer')(annotator)
    return Flow([writer], flow_type = BATCH)

if __name__ == "__main__":
    # Local run (needs a NATS server): one subprocess per node, talking to NATS.
    from videoflow.engines.local import LocalProcessEngine
    flow = build_flow()
    flow.run(LocalProcessEngine())
    flow.join()

# Deploy to Kubernetes (one workload per node) with the videoflow CLI:
#   videoflow deploy my_flow.py:build_flow --nats nats://nats:4222 --image <your-image>
```


