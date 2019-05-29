# videoflow-contrib: Videoflow community contributions

[![Build Status](https://travis-ci.org/videoflow/videoflow-contrib.svg?branch=master)](https://travis-ci.org/videoflow/videoflow-contrib)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/videoflow/videoflow-contrib/blob/master/LICENSE)

This library is the official extension repository for the Videoflow library. 
It contains additional consumers, producers, processors, subflows, etc. which are not yet available within Videoflow itself. 
All of these additional modules can be used in conjunction with core Videoflow flows.
This is done in the interest of keeping Videoflow succinct, clean, and simple, with as minimal dependencies to third-party
libraries as necessaries.

This contribution repository is both the proving ground for new functionality, and the archive for functionality that (while useful) may not fit well into the Videoflow paradigm.

## Installation
### Install videoflow_contrib from videoflow/videoflow-contrib

```
git clone https://github.com/videoflow/videoflow-contrib.git
cd videoflow-contrib
python3 setup.py install
```

Alternatively, using pip:

```
sudo pip3 install git+https://github.com/videoflow/videoflow-contrib.git
```

To uninstall:

```
pip3 uninstall videoflow_contrib
```

## Contributing
For contributing guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md). 
Also, whenever you add a new contribution, be sure to also update the [CODEOWNERS](CODEOWNERS) file.  Doing so will, in the future, tag you whenever an issue or a pull request about your feature is opened.

## Example Usage
Consumers, producers and processors from the Videoflow-contrib library are used
in the same way as the components within Videoflow itself.

```
from videoflow.core import Flow
from videoflow.producers import IntProducer
from videoflow_contrib.processors.identity import IdentityProcessor
from videoflow.consumers import CommandlineConsumer

producer = IntProducer(0, 40, 0.1)
identity = IdentityProcessor()(producer)
printer = CommandlineConsumer()(identity)
flow = Flow([producer], [printer])
flow.run()
flow.join()

```


