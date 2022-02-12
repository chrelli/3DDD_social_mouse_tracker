DeepPoseKit Annotator: a toolkit for annotating images with user-defined keypoints
============
This package is part of [DeepPoseKit](https://github.com/jgraving/deepposekit)

Note: This software is still in early-release development. Expect some adventures.

Annotation Hotkeys
------------
* <kbd>+</kbd><kbd>-</kbd> = rescale image by ±10%
* <kbd>left mouse button</kbd> = move active keypoint to cursor location
* <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> = move active keypoint 1px or 10px
* <kbd>space</kbd> = change <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> mode (swaps between 1px or 10px movements)
* <kbd>J</kbd><kbd>L</kbd> = next or previous image
* <kbd><</kbd><kbd>></kbd> = jump 10 images forward or backward
* <kbd>I</kbd>,<kbd>K</kbd> or <kbd>tab</kbd>, <kbd>shift</kbd>+<kbd>tab</kbd> = switch active keypoint
* <kbd>R</kbd> = mark image as unannotated ("reset")
* <kbd>F</kbd> = mark image as annotated ("finished")
* <kbd>esc</kbd> or <kbd>Q</kbd> = quit

License
------------
Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/deepposekit-annotator/blob/master/LICENSE.md) for details.

Installation
------------

Install the development version:
```bash
pip install git+https://www.github.com/jgraving/deepposekit-annotator.git
```

