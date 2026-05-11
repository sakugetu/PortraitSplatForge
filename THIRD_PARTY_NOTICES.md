# Third-Party License Notices

Portrait Splat Forge itself is licensed under the MIT License. That license
applies only to the code in this repository.

This project can optionally call external 3D Gaussian Splatting, view synthesis,
segmentation, and image-generation tools. Those external projects, model
weights, checkpoints, datasets, APIs, and generated assets may be governed by
their own licenses and usage restrictions. Users are responsible for reviewing
and complying with the licenses of any backend, model, checkpoint, dataset, API,
or asset they configure.

This document is a practical notice, not legal advice.

## Apple SHARP

The Apple SHARP backend integration in this repository is only a wrapper. This
repository does not include Apple's SHARP model weights.

Apple's `apple/Sharp` model is distributed under the Apple Machine Learning
Research Model License (`apple-amlr`). That license limits use of the model and
model derivatives to research purposes, described as non-commercial scientific
research and academic development. It does not include commercial exploitation,
product development, or use in a commercial product or service.

If you enable `SHARP_COMMAND`, download Apple's checkpoint, or otherwise use
Apple SHARP, you must comply with Apple's model license and attribution
requirements.

Relevant upstream references:

- Apple SHARP repository: https://github.com/apple/ml-sharp
- Apple SHARP model license: https://huggingface.co/apple/Sharp/blob/main/LICENSE

## Other Optional 3DGS Backends

Portrait Splat Forge includes wrappers or command hooks for optional external
backends such as InstantSplat, VGGT + gsplat, OpenSplat-compatible commands,
GaussianObject-style pipelines, and custom commands.

These projects are not vendored into this repository unless explicitly present
as source files here. Installing or configuring them may require accepting their
own licenses. Some research repositories and model checkpoints may restrict
commercial use, redistribution, or derivative works.

Before using an external backend, check at least:

- the source-code license,
- the model/checkpoint license,
- dataset licenses,
- dependency licenses,
- whether commercial use is allowed,
- whether redistribution of outputs, checkpoints, or derivatives is allowed.

## OpenAI API

OpenAI API usage is optional. If enabled, uploaded images may be sent to the
configured OpenAI model for view generation. Use of OpenAI services is governed
by the applicable OpenAI terms and policies, not by this repository's MIT
License.

## Generated Outputs

The license status of generated `.ply`, `.splat`, `.splat.json`, images, or
videos can depend on:

- the source input media,
- the external backend used,
- model/checkpoint terms,
- API provider terms,
- dataset or asset rights.

Do not assume generated outputs are commercially usable solely because this
repository is MIT licensed.
