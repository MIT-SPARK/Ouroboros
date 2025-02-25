# Spark VLC

## Installation

First install Eigen and pkg-config
```bash
apt install libeigen3-dev pkg-config
```

Then clone this repo, and install the `opengv` submodule with  `git submodule
update --init --recursive`.  Then install Ouroboros by running

```bash
pip install .
```

You can run the tests with
```
pytest --ignore=third_party --ignore=extra/ouroboros_ros
```

Individual components of the VLC pipeline may have extra dependencies. You
probably want to install:

1. `LightGlue` (https://github.com/cvg/LightGlue)
2. `pytorch`, `pytorch-lightning`, `pytorch-metric-learning`, and `torchvision` via pip (needed for Salad, probably others)


See `examples/vlc_server_driver_example.py` for the best example of how to use
the VLC Server abstraction.

## ROS Integration

Ouroboros is integrated with ROS in `extras/ouroboros_ros`, which is a valid
ROS package. Once you have built `ouroboros_ros` in your ROS workspace, you can
run the Ouroboros ROS node with `roslaunch ouroboros_ros
vlc_server_node.launch`. When using the node with you datasets, you will need
to properly remap the following topics:
* `~image_in`
* `~camera_info`
* `~depth_in`

You will also need to specify a couple of coordinate frames by rosparam:
* `fixed_frame` -- Frame to consider "fixed" for pose hints (mostly for GT debugging or baseline comparisons)
* `hint_body_frame` -- Frame to consider body for pose hints
* `body_frame` -- Robot body frame, the frame that loop closures are computed with respect to
* `camera_frame`

## Plugins and Configuration

Ouroboros models the VLC pipeline as
```
Input Image --> Place Recognition --> Keypoint / Descriptor Extraction -->
    --> Query-Match Keypoint Association --> Pose Recovery --> Output Looop Closures
```

The recognition, keypoint/descriptor, association, and pose recovery methods
can all be swapped out. Ouroboros uses a plugin-based system that uses the
provided configuration to decide which module to load for each of these parts
of the pipeline. These plugins are also auto-discovered, meaning that it is
possible to extend Ouroboros without needs to add any new code to the Ouroboros
repo itself.

We will use the Salad place recognition module as an example of how to
implement a custom module. It is implemented
[here](src/ouroboros_salad/salad_model.py). A plugin should be a class (here
`SaladModel`) that takes a configuration struct as an argument in the
constructor. It must implement a function `infer`, which will be called at the
approriate part of the VLC pipeline. For examples of the specific interface
that `infer` must have for each part of the pipeline, check refer to the
"ground truth" example modules [here](src/ouroboros_gt). A plugin should also
have a `load` method to create an instant of the class from a configuration.

Next, we need to define a configuration (here `SaladModelConfig`) which
inherits from `ouroboros.config.Config`. This should be a dataclass with any
configuration information that you would like to set from a file. It needs to
use the `register_config` decorator to declare that it is a plugin of type
`place_model`, with name `Salad`, and can be constructed into a class with
constructor `SaladModel`. The resulting plugin can be loaded from a config file
such as [vlc\_driver\_config.yaml](examples/config/vlc_driver_config.yaml) (see
`VlcDriverConfig.load` in
[vlc\_server\_driver\_example.py](examples/vlc_server_driver_example.py)).
Plugins are automatically discovered in any top-level imports from packages
that start with `ouroboros_`, so you can use plugins with Ouroboros even
without adding them to the Ouroboros codebase.

Note that it is possible to have recursive configurations, see e.g.
`VlcDriverConfig` in
[vlc\_server\_driver\_example.py](examples/vlc_server_driver_example.py).

# Library Building Blocks

The ROS node implementations are the highest-level and most "out-of-the-box"
solutions that we provide, but they are pretty thin wrappers around the
VlcServer abstraction. There is also a lower-level VLC Database abstraction
that handles the storage and querying of images, embeddings, loop closures,
etc. Currently this database abstraction is implemented in a very naive way,
but the implementation should be able to be improved while maintaining exactly
the same interface.


## VlcServer

The VlcServer encapsulates the Visual Loop Closure pipeline.  First, a sensor
needs to be registered with the VlcServer with

```python
session_id = vlc_server.register_camera(robot_id, camera_config, epoch_ns)
```

The `session_id` is a unique identifier for a camera in the current session.
When new frames should be checked and stored for VLC, you can call
`add_and_query_frame`:

```python
loop_closure = vlc_server.add_and_query_frame(session_id, image, epoch_ns)
```

For basic use-cases, this is the only function you should need. However, the
VlcServer also provides slightly lower-level functions for more advanced
functionality, such as storing image embeddings without the actual image which
is a common need in distributed VLC systems.


## VlcDb

`VlcDb` provides an abstraction for storing and querying data related to
sensors, images, and loop closures. Philosophically, `VlcDb` represents a
relational database, although it is currently not implemented with a "real"
database backend. There are four tables:
* Image Table
* Loop Closure Table
* Session Table
* Camera Table

### Image Table

The image table stores the information related to a single image frame. This
includes the RGB(D) image, and any byproducts like embedding vector, keypoints,
descriptors, etc. The Image Table is the most relevant place for making
improvements -- we need to execute vector queries based on the embedding
vectors in this table, and serializing these images to disk is not handled yet.

`add_image` and `get_image` are the most relevant interfaces.

### Loop Closure Table

The loop closure table stores the history of loop closures that the robot has
detected. This is pretty straightforward and not extensively used right now.

`get_lc` and `iterate_lcs` are useful if you want to reason over loop closures found so far.

### Session Table

This tracks each session, which is specific to a camera and continuous period of operation.

See `add_session`.

### Camera Table

This tracks each camera, for example intrinsics and which robot the camera is associated with.

See `add_camera`.

### Queries

The VlcDb provides an interface to several kinds of queries. The primary
purpose of the query is to find images matching the supplied image embedding
vector. However, there are certain filters that we might need to apply to these
queries. For example, often we want the closest match that was observed more
than N seconds ago. There are also cases where we would like to support batch
queries (e.g. in multisession context) to increase speed.

The most VLC-specific query is probably `query_embeddings_max_time`, which
queries for the single closest embedding with time at most T. This is a very
common query when running an online VLC module. There is also
`query_embeddings` to find the closest embedding across any time. There are
several other query functions, which allow for batch queries or filtering based
on custom predicate function. Note that the batch queries *might* be faster
than the single versions, although it depends on when we get around to
optimizing the backend. The filter functions with a custom callable should be
expected to be quite a bit slower than versions that don't use a custom
callable.
