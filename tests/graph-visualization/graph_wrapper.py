import asyncio
import dataclasses
from typing import Literal, Optional, Hashable

import networkx
from starlette.websockets import WebSocket


# https://github.com/vasturiano/3d-force-graph/tree/master

"""
### Input JSON syntax

```json
{
    "nodes": [
        {
          "id": "id1",
          "name": "name1",
          "val": 1
        },
        {
          "id": "id2",
          "name": "name2",
          "val": 10
        },
        ...
    ],
    "links": [
        {
            "source": "id1",
            "target": "id2"
        },
        ...
    ]
}
```
"""


@dataclasses.dataclass
class ConfigOptions:
    """
    Configuration options for the graph visualization.

    controlType: str = "trackball"
    Which type of control to use to control the camera. Choice between
    [trackball](https://threejs.org/examples/misc_controls_trackball.html),
    [orbit](https://threejs.org/examples/#misc_controls_orbit) or
    [fly](https://threejs.org/examples/misc_controls_fly.html).

    rendererConfig: dict[str, any] = {"antialias": True, "alpha": True}
    Configuration parameters to pass to the
    [ThreeJS WebGLRenderer](https://threejs.org/docs/#api/en/renderers/WebGLRenderer) constructor.

    extraRenderers: list[str] = []
    If you wish to include custom objects that require a dedicated renderer besides `WebGL`, such as
    [CSS3DRenderer](https://threejs.org/docs/#examples/en/renderers/CSS3DRenderer), include in this array those extra
    renderer instances.
    """

    controlType: str = "trackball"
    rendererConfig: dict[str, any] = dataclasses.field(default_factory=lambda: {"antialias": True, "alpha": True})
    extraRenderers: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class GraphData:
    nodes: list[dict[str, any]]
    links: list[dict[str, any]]


@dataclasses.dataclass
class Position:
    x: float
    y: float
    z: float


class GraphManager:
    def __init__(self, websocket: WebSocket):
        self._initialized = False

        self._dag_mode: str | None = None
        self.websocket = websocket
        self._graph = networkx.Graph()

        self._last_return_value = None
        self._message_id = 0
        self._confirmation_id = -1
        self._event = asyncio.Event()

        self._message_listener: asyncio.Task | None = None


    async def handle_messages(self) -> None:
        while True:
            message = await self.websocket.receive_json()
            print(f"Received message: {message}")

            if message.get("type") == "confirmation":
                await self._confirmation_handler(message)

            if message.get("type") == "event":
                print(f"Received event: {message}")

    async def start_listeners(self) -> None:
        self._message_listener = asyncio.create_task(self.handle_messages())

    async def _send_command(
            self, command: str, positional_arguments: list[any] | None = None,
            keyword_arguments: dict[str, any] | None = None, is_native: bool = True) -> any:

        """
        Send a command to the frontend and wait for confirmation.
        :param is_native:
        """

        if command != "initGraph" and not self._initialized:
            raise Exception("The graph must be initialized first.")

        if is_native:
            await self.websocket.send_json({
                "type": "nativeCommand",
                "command": command,
                "positionalArguments": positional_arguments or list(),
                "keywordArguments": keyword_arguments or dict(),
                "messageId": self._message_id,
            })
        else:
            await self.websocket.send_json({
                "type": "additionalCommand",
                "command": command,
                "positionalArguments": positional_arguments or list(),
                "keywordArguments": keyword_arguments or dict(),
                "messageId": self._message_id,
            })

        self._confirmation_id = self._message_id  # Expecting this ID in the confirmation
        self._message_id += 1

        print(f"Sent command: {command} with message ID: {self._confirmation_id}. Waiting for confirmation...")

        self._event.clear()  # Reset event for the new command
        await self._event.wait()  # Wait until the event is set by the confirmation handler
        return self._last_return_value

    async def _confirmation_handler(self, message: dict) -> None:
        """Handle incoming messages, check if they confirm the sent messageId."""
        if message.get("messageId") == self._confirmation_id:
            print(f"Received confirmation for message ID: {self._confirmation_id}")
            self._last_return_value = message.get("returnValue")
            self._event.set()  # Set the event to release the waiting command

    async def sync_graph(self) -> None:
        # Sync graph data with the frontend

        graph_data = GraphData(
            nodes=[{"id": n, "label": str(n)} for n in self._graph.nodes()],
            links=[{"source": u, "target": v} for u, v in self._graph.edges()]
        )
        # return await self.set_graph_data(graph_data)
        return await self._send_command("synchronizeGraph", [graph_data.nodes, graph_data.links], is_native=False)

    # initialization
    async def initialize_graph(self, config_options: ConfigOptions | None = None) -> None:
        if self._initialized:
            raise Exception("The graph is already initialized.")

        if self._message_listener is None:
            await self.start_listeners()

        # Initialize the graph with specific configuration
        config_dict = dataclasses.asdict(config_options or ConfigOptions())

        return_value = await self._send_command("initGraph", None, config_dict, is_native=True)
        self._initialized = True
        await self.sync_graph()
        return return_value

    async def close(self) -> None:
        if self._message_listener is not None:
            self._message_listener.cancel()

        await asyncio.gather(self._message_listener, return_exceptions=True)

    # data input
    async def set_graph_data(self, graph_data: GraphData) -> None:
        """
        Setter for graph data structure (see below for syntax details). Can also be used to apply
        [incremental updates](https://bl.ocks.org/vasturiano/2f602ea6c51c664c29ec56cbe2d6a5f6).
        """
        return await self._send_command(
            "graphData", None, dataclasses.asdict(graph_data), is_native=True)

    async def get_graph_data(self) -> GraphData:
        """
        Getter for graph data structure (see below for syntax details).
        """
        graph_dict = await self._send_command("graphData", is_native=True)
        return GraphData(**graph_dict)

    async def json_url(self, url: str) -> None:
        """
        URL of JSON file to load graph data directly from, as an alternative to specifying graphData directly.
        """
        return await self._send_command("jsonUrl", [url], is_native=True)

    async def node_id(self, accessor: str = "id") -> None:
        """
        Node object accessor attribute for unique node id (used in link objects source/target).
        """
        return await self._send_command("nodeId", [accessor], is_native=True)

    async def link_source(self, accessor: str = "source") -> None:
        """
        Link object accessor attribute referring to id of source node. 
        """
        return await self._send_command("linkSource", [accessor], is_native=True)

    async def link_target(self, accessor: str = "target") -> None:
        """
        Link object accessor attribute referring to id of target node.
        """
        return await self._send_command("linkTarget", [accessor], is_native=True)

    # container layout
    async def set_width(self, width: int) -> any:
        """
        Setter for the canvas width.
        """
        return await self._send_command("width", [width], is_native=True)

    async def get_width(self) -> int:
        """
        Getter for the canvas width.
        """
        return await self._send_command("width", is_native=True)

    async def set_height(self, height: int) -> any:
        """
        Setter for the canvas height.
        """
        return await self._send_command("height", [height], is_native=True)

    async def get_height(self) -> int:
        """
        Getter for the canvas height.
        """
        return await self._send_command("height", is_native=True)

    async def set_background_color(self, color: str = "#000011") -> None:
        """
        Setter for the chart background color.
        """
        return await self._send_command("backgroundColor", [color], is_native=True)

    async def get_background_color(self) -> str:
        """
        Getter for the chart background color.
        """
        return await self._send_command("backgroundColor", is_native=True)

    async def set_nav_info(self, show: bool = True) -> None:
        """
        Setter for whether to show the navigation controls footer info.
        """
        return await self._send_command("showNavInfo", [show], is_native=True)

    async def get_nav_info(self) -> bool:
        """
        Getter for whether to show the navigation controls footer info.
        """
        return await self._send_command("showNavInfo", is_native=True)

    # node styling
    async def set_node_rel_size(self, size: int = 4) -> None:
        """
        Setter for the ratio of node sphere volume (cubic px) per value unit.
        """
        return await self._send_command("nodeRelSize", [size], is_native=True)

    async def get_node_rel_size(self) -> int:
        """
        Getter for the ratio of node sphere volume (cubic px) per value unit.
        """
        return await self._send_command("nodeRelSize", is_native=True)

    async def node_val(self, value: str | int | float = "val") -> None:
        """
        Node object accessor function, attribute or a numeric constant for the node numeric value (affects sphere
        volume).
        """
        return await self._send_command("nodeVal", [value], is_native=True)

    async def node_label(self, label: str = "name") -> None:
        """
        Node object accessor function or attribute for name (shown in label). Supports plain text or HTML content.
        Note that this method uses `innerHTML` internally, so make sure to pre-sanitize any user-input content to
        prevent XSS vulnerabilities.
        """
        return await self._send_command("nodeLabel", [label], is_native=True)

    async def node_visibility(self, visibility: bool = "True") -> None:
        """
        Node object accessor function, attribute or a boolean constant for whether to display the node.
        """
        return await self._send_command("nodeVisibility", [visibility], is_native=True)

    async def node_color(self, color: str = "color") -> None:
        """
        Node object accessor function or attribute for node color (affects sphere color). 
        """
        return await self._send_command("nodeColor", [color], is_native=True)

    async def node_auto_color_by(self, node_property: str) -> None:
        """
        Node object accessor function (`fn(node)`) or attribute (e.g. `'type'`) to automatically group colors by. Only
        affects nodes without a color attribute.
        """
        return await self._send_command("nodeAutoColorBy", [node_property], is_native=True)

    async def set_node_opacity(self, opacity: float = .75) -> None:
        """
        Setter for the node sphere opacity, between [0, 1].
        """
        return await self._send_command("nodeOpacity", [opacity], is_native=True)

    async def get_node_opacity(self) -> float:
        """
        Getter for the node sphere opacity, between [0, 1].
        """
        return await self._send_command("nodeOpacity", is_native=True)

    async def set_node_resolution(self, resolution: int = 8) -> None:
        """
        Setter for the geometric resolution of each node, expressed in how many slice segments to divide the
        circumference. Higher values yield smoother spheres.
        """
        return await self._send_command("nodeResolution", [resolution], is_native=True)

    async def get_node_resolution(self) -> int:
        """
        Getter for the geometric resolution of each node, expressed in how many slice segments to divide the
        circumference. Higher values yield smoother spheres.
        """
        return await self._send_command("nodeResolution", is_native=True)

    async def node_three_object(self, object3d: str) -> None:
        """
        Node object accessor function or attribute for generating a custom 3d object to render as graph nodes. Should
        return an instance of [ThreeJS Object3d](https://threejs.org/docs/index.html#api/core/Object3D). If a falsy
        value is returned, the default 3d object type will be used instead for that node.
        """
        return await self._send_command("nodeThreeObject", [object3d], is_native=True)

    async def node_three_object_extend(self, extend: bool = False) -> None:
        """
        Node object accessor function, attribute or a boolean value for whether to replace the default node when
        using a custom nodeThreeObject (`false`) or to extend it (`true`).
        """
        return await self._send_command("nodeThreeObjectExtend", [extend], is_native=True)

    # link styling
    async def link_label(self, label: str = "name") -> None:
        """
        Link object accessor function or attribute for name (shown in label). Supports plain text or HTML content.
        Note that this method uses `innerHTML` internally, so make sure to pre-sanitize any user-input content to
        prevent XSS vulnerabilities.
        """
        return await self._send_command("linkLabel", [label], is_native=True)

    async def link_visibility(self, visibility: bool = True) -> None:
        """
        Link object accessor function, attribute or a boolean constant for whether to display the link line. A value
        of `false` maintains the link force without rendering it.
        """
        return await self._send_command("linkVisibility", [visibility], is_native=True)

    async def link_color(self, color: str = "color") -> None:
        """
        Link object accessor function or attribute for line color. 
        """
        return await self._send_command("linkColor", [color], is_native=True)

    async def link_auto_color_by(self, property_or_function: str) -> None:
        """
        Link object accessor function (`fn(link)`) or attribute (e.g. `'type'`) to automatically group colors by. Only
        affects links without a color attribute.
        """
        return await self._send_command("linkAutoColorBy", [property_or_function], is_native=True)

    async def set_link_opacity(self, opacity: float = 0.2) -> None:
        """
        Setter for line opacity of links, between [0,1].
        """
        return await self._send_command("linkOpacity", [opacity], is_native=True)

    async def get_link_opacity(self) -> float:
        """
        Getter for line opacity of links, between [0,1].
        """
        return await self._send_command("linkOpacity", is_native=True)

    async def link_width(self, width: float = 0) -> None:
        """
        Link object accessor function, attribute or a numeric constant for the link line width. A value of zero will
        render a [ThreeJS Line](https://threejs.org/docs/#api/objects/Line) whose width is constant (`1px`) regardless
        of distance. Values are rounded to the nearest decimal for indexing purposes.
        """
        return await self._send_command("linkWidth", [width], is_native=True)

    async def set_link_resolution(self, resolution: int = 6) -> None:
        """
        Setter for the geometric resolution of each link, expressed in how many radial segments to divide the
        cylinder. Higher values yield smoother cylinders. Applicable only to links with positive width.
        """
        return await self._send_command("linkResolution", [resolution], is_native=True)

    async def get_link_resolution(self) -> int:
        """
        Getter for the geometric resolution of each link, expressed in how many radial segments to divide the
        cylinder. Higher values yield smoother cylinders. Applicable only to links with positive width.
        """
        return await self._send_command("linkResolution", is_native=True)

    async def link_curvature(self, curvature: float = 0) -> None:
        """
        Link object accessor function, attribute or a numeric constant for the curvature radius of the link line.
        Curved lines are represented as 3D bezier curves, and any numeric value is accepted. A value of `0` renders a
        straight line. `1` indicates a radius equal to half of the line length, causing the curve to approximate a
        semi-circle. For self-referencing links (`source` equal to `target`) the curve is represented as a loop around
        the node, with length proportional to the curvature value. Lines are curved clockwise for positive values,
        and counter-clockwise for negative values. Note that rendering curved lines is purely a visual effect and
        does not affect the behavior of the underlying forces.
        """
        return await self._send_command("linkCurvature", [curvature], is_native=True)

    async def link_curve_rotation(self, rotation: float = 0) -> None:
        """
        Link object accessor function, attribute or a numeric constant for the rotation along the line axis to apply
        to the curve. Has no effect on straight lines. At `0` rotation, the curve is oriented in the direction of the
        intersection with the `XY` plane. The rotation angle (in radians) will rotate the curved line clockwise around
        the "start-to-end" axis from this reference orientation.
        """
        return await self._send_command("linkCurveRotation", [rotation], is_native=True)

    async def link_material(self, material: str) -> None:
        """
        Link object accessor function or attribute for specifying a custom material to style the graph links with.
        Should return an instance of [ThreeJS Material](https://threejs.org/docs/#api/materials/Material). If a falsy
        value is returned, the default material will be used instead for that link.
        
        default link material is [MeshLambertMaterial](https://threejs.org/docs/#api/materials/MeshLambertMaterial)
        styled according to `color` and `opacity`.
        """
        return await self._send_command("linkMaterial", [material], is_native=True)

    async def link_three_object(self, object3d: str) -> None:
        """
        Link object accessor function or attribute for generating a custom 3d object to render as graph links. Should
        return an instance of [ThreeJS Object3d](https://threejs.org/docs/index.html#api/core/Object3D). If a falsy
        value is returned, the default 3d object type will be used instead for that link.

        default link object is a line or cylinder, sized according to `width` and styled according to `material`.
        """
        
        return await self._send_command("linkThreeObject", [object3d], is_native=True)

    async def link_three_object_extend(self, extend: bool = False) -> None:
        """
        Link object accessor function, attribute or a boolean value for whether to replace the default link when
        using a custom `linkThreeObject` (`false`) or to extend it (`true`).
        """
        return await self._send_command("linkThreeObjectExtend", [extend], is_native=True)

    async def set_link_position_update(self, update_function: str) -> None:
        """
        Setter for the custom function to call for updating the position of links at every render iteration. It
        receives the respective link `ThreeJS Object3d`, the `start` and `end` coordinates of the link (`{x,y,z}` each),
        and the link's `data`. If the function returns a truthy value, the regular position update function will not
        run for that link.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def get_link_position_update(self) -> str:
        """
        Getter for the custom function to call for updating the position of links at every render iteration. It
        receives the respective link `ThreeJS Object3d`, the `start` and `end` coordinates of the link (`{x,y,z}` each),
        and the link's `data`. If the function returns a truthy value, the regular position update function will not
        run for that link.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def link_directional_arrow_length(self, length: float = 0) -> None:
        """
        Link object accessor function, attribute or a numeric constant for the length of the arrow head indicating
        the link directionality. The arrow is displayed directly over the link line, and points in the direction of
        `source` > `target`. A value of `0` hides the arrow.
        """
        return await self._send_command("linkDirectionalArrowLength", [length], is_native=True)

    async def link_directional_arrow_color(self, color: str = "color") -> None:
        """
        Link object accessor function or attribute for the color of the arrow head. 
        """
        return await self._send_command("linkDirectionalArrowColor", [color], is_native=True)

    async def link_directional_arrow_rel_pos(self, position: float = .5) -> None:
        """
        Link object accessor function, attribute or a numeric constant for the longitudinal position of the arrow
        head along the link line, expressed as a ratio between `0` and `1`, where `0` indicates immediately next to
        the `source` node, `1` next to the `target` node, and `0.5` right in the middle.
        """
        return await self._send_command("linkDirectionalArrowRelPos", [position], is_native=True)

    async def set_link_directional_arrow_resolution(self, resolution: int = 8) -> None:
        """
        Setter for the geometric resolution of the arrow head, expressed in how many slice segments to divide the 
        cone base circumference. Higher values yield smoother arrows.
        """
        return await self._send_command("linkDirectionalArrowResolution", [resolution], is_native=True)

    async def get_link_directional_arrow_resolution(self) -> int:
        """
        Getter for the geometric resolution of the arrow head, expressed in how many slice segments to divide the
        cone base circumference. Higher values yield smoother arrows.
        """
        return await self._send_command("linkDirectionalArrowResolution", is_native=True)

    async def link_directional_particles(self, count: int = 0) -> None:
        """
        Link object accessor function, attribute or a numeric constant for the number of particles (small spheres) to
        display over the link line. The particles are distributed equi-spaced along the line, travel in the direction
        `source` > `target`, and can be used to indicate link directionality.
        """
        return await self._send_command("linkDirectionalParticles", [count], is_native=True)

    async def link_directional_particle_speed(self, speed: float = 0.01) -> None:
        """
        Link object accessor function, attribute or a numeric constant for the directional particles speed,
        expressed as the ratio of the link length to travel per frame. Values above `0.5` are discouraged.
        """
        return await self._send_command("linkDirectionalParticleSpeed", [speed], is_native=True)

    async def link_directional_particle_width(self, width: float = 0.5) -> None:
        """
        Link object accessor function, attribute or a numeric constant for the directional particles width. Values
        are rounded to the nearest decimal for indexing purposes.
        """
        return await self._send_command("linkDirectionalParticleWidth", [width], is_native=True)

    async def link_directional_particle_color(self, color: str = "color") -> None:
        """
        Link object accessor function or attribute for the directional particles color. 
        """
        return await self._send_command("linkDirectionalParticleColor", [color], is_native=True)

    async def set_link_directional_particle_resolution(self, resolution: int = 4) -> None:
        """
        Setter for the geometric resolution of each directional particle, expressed in how many slice segments to
        divide the circumference. Higher values yield smoother particles.
        """
        return await self._send_command("linkDirectionalParticleResolution", [resolution], is_native=True)

    async def get_link_directional_particle_resolution(self) -> int:
        """
        Getter for the geometric resolution of each directional particle, expressed in how many slice segments to
        divide the circumference. Higher values yield smoother particles.
        """
        return await self._send_command("linkDirectionalParticleResolution", is_native=True)

    async def emit_particle(self, edge) -> None:
        """
        An alternative mechanism for generating particles, this method emits a non-cyclical single particle within a
        specific link. The emitted particle shares the styling (speed, width, color) of the regular particle props. A
        valid `link` object that is included in `graphData` should be passed as a single parameter.
        """
        return await self._send_command("emitParticle", [edge], is_native=True)

    # render control
    async def pause_animation(self) -> None:
        """
        Pauses the rendering cycle of the component, effectively freezing the current view and cancelling all user
        interaction. This method can be used to save performance in circumstances when a static image is sufficient.
        """
        return await self._send_command("pauseAnimation", is_native=True)

    async def resume_animation(self) -> None:
        """
        Resumes the rendering cycle of the component, and re-enables the user interaction. This method can be used
        together with `pauseAnimation` for performance optimization purposes.
        """
        return await self._send_command("resumeAnimation", is_native=True)

    async def set_camera_position(
            self, x: float | None = None, y: float | None = None, z: float | None = None,
            position: Position | None = None, duration_ms: int | None = None

    ) -> None:
        """
        Setter for the camera position, in terms of `x`, `y`, `z` coordinates. Each of the coordinates is optional,
        allowing for motion in just some dimensions. The optional second argument can be used to define the direction
        that the camera should aim at, in terms of an `{x,y,z}` point in the 3D space. The 3rd optional argument
        defines the duration of the transition (in ms) to animate the camera motion. A value of 0 (default) moves the
        camera immediately to the final position.

        By default the camera will face the center of the graph at a `z`
        distance proportional to the amount of nodes in the system.
        """
        camera_position = dict()
        if x is not None:
            camera_position["x"] = x
        if y is not None:
            camera_position["y"] = y
        if z is not None:
            camera_position["z"] = z

        arguments = [camera_position]
        if position is not None:
            arguments.append(dataclasses.asdict(position))

        if duration_ms is not None:
            arguments.append(duration_ms)

        return await self._send_command("cameraPosition", arguments, is_native=True)

    async def get_camera_position(self) -> Position:
        """
        Getter for the camera position, in terms of `x`, `y`, `z` coordinates.
        """
        camera_position = await self._send_command("cameraPosition", is_native=True)
        return Position(**camera_position)

    async def zoom_to_fit(self, duration_ms: int = 0, padding_px: int = 10) -> None:
        """
        Automatically moves the camera so that all of the nodes become visible within its field of view,
        aiming at the graph center (0,0,0). If no nodes are found no action is taken. It accepts three optional
        arguments: the first defines the duration of the transition (in ms) to animate the camera motion (default:
        0ms). The second argument is the amount of padding (in px) between the edge of the canvas and the outermost
        node location (default: 10px). The third argument specifies a custom node filter: `node => <boolean>`,
        which should return a truthy value if the node is to be included. This can be useful for focusing on a
        portion of the graph.
        """
        return await self._send_command("zoomToFit", [duration_ms, padding_px], is_native=True)

    async def post_processing_composer(self) -> None:
        """
        Access the [post-processing composer](https://threejs.org/docs/#examples/en/postprocessing/EffectComposer).
        Use this to add post-processing [rendering effects](
        https://github.com/mrdoob/three.js/tree/dev/examples/jsm/postprocessing) to the scene. By default the
        composer has a single pass ([RenderPass](
        https://github.com/mrdoob/three.js/blob/dev/examples/jsm/postprocessing/RenderPass.js)) that directly renders
        the scene without any effects.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def set_lights(self, lights: list) -> None:
        """
        Setter for the list of lights to use in the scene. Each item should be an instance of
        [Light](https://threejs.org/docs/#api/en/lights/Light).

        [AmbientLight](https://threejs.org/docs/?q=ambient#api/en/lights/AmbientLight) +
        [DirectionalLight](https://threejs.org/docs/#api/en/lights/DirectionalLight) (from above)
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def get_lights(self) -> None:
        """
        Getter for the list of lights to use in the scene.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def scene(self) -> None:
        """
        Access the internal ThreeJS [Scene](https://threejs.org/docs/#api/scenes/Scene). Can be used to extend the
        current scene with additional objects not related to 3d-force-graph.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def camera(self) -> None:
        """
        Access the internal ThreeJS [Camera](https://threejs.org/docs/#api/cameras/PerspectiveCamera).
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def renderer(self) -> None:
        """
        Access the internal ThreeJS [WebGL renderer](https://threejs.org/docs/#api/renderers/WebGLRenderer).
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def controls(self) -> None:
        """
        Access the internal ThreeJS controls object.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def refresh(self) -> None:
        """
        Redraws all the nodes/links.
        """
        return await self._send_command("refresh", is_native=True)

    # force engine configuration
    async def set_force_engine(self, engine: str = "d3") -> None:
        """
        Setter for which force-simulation engine to use ([*d3*](https://github.com/vasturiano/d3-force-3d) or
        [*ngraph*](https://github.com/anvaka/ngraph.forcelayout)).
        """
        return await self._send_command("forceEngine", [engine], is_native=True)

    async def get_force_engine(self) -> str:
        """
        Getter for which force-simulation engine to use ([*d3*](https://github.com/vasturiano/d3-force-3d) or
        [*ngraph*](https://github.com/anvaka/ngraph.forcelayout)).
        """
        return await self._send_command("forceEngine", is_native=True)

    async def set_num_dimensions(self, dimensions: int = 3) -> None:
        """
        Setter for number of dimensions to run the force simulation on (1, 2 or 3).
        """
        return await self._send_command("numDimensions", [dimensions], is_native=True)

    async def get_num_dimensions(self) -> int:
        """
        Getter for number of dimensions to run the force simulation on (1, 2 or 3).
        """
        return await self._send_command("numDimensions", is_native=True)

    async def dag_mode(
            self, mode: str = Literal["td", "bu", "lr", "rl", "zout", "zin", "radialout", "radialin"]) -> None:
        """
        Apply layout constraints based on the graph directionality. Only works correctly for [DAG](
        https://en.wikipedia.org/wiki/Directed_acyclic_graph) graph structures (without cycles). Choice between `td`
        (top-down), `bu` (bottom-up), `lr` (left-to-right), `rl` (right-to-left), `zout` (near-to-far),
        `zin` (far-to-near), `radialout` (outwards-radially) or `radialin` (inwards-radially).
        """
        if not self._graph.is_directed():
            raise ValueError("The graph must be directed to use the DAG mode.")
        self._dag_mode = mode
        return await self._send_command("dagMode", [mode], is_native=True)

    async def deg_level_distance(self, distance: int) -> None:
        """
        If `dagMode` is engaged, this specifies the distance between the different graph depths.
        """
        return await self._send_command("dagLevelDistance", [distance], is_native=True)

    async def dag_node_filter(self, fn: callable) -> None:
        """
        Node accessor function to specify nodes to ignore during the DAG layout processing. This accessor method
        receives a node object and should return a `boolean` value indicating whether the node is to be included.
        Excluded nodes will be left unconstrained and free to move in any direction.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def on_dag_error(self, fn: callable) -> None:
        """
        Callback to invoke if a cycle is encountered while processing the data structure for a DAG layout. The loop
        segment of the graph is included for information, as an array of node ids. By default an exception will be
        thrown whenever a loop is encountered. You can override this method to handle this case externally and allow
        the graph to continue the DAG processing. Strict graph directionality is not guaranteed if a loop is
        encountered and the result is a best effort to establish a hierarchy.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def set_d3_alpha_min(self, alpha_min: float = 0.) -> None:
        """
        Setter for the [simulation alpha min](https://github.com/vasturiano/d3-force-3d#simulation_alphaMin) 
        parameter, only applicable if using the d3 simulation engine.
        """
        return await self._send_command("d3AlphaMin", [alpha_min], is_native=True)

    async def get_d3_alpha_min(self) -> float:
        """
        Getter for the [simulation alpha min](https://github.com/vasturiano/d3-force-3d#simulation_alphaMin)
        parameter, only applicable if using the d3 simulation engine. 
        """
        return await self._send_command("d3AlphaMin", is_native=True)

    async def set_d3_alpha_decay(self, alpha_decay: float = .0228) -> None:
        """
        Setter for the [simulation intensity decay](https://github.com/vasturiano/d3-force-3d#simulation_alphaDecay)
        parameter, only applicable if using the d3 simulation engine.
        """
        return await self._send_command("d3AlphaDecay", [alpha_decay], is_native=True)

    async def get_d3_alpha_decay(self) -> float:
        """
        Getter for the [simulation intensity decay](https://github.com/vasturiano/d3-force-3d#simulation_alphaDecay)
        parameter, only applicable if using the d3 simulation engine.
        """
        return await self._send_command("d3AlphaDecay", is_native=True)

    async def set_d3_velocity_decay(self, velocity_decay: float = .4) -> None:
        """
        Setter for the nodes' [velocity decay](https://github.com/vasturiano/d3-force-3d#simulation_velocityDecay) 
        that simulates the medium resistance, only applicable if using the d3 simulation engine.
        """
        return await self._send_command("d3VelocityDecay", [velocity_decay], is_native=True)

    async def get_d3_velocity_decay(self) -> float:
        """
        Getter for the nodes' [velocity decay](https://github.com/vasturiano/d3-force-3d#simulation_velocityDecay)
        that simulates the medium resistance, only applicable if using the d3 simulation engine.
        """
        return await self._send_command("d3VelocityDecay", is_native=True)

    async def set_d3_force(self, force_strength: str) -> None:
        """
        Setter for the internal forces that control the d3 simulation engine. Follows the same interface as
        `d3-force-3d`'s [simulation.force](https://github.com/vasturiano/d3-force-3d#simulation_force). Three forces
        are included by default: `'link'` (based on [forceLink](
        https://github.com/vasturiano/d3-force-3d#forceLink)), `'charge'` (based on [forceManyBody](
        https://github.com/vasturiano/d3-force-3d#forceManyBody)) and `'center'` (based on [forceCenter](
        https://github.com/vasturiano/d3-force-3d#forceCenter)). Each of these forces can be reconfigured,
        or new forces can be added to the system. This method is only applicable if using the d3 simulation engine.
        """
        return await self._send_command("d3Force", [force_strength], is_native=True)

    async def get_d3_force(self) -> str:
        """
        Getter for the internal forces that control the d3 simulation engine.
        """
        return await self._send_command("d3Force", is_native=True)

    async def d3_reheat_simulation(self) -> None:
        """
        Reheats the force simulation engine, by setting the `alpha` value to `1`. Only applicable if using the d3
        simulation engine.
        """
        return await self._send_command("d3ReheatSimulation", is_native=True)

    async def ngraph_physics(self, obj: any) -> None:
        """
        Specify custom physics configuration for ngraph, according to its [configuration object](
        https://github.com/anvaka/ngraph.forcelayout#configuring-physics) syntax. This method is only applicable if
        using the ngraph simulation engine.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def set_warmup_ticks(self, ticks: int = 0) -> None:
        """
        Setter for number of layout engine cycles to dry-run at ignition before starting to render. 
        """
        return await self._send_command("warmupTicks", [ticks], is_native=True)

    async def get_warmup_ticks(self) -> int:
        """
        Getter for number of layout engine cycles to dry-run at ignition before starting to render.
        """
        return await self._send_command("warmupTicks", is_native=True)

    async def set_cooldown_ticks(self, ticks: int = 0) -> None:
        """
        Setter for how many build-in frames to render before stopping and freezing the layout engine. 
        """
        return await self._send_command("cooldownTicks", [ticks], is_native=True)

    async def get_cooldown_ticks(self) -> int:
        """
        Getter for how many build-in frames to render before stopping and freezing the layout engine.
        """
        return await self._send_command("cooldownTicks", is_native=True)

    async def set_cooldown_time(self, time: int = 15000) -> None:
        """
        Setter for how long (ms) to render for before stopping and freezing the layout engine.
        """
        return await self._send_command("cooldownTime", [time], is_native=True)

    async def get_cooldown_time(self) -> int:
        """
        Getter for how long (ms) to render for before stopping and freezing the layout engine.
        """
        return await self._send_command("cooldownTime", is_native=True)

    async def on_engine_tick(self, fn: callable) -> None:
        """
        Callback function invoked at every tick of the simulation engine. 
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def on_engine_stop(self, fn: callable) -> None:
        """
        Callback function invoked when the simulation engine stops and the layout is frozen. 
        """
        raise NotImplementedError("This function is not implemented yet.")

    # interaction
    async def on_node_click(self, fn: callable) -> None:
        """
        Callback function for node (left-button) clicks. The node object and the event object are included as
        arguments `onNodeClick(node, event)`.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def on_node_right_click(self, fn: callable) -> None:
        """
        Callback function for node right-clicks. The node object and the event object are included as arguments
        `onNodeRightClick(node, event)`.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def on_node_hover(self, fn: callable) -> None:
        """
        Callback function for node mouse over events. The node object (or `null` if there's no node under the mouse
        line of sight) is included as the first argument, and the previous node object (or null) as second argument:
        `onNodeHover(node, prevNode)`.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def on_node_drag(self, fn: callable) -> None:
        """
        Callback function for node drag interactions. This function is invoked repeatedly while dragging a node,
        every time its position is updated. The node object is included as the first argument, and the change in
        coordinates since the last iteration of this function are included as the second argument in format {x,y,
        z}: `onNodeDrag(node, translate)`.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def on_node_drag_end(self, fn: callable) -> None:
        """
        Callback function for the end of node drag interactions. This function is invoked when the node is released.
        The node object is included as the first argument, and the entire change in coordinates from initial location
        are included as the second argument in format {x,y,z}: `onNodeDragEnd(node, translate)`.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def on_link_click(self, fn: callable) -> None:
        """
        Callback function for link (left-button) clicks. The link object and the event object are included as
        arguments `onLinkClick(link, event)`.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def on_link_right_click(self, fn: callable) -> None:
        """
        Callback function for link right-clicks. The link object and the event object are included as arguments
        `onLinkRightClick(link, event)`.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def on_link_hover(self, fn: callable) -> None:
        """
        Callback function for link mouse over events. The link object (or `null` if there's no link under the mouse
        line of sight) is included as the first argument, and the previous link object (or null) as second argument:
        `onLinkHover(link, prevLink)`.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def on_background_click(self, fn: callable) -> None:
        """
        Callback function for click events on the empty space between the nodes and links. The event object is
        included as single argument `onBackgroundClick(event)`.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def on_background_right_click(self, fn: callable) -> None:
        """
        Callback function for right-click events on the empty space between the nodes and links. The event object is
        included as single argument `onBackgroundRightClick(event)`.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def link_hover_precision(self, precision: int = 1) -> None:
        """
        Whether to display the link label when gazing the link closely (low value) or from far away (high value).
        """
        return await self._send_command("linkHoverPrecision", [precision], is_native=True)

    async def set_enable_pointer_interaction(self, enable: bool = True) -> None:
        """
        Setter for whether to enable the mouse tracking events. This activates an internal tracker of the canvas
        mouse position and enables the functionality of object hover/click and tooltip labels, at the cost of
        performance. If you're looking for maximum gain in your graph performance it's recommended to switch off this
        property.
        """
        return await self._send_command("enablePointerInteraction", [enable], is_native=True)

    async def get_enable_pointer_interaction(self) -> bool:
        """
        Getter for whether to enable the mouse tracking events.
        """
        return await self._send_command("enablePointerInteraction", is_native=True)

    async def set_enable_node_drag(self, enable: bool = True) -> None:
        """
        Setter for whether to enable the user interaction to drag nodes by click-dragging. Only supported on the `d3`
        force engine. If enabled, every time a node is dragged the simulation is re-heated so the other nodes react
        to the changes. Only applicable if enablePointerInteraction is `true` and using the `d3` force engine.
        """
        return await self._send_command("enableNodeDrag", [enable], is_native=True)

    async def get_enable_node_drag(self) -> bool:
        """
        Getter for whether to enable the user interaction to drag nodes by click-dragging.
        """
        return await self._send_command("enableNodeDrag", is_native=True)

    async def set_enable_navigation_controls(self, enable: bool = True) -> None:
        """
        Setter for whether to enable the trackball navigation controls used to move the camera using mouse
        interactions (rotate/zoom/pan).
        """
        return await self._send_command("enableNavigationControls", [enable], is_native=True)

    async def get_enable_navigation_controls(self) -> bool:
        """
        Getter for whether to enable the trackball navigation controls used to move the camera using mouse
        interactions (rotate/zoom/pan).
        """
        return await self._send_command("enableNavigationControls", is_native=True)

    # utility
    async def get_graph_bbox(
            self, node_filter: Optional[callable] = None) -> dict[Literal["x", "y", "z"], tuple[float, float]] | None:
        """
        Returns the current bounding box of the nodes in the graph, formatted as `{ x: [<num>, <num>], y: [<num>,
        <num>], z: [<num>, <num>] }`. If no nodes are found, returns `null`. Accepts an optional argument to define a
        custom node filter: `node => <boolean>`, which should return a truthy value if the node is to be included.
        This can be useful to calculate the bounding box of a portion of the graph.
        """
        raise NotImplementedError("This function is not implemented yet.")

    async def graph_to_screen_coords(self, x: float, y: float, z: float) -> dict[Literal["x", "y"], float]:
        """
        Utility method to translate node coordinates to the viewport domain. Given a set of `x`,`y`,`z` graph
        coordinates, returns the current equivalent `{x, y}` in viewport coordinates.
        """
        return await self._send_command("graphToScreenCoords", [x, y, z], is_native=True)

    async def screen_to_graph_coords(self, x: float, y: float, distance: float) -> dict[Literal["x", "y", "z"], float]:
        """
        Utility method to translate viewport distance coordinates to the graph domain. Given a pair of `x`,
        `y` screen coordinates and distance from the camera, returns the current equivalent `{x, y, z}` in the domain
        of graph node coordinates.
        """
        return await self._send_command("screenToGraphCoords", [x, y, distance], is_native=True)

    # additional
    async def add_node(self, node: dict, node_id: Hashable | None = None) -> None:
        """
        Add a node to the graph. If `node_id` is provided, any existing node with the same `node_id` will be
        replaced with the new node.
        """
        node_id = node_id or self._graph.number_of_nodes()
        self._graph.add_node(node_id, **node)
        return await self._send_command(
            "addNode", [node], {"node_id": node_id}, is_native=False
        )

    async def add_link(self, source_id: Hashable, target_id: Hashable, link: dict | None = None) -> None:
        """
        Add a link to the graph.
        """
        if link is None:
            self._graph.add_edge(source_id, target_id)

        else:
            if "source" in link:
                raise ValueError("The source key is not allowed in the link dictionary.")

            if "target" in link:
                raise ValueError("The target key is not allowed in the link dictionary.")

            self._graph.add_edge(source_id, target_id, **link)

        return await self._send_command(
            "addLink", [source_id, target_id], keyword_arguments=link, is_native=False
        )

    async def remove_node(self, node_id: Hashable) -> None:
        """
        Remove a node from the graph.
        """
        self._graph.remove_node(node_id)
        return await self._send_command("removeNode", [node_id], is_native=False)

    async def remove_link(self, source_id: Hashable, target_id: Hashable) -> None:
        """
        Remove a link from the graph.
        """
        self._graph.remove_edge(source_id, target_id)
        return await self._send_command(
            "removeLink", [source_id, target_id], is_native=False)
