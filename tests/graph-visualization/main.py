import asyncio

import random
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket
import uvicorn
from graph_wrapper import GraphManager


class MyGraph(GraphManager):
    def __init__(self, websocket: WebSocket):
        super().__init__(websocket)
        #self.graph.add_nodes_from([0, 1, 2, 3, 4, 5])
        #self.graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])

    async def add_node(self, node_id: int, label: str) -> None:
        self.graph.add_node(node_id)
        return await self.sync_graph()

    async def remove_node(self, node_id: int) -> None:
        self.graph.remove_node(node_id)
        return await self.sync_graph()

    async def add_edge(self, source: int, target: int) -> None:
        self.graph.add_edge(source, target)
        return await self.sync_graph()

    async def remove_edge(self, source: int, target: int) -> None:
        self.graph.remove_edge(source, target)
        return await self.sync_graph()


async def handle_events(websocket: WebSocket) -> None:
    while True:
        message = await websocket.receive_json()
        print(f"Received message: {message}")

        if message.get("type") == "event":
            print(f"Received event: {message}")


# WebSocket route for handling graph operations
async def ws_channel(websocket: WebSocket) -> None:
    await websocket.accept()
    print("WebSocket connection established.")

    graph_manager = MyGraph(websocket)
    confirmation_listener = asyncio.create_task(graph_manager.handle_confirmations())
    event_listener = asyncio.create_task(handle_events(websocket))

    await graph_manager.initialize_graph()
    await graph_manager.sync_graph()

    # https://github.com/vasturiano/3d-force-graph/blob/master/example/dynamic/index.html
    # https://vasturiano.github.io/3d-force-graph/example/dynamic/
    # https://github.com/vasturiano/force-graph

    for i in range(50):
        await asyncio.sleep(1)
        await graph_manager.add_node(i, f"Node {i}")

        if i >= 1:
            random_node = random.randint(0, i - 1)
            await graph_manager.add_edge(i, random_node)

    await confirmation_listener
    await event_listener


async def visualization(request: Request) -> HTMLResponse:
    with open('templates/index.html', mode='r') as f:
        html_content = f.read()
    return HTMLResponse(html_content)


async def example(request: Request) -> HTMLResponse:
    with open('templates/dynamic.html', mode='r') as f:
        html_content = f.read()
    return HTMLResponse(html_content)

routes = [
    Route("/", endpoint=visualization),
    Route("/example", endpoint=example),
    WebSocketRoute("/ws", endpoint=ws_channel)
]

app = Starlette(debug=True, routes=routes)


if __name__ == '__main__':
    # todo
    #  implement 2d graph
    #  remove await for methods without return value

    # https://chat.openai.com/c/8b9bbd01-8e85-4012-a2f4-39fd02723625
    # https://github.com/vasturiano/3d-force-graph
    # https://github.com/vasturiano/3d-force-graph/blob/master/example/dynamic/index.html

    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
