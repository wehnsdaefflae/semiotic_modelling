import asyncio

import random
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket
import uvicorn
from graph_wrapper import GraphManager


async def visualization(request: Request) -> HTMLResponse:
    with open('templates/index.html', mode='r') as f:
        html_content = f.read()
    return HTMLResponse(html_content)


async def example(request: Request) -> HTMLResponse:
    with open('templates/dynamic.html', mode='r') as f:
        html_content = f.read()
    return HTMLResponse(html_content)


# WebSocket route for handling graph operations
async def main(websocket: WebSocket) -> None:
    await websocket.accept()
    print("WebSocket connection established.")

    graph_manager = GraphManager(websocket)
    await graph_manager.initialize_graph()

    for i in range(50):
        await asyncio.sleep(1)
        nodes = list(graph_manager._graph.nodes)
        await graph_manager.add_node({"Label": f"Node {i}"}, node_id=i)

        if i >= 1:
            random_node = random.choice(nodes)
            print(f"Adding link from {random_node} to {i}")
            await graph_manager.add_link(random_node, i)

            if random.random() < .5:
                await graph_manager.remove_node(i)

    await graph_manager.close()


routes = [
    Route("/", endpoint=visualization),
    Route("/example", endpoint=example),
    WebSocketRoute("/ws", endpoint=main)
]

app = Starlette(debug=True, routes=routes)


if __name__ == '__main__':
    # todo
    #  make visualization code invisible
    #    subclass networkx graphs
    #  mirror networkx functions in visualization
    #  text and fix 3d-force-graph functions
    #
    #  implement 2d graph
    #  remove await for methods without return value

    # https://chat.openai.com/c/8b9bbd01-8e85-4012-a2f4-39fd02723625
    # https://github.com/vasturiano/3d-force-graph
    # https://github.com/vasturiano/3d-force-graph/blob/master/example/dynamic/index.html

    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
