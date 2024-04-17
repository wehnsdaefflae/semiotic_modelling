import asyncio

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket
import uvicorn
from graph_wrapper import GraphManager


# WebSocket route for handling graph operations
async def ws_channel(websocket: WebSocket) -> None:
    await websocket.accept()
    print("WebSocket connection established.")

    graph_manager = GraphManager(websocket)
    listener_task = asyncio.create_task(graph_manager.handle_websocket_messages())

    await graph_manager.initialize_graph()

    await graph_manager.sync_graph()

    await listener_task


async def homepage(request: Request) -> HTMLResponse:
    with open('templates/index.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(html_content)


routes = [
    Route("/", endpoint=homepage),
    WebSocketRoute("/ws", endpoint=ws_channel)
]

app = Starlette(debug=True, routes=routes)


if __name__ == '__main__':
    # https://chat.openai.com/c/8b9bbd01-8e85-4012-a2f4-39fd02723625
    # https://github.com/vasturiano/3d-force-graph
    # https://github.com/vasturiano/3d-force-graph/blob/master/example/dynamic/index.html

    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
