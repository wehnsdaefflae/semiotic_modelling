from starlette.applications import Starlette
from starlette.responses import HTMLResponse
import uvicorn
import networkx as nx
import json

from starlette.routing import Route, WebSocketRoute

# Create a NetworkX graph
G = nx.Graph()
G.add_node("1", name="Node 1")
G.add_node("2", name="Node 2")
G.add_edge("1", "2")


def graph_to_json(graph):
    # Convert graph nodes and edges to a format suitable for the frontend
    nodes = [{"id": node, "name": data['name']} for node, data in graph.nodes(data=True)]
    links = [{"source": u, "target": v} for u, v in graph.edges()]
    return json.dumps({"nodes": nodes, "links": links})


async def homepage(request):
    with open('templates/index.html', 'r') as f:
        html_content = f.read()
    return HTMLResponse(html_content)


async def ws_endpoint(websocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        command = json.loads(data)
        if command['action'] == 'add_node':
            new_id = str(len(G.nodes) + 1)
            G.add_node(new_id, name=f"Node {new_id}")
            G.add_edge(list(G.nodes)[-2], new_id)  # Connect the new node to the last one
            # Send updated graph data
            await websocket.send_text(graph_to_json(G))
        elif command['action'] == 'close':
            await websocket.close()
            break


routes = [
    Route("/", endpoint=homepage),
    WebSocketRoute("/ws", endpoint=ws_endpoint)
]

app = Starlette(debug=True, routes=routes)

if __name__ == '__main__':
    # https://chat.openai.com/c/8b9bbd01-8e85-4012-a2f4-39fd02723625
    # https://github.com/vasturiano/3d-force-graph
    # https://github.com/vasturiano/3d-force-graph/blob/master/example/dynamic/index.html


    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info", reload=True)
