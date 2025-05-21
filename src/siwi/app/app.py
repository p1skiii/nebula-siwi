from flask import Flask
from siwi.app import register_subgraph_routes
from siwi.app import app as flask_app

app = flask_app
register_subgraph_routes(app)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)