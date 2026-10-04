"""Reference Laya HTTP server for laya-go's default transport.

One process holds the checkpoints, every caller shares them. Standard library only, beyond
laya itself, so there is nothing to install and nothing to read before trusting it.

    $ pip install laya
    $ python3 python/server.py --preload english,multilingual --device cuda

    POST /v1/predict  {"state": ..., "questions": {...}, "model": "", "task": "", "lang": "", "min_confidence": 0.8}
    POST /v1/route    the same body, answered without a forward pass
    POST /v1/warm     {"models": ["english"]}
    GET  /healthz

Preloading matters more than it looks: a cold checkpoint build costs seconds, answering costs
tens of milliseconds, and at the default max_loaded=1 traffic that alternates languages
rebuilds a model on every request.
"""

import argparse
import json
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_BODY = 8 << 20


def blank_to_none(value):
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class Engine:
    """A Router plus the lock that keeps one forward pass on the device at a time."""

    def __init__(self, args):
        from laya.router import Router

        self.lock = threading.Lock()
        self.config = {}
        if args.head_max_len:
            self.config["head_max_len"] = args.head_max_len
        if args.max_len:
            self.config["max_len"] = args.max_len
        self.configured = set()
        self.router = Router(
            device=blank_to_none(args.device),
            token=blank_to_none(args.token),
            max_loaded=args.max_loaded,
            default=args.default,
            auto_task_detection=args.auto_task_detection,
            standalone_repos=args.standalone_repos,
        )
        if args.preload:
            names = [n.strip() for n in args.preload.split(",") if n.strip()]
            self.router.preload(names or None)

    def agent(self, name):
        agent = self.router.load(name)
        key = id(agent)
        if self.config and key not in self.configured:
            agent.cfg.update(self.config)
            self.configured.add(key)
        return agent

    def route(self, body):
        return dict(
            self.router.route(
                body.get("state"),
                body.get("questions") or {},
                model=blank_to_none(body.get("model")),
                task=blank_to_none(body.get("task")),
                lang=blank_to_none(body.get("lang")),
                lang_guess=blank_to_none(body.get("lang_guess")),
            )
        )

    def predict(self, body):
        questions = body.get("questions") or {}
        if not questions:
            raise ValueError("no questions")
        kwargs = {}
        if body.get("model") is not None:
            kwargs["model"] = blank_to_none(body.get("model"))
        if body.get("task") is not None:
            kwargs["task"] = blank_to_none(body.get("task"))
        if body.get("lang") is not None:
            kwargs["lang"] = blank_to_none(body.get("lang"))
        if body.get("lang_guess") is not None:
            kwargs["lang_guess"] = blank_to_none(body.get("lang_guess"))
        if body.get("max_len"):
            kwargs["max_len"] = int(body["max_len"])
        if body.get("head_max_len"):
            kwargs["head_max_len"] = int(body["head_max_len"])
        if body.get("min_confidence") is not None:
            kwargs["min_confidence"] = body["min_confidence"]

        with self.lock:
            return self.router.predict(
                body.get("state"),
                questions,
                **kwargs,
            )

    def warm(self, body):
        models = body.get("models") or None
        with self.lock:
            self.router.preload(list(models) if models else None)
        return {"loaded": self.router.loaded}


class Handler(BaseHTTPRequestHandler):
    server_version = "laya-serve"
    engine = None

    def reply(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/healthz", "/v1/healthz"):
            self.reply(200, {"status": "ok", "loaded": self.engine.router.loaded})
            return
        self.reply(404, {"error": "no such endpoint: %s" % self.path})

    def do_POST(self):
        route = {
            "/v1/predict": self.engine.predict,
            "/v1/route": self.engine.route,
            "/v1/warm": self.engine.warm,
        }.get(self.path.rstrip("/"))
        if route is None:
            self.reply(404, {"error": "no such endpoint: %s" % self.path})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self.reply(413, {"error": "body too large"})
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:
            self.reply(400, {"error": "malformed JSON: %s" % exc})
            return

        try:
            self.reply(200, route(body))
        except (ValueError, KeyError, TypeError) as exc:
            self.reply(422, {"error": str(exc), "type": type(exc).__name__})
        except Exception as exc:
            traceback.print_exc()
            self.reply(500, {"error": str(exc), "type": type(exc).__name__})

    def log_message(self, fmt, *args):
        sys.stderr.write("[laya-serve] %s\n" % (fmt % args))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8600)
    parser.add_argument("--device", default=None, help="cuda, cpu or mps; autodetected by default")
    parser.add_argument("--preload", default=None,
                        help="comma-separated checkpoints to build at startup, or 'all'")
    parser.add_argument("--max-loaded", type=int, default=1, dest="max_loaded")
    parser.add_argument("--default", default="english")
    parser.add_argument("--auto-task-detection", action="store_true", dest="auto_task_detection")
    parser.add_argument("--standalone-repos", action="store_true", dest="standalone_repos")
    parser.add_argument("--head-max-len", type=int, default=None, dest="head_max_len",
                        help="option token budget; raise to 512 for more than ~20 options")
    parser.add_argument("--max-len", type=int, default=None, dest="max_len")
    parser.add_argument("--token", default=None, help="Hugging Face token, else $HF_TOKEN")
    args = parser.parse_args()

    if args.preload == "all":
        args.preload = "english,multilingual,typed-decisions"

    Handler.engine = Engine(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("[laya-serve] listening on http://%s:%d (loaded: %s)"
          % (args.host, args.port, Handler.engine.router.loaded or "none yet"), file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
