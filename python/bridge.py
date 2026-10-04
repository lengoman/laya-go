"""Sidecar bridge between laya-go and the laya Python package.

laya-go embeds this file and runs it as `python3 -u -c <source> <options-json>`, then speaks
newline-delimited JSON over stdin and stdout: one request object per line in, one response
object per line out. It is also runnable by hand, which is the easiest way to see what the Go
side sees:

    $ pip install laya
    $ echo '{"op":"route","state":"मुझे दो बार शुल्क लिया गया"}' | python3 python/bridge.py

Requests
    {"op": "predict", "state": ..., "questions": {...}, "model": "", "task": "", "lang": "", "min_confidence": 0.8}
    {"op": "route",   ... the same fields, answered without a forward pass}
    {"op": "warm",    "models": ["english", "multilingual"]}
    {"op": "ping"}
    {"op": "shutdown"}

Responses
    {"ok": true,  "result": {...}}
    {"ok": false, "error": {"type": "ValueError", "message": "...", "traceback": "..."}}

Anything laya prints -- the CUDA fallback warning, a download bar -- would corrupt that stream,
so stdout is rebound to stderr before laya is imported and the real stdout is kept private to
the protocol.
"""

import json
import sys
import traceback

_PROTOCOL = sys.stdout
sys.stdout = sys.stderr


def send(payload):
    _PROTOCOL.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    _PROTOCOL.flush()


def failure(exc):
    return {
        "ok": False,
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
    }


def blank_to_none(value):
    """Treat Go's zero value for a string the way Python treats an absent argument."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class Bridge:
    def __init__(self, options):
        from laya.router import Router

        self.config = options.get("config") or {}
        self.configured = set()
        self.router = Router(
            device=blank_to_none(options.get("device")),
            token=blank_to_none(options.get("token")),
            max_loaded=int(options.get("max_loaded") or 1),
            default=blank_to_none(options.get("default")) or "english",
            auto_task_detection=bool(options.get("auto_task_detection")),
            standalone_repos=bool(options.get("standalone_repos")),
        )
        preload = options.get("preload") or []
        if preload:
            self.router.preload(list(preload))

    def agent(self, name):
        """Load a checkpoint and apply any config overrides exactly once."""
        agent = self.router.load(name)
        key = id(agent)
        if self.config and key not in self.configured:
            agent.cfg.update(self.config)
            self.configured.add(key)
        return agent

    def route(self, request):
        return dict(
            self.router.route(
                request.get("state"),
                request.get("questions") or {},
                model=blank_to_none(request.get("model")),
                task=blank_to_none(request.get("task")),
                lang=blank_to_none(request.get("lang")),
                lang_guess=blank_to_none(request.get("lang_guess")),
            )
        )

    def predict(self, request):
        questions = request.get("questions") or {}
        if not questions:
            raise ValueError("no questions")
        kwargs = {}
        if request.get("model") is not None:
            kwargs["model"] = blank_to_none(request.get("model"))
        if request.get("task") is not None:
            kwargs["task"] = blank_to_none(request.get("task"))
        if request.get("lang") is not None:
            kwargs["lang"] = blank_to_none(request.get("lang"))
        if request.get("lang_guess") is not None:
            kwargs["lang_guess"] = blank_to_none(request.get("lang_guess"))
        if request.get("max_len"):
            kwargs["max_len"] = int(request["max_len"])
        if request.get("head_max_len"):
            kwargs["head_max_len"] = int(request["head_max_len"])
        if request.get("min_confidence") is not None:
            kwargs["min_confidence"] = request["min_confidence"]

        return self.router.predict(
            request.get("state"),
            questions,
            **kwargs,
        )

    def warm(self, request):
        models = request.get("models") or None
        self.router.preload(list(models) if models else None)
        return {"loaded": self.router.loaded}

    def ping(self, _request):
        return {"loaded": self.router.loaded, "device": str(getattr(self.router, "device", "") or "auto")}

    def handle(self, request):
        op = request.get("op") or "predict"
        if op == "predict":
            return self.predict(request)
        if op == "route":
            return self.route(request)
        if op == "warm":
            return self.warm(request)
        if op == "ping":
            return self.ping(request)
        raise ValueError("unknown op %r" % op)


def main():
    options = json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else {}

    try:
        import laya  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without laya installed
        payload = failure(exc)
        payload["event"] = "ready"
        send(payload)
        return 1

    try:
        bridge = Bridge(options)
    except Exception as exc:
        payload = failure(exc)
        payload["event"] = "ready"
        send(payload)
        return 1

    send({
        "ok": True,
        "event": "ready",
        "python": sys.version.split()[0],
        "laya": getattr(laya, "__version__", "unknown"),
        "loaded": bridge.router.loaded,
    })

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except Exception as exc:
            send(failure(exc))
            continue
        if request.get("op") == "shutdown":
            return 0
        try:
            send({"ok": True, "result": bridge.handle(request)})
        except Exception as exc:
            send(failure(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
