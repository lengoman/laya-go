"""A bridge that speaks the sidecar protocol without loading a 421M parameter checkpoint.

Everything the Go side does -- the ready handshake, one reply per request, Python exceptions
as errors, an interpreter that dies mid-run -- is exercised here, so the transport is tested
without laya installed and without a GPU.
"""

import json
import sys
import traceback

PROTOCOL = sys.stdout
sys.stdout = sys.stderr

OPTIONS = json.loads(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] else {}
LOADED = list(OPTIONS.get("preload") or [])


def send(payload):
    PROTOCOL.write(json.dumps(payload) + "\n")
    PROTOCOL.flush()


def routing(request):
    return {
        "model": request.get("model") or "english",
        "repo": "convaiinnovations/laya",
        "reason": "stub",
        "detection": None,
        "workflow": None,
    }


def predict(request):
    state = request.get("state")
    if state == "boom":
        raise ValueError("stub failure")
    if state == "oom":
        raise RuntimeError("CUDA out of memory")
    if state == "crash":
        PROTOCOL.close()
        sys.exit(3)

    answers = {}
    for qid, question in (request.get("questions") or {}).items():
        kind = question.get("type")
        if kind == "choice":
            options = list(question.get("criteria") or {})
            answers[qid] = {
                "type": "choice",
                "choice": options[0],
                "probabilities": {name: 1.0 if i == 0 else 0.0 for i, name in enumerate(options)},
                "confidence": 0.9,
                "action": {"act_probability": 0.5},
            }
        elif kind == "score":
            levels = list(question.get("criteria") or [])
            answers[qid] = {
                "type": "score",
                "score": 1.5,
                "legend": {str(i): str(level) for i, level in enumerate(levels)},
                "probabilities": {str(i): 1.0 / max(1, len(levels)) for i in range(len(levels))},
                "confidence": 0.7,
                "action": {"act_probability": 0.5},
            }
        else:
            answers[qid] = {
                "type": "noul",
                "noul": 0.75,
                "confidence": 0.75,
                "action": {"act_probability": 0.5},
            }

    return {
        "model": "laya-stub",
        "answers": answers,
        "usage": {"input_tokens": 42, "output_tokens": 0},
        "routing": routing(request),
        "echo": {"options": OPTIONS, "lang": request.get("lang"), "task": request.get("task")},
    }


def handle(request):
    op = request.get("op") or "predict"
    if op == "predict":
        return predict(request)
    if op == "route":
        return routing(request)
    if op == "warm":
        for name in request.get("models") or ["english"]:
            if name not in LOADED:
                LOADED.append(name)
        return {"loaded": LOADED}
    if op == "ping":
        return {"loaded": LOADED, "device": OPTIONS.get("device") or "auto"}
    raise ValueError("unknown op %r" % op)


send({"ok": True, "event": "ready", "python": sys.version.split()[0], "laya": "stub", "loaded": LOADED})

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    request = json.loads(line)
    if request.get("op") == "shutdown":
        break
    try:
        send({"ok": True, "result": handle(request)})
    except Exception as exc:
        send({
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        })
