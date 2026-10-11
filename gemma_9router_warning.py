"""Gemma 4 26B A4B IT webcam alerts through 9Router.

Install: pip install opencv-python requests
Start 9Router first, then:
  export NINEROUTER_API_KEY="your-dashboard-api-key"
  python gemma_9router_warning.py --list-models
  python gemma_9router_warning.py --model "exact-model-id-from-9router" --debug

If --model is omitted, select Gemma 4 26B A4B IT only when its ID is
unambiguous in /v1/models. Images are sent to the configured provider.
Q / Esc exits. No depth estimation or audio output is included.
stdout contains alerts only; diagnostics go to stderr.
"""
import argparse
import base64
import json
import os
import sys
import threading
import time

import cv2
import requests

BASE_URL = "http://127.0.0.1:20128/v1"
MODEL_NAME = "gemma-4-26b-a4b-it"
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
MAX_OUTPUT_TOKENS = 256
# Experimental freshness limit, NOT a validated emergency-warning deadline.
MAX_RESULT_AGE_S = 5.0
REPEAT_ALERT_S = 5.0
# Visual approximation of the path: central 60% of the image, not measured depth.
PATH_LEFT, PATH_RIGHT = 0.2, 0.8
TARGET_OBJECTS = {"person", "table", "chair"}
POSITIONS = {"left", "center", "right"}
VALID_OBSTACLES = {f"{obj} {pos}" for obj in TARGET_OBJECTS for pos in POSITIONS}
PROMPT = (
    "Is a person, table, or chair blocking the walking path in the central 60% of this image? "
    "Ignore other objects. Answer only: person/table/chair followed by left/center/right "
    "for its image position. If none blocks the path, answer none. If unsure, answer unknown."
)


def parse_answer(raw):
    """Require a target class and position; never extract them from a sentence."""
    text = " ".join(raw.strip().lower().strip('". ').split())
    return text if text in VALID_OBSTACLES or text in {"none", "unknown"} else "unknown"


class AlertGate:
    """Suppress repeats; two clear observations rearm, uncertainty does not."""
    def __init__(self):
        self.position = None
        self.last_emitted = float("-inf")
        self.clear_count = 0

    def update(self, answer, age, now):
        if age > MAX_RESULT_AGE_S or answer == "unknown":
            self.clear_count = 0
            return None
        if answer == "none":
            self.clear_count += 1
            if self.clear_count >= 2:
                self.position = None
            return None
        self.clear_count = 0
        if answer != self.position or now - self.last_emitted >= REPEAT_ALERT_S:
            self.position = answer
            self.last_emitted = now
            return {"object": True, "position": answer.split()[1]}
        return None


def log(message):
    print(message, file=sys.stderr, flush=True)


class RouterClient:
    def __init__(self, base_url, api_key, timeout, max_tokens):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.session = requests.Session()
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"

    def request(self, method, path, **kwargs):
        response = self.session.request(
            method, self.base_url + path, timeout=(5, self.timeout), **kwargs)
        if not response.ok:
            hints = {401: "Kiểm tra API key của 9Router.",
                     403: "Kiểm tra quyền truy cập provider/model.",
                     404: "Kiểm tra endpoint và model ID.",
                     429: "Provider đang giới hạn lượt gọi hoặc hết quota."}
            raise RuntimeError(f"HTTP {response.status_code}. " +
                               hints.get(response.status_code, "Kiểm tra log 9Router/provider."))
        return response.json()

    def models(self):
        body = self.request("GET", "/models")
        return [item["id"] for item in body.get("data", [])
                if isinstance(item, dict) and isinstance(item.get("id"), str)]

    def analyze(self, frame, model):
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise RuntimeError("Không mã hóa được ảnh JPEG")
        image_url = "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")
        body = self.request("POST", "/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]}],
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "stream": False,
        })
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError("API không trả về choices; kiểm tra provider/model hỗ trợ ảnh")
        content = choices[0].get("message", {}).get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return " ".join(item.get("text", "") for item in content
                            if isinstance(item, dict) and item.get("type") == "text").strip()
        # Empty/truncated output must never be treated as a clear path.
        return "unknown"

    def close(self):
        self.session.close()


def positive_float(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("Giá trị phải hữu hạn và lớn hơn 0")
    return number


def main():
    parser = argparse.ArgumentParser(description="Gemma via 9Router person/table/chair obstacle alerts")
    parser.add_argument("--debug", action="store_true", help="Show raw responses on stderr")
    parser.add_argument("--base-url", default=os.getenv("NINEROUTER_BASE_URL", BASE_URL))
    parser.add_argument("--model", default=os.getenv("NINEROUTER_MODEL"),
                        help="Exact model ID or combo configured in 9Router")
    parser.add_argument("--list-models", action="store_true", help="List model IDs on stderr, then exit")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--timeout", type=positive_float, default=20.0, help="API read timeout in seconds")
    parser.add_argument("--interval", type=positive_float, default=1.0,
                        help="Minimum seconds between request starts")
    parser.add_argument("--max-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens phải lớn hơn 0")
    client = RouterClient(args.base_url, os.getenv("NINEROUTER_API_KEY", ""),
                          args.timeout, args.max_tokens)
    try:
        if args.list_models or not args.model:
            models = client.models()
            if args.list_models:
                for model_id in models:
                    log(model_id)
                client.close()
                return 0
            candidates = [model_id for model_id in models
                          if model_id.lower().split("/")[-1].split(":")[0] == MODEL_NAME]
            if len(candidates) != 1:
                raise RuntimeError("Không chọn được duy nhất Gemma 4 26B A4B IT. "
                                   "Chạy --list-models rồi truyền --model với ID chính xác.")
            args.model = candidates[0]
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        log(f"Không kết nối/chọn được model: {exc}")
        client.close()
        return 1
    log(f"9Router: {client.base_url} | Model: {args.model}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        cap.release()
        log("Không mở được webcam. Kiểm tra --camera.")
        client.close()
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    stop = threading.Event()
    condition = threading.Condition()
    # One slot: camera replaces old frames, never queues them for inference.
    state = {"frame": None, "captured": 0.0, "sequence": 0,
             "result": None, "status": "Connecting to 9Router", "fatal": None}

    def worker():
        gate = AlertGate()
        last_sequence = -1
        next_request = 0.0
        failures = 0
        log("Sẵn sàng. Chỉ cảnh báo người/bàn/ghế cản lối đi; Q / Esc để thoát.")
        try:
            while not stop.is_set():
                if stop.wait(max(0.0, next_request - time.perf_counter())):
                    break
                with condition:
                    condition.wait_for(lambda: stop.is_set() or
                        (state["frame"] is not None and state["sequence"] != last_sequence))
                    if stop.is_set():
                        break
                    frame = state["frame"].copy()
                    captured = state["captured"]
                    last_sequence = state["sequence"]
                    state["status"] = "Analyzing via 9Router"
                started = time.perf_counter()
                next_request = started + args.interval
                try:
                    raw = client.analyze(frame, args.model)
                    answer = parse_answer(raw)
                    finished = time.perf_counter()
                    failures = 0
                    if stop.is_set():
                        break
                    age = finished - captured
                    with condition:
                        state["result"] = {"answer": answer, "captured": captured,
                            "elapsed": finished - started,
                            "preview": cv2.resize(frame, (192, 144))}
                        state["status"] = "Running"
                    alert = gate.update(answer, age, finished)
                    if alert is not None:
                        print(json.dumps(alert, separators=(",", ":")), flush=True)
                    if args.debug:
                        log(f"raw={raw[:200]!r} parsed={answer} "
                            f"process={finished-started:.2f}s age={age:.2f}s")
                except (requests.RequestException, RuntimeError, ValueError, KeyError, TypeError) as exc:
                    gate.update("unknown", 0, time.perf_counter())
                    failures += 1
                    delay = min(30.0, 2.0 ** min(failures, 5))
                    with condition:
                        state["result"] = None
                        state["status"] = f"API error - retry in {delay:.0f}s"
                    log(f"Lỗi API: {exc}")
                    next_request = max(next_request, time.perf_counter() + delay)
        except Exception as exc:
            with condition:
                state["fatal"] = str(exc)
            log(f"Lỗi runtime: {exc}")
        finally:
            client.close()

    thread = threading.Thread(target=worker, name="gemma-api")
    thread.start()
    log("Webcam chạy liên tục. Đang kết nối Gemma qua 9Router...")
    try:
        while True:
            ret, frame = cap.read()
            captured = time.perf_counter()  # Receipt time; camera buffering is not measured.
            if not ret:
                log("Không đọc được frame webcam.")
                break
            with condition:
                state["frame"] = frame
                state["captured"] = captured
                state["sequence"] += 1
                result, status, fatal = state["result"], state["status"], state["fatal"]
                condition.notify()
            if fatal is not None:
                break
            lines = ["AUTO | Q / Esc: Quit", status]
            color = (0, 220, 255)
            if result is not None:
                age = captured - result["captured"]
                if age > MAX_RESULT_AGE_S:
                    lines.append("STALE - waiting for a new result")
                elif result["answer"] == "unknown":
                    lines.append("UNCERTAIN - no safety conclusion")
                elif result["answer"] in VALID_OBSTACLES:
                    position = result["answer"].split()[1]
                    lines.append(f"OBSTACLE: {position}")
                lines.append(f"Process: {result['elapsed']:.2f}s | Image age: {age:.2f}s")
            display = frame.copy()
            h, w = display.shape[:2]
            for fraction in (PATH_LEFT, PATH_RIGHT):
                x = int(w * fraction)
                cv2.line(display, (x, 0), (x, h), (120, 120, 120), 1)
            # Preview the exact analyzed frame, not the current live image.
            if result is not None and display.shape[0] >= 180 and display.shape[1] >= 212:
                h, w = display.shape[:2]
                preview = result["preview"].copy()
                for fraction in (PATH_LEFT, PATH_RIGHT):
                    x = int(preview.shape[1] * fraction)
                    cv2.line(preview, (x, 0), (x, preview.shape[0]), (120, 120, 120), 1)
                display[h-154:h-10, w-202:w-10] = preview
                cv2.putText(display, "ANALYZED FRAME", (w-202, h-160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            for i, line in enumerate(lines):
                cv2.putText(display, line, (10, 28 + i * 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 2)
            cv2.imshow("SenseAble - Gemma 9Router Auto", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if cv2.getWindowProperty("SenseAble - Gemma 9Router Auto", cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        with condition:
            condition.notify_all()
        cap.release()
        cv2.destroyAllWindows()
        if thread.is_alive():
            log("Đang chờ yêu cầu API hiện tại kết thúc (hoặc timeout)...")
        thread.join()
        log("Đã thoát.")
    return 1 if state["fatal"] else 0


if __name__ == "__main__":
    sys.exit(main())