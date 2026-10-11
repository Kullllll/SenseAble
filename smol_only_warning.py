"""SmolVLM webcam: experimental person/table/chair visual-path alerts.

Run in the existing virtual environment: python liteRT_webcam.py
Q / Esc exits. No depth estimation or audio output is included.
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import time

import cv2
import litert_lm

MODEL_PATH = os.path.expanduser("~/.litert-lm/models/smolvlm2-500m/model.litertlm")
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
MAX_OUTPUT_TOKENS = 6
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


def main():
    parser = argparse.ArgumentParser(description="Experimental SmolVLM person/table/chair obstacle alerts")
    parser.add_argument("--debug", action="store_true", help="Show raw responses on stderr")
    args = parser.parse_args()
    if not os.path.isfile(MODEL_PATH):
        log(f"Không tìm thấy model: {MODEL_PATH}")
        return
    if hasattr(litert_lm, "set_min_log_severity"):
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        log("Không mở được webcam. Kiểm tra CAMERA_INDEX.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    stop = threading.Event()
    condition = threading.Condition()
    # One slot: camera replaces old frames, never queues them for inference.
    state = {"frame": None, "captured": 0.0, "sequence": 0,
             "result": None, "status": "Loading model", "fatal": None}

    def worker():
        try:
            # Engine and conversations are owned exclusively by this worker.
            with tempfile.TemporaryDirectory(prefix="senseable-") as tmp:
                image_path = os.path.join(tmp, "latest.jpg")
                with litert_lm.Engine(
                    MODEL_PATH, backend=litert_lm.Backend.CPU(),
                    vision_backend=litert_lm.Backend.CPU(),
                ) as engine:
                    log("Model sẵn sàng. Chỉ cảnh báo người/bàn/ghế cản lối đi; Q / Esc để thoát.")
                    gate = AlertGate()
                    last_sequence = -1
                    while not stop.is_set():
                        with condition:
                            condition.wait_for(lambda: stop.is_set() or
                                (state["frame"] is not None and state["sequence"] != last_sequence))
                            if stop.is_set():
                                break
                            frame = state["frame"].copy()
                            captured = state["captured"]
                            last_sequence = state["sequence"]
                            state["status"] = "Analyzing latest frame"
                        started = time.perf_counter()
                        try:
                            # Reuse one temporary image; do not accumulate snapshots.
                            if not cv2.imwrite(image_path, frame):
                                raise RuntimeError("Không ghi được ảnh tạm")
                            with engine.create_conversation(max_output_tokens=MAX_OUTPUT_TOKENS) as conv:
                                response = conv.send_message(litert_lm.Contents.of(
                                    PROMPT, litert_lm.Content.ImageFile(absolute_path=image_path)))
                            raw = " ".join(item.get("text", "") for item in
                                response.get("content", []) if item.get("type", "text") == "text").strip()
                            answer = parse_answer(raw)
                            finished = time.perf_counter()
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
                        except Exception as exc:
                            with condition:
                                state["result"] = None
                                state["status"] = "Inference error"
                            log(f"Lỗi phân tích: {exc}")
                            # Avoid a tight retry loop on persistent errors.
                            stop.wait(0.5)
        except Exception as exc:
            with condition:
                state["fatal"] = str(exc)
            log(f"Lỗi khởi tạo/runtime: {exc}")

    thread = threading.Thread(target=worker, name="smol-inference")
    thread.start()
    log("Webcam chạy liên tục. Đang nạp SmolVLM...")
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
            cv2.imshow("SenseAble - SmolVLM Auto", display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            if cv2.getWindowProperty("SenseAble - SmolVLM Auto", cv2.WND_PROP_VISIBLE) < 1:
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
            log("Đang chờ lượt suy luận hiện tại kết thúc để giải phóng model...")
        thread.join()
        log("Đã thoát.")


if __name__ == "__main__":
    main()