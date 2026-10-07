"""SmolVLM webcam: automatic latest-frame main-object recognition.

Run in the existing virtual environment: python liteRT_webcam.py
Q / Esc exits. No depth estimation or audio output is included.
"""
import os
import re
import tempfile
import threading
import time

import cv2
import litert_lm

MODEL_PATH = os.path.expanduser("~/.litert-lm/models/smolvlm2-500m/model.litertlm")
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
MAX_OUTPUT_TOKENS = 8
# Experimental display expiry, NOT a validated emergency-warning deadline.
MAX_RESULT_AGE_S = 5.0
PROMPT = (
    "What is the main object in this image? "
    "Answer only its name in 1 to 3 words, or unknown."
)


def parse_answer(raw):
    """Validate a short label, not its visual correctness. Never extract a guessed label from prose."""
    text = raw.strip().strip('"').strip().rstrip('.').strip()
    match = re.fullmatch(r"[A-Za-z]+(?:[- ][A-Za-z]+){0,2}", text)
    if not match:
        return None
    obj = text.lower()
    prose_words = {"the", "a", "an", "is", "are", "this", "that", "it", "image", "object", "looks"}
    if (set(obj.split()) & prose_words or
            obj in {"none", "uncertain", "unknown", "clear", "no obstacle", "nothing"}):
        return None
    return obj


def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"Không tìm thấy model: {MODEL_PATH}", flush=True)
        return
    if hasattr(litert_lm, "set_min_log_severity"):
        litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        cap.release()
        print("Không mở được webcam. Kiểm tra CAMERA_INDEX.", flush=True)
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
                    print("Model sẵn sàng. Tự động phân tích; Q / Esc để thoát.", flush=True)
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
                            if age > MAX_RESULT_AGE_S:
                                message = "Bỏ kết quả cũ"
                            elif answer is None:
                                message = "UNKNOWN / đầu ra chưa hợp lệ"
                            else:
                                message = f"Nhận diện: {answer}"
                            print(f"{message} | xử lý {finished-started:.2f}s | tuổi ảnh {age:.2f}s", flush=True)
                            print(f"  Output thô: {raw[:200]!r}", flush=True)
                        except Exception as exc:
                            with condition:
                                state["result"] = None
                                state["status"] = "Inference error"
                            print(f"Lỗi phân tích: {exc}", flush=True)
                            # Avoid a tight retry loop on persistent errors.
                            stop.wait(0.5)
        except Exception as exc:
            with condition:
                state["fatal"] = str(exc)
            print(f"Lỗi khởi tạo/runtime: {exc}", flush=True)

    thread = threading.Thread(target=worker, name="smol-inference")
    thread.start()
    print("Webcam chạy liên tục. Đang nạp SmolVLM...", flush=True)
    try:
        while True:
            ret, frame = cap.read()
            captured = time.perf_counter()  # Receipt time; camera buffering is not measured.
            if not ret:
                print("Không đọc được frame webcam.", flush=True)
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
                elif result["answer"] is None:
                    lines.append("UNCERTAIN - no safety conclusion")
                else:
                    lines.append(f"OBJECT: {result['answer']}")
                lines.append(f"Process: {result['elapsed']:.2f}s | Image age: {age:.2f}s")
            display = frame.copy()
            # Preview the exact analyzed frame, not the current live image.
            if result is not None and display.shape[0] >= 180 and display.shape[1] >= 212:
                h, w = display.shape[:2]
                preview = result["preview"].copy()
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
            print("Đang chờ lượt suy luận hiện tại kết thúc để giải phóng model...", flush=True)
        thread.join()
        print("Đã thoát.", flush=True)


if __name__ == "__main__":
    main()