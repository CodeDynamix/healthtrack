import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys, json, traceback
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# You can tune these
CONF_THRES = 0.25        # detection threshold
MIN_VALID_SCORE = 0.30   # below this -> treated as "not valid food"

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def fail(reason, conf=0.0, debug=None):
    out = {
        "success": False,
        "error": "Please upload or capture a valid food image.",
        "reason": reason,
        "confidence": round(float(conf), 4),
        "script_version": "yolo_tflite_parser_v2"
    }
    if debug is not None:
        out["debug"] = debug
    print(json.dumps(out, ensure_ascii=False))
    sys.exit(0)

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH  = os.path.join(BASE_DIR, "best_float32.tflite")
    LABELS_PATH = os.path.join(BASE_DIR, "labels.txt")

    if len(sys.argv) < 2:
        fail("no_image_path")

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        fail("image_not_found")

    if not os.path.exists(MODEL_PATH):
        fail("model_not_found", debug={"expected_path": MODEL_PATH})

    if not os.path.exists(LABELS_PATH):
        fail("labels_not_found", debug={"expected_path": LABELS_PATH})

    labels = load_labels(LABELS_PATH)

    # Use TensorFlow Lite Interpreter
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    in_det  = interpreter.get_input_details()
    out_det = interpreter.get_output_details()

    in_h = int(in_det[0]["shape"][1])
    in_w = int(in_det[0]["shape"][2])
    in_dtype = in_det[0]["dtype"]

    # Prepare image
    img = Image.open(image_path).convert("RGB").resize((in_w, in_h))
    x = np.array(img)

    if in_dtype == np.float32:
        x = x.astype(np.float32) / 255.0
    elif in_dtype == np.float16:
        x = x.astype(np.float16) / np.float16(255.0)
    else:
        x = x.astype(in_dtype)

    x = np.expand_dims(x, 0)

    interpreter.set_tensor(in_det[0]["index"], x)
    interpreter.invoke()

    out = np.array(interpreter.get_tensor(out_det[0]["index"]))  # [1, C, 8400] usually
    if out.ndim != 3:
        fail("bad_output_shape", debug={"shape": str(out.shape)})

    out = out[0]  # [C, 8400]
    C = int(out.shape[0])

    # ---------------------------------------------------------
    # ✅ Detect output format automatically:
    # Format A: [4 box + 1 obj + nc classes] => C = nc + 5
    # Format B: [4 box + nc classes]         => C = nc + 4
    # ---------------------------------------------------------
    has_obj = None
    nc = None

    # Prefer matching labels count first
    if C - 5 == len(labels):
        has_obj = True
        nc = C - 5
    elif C - 4 == len(labels):
        has_obj = False
        nc = C - 4
    else:
        # If labels mismatch, infer nc from output and truncate labels if needed
        if C > 5:
            # Choose the more common YOLOv8 layout first (nc+4) then (nc+5)
            # We'll pick whichever produces a reasonable class count
            nc4 = C - 4
            nc5 = C - 5
            # pick positive one, prefer nc4
            if nc4 > 0:
                has_obj = False
                nc = nc4
            elif nc5 > 0:
                has_obj = True
                nc = nc5
            else:
                fail("bad_channel_count", debug={"channels": C})
        else:
            fail("bad_channel_count", debug={"channels": C})

        # Fix labels length to not crash
        if len(labels) > nc:
            labels = labels[:nc]
        elif len(labels) < nc:
            # pad missing labels so it still runs
            labels = labels + [f"class_{i}" for i in range(len(labels), nc)]

    # Transpose to [8400, C]
    preds = out.T

    best_score = 0.0
    best_cls = -1

    # Loop all candidates
    for row in preds:
        if has_obj:
            obj = sigmoid(float(row[4]))
            class_start = 5
        else:
            obj = 1.0
            class_start = 4

        # Class scores
        cls_raw = row[class_start:class_start+nc].astype(np.float32)

        # Some exports already output probabilities; sigmoid won't hurt much,
        # but for safety we keep sigmoid to match logits-style output.
        cls_scores = sigmoid(cls_raw)

        cls_id = int(np.argmax(cls_scores))
        cls_conf = float(cls_scores[cls_id])

        score = obj * cls_conf
        if score > best_score:
            best_score = score
            best_cls = cls_id

    if best_cls < 0 or best_score < CONF_THRES:
        fail("no_detection", conf=best_score, debug={
            "best_score": float(best_score),
            "has_obj": has_obj,
            "channels": C,
            "nc": int(nc),
            "labels": len(labels)
        })

    if best_score < MIN_VALID_SCORE:
        fail("low_confidence_not_food", conf=best_score, debug={
            "best_score": float(best_score),
            "has_obj": has_obj,
            "channels": C,
            "nc": int(nc),
            "labels": len(labels)
        })

    food = labels[int(best_cls)]

    print(json.dumps({
        "success": True,
        "food": food,
        "confidence": round(float(best_score), 4),
        "class_id": int(best_cls),
        "model_classes": int(nc),
        "has_objectness": bool(has_obj),
        "channels": int(C),
        "script_version": "yolo_tflite_parser_v2"
    }, ensure_ascii=False))

except Exception as e:
    print(json.dumps({
        "success": False,
        "error": str(e),
        "trace": traceback.format_exc(),
        "script_version": "yolo_tflite_parser_v2"
    }, ensure_ascii=False))