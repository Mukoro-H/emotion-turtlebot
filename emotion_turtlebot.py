#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nhận diện biểu cảm khuôn mặt qua webcam -> điều khiển TurtleBot qua ROS 2 (/cmd_vel).

    Biểu cảm      Hành động
    happy         Cong sang trái (liên tục khi còn giữ biểu cảm)
    sad           Cong sang phải (liên tục khi còn giữ biểu cảm)
    neutral       Dừng tại chỗ
    angry         Lùi nhẹ (~0.10 m) rồi dừng
    surprise      Xoay đúng 1 vòng (360°) tại chỗ rồi dừng
    fear          Tiến nhẹ (~0.10 m) rồi dừng
    (disgust      -> coi như neutral)
    (mất mặt      -> dừng)

Chạy:
    python3 emotion_turtlebot.py
    python3 emotion_turtlebot.py --ros-args -p backend:=fer -p cmd_vel_topic:=/cmd_vel

Phím tắt trong cửa sổ camera:
    q / ESC : thoát (robot được dừng trước khi thoát)
    SPACE   : bật/tắt dừng khẩn cấp (E-STOP)
"""

import math
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

# --------------------------------------------------------------------------- #
#  Hằng số hiển thị
# --------------------------------------------------------------------------- #
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
TARGET_EMOTIONS = {'happy', 'sad', 'neutral', 'angry', 'surprise', 'fear'}

# cv2.putText không vẽ được chữ có dấu -> dùng tiếng Việt không dấu trên màn hình
LABEL_VI = {
    'happy': 'Vui ve', 'sad': 'Buon', 'neutral': 'Binh thuong',
    'angry': 'Tuc gian', 'surprise': 'Ngac nhien', 'fear': 'So hai',
    'disgust': 'Ghe so',
}
COLORS = {  # BGR
    'happy': (0, 200, 255), 'sad': (255, 120, 0), 'neutral': (200, 200, 200),
    'angry': (0, 0, 255), 'surprise': (255, 0, 255), 'fear': (0, 180, 0),
    'disgust': (0, 100, 100),
}


@dataclass
class Action:
    label: str        # hiển thị trên màn hình
    linear: float     # m/s
    angular: float    # rad/s (dương = quay trái)
    kind: str         # 'continuous' | 'distance' | 'rotation'
    target: float = 0.0  # m (distance) hoặc rad (rotation)


def wrap_angle(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


# --------------------------------------------------------------------------- #
#  Luồng nhận diện biểu cảm (chạy riêng để video không bị giật)
# --------------------------------------------------------------------------- #
class EmotionWorker(threading.Thread):
    """Luôn xử lý khung hình MỚI NHẤT; khung cũ bị bỏ qua nếu mô hình chưa xử lý kịp."""

    def __init__(self, backend: str, detector_backend: str, min_face_conf: float):
        super().__init__(daemon=True)
        self.backend = backend.lower()
        self.detector_backend = detector_backend
        self.min_face_conf = min_face_conf
        self._cond = threading.Condition()
        self._frame = None
        self._result = None          # (timestamp, face_dict | None)
        self._running = True
        self.ready = threading.Event()
        self.error: Optional[Exception] = None
        self.infer_fps = 0.0

    # --- API cho luồng chính ---
    def submit(self, frame):
        with self._cond:
            self._frame = frame
            self._cond.notify()

    def latest(self):
        with self._cond:
            return self._result

    def stop(self):
        with self._cond:
            self._running = False
            self._cond.notify()

    # --- nội bộ ---
    def _load(self):
        if self.backend == 'deepface':
            from deepface import DeepFace
            self._df = DeepFace
            # Warm-up: lần đầu sẽ tải trọng số mô hình về ~/.deepface
            dummy = np.zeros((224, 224, 3), np.uint8)
            DeepFace.analyze(dummy, actions=['emotion'],
                             detector_backend=self.detector_backend,
                             enforce_detection=False, silent=True)
        elif self.backend == 'fer':
            from fer import FER
            self._fer = FER(mtcnn=False)
        else:
            raise ValueError(f"backend không hợp lệ: '{self.backend}' (chọn deepface | fer)")

    def _analyze(self, frame) -> Optional[dict]:
        h, w = frame.shape[:2]
        faces = []
        if self.backend == 'deepface':
            out = self._df.analyze(frame, actions=['emotion'],
                                   detector_backend=self.detector_backend,
                                   enforce_detection=False, silent=True)
            if isinstance(out, dict):
                out = [out]
            for r in out:
                reg = r.get('region', {}) or {}
                x, y = int(reg.get('x', 0)), int(reg.get('y', 0))
                bw, bh = int(reg.get('w', 0)), int(reg.get('h', 0))
                conf = float(r.get('face_confidence', 1.0) or 0.0)
                # enforce_detection=False: không có mặt thì DeepFace trả về cả khung hình, conf = 0
                if conf <= self.min_face_conf or bw <= 0 or (bw >= 0.95 * w and bh >= 0.95 * h):
                    continue
                scores = {k: float(v) for k, v in r['emotion'].items()}  # đã là %
                faces.append({'box': (x, y, bw, bh), 'scores': scores})
        else:  # fer
            for r in self._fer.detect_emotions(frame):
                x, y, bw, bh = [int(v) for v in r['box']]
                scores = {k: float(v) * 100.0 for k, v in r['emotions'].items()}
                faces.append({'box': (x, y, bw, bh), 'scores': scores})

        if not faces:
            return None
        face = max(faces, key=lambda f: f['box'][2] * f['box'][3])  # mặt lớn nhất = gần camera nhất
        face['dominant'] = max(face['scores'], key=face['scores'].get)
        return face

    def run(self):
        try:
            self._load()
        except Exception as e:  # noqa: BLE001
            self.error = e
            self.ready.set()
            return
        self.ready.set()

        while True:
            with self._cond:
                while self._running and self._frame is None:
                    self._cond.wait(0.1)
                if not self._running:
                    return
                frame, self._frame = self._frame, None

            t0 = time.time()
            try:
                face = self._analyze(frame)
            except Exception:  # noqa: BLE001  (lỗi lẻ tẻ 1 khung -> coi như không thấy mặt)
                face = None
            dt = max(time.time() - t0, 1e-3)
            self.infer_fps = 1.0 / dt if self.infer_fps == 0 else 0.8 * self.infer_fps + 0.2 / dt

            with self._cond:
                self._result = (time.time(), face)


# --------------------------------------------------------------------------- #
#  Làm mượt: chỉ đổi biểu cảm khi đa số các lần nhận diện gần nhất đồng ý
# --------------------------------------------------------------------------- #
class EmotionStabilizer:
    def __init__(self, window: int, ratio: float, min_score: float):
        self.hist = deque(maxlen=max(1, window))
        self.need = max(1, math.ceil(ratio * self.hist.maxlen))
        self.min_score = min_score
        self.stable: Optional[str] = None   # None = không có mặt / chưa chắc

    def update(self, face: Optional[dict]) -> Optional[str]:
        if face is None:
            label = None
        else:
            emo = face['dominant']
            if emo not in TARGET_EMOTIONS:
                emo = 'neutral'
            label = emo if face['scores'].get(face['dominant'], 0) >= self.min_score else 'uncertain'
        self.hist.append(label)

        top, count = Counter(self.hist).most_common(1)[0]
        if count >= self.need and top != 'uncertain':
            self.stable = top
        return self.stable


# --------------------------------------------------------------------------- #
#  ROS 2 node điều khiển
# --------------------------------------------------------------------------- #
class EmotionTurtleBot(Node):
    def __init__(self):
        super().__init__('emotion_turtlebot')
        P = lambda n, v: self.declare_parameter(n, v).value  # noqa: E731

        # Nhận diện
        self.backend = P('backend', 'deepface')               # deepface | fer
        self.detector_backend = P('detector_backend', 'opencv')
        self.min_face_conf = float(P('min_face_conf', 0.0))
        self.window = int(P('smoothing_window', 5))
        self.ratio = float(P('smoothing_ratio', 0.6))
        self.min_score = float(P('min_emotion_score', 40.0))  # %
        self.face_lost_timeout = float(P('face_lost_timeout', 1.0))
        # Camera
        self.camera_index = int(P('camera_index', 0))
        self.frame_width = int(P('frame_width', 640))
        self.frame_height = int(P('frame_height', 480))
        # ROS
        cmd_topic = P('cmd_vel_topic', '/cmd_vel')
        odom_topic = P('odom_topic', '/odom')
        self.use_stamped = bool(P('use_stamped', False))      # True cho TurtleBot 4 Jazzy
        rate = float(P('control_rate', 20.0))
        # Chuyển động
        curve_v = float(P('curve_linear', 0.12))
        curve_w = float(P('curve_angular', 0.6))
        step_v = float(P('step_speed', 0.08))
        step_d = float(P('step_distance', 0.10))
        spin_w = float(P('spin_speed', 1.0))

        self.actions = {
            'happy':    Action('Cong sang trai', curve_v, +curve_w, 'continuous'),
            'sad':      Action('Cong sang phai', curve_v, -curve_w, 'continuous'),
            'neutral':  Action('Dung tai cho', 0.0, 0.0, 'continuous'),
            'angry':    Action('Lui nhe', -step_v, 0.0, 'distance', step_d),
            'surprise': Action('Xoay 1 vong', 0.0, spin_w, 'rotation', 2 * math.pi),
            'fear':     Action('Tien nhe', step_v, 0.0, 'distance', step_d),
        }

        self.pub = self.create_publisher(TwistStamped if self.use_stamped else Twist, cmd_topic, 10)
        self.create_subscription(Odometry, odom_topic, self._odom_cb, 10)
        self.create_timer(1.0 / rate, self._control_loop)

        self._lock = threading.Lock()
        self._requested: Optional[str] = None   # do luồng camera ghi
        self._estop = False
        self._status = {'action': 'Khoi dong...', 'lin': 0.0, 'ang': 0.0, 'odom': False, 'estop': False}

        # Trạng thái hành động rời rạc (lùi / tiến / xoay)
        self._active: Optional[str] = None
        self._last_discrete: Optional[str] = None
        self._odom = None
        self._odom_time = 0.0

        self.get_logger().info(
            f"Publish {'TwistStamped' if self.use_stamped else 'Twist'} -> {cmd_topic}, odom <- {odom_topic}")

    # ---- giao tiếp với luồng camera ----
    def set_emotion(self, emo: Optional[str]):
        with self._lock:
            self._requested = emo

    def toggle_estop(self):
        with self._lock:
            self._estop = not self._estop
            state = self._estop
        self.get_logger().warn(f'E-STOP {"BẬT" if state else "TẮT"}')

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    # ---- odom ----
    def _odom_cb(self, msg: Odometry):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self._odom = (p.x, p.y, yaw)
        self._odom_time = time.time()

    def _odom_ok(self) -> bool:
        return self._odom is not None and time.time() - self._odom_time < 0.5

    # ---- hành động rời rạc ----
    def _start_discrete(self, emo: str):
        act = self.actions[emo]
        self._active = emo
        self._t0 = time.time()
        self._use_odom = self._odom_ok()
        speed = abs(act.linear) if act.kind == 'distance' else abs(act.angular)
        self._expected = act.target / speed           # thời gian lý thuyết
        self._timeout = self._expected * 2.0 + 2.0     # chặn trên khi dùng odom
        if self._use_odom:
            self._x0, self._y0, self._prev_yaw = self._odom
            self._yaw_acc = 0.0
        self.get_logger().info(
            f"Bắt đầu: {act.label} ({'odom' if self._use_odom else 'theo thời gian'})")

    def _discrete_progress(self, act: Action) -> float:
        """Trả về tiến độ 0..1."""
        elapsed = time.time() - self._t0
        if self._use_odom:
            if elapsed > self._timeout or not self._odom_ok():
                self.get_logger().warn('Odom quá hạn/mất tín hiệu -> kết thúc hành động')
                return 1.0
            x, y, yaw = self._odom
            if act.kind == 'distance':
                return math.hypot(x - self._x0, y - self._y0) / act.target
            d = wrap_angle(yaw - self._prev_yaw)
            self._prev_yaw = yaw
            self._yaw_acc += d * math.copysign(1.0, act.angular)
            return (self._yaw_acc + 0.03) / act.target   # dung sai ~2°
        return elapsed / self._expected

    # ---- vòng điều khiển (timer, chạy trong luồng spin) ----
    def _control_loop(self):
        with self._lock:
            req, estop = self._requested, self._estop

        lin = ang = 0.0
        if estop:
            self._active = None
            label = 'E-STOP'
        elif self._active is not None:
            act = self.actions[self._active]
            prog = self._discrete_progress(act)
            if prog >= 1.0:
                self.get_logger().info(f'Hoàn thành: {act.label}')
                self._active = None
                label = f'Xong: {act.label}'
            else:
                lin, ang = act.linear, act.angular
                label = f'{act.label} {int(prog * 100)}%'
        elif req is None:
            self._last_discrete = None
            label = 'Khong thay mat -> Dung'
        else:
            act = self.actions[req]
            if act.kind == 'continuous':
                self._last_discrete = None
                lin, ang = act.linear, act.angular
                label = act.label
            elif req != self._last_discrete:
                # Mỗi "lượt" biểu cảm chỉ kích hoạt 1 lần; muốn lặp lại thì đổi biểu cảm khác rồi quay lại
                self._last_discrete = req
                self._start_discrete(req)
                lin, ang = act.linear, act.angular
                label = act.label
            else:
                label = f'{act.label} (xong, doi bieu cam)'

        self._publish(lin, ang)
        with self._lock:
            self._status = {'action': label, 'lin': lin, 'ang': ang,
                            'odom': self._odom_ok(), 'estop': estop}

    def _publish(self, lin: float, ang: float):
        t = Twist()
        t.linear.x = float(lin)
        t.angular.z = float(ang)
        if self.use_stamped:
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = 'base_link'
            m.twist = t
            self.pub.publish(m)
        else:
            self.pub.publish(t)

    def stop_robot(self):
        for _ in range(5):
            self._publish(0.0, 0.0)
            time.sleep(0.02)


# --------------------------------------------------------------------------- #
#  Vẽ giao diện
# --------------------------------------------------------------------------- #
def put(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_overlay(frame, face, stable, status, fps, infer_fps, loading):
    h, w = frame.shape[:2]

    # Panel xác suất
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (230, 30 + 22 * len(EMOTIONS)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    put(frame, 'Xac suat bieu cam', (8, 20), 0.5, (255, 255, 255))
    for i, emo in enumerate(EMOTIONS):
        y0 = 32 + 22 * i
        score = face['scores'].get(emo, 0.0) if face else 0.0
        c = COLORS[emo]
        cv2.rectangle(frame, (95, y0), (95 + int(1.2 * score), y0 + 14), c, -1)
        put(frame, LABEL_VI[emo], (8, y0 + 12), 0.42, c)
        put(frame, f'{score:4.0f}', (190, y0 + 12), 0.42)

    # Khung mặt + nhãn (vẽ sau panel để luôn nằm trên cùng)
    if face is not None:
        x, y, bw, bh = face['box']
        emo = face['dominant']
        c = COLORS.get(emo, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), c, 2)
        put(frame, f"{LABEL_VI.get(emo, emo)} {face['scores'][emo]:.0f}%",
            (x, max(20, y - 8)), 0.7, c, 2)

    # Thanh trạng thái phía dưới
    cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), -1)
    st_txt = LABEL_VI.get(stable, '---') if stable else 'Khong co mat'
    put(frame, f'Bieu cam on dinh: {st_txt}', (10, h - 48), 0.6,
        COLORS.get(stable, (255, 255, 255)), 2)
    act_color = (0, 0, 255) if status['estop'] else (0, 255, 255)
    put(frame, f"Hanh dong: {status['action']}", (10, h - 24), 0.6, act_color, 2)
    put(frame, f"v={status['lin']:+.2f} m/s  w={status['ang']:+.2f} rad/s", (w - 290, h - 48), 0.5)
    put(frame, f"Cam {fps:4.1f} fps | AI {infer_fps:4.1f} fps | odom {'OK' if status['odom'] else '--'}",
        (w - 290, h - 24), 0.45)

    if loading:
        put(frame, 'Dang tai mo hinh nhan dien...', (w // 2 - 170, h // 2), 0.8, (0, 255, 255), 2)
    if status['estop']:
        put(frame, 'E-STOP (SPACE de tiep tuc)', (w // 2 - 170, 60), 0.8, (0, 0, 255), 2)
    return frame


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    rclpy.init(args=sys.argv)
    node = EmotionTurtleBot()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    cap = cv2.VideoCapture(node.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, node.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, node.frame_height)
    if not cap.isOpened():
        node.get_logger().error(f'Không mở được webcam index {node.camera_index}')
        node.stop_robot()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
        return 1

    worker = EmotionWorker(node.backend, node.detector_backend, node.min_face_conf)
    worker.start()
    node.get_logger().info(f'Đang tải mô hình {node.backend} ...')

    stabilizer = EmotionStabilizer(node.window, node.ratio, node.min_score)
    win = 'Emotion -> TurtleBot  (q: thoat, SPACE: E-STOP)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    face, stable, last_ts = None, None, 0.0
    fps, t_prev, read_fail = 0.0, time.time(), 0
    announced = False

    try:
        while rclpy.ok():
            ok, frame = cap.read()
            if not ok:
                read_fail += 1
                node.set_emotion(None)
                if read_fail > 50:
                    node.get_logger().error('Mất tín hiệu webcam')
                    break
                time.sleep(0.02)
                continue
            read_fail = 0
            frame = cv2.flip(frame, 1)  # hiển thị kiểu gương

            if worker.error is not None:
                raise RuntimeError(f'Lỗi tải mô hình: {worker.error}')
            loading = not worker.ready.is_set()
            if not loading:
                if not announced:
                    node.get_logger().info('Mô hình sẵn sàng.')
                    announced = True
                worker.submit(frame.copy())

            res = worker.latest()
            if res is not None and res[0] != last_ts:
                last_ts, face = res
                stable = stabilizer.update(face)

            # An toàn: nếu kết quả nhận diện quá cũ (worker treo) -> coi như không có mặt
            fresh = (time.time() - last_ts) < node.face_lost_timeout
            node.set_emotion(stable if fresh else None)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
            t_prev = now

            draw_overlay(frame, face if fresh else None, stable if fresh else None,
                         node.status(), fps, worker.infer_fps, loading)
            cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord(' '):
                node.toggle_estop()
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Dừng robot và thoát...')
        node.set_emotion(None)
        worker.stop()
        node.stop_robot()
        cap.release()
        cv2.destroyAllWindows()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
