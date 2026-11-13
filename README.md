# Điều khiển TurtleBot bằng biểu cảm khuôn mặt (ROS 2)

Webcam → DeepFace/FER nhận diện biểu cảm → làm mượt → publish `geometry_msgs/Twist` lên `/cmd_vel`.

| Biểu cảm | Hành động | Kiểu |
|---|---|---|
| Happy | Cong sang trái (v = 0.12 m/s, ω = +0.6 rad/s) | Liên tục khi còn giữ biểu cảm |
| Sad | Cong sang phải (v = 0.12 m/s, ω = −0.6 rad/s) | Liên tục |
| Neutral | Dừng tại chỗ | Liên tục |
| Angry | Lùi 0.10 m rồi dừng | Chạy 1 lần |
| Surprise | Xoay đúng 360° rồi dừng | Chạy 1 lần |
| Fear | Tiến 0.10 m rồi dừng | Chạy 1 lần |
| Disgust / không thấy mặt | Dừng | — |

**Thiết kế chính**
- **Nhận diện chạy ở luồng riêng**: video vẫn mượt dù DeepFace chỉ đạt ~5–10 fps trên CPU.
- **Làm mượt bằng bỏ phiếu đa số**: robot chỉ đổi hành động khi ≥ 60% trong 5 lần nhận diện gần nhất cùng ra một biểu cảm với độ tin cậy ≥ 40%. Nhờ vậy robot không bị giật khi kết quả dao động.
- **Hành động rời rạc dùng odometry (`/odom`)**: lùi/tiến theo quãng đường thực và xoay theo góc thực (tích luỹ yaw). Nếu không có odom thì tự chuyển sang tính theo thời gian. Mỗi lần giữ biểu cảm chỉ chạy một lần; muốn lặp lại thì đổi sang biểu cảm khác rồi quay lại.
- **An toàn**: mất mặt hoặc kết quả quá cũ (> 1 s) thì dừng; phím SPACE là E-STOP; khi thoát, node gửi lệnh vận tốc 0.

## Cài đặt (Ubuntu 22.04 + ROS 2 Humble)

```bash
source /opt/ros/humble/setup.bash
pip install deepface tf-keras opencv-python
# (tuỳ chọn) dùng FER thay DeepFace:
pip install fer
```

> Nếu lỗi kiểu *"A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x"*, chạy `pip install "numpy<2"`.
> Lần chạy đầu tiên DeepFace sẽ tải trọng số mô hình biểu cảm về `~/.deepface`, nên cần có mạng.

## Chạy

**1. Bật robot**

- TurtleBot 4 mô phỏng:
  ```bash
  ros2 launch turtlebot4_ignition_bringup turtlebot4_ignition.launch.py
  # robot khởi động trên dock -> undock trước:
  ros2 action send_goal /undock irobot_create_msgs/action/Undock {}
  ```
- TurtleBot 3 mô phỏng:
  ```bash
  export TURTLEBOT3_MODEL=burger
  ros2 launch turtlebot3_gazebo empty_world.launch.py
  ```
- Robot thật: bringup như bình thường, đảm bảo cùng `ROS_DOMAIN_ID` với máy chạy webcam.

**2. Chạy node** (trên máy có webcam):

```bash
source /opt/ros/humble/setup.bash
python3 emotion_turtlebot.py
```

Ví dụ thêm tham số:

```bash
# Dùng FER, webcam thứ 2
python3 emotion_turtlebot.py --ros-args -p backend:=fer -p camera_index:=1

# TurtleBot 4 có namespace
python3 emotion_turtlebot.py --ros-args -p cmd_vel_topic:=/robot1/cmd_vel -p odom_topic:=/robot1/odom

# TurtleBot 4 trên ROS 2 Jazzy (cmd_vel dùng TwistStamped)
python3 emotion_turtlebot.py --ros-args -p use_stamped:=true
```

Kiểm tra lệnh gửi đi: `ros2 topic echo /cmd_vel`

**Phím tắt:** `q`/`ESC` để thoát · `SPACE` để bật/tắt E-STOP

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `backend` | `deepface` | `deepface` hoặc `fer` |
| `detector_backend` | `opencv` | Bộ dò mặt của DeepFace (`opencv` nhanh; `retinaface`/`mtcnn` chính xác hơn nhưng chậm) |
| `camera_index` | `0` | Chỉ số webcam |
| `frame_width` / `frame_height` | `640` / `480` | Độ phân giải camera |
| `cmd_vel_topic` | `/cmd_vel` | Topic điều khiển |
| `odom_topic` | `/odom` | Topic odometry |
| `use_stamped` | `false` | Publish `TwistStamped` thay vì `Twist` |
| `control_rate` | `20.0` | Tần số publish (Hz) |
| `curve_linear` / `curve_angular` | `0.12` / `0.6` | Vận tốc cho quỹ đạo cong (bán kính ≈ 0.2 m) |
| `step_speed` / `step_distance` | `0.08` / `0.10` | Tốc độ và quãng đường khi tiến/lùi nhẹ |
| `spin_speed` | `1.0` | Tốc độ xoay một vòng (rad/s) |
| `smoothing_window` / `smoothing_ratio` | `5` / `0.6` | Cửa sổ bỏ phiếu và tỉ lệ đồng thuận |
| `min_emotion_score` | `40.0` | Độ tin cậy tối thiểu (%) để chấp nhận một lần nhận diện |
| `face_lost_timeout` | `1.0` | Mất kết quả lâu hơn mức này (s) thì dừng |

## Lưu ý

- Muốn robot phản ứng nhanh hơn thì giảm `smoothing_window` (ví dụ 3); muốn ổn định hơn thì tăng lên.
- "Fear" và "Surprise" dễ bị nhầm với nhau. Hãy biểu cảm rõ (mở to mắt và miệng cho Surprise) và để đủ sáng chính diện.
- TurtleBot 4 / Create 3 thật có cơ chế an toàn hạn chế lùi. Nếu robot không chịu lùi, kiểm tra tham số `safety_override` của Create 3 (ví dụ đặt `backup_only`).
