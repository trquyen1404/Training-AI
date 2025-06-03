import cv2
import mediapipe as mp
import os
import string
import time

# --- CÁC THAM SỐ ---
IMAGE_DATA_PATH = "image_data_v2"  # Dùng thư mục mới để tránh lẫn lộn
NUM_SAMPLES_PER_CLASS = 200  # Số lượng ảnh HỢP LỆ cần thu thập cho mỗi lớp
MIN_DETECTION_CONFIDENCE = 0.6  # Giảm nhẹ ngưỡng tin cậy để linh hoạt hơn

# --- MÀU SẮC CHO PHẢN HỒI ---
COLOR_INFO = (255, 255, 0)  # Cyan cho thông tin
COLOR_OK = (0, 255, 0)  # Xanh lá cho thành công
COLOR_ERROR = (0, 0, 255)  # Đỏ cho lỗi/cảnh báo

# Khởi tạo
os.makedirs(IMAGE_DATA_PATH, exist_ok=True)
characters = list(string.ascii_uppercase) + [str(i) for i in range(10)]
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Lỗi: Không thể mở camera.")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=0.5
)

# --- BẮT ĐẦU QUÁ TRÌNH ---
for char_to_collect in characters:
    char_dir = os.path.join(IMAGE_DATA_PATH, char_to_collect)
    os.makedirs(char_dir, exist_ok=True)

    # --- GIAI ĐOẠN 1: CHUẨN BỊ ---
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)

        cv2.putText(frame, f"Chuan bi cho ky tu: '{char_to_collect}'", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    COLOR_INFO, 2)
        cv2.putText(frame, "Nhan 'S' de bat dau. Nhan 'Q' de thoat.", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    COLOR_INFO, 2)
        cv2.imshow("Data Collection - Robust Version", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit("Đã thoát chương trình.")
        if key == ord('s'):
            for i in range(3, 0, -1):
                ret, frame = cap.read();
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Bat dau trong {i}...", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_OK, 3)
                cv2.imshow("Data Collection - Robust Version", frame);
                cv2.waitKey(1000)
            break

    # --- GIAI ĐOẠN 2: TỰ ĐỘNG CHỤP CÓ KIỂM SOÁT CHẤT LƯỢNG ---
    print(f"--- Bat dau thu thap cho '{char_to_collect}' ---")
    sample_count = 0
    while sample_count < NUM_SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        if not ret: break

        frame_for_display = cv2.flip(frame, 1)

        # Xử lý để nhận dạng
        rgb_frame = cv2.cvtColor(frame_for_display, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # --- CỔNG KIỂM SOÁT CHẤT LƯỢNG ---
        if results.multi_hand_landmarks:
            # Nếu tìm thấy tay -> LƯU ẢNH
            mp_drawing.draw_landmarks(frame_for_display, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            # Lưu ảnh gốc (chưa có landmark vẽ lên)
            img_name = os.path.join(char_dir, f'{sample_count}.jpg')
            cv2.imwrite(img_name, frame)

            sample_count += 1

            # Hiển thị trạng thái thành công
            cv2.putText(frame_for_display, f"DA LUU: {sample_count}/{NUM_SAMPLES_PER_CLASS}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_OK, 2)
        else:
            # Nếu không tìm thấy tay -> HIỂN THỊ CẢNH BÁO
            cv2.putText(frame_for_display, "KHONG TIM THAY TAY!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ERROR,
                        2)
            cv2.putText(frame_for_display, "Vui long kiem tra anh sang/goc quay", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, COLOR_ERROR, 2)

        cv2.imshow("Data Collection - Robust Version", frame_for_display)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    if sample_count < NUM_SAMPLES_PER_CLASS:
        print("Đã dừng giữa chừng. Thoát chương trình.")
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
print("--- HOAN THANH TOAN BO QUA TRINH THU THAP DU LIEU ---")