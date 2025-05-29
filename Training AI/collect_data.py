import cv2
import mediapipe as mp
import csv
import os
import string

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Thư mục lưu dữ liệu
DATA_PATH = "data"
os.makedirs(DATA_PATH, exist_ok=True)

# Danh sách ký tự cần thu thập (chữ cái HOA và số)
characters_alpha = list(string.ascii_uppercase)
characters_numeric = [str(i) for i in range(10)]  # ['0', '1', ..., '9']
characters = characters_alpha + characters_numeric  # Nối hai danh sách

cap = cv2.VideoCapture(0)

# Thay đổi max_num_hands thành 2 để phát hiện cả hai tay
with mp_hands.Hands(
        max_num_hands=2,  # CHO PHÉP NHẬN DIỆN 2 TAY
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
) as hands:
    for char_to_collect in characters:  # Đổi tên biến để rõ ràng hơn
        print(f"--- Bắt đầu thu thập dữ liệu cho ký tự/số: {char_to_collect} ---")
        print("Giơ tay (một hoặc cả hai) trước camera và nhấn 's' để ghi 1 mẫu dữ liệu.")
        print("Mỗi tay được phát hiện sẽ được lưu như một mẫu riêng.")
        print("Nhấn 'n' để chuyển sang ký tự/số tiếp theo.")
        print("Nhấn 'q' để thoát chương trình.")

        file_path = os.path.join(DATA_PATH, f"{char_to_collect}.csv")
        # Mở file CSV ở chế độ 'a' để ghi nối tiếp
        with open(file_path, 'a', newline='') as f:
            csv_writer = csv.writer(f)

            sample_count = 0  # Đếm số mẫu đã thu thập cho ký tự hiện tại
            # Đọc số mẫu đã có nếu file tồn tại và không rỗng
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                with open(file_path, 'r') as temp_f:
                    sample_count = sum(1 for row in temp_f)
            print(f"Đã có {sample_count} mẫu cho '{char_to_collect}'.")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # Vẽ các điểm mốc cho TẤT CẢ các tay được phát hiện
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),  # Màu điểm mốc
                            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)  # Màu đường nối
                        )

                # Hiển thị thông tin
                cv2.putText(frame, f"Collecting '{char_to_collect}' - Samples: {sample_count}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "S: Save | N: Next | Q: Quit",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Data Collection - 2 Hands & Numbers", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Đã thoát chương trình.")
                    exit()

                if key == ord('n'):
                    print(f"Hoàn thành cho '{char_to_collect}'. Chuyển sang ký tự/số tiếp theo.")
                    break  # Thoát vòng lặp while True để sang ký tự tiếp theo

                if key == ord('s'):
                    if results.multi_hand_landmarks:
                        # Lặp qua từng tay được phát hiện và lưu dữ liệu
                        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                            landmarks = []
                            for lm in hand_landmarks.landmark:
                                landmarks.extend([lm.x, lm.y, lm.z])

                            # Thêm nhãn ký tự/số vào cuối
                            landmarks.append(char_to_collect)

                            # Ghi vào file CSV
                            csv_writer.writerow(landmarks)
                            sample_count += 1
                            print(f"Đã lưu mẫu #{sample_count} (tay {hand_idx + 1}) cho ký tự/số '{char_to_collect}'")
                    else:
                        print("Không phát hiện thấy tay nào để lưu mẫu!")

cap.release()
cv2.destroyAllWindows()
print("Hoàn thành toàn bộ quá trình thu thập dữ liệu.")