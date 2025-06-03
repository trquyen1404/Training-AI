import cv2
import mediapipe as mp
import joblib
import time
from collections import deque
import speech_recognition as sr
import pyttsx3
import threading
import numpy as np
import os
import statistics  # SỬ DỤNG THƯ VIỆN NÀY ĐỂ THAY THẾ SCIPY.STATS

# --- Biến cờ và biến trạng thái toàn cục ---
speech_thread_active = False
tts_thread_active = False
text_display = ""
current_mode = None


# --- Các hàm chức năng cho STT và TTS ---
def nghe_va_chuyen_thanh_van_ban():
    """Chức năng nhận dạng giọng nói từ micro."""
    global speech_thread_active
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    recognized_text_output = None
    with mic as source:
        print("DEBUG_STT: Đang điều chỉnh tiếng ồn môi trường...")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=2)
        except Exception as e:
            print(f"DEBUG_STT: Lỗi khi điều chỉnh tiếng ồn: {e}")
            speech_thread_active = False
            return None
        print("DEBUG_STT: 🎤 Vui lòng nói...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            print("DEBUG_STT: Không có âm thanh nào được phát hiện.")
            speech_thread_active = False
            return None
        except Exception as e:
            print(f"DEBUG_STT: Lỗi khi nghe từ micro: {e}")
            speech_thread_active = False
            return None
    try:
        print("DEBUG_STT: 🔄 Đang xử lý giọng nói...")
        recognized_text_output = recognizer.recognize_google(audio, language="vi-VN")
        print(f"DEBUG_STT: ✅ Bạn đã nói: {recognized_text_output}")
    except sr.UnknownValueError:
        print("DEBUG_STT: ⚠️ Không thể nhận dạng được giọng nói.")
    except sr.RequestError as e:
        print(f"DEBUG_STT: Lỗi kết nối tới dịch vụ Google; {e}")
    except Exception as e:
        print(f"DEBUG_STT: Lỗi không xác định khi nhận dạng: {e}")
    speech_thread_active = False
    return recognized_text_output


def doc_van_ban(text_to_speak):
    """Chức năng chuyển văn bản thành giọng nói."""
    global tts_thread_active
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        vietnamese_voice_id = next(
            (voice.id for voice in voices if "vietnamese" in voice.name.lower() or "an" in voice.name.lower()), None)
        if vietnamese_voice_id:
            engine.setProperty('voice', vietnamese_voice_id)
        else:
            print("DEBUG_TTS: ⚠️ Không tìm thấy giọng TV, dùng giọng mặc định.")
        engine.say(text_to_speak)
        engine.runAndWait()
    except Exception as e:
        print(f"DEBUG_TTS: Lỗi khi đọc văn bản: {e}")
    finally:
        tts_thread_active = False


def stt_thread_target():
    """Hàm mục tiêu cho luồng nhận dạng giọng nói."""
    global text_display
    recognized_text = nghe_va_chuyen_thanh_van_ban()
    if recognized_text:
        if text_display and not text_display.endswith(" "):
            text_display += " " + recognized_text
        else:
            text_display += recognized_text


def tts_thread_target(text_to_speak):
    """Hàm mục tiêu cho luồng đọc văn bản."""
    doc_van_ban(text_to_speak)


def display_mode_selection_screen():
    """Hiển thị màn hình chọn chế độ và chờ người dùng nhập."""
    screen_width, screen_height = 800, 600
    background = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_text = "CHON CHE DO NHAP LIEU"
    (text_w, text_h), _ = cv2.getTextSize(title_text, font, 1.2, 3)
    cv2.putText(background, title_text, ((screen_width - text_w) // 2, 100), font, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
    options = ["1: NHAN DANG CU CHI TAY", "2: NOI CHUYEN SANG VAN BAN", "3: GO VAN BAN BANG BAN PHIM"]
    for i, option in enumerate(options):
        (text_w, text_h), _ = cv2.getTextSize(option, font, 1, 2)
        cv2.putText(background, option, ((screen_width - text_w) // 2, 250 + i * 50), font, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
    cv2.putText(background, "Nhan 'Q' de thoat chuong trinh.", (50, screen_height - 50), font, 0.7, (150, 150, 150), 1,
                cv2.LINE_AA)
    cv2.imshow("Chon Che Do", background)
    selected_mode = None
    while selected_mode is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            selected_mode = 'gesture'
        elif key == ord('2'):
            selected_mode = 'speech'
        elif key == ord('3'):
            selected_mode = 'typing'
        elif key == ord('q'):
            selected_mode = 'quit'
    cv2.destroyWindow("Chon Che Do")
    return selected_mode


def print_instructions():
    """In ra hướng dẫn sử dụng cho người dùng (phiên bản cuối cùng)."""
    print("\n" + "=" * 50)
    print("--- HƯỚNG DẪN SỬ DỤNG CHUNG ---")
    print("  'q'      : Thoát toàn bộ chương trình")
    print("  'm'      : Quay lại màn hình chọn chế độ (Menu)")
    print("  'f'      : Bật/Tắt chế độ Toàn màn hình (Fullscreen)")
    print("  'r'      : Xóa (Reset) toàn bộ văn bản hiện tại")
    print("  'b'      : Xóa lùi (Backspace) ký tự cuối cùng")
    print("  't'      : Đọc (Text-to-speech) văn bản hiện tại")
    print("  SPACEBAR : Thêm dấu cách")
    print("\n--- TÙY CHỌN THEO CHẾ ĐỘ ---")
    print("  CHẾ ĐỘ CỬ CHỈ: Giữ ký hiệu ổn định 2 giây để xác nhận.")
    print("  CHẾ ĐỘ NÓI    : Nhấn 'l' để bắt đầu nghe (Listen).")
    print("  CHẾ ĐỘ GÕ     : Gõ phím bình thường.")
    print("=" * 50 + "\n")


def main_app():
    """Hàm chính chạy toàn bộ ứng dụng."""
    global text_display, speech_thread_active, tts_thread_active, current_mode

    is_fullscreen = False
    WINDOW_NAME = "Giao Dien Dieu Khien"

    current_mode = display_mode_selection_screen()
    if current_mode == 'quit': return

    text_display = ""
    candidate_char = ""
    candidate_char_start_time = 0
    CONFIRMATION_DURATION = 2.0
    last_char_added_to_text_display = ""
    last_time_char_added_to_text_display = 0
    MIN_INTERVAL_FOR_REPEATED_CHAR_IN_TEXT = 1.5

    # Hàng đợi để làm mượt dự đoán
    prediction_history = deque(maxlen=5)

    model = None
    if current_mode == 'gesture':
        try:
            model = joblib.load("sign_model.pkl")
            print("DEBUG_MAIN: ✅ Model 'sign_model.pkl' đã được tải.")
        except Exception as e:
            print(f"DEBUG_MAIN: ❌ Lỗi khi tải model: {e}");
            return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("DEBUG_MAIN: ❌ LỖI: Không thể mở webcam."); return

    mic_icon, tts_icon = None, None
    try:
        ICON_SIZE = (50, 50)
        if os.path.exists("mic_on.png"): mic_icon = cv2.resize(cv2.imread("mic_on.png", cv2.IMREAD_UNCHANGED),
                                                               ICON_SIZE)
        if os.path.exists("tts_on.png"): tts_icon = cv2.resize(cv2.imread("tts_on.png", cv2.IMREAD_UNCHANGED),
                                                               ICON_SIZE)
    except Exception as e:
        print(f"DEBUG_MAIN: ⚠️ LỖI: Không tải được icon. Lỗi: {e}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print_instructions()

    with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            stable_prediction = ""

            if current_mode == 'gesture' and model and results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmarks_data = []
                base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, \
                hand_landmarks.landmark[0].z
                for lm in hand_landmarks.landmark:
                    landmarks_data.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

                raw_prediction = model.predict([landmarks_data])[0]
                prediction_history.append(raw_prediction)

                # --- ĐOẠN CODE ĐÃ ĐƯỢC SỬA LỖI ---
                if len(prediction_history) == prediction_history.maxlen:
                    try:
                        stable_prediction = statistics.mode(prediction_history)
                    except statistics.StatisticsError:
                        # Nếu không có giá trị mode duy nhất (ví dụ: ['A','A','B','B']),
                        # lấy giá trị cuối cùng làm dự đoán tạm thời.
                        stable_prediction = prediction_history[-1]
                # ----------------------------------

                if stable_prediction:
                    if stable_prediction == candidate_char:
                        if (time.time() - candidate_char_start_time) >= CONFIRMATION_DURATION:
                            if stable_prediction != last_char_added_to_text_display or (
                                    time.time() - last_time_char_added_to_text_display > MIN_INTERVAL_FOR_REPEATED_CHAR_IN_TEXT):
                                text_display += stable_prediction
                                last_char_added_to_text_display = stable_prediction
                                last_time_char_added_to_text_display = time.time()
                                prediction_history.clear()
                    else:
                        candidate_char = stable_prediction
                        candidate_char_start_time = time.time()
            else:
                prediction_history.clear()
                candidate_char = ""

            # --- Giao diện Overlay ---
            overlay_height = 120;
            cv2.rectangle(frame, (5, 10), (frame_width - 5, overlay_height), (255, 255, 255), -1)
            mode_status_text = f"MODE: {current_mode.upper()}"
            if current_mode == 'gesture': mode_status_text += f" | Dang cho: {candidate_char}" if candidate_char else ""
            cv2.putText(frame, mode_status_text, (15, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200), 2)
            display_text_on_frame = "..." + text_display[-(int((frame_width - 30) / 12) - 3):] if len(
                text_display) > int((frame_width - 30) / 12) else text_display
            cv2.putText(frame, f"Van ban: {display_text_on_frame}", (15, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0),
                        2)

            def draw_icon(target_frame, icon, x, y):
                try:
                    h, w, channels = icon.shape
                    if channels < 4: return
                    alpha = icon[:, :, 3] / 255.0
                    roi = target_frame[y:y + h, x:x + w]
                    for c in range(3): roi[:, :, c] = (alpha * icon[:, :, c] + (1.0 - alpha) * roi[:, :, c])
                except Exception:
                    pass

            if speech_thread_active and mic_icon is not None: draw_icon(frame, mic_icon,
                                                                        frame_width - ICON_SIZE[0] - 15, 15)
            if tts_thread_active and tts_icon is not None: draw_icon(frame, tts_icon, frame_width - ICON_SIZE[0] - 15,
                                                                     15)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            # --- Khối xử lý phím bấm ---
            if key == ord('q'):
                break
            elif key == ord('m'):
                current_mode = display_mode_selection_screen()
                if current_mode == 'quit': break
                text_display = "";
                model = None;
                prediction_history.clear();
                candidate_char = ""
                if current_mode == 'gesture':
                    try:
                        model = joblib.load("sign_model.pkl")
                    except Exception as e:
                        print(f"Lỗi tải lại model: {e}")
                print_instructions();
                continue
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
            elif key == ord('r'):
                text_display = ""
            elif key == ord('b'):
                text_display = text_display[:-1]
            elif key == ord(' '):
                text_display += " "
            elif key == ord('t'):
                if not tts_thread_active and text_display:
                    tts_thread_active = True;
                    threading.Thread(target=tts_thread_target, args=(text_display,)).start()

            if current_mode == 'speech' and key == ord('l'):
                if not speech_thread_active:
                    speech_thread_active = True;
                    threading.Thread(target=stt_thread_target).start()
            elif current_mode == 'typing':
                if 32 <= key <= 126: text_display += chr(key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_app()