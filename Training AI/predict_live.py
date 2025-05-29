import cv2
import mediapipe as mp
import joblib
import time
from collections import deque
import speech_recognition as sr
import pyttsx3
import threading
import numpy as np  # Giữ lại import này như trong code bạn gửi lần trước

# --- Biến cờ để kiểm soát luồng ---
speech_thread_active = False
tts_thread_active = False
text_display = ""  # Khai báo text_display ở phạm vi global để các luồng có thể truy cập


# --- Các hàm chức năng cho STT và TTS (giữ nguyên từ các phiên bản trước) ---
def nghe_va_chuyen_thanh_van_ban():
    global speech_thread_active, text_display  # Thêm text_display vào global nếu hàm này sửa nó
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    recognized_text_output = None
    with mic as source:
        print("DEBUG_STT: Đang điều chỉnh tiếng ồn môi trường...")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=1)
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
        print(f"DEBUG_STT:  Lỗi kết nối tới dịch vụ Google; {e}")
    except Exception as e:
        print(f"DEBUG_STT: Lỗi không xác định khi nhận dạng: {e}")
    speech_thread_active = False  # Đảm bảo cờ được reset ở cuối hàm này
    return recognized_text_output


def doc_van_ban(text_to_speak):
    global tts_thread_active
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        vietnamese_voice_id = None
        # (Logic tìm giọng nói tiếng Việt giữ nguyên)
        for voice in voices:
            if "vietnamese" in voice.name.lower() or "an" in voice.name.lower():
                vietnamese_voice_id = voice.id
                break
            if vietnamese_voice_id: break
            for lang in getattr(voice, 'languages', []):
                if isinstance(lang, bytes) and b'vi' in lang:
                    vietnamese_voice_id = voice.id; break
                elif isinstance(lang, str) and 'vi' in lang:
                    vietnamese_voice_id = voice.id; break
            if vietnamese_voice_id: break
        if vietnamese_voice_id:
            engine.setProperty('voice', vietnamese_voice_id)
        else:
            print("DEBUG_TTS: ⚠️ Không tìm thấy giọng TV, dùng giọng mặc định.")

        print(f"DEBUG_TTS: Đang đọc: '{text_to_speak}'")
        engine.say(text_to_speak)
        engine.runAndWait()
    except Exception as e:
        print(f"DEBUG_TTS: Lỗi khi đọc văn bản: {e}")
    finally:
        tts_thread_active = False  # Đảm bảo cờ được reset ở cuối hàm này


def stt_thread_target():
    global text_display  # text_display được sửa đổi từ luồng này
    print("DEBUG_THREAD: 🎤 Luồng STT BẮT ĐẦU.")
    recognized_text = nghe_va_chuyen_thanh_van_ban()
    if recognized_text:
        # Đảm bảo text_display được truy cập một cách an toàn nếu cần,
        # nhưng với string append thì thường là ổn trong Python cho kịch bản đơn giản.
        current_text_snapshot = text_display  # Đọc giá trị hiện tại
        if current_text_snapshot and not current_text_snapshot.endswith(" "):
            new_text = current_text_snapshot + " " + recognized_text
        else:
            new_text = current_text_snapshot + recognized_text
        text_display = new_text  # Gán lại giá trị mới
        print(f"DEBUG_THREAD: ➕ text_display cập nhật (STT): '{text_display}'")
    print("DEBUG_THREAD: 🎤 Luồng STT KẾT THÚC.")


def tts_thread_target(text_to_speak):
    # Hàm này không sửa đổi biến global, chỉ đọc text_to_speak
    print(f"DEBUG_THREAD: 🔊 Luồng TTS BẮT ĐẦU cho: '{text_to_speak}'")
    doc_van_ban(text_to_speak)
    print("DEBUG_THREAD: 🔊 Luồng TTS KẾT THÚC.")


# --- PHIÊN BẢN main_app() VỚI LOGIC CHỜ 2 GIÂY ---
def main_app():
    global text_display, speech_thread_active, tts_thread_active
    # text_display đã được khai báo global ở trên và sẽ được khởi tạo trong main_app nếu cần
    text_display = ""  # Khởi tạo lại mỗi khi chạy main_app

    candidate_char = ""
    candidate_char_start_time = 0
    CONFIRMATION_DURATION = 2.0

    last_char_added_to_text_display = ""
    last_time_char_added_to_text_display = 0
    MIN_INTERVAL_FOR_REPEATED_CHAR_IN_TEXT = 1.5

    try:
        model = joblib.load("sign_model.pkl")
        print("DEBUG_MAIN: ✅ Model 'sign_model.pkl' đã được tải.")
        print(f"DEBUG_MAIN: Model có thể dự đoán các lớp: {list(model.classes_)}")  # Chuyển sang list để dễ đọc hơn
    except FileNotFoundError:
        print("DEBUG_MAIN:  LỖI: Không tìm thấy file 'sign_model.pkl'.")
        return
    except Exception as e:
        print(f"DEBUG_MAIN:  Lỗi khi tải model: {e}")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("DEBUG_MAIN:  LỖI: Không thể mở webcam.")
        return

    print_instructions_2_seconds()

    with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("DEBUG_MAIN:  Lỗi: Không thể đọc frame.")
                break

            frame = cv2.flip(frame, 1)
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            char_confirmed_for_ky_tu_field = ""

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )
                    landmarks_data = []
                    for lm in hand_landmarks.landmark:
                        landmarks_data.extend([lm.x, lm.y, lm.z])

                    if len(landmarks_data) == 63:
                        try:
                            raw_prediction = model.predict([landmarks_data])[0]
                            # print(f"DEBUG: Dự đoán thô: {raw_prediction}") # Có thể bỏ comment nếu muốn xem liên tục

                            if raw_prediction == candidate_char:
                                if (time.time() - candidate_char_start_time) >= CONFIRMATION_DURATION:
                                    char_confirmed_for_ky_tu_field = candidate_char
                                    print(
                                        f"DEBUG: Ký tự XÁC NHẬN (sau {CONFIRMATION_DURATION}s): '{char_confirmed_for_ky_tu_field}'")

                                    current_time = time.time()
                                    if char_confirmed_for_ky_tu_field and \
                                            (char_confirmed_for_ky_tu_field != last_char_added_to_text_display or \
                                             (
                                                     current_time - last_time_char_added_to_text_display > MIN_INTERVAL_FOR_REPEATED_CHAR_IN_TEXT)):

                                        temp_text_display = text_display  # Đọc text_display hiện tại
                                        if temp_text_display and not temp_text_display.endswith(
                                                " ") and char_confirmed_for_ky_tu_field != "":
                                            temp_text_display += " "
                                        temp_text_display += char_confirmed_for_ky_tu_field
                                        text_display = temp_text_display  # Gán lại giá trị mới

                                        last_char_added_to_text_display = char_confirmed_for_ky_tu_field
                                        last_time_char_added_to_text_display = current_time
                                        print(f"DEBUG: text_display (SAU 2s XÁC NHẬN) cập nhật: '{text_display}'")
                            else:
                                candidate_char = raw_prediction
                                candidate_char_start_time = time.time()
                                # print(f"DEBUG: Ứng viên mới: '{candidate_char}' lúc {candidate_char_start_time:.2f}")

                        except Exception as e:
                            print(f"DEBUG_MAIN: Lỗi dự đoán: {e}")
            else:
                if candidate_char != "":
                    # print(f"DEBUG: Không có tay, reset ứng viên '{candidate_char}'")
                    candidate_char = ""
                    candidate_char_start_time = 0

            overlay_height = 120
            cv2.rectangle(frame, (5, 10), (frame_width - 5, overlay_height), (255, 255, 255), -1)
            alpha = 0.7
            frame_slice = frame[10:overlay_height, 5:frame_width - 5]
            if frame_slice.shape[0] > 0 and frame_slice.shape[1] > 0:
                cv2.addWeighted(frame_slice, alpha, frame_slice.copy(), 1 - alpha, 0, frame_slice)

            cv2.putText(frame, f"Ky tu: {char_confirmed_for_ky_tu_field}", (15, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 200), 2, cv2.LINE_AA)

            current_text_display_snapshot = text_display  # Đọc giá trị text_display để hiển thị
            display_text_on_frame = current_text_display_snapshot
            max_display_len_on_frame = int((frame_width - 30) / 12)
            if len(current_text_display_snapshot) > max_display_len_on_frame:
                display_text_on_frame = "..." + current_text_display_snapshot[-(max_display_len_on_frame - 3):]
            cv2.putText(frame, f"Van ban: {display_text_on_frame}", (15, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            # print(f"DEBUG_DRAW: Vẽ Ky tu: '{char_confirmed_for_ky_tu_field}', Van ban: '{display_text_on_frame}'")

            cv2.imshow("Nhan dang - Xac nhan sau 2 giay", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("DEBUG_MAIN: Thoát chương trình.")
                break
            elif key == ord('r'):
                text_display = ""
                last_char_added_to_text_display = ""
                candidate_char = ""
                print("DEBUG_MAIN: Đã xóa nội dung văn bản.")
            elif key == ord('b'):
                temp_text_display_b = text_display
                if temp_text_display_b:
                    if temp_text_display_b.endswith(" "): temp_text_display_b = temp_text_display_b[:-1]
                    if temp_text_display_b: temp_text_display_b = temp_text_display_b[:-1]
                    text_display = temp_text_display_b
                    last_char_added_to_text_display = ""
                    candidate_char = ""
                    print("DEBUG_MAIN: Đã xóa ký tự cuối.")
                else:
                    print("DEBUG_MAIN: Không có văn bản để xóa.")
            elif key == ord('t'):
                if not tts_thread_active:
                    current_text_to_speak = text_display  # Lấy snapshot để truyền vào luồng
                    if current_text_to_speak:
                        tts_thread_active = True
                        thread_tts = threading.Thread(target=tts_thread_target, args=(current_text_to_speak,))
                        thread_tts.daemon = True
                        thread_tts.start()
                    else:
                        print("DEBUG_MAIN: Không có văn bản để đọc (TTS).")
                else:
                    print("DEBUG_MAIN: ⚠️ Luồng TTS đang chạy.")
            elif key == ord('l'):
                if not speech_thread_active:
                    speech_thread_active = True
                    thread_stt = threading.Thread(
                        target=stt_thread_target)  # Luồng STT sẽ tự cập nhật global text_display
                    thread_stt.daemon = True
                    thread_stt.start()
                else:
                    print("DEBUG_MAIN: ⚠️ Luồng STT đang chạy.")

    cap.release()
    cv2.destroyAllWindows()
    print("DEBUG_MAIN: Chương trình đã kết thúc.")


def print_instructions_2_seconds():
    print("\n--- Hướng dẫn sử dụng (XÁC NHẬN SAU 2 GIÂY) ---")
    print("  Giơ tay làm ký hiệu (A-Z, 0-9) trước camera.")
    print("  GIỮ KÝ HIỆU ỔN ĐỊNH TRONG KHOẢNG 2 GIÂY ĐỂ HIỂN THỊ.")
    print("  QUAN SÁT CONSOLE để xem các thông báo DEBUG.")
    print("  'q': Thoát | 'r': Xóa text | 'b': Xóa ký tự cuối")
    print("  'l': Nghe (STT) | 't': Nói (TTS)")
    print("-----------------------------------------------------\n")


if __name__ == "__main__":
    main_app()