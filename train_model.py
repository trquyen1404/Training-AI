import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Thư mục chứa dữ liệu đã thu thập
DATA_PATH = "data"

def train_model():
    print("--- Bắt đầu quá trình huấn luyện model ---")
    all_data = pd.DataFrame() # DataFrame để chứa tất cả dữ liệu

    # Kiểm tra xem thư mục DATA_PATH có tồn tại không
    if not os.path.exists(DATA_PATH):
        print(f" LỖI: Thư mục '{DATA_PATH}' không tồn tại. Vui lòng thu thập dữ liệu trước.")
        return

    # Đọc toàn bộ dữ liệu từ các file .csv trong thư mục DATA_PATH
    print(f"Đang đọc dữ liệu từ thư mục: '{DATA_PATH}'...")
    file_list = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

    if not file_list:
        print(f" LỖI: Không tìm thấy file .csv nào trong thư mục '{DATA_PATH}'.")
        return

    for file_name in file_list:
        file_path = os.path.join(DATA_PATH, file_name)
        if os.path.getsize(file_path) > 0: # Chỉ đọc file không rỗng
            try:
                df_temp = pd.read_csv(file_path, header=None)
                all_data = pd.concat([all_data, df_temp], ignore_index=True)
            except pd.errors.EmptyDataError:
                print(f" CẢNH BÁO: File '{file_name}' rỗng hoặc có lỗi, bỏ qua.")
            except Exception as e:
                print(f" Lỗi khi đọc file {file_name}: {e}")
        else:
            print(f" CẢNH BÁO: File '{file_name}' rỗng, bỏ qua.")

    if all_data.empty:
        print(" LỖI: Không có dữ liệu hợp lệ để huấn luyện sau khi đọc tất cả các file.")
        return

    print(f"Tổng số mẫu dữ liệu đã đọc: {len(all_data)}")
    print(f"Các lớp (nhãn) có trong dữ liệu: {sorted(all_data.iloc[:, -1].unique())}")

    # Tách đặc trưng (X) và nhãn (y)
    # Giả định 63 cột đầu là đặc trưng (tọa độ điểm mốc), cột cuối cùng là nhãn
    X = all_data.iloc[:, :-1]
    y = all_data.iloc[:, -1]

    print(f"Số lượng đặc trưng (features) trên mỗi mẫu: {X.shape[1]}")

    # Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
    # test_size=0.2 nghĩa là 20% dữ liệu dùng để kiểm tra, 80% để huấn luyện
    # random_state để đảm bảo kết quả phân chia giống nhau mỗi lần chạy (nếu cần tái lặp)
    # stratify=y giúp đảm bảo tỷ lệ các lớp trong tập huấn luyện và kiểm tra là tương đồng (quan trọng nếu dữ liệu mất cân bằng)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError as e:
        print(f" LỖI khi phân chia dữ liệu: {e}. Có thể do một số lớp có quá ít mẫu.")
        print("Vui lòng kiểm tra lại dữ liệu hoặc thu thập thêm mẫu cho các lớp ít dữ liệu.")
        return


    print(f"Số mẫu huấn luyện: {len(X_train)}")
    print(f"Số mẫu kiểm tra: {len(X_test)}")

    # Khởi tạo và huấn luyện model RandomForestClassifier
    # n_estimators: số lượng cây trong rừng
    # Bạn có thể thử nghiệm với các tham số khác của RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    print("Đang huấn luyện model RandomForestClassifier...")
    model.fit(X_train, y_train)
    print("✅ Huấn luyện model hoàn thành.")

    # Lưu model đã huấn luyện
    model_filename = "sign_model.pkl"
    joblib.dump(model, model_filename)
    print(f"✅ Model đã được lưu vào file: '{model_filename}'")

    # Đánh giá model trên tập kiểm tra
    print("Đang đánh giá model trên tập kiểm tra...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f" Độ chính xác (Accuracy) trên tập kiểm tra: {accuracy:.4f}")

    # Hiển thị báo cáo chi tiết hơn (precision, recall, f1-score cho từng lớp)
    print("\nBáo cáo phân loại chi tiết:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("--- Kết thúc quá trình huấn luyện ---")

if __name__ == "__main__":
    train_model()