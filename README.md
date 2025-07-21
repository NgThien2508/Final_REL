# Controllable Music Playlist Generation với Reinforcement Learning

## 🎵 Mô tả dự án

Hệ thống tạo playlist thông minh sử dụng Reinforcement Learning (DQN) để tạo ra các playlist có thể kiểm soát được dựa trên sở thích người dùng. Hệ thống thu thập dữ liệu từ Spotify API bao gồm nhạc Việt Nam và quốc tế.

## 🚀 Tính năng chính

- **Thu thập dữ liệu tự động**: Lấy 10,000+ bài hát từ Spotify API
- **Nhạc Việt Nam và quốc tế**: Hỗ trợ đa dạng thể loại âm nhạc
- **Web UI thân thiện**: Giao diện đẹp và dễ sử dụng
- **Tùy chỉnh playlist**: Kiểm soát thể loại, độ phổ biến, năm phát hành
- **Reinforcement Learning**: Sử dụng DQN để tối ưu hóa playlist
- **Đánh giá chất lượng**: Hệ thống tự động đánh giá playlist

## 📋 Yêu cầu hệ thống

- Python 3.8+
- Windows 10/11
- Kết nối internet
- Tài khoản Spotify (để lấy Client ID và Secret)

## 🛠️ Cài đặt

### Bước 1: Clone dự án
```bash
git clone <repository-url>
cd Final_REL
```

### Bước 2: Tạo môi trường ảo (khuyến nghị)
```bash
# Tạo môi trường ảo
python -m venv venv

# Kích hoạt môi trường ảo (Windows)
venv\Scripts\activate

# Kích hoạt môi trường ảo (Linux/Mac)
source venv/bin/activate
```

### Bước 3: Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình Spotify API
File `config.py` đã được cấu hình sẵn với:
- Client ID: `c95b43c04ce94a2982d503ebd99adfeb`
- Client Secret: `f109da7340df466198a876778aad4559`

## 🎯 Hướng dẫn sử dụng

### Bước 1: Khởi động web server
```bash
python app.py
```

Mở trình duyệt và truy cập: `http://localhost:5000`

### Bước 2: Thu thập dữ liệu
1. Trên web UI, nhấn nút **"Thu thập dữ liệu"**
2. Chờ quá trình hoàn tất (có thể mất 10-15 phút)
3. Hệ thống sẽ thu thập:
   - 6,000 bài hát theo thể loại (pop, rock, hip-hop, etc.)
   - 2,000 bài hát Việt Nam
   - 2,000 bài hát quốc tế (K-pop, J-pop, etc.)

### Bước 3: Training model
1. Nhấn nút **"Bắt đầu Training"**
2. Chọn số episodes (khuyến nghị: 500-1000)
3. Chờ quá trình training hoàn tất
4. Model sẽ được lưu vào thư mục `models/`

### Bước 4: Tạo playlist
1. **Tùy chỉnh playlist**:
   - Độ dài: 5-50 bài hát
   - Thể loại: Chọn một hoặc nhiều thể loại
   - Độ phổ biến: 0-100
   - Năm phát hành: Từ 1950-2024
   - Bài hát khởi đầu: Tìm kiếm và chọn (tùy chọn)

2. Nhấn **"Tạo Playlist"**

3. Xem kết quả và đánh giá chất lượng

## 📁 Cấu trúc dự án

```
Final_REL/
├── app.py                      # Flask web application
├── config.py                   # Cấu hình hệ thống
├── spotify_data_collector.py   # Thu thập dữ liệu Spotify
├── models.py                   # ML/RL models
├── playlist_generator.py       # Logic tạo playlist
├── requirements.txt            # Thư viện cần thiết
├── templates/
│   └── index.html             # Web UI
├── data/                      # Dữ liệu (tự động tạo)
│   ├── playlists.json        # Dữ liệu bài hát
│   ├── embeddings.txt        # Embeddings
│   ├── raw_data.txt          # Dữ liệu thô
│   └── metrics.txt           # Metrics
├── models/                    # Model đã train (tự động tạo)
└── README.md                 # Hướng dẫn này
```

## 🔧 Cấu hình nâng cao

### Thay đổi cấu hình trong `config.py`:

```python
# Số lượng bài hát thu thập
TARGET_SONGS_COUNT = 10000

# Cấu hình model
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LEARNING_RATE = 0.001

# Cấu hình RL
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
```

### Thêm thể loại nhạc mới:

```python
# Trong config.py
GENRES = [
    "pop", "rock", "hip-hop", "electronic", "jazz", "classical", 
    "country", "r&b", "folk", "reggae", "blues", "metal",
    "your_new_genre"  # Thêm thể loại mới
]
```

## 🎮 Sử dụng qua Command Line

### Thu thập dữ liệu:
```bash
python spotify_data_collector.py
```

### Training model:
```bash
python playlist_generator.py
```

### Test tạo playlist:
```python
from playlist_generator import PlaylistGenerator

generator = PlaylistGenerator()
generator.load_data()
generator.train_rl_model(episodes=500)

# Tạo playlist với constraints
constraints = {
    'genre': ['pop', 'vietnamese'],
    'min_popularity': 70,
    'min_year': 2020
}

playlist = generator.generate_playlist(length=15, constraints=constraints)
print(f"Tạo được {len(playlist)} bài hát")
```

## 📊 Đánh giá chất lượng

Hệ thống đánh giá playlist dựa trên:
- **Độ tương đồng** (40%): Tính nhất quán giữa các bài hát
- **Đa dạng** (30%): Tránh lặp lại quá nhiều
- **Độ phổ biến** (30%): Chất lượng bài hát

Điểm tối đa: 10/10

## 🐛 Xử lý lỗi thường gặp

### Lỗi Spotify API:
```
Error: Invalid client credentials
```
**Giải pháp**: Kiểm tra Client ID và Secret trong `config.py`

### Lỗi memory:
```
OutOfMemoryError
```
**Giải pháp**: Giảm `TARGET_SONGS_COUNT` hoặc `BATCH_SIZE`

### Lỗi training chậm:
**Giải pháp**: 
- Giảm số episodes
- Sử dụng GPU (nếu có)
- Tăng `BATCH_SIZE`

## 📈 Kết quả mong đợi

Sau khi hoàn tất:
- **Dữ liệu**: ~10,000 bài hát với metadata đầy đủ
- **Model**: DQN model đã train sẵn
- **Playlist**: Chất lượng cao với điểm 7-9/10
- **Thời gian tạo playlist**: < 5 giây

## 🤝 Đóng góp

1. Fork dự án
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Tạo Pull Request

## 📄 License

Dự án này được phát hành dưới MIT License.

## 📞 Hỗ trợ

Nếu gặp vấn đề, vui lòng:
1. Kiểm tra phần "Xử lý lỗi thường gặp"
2. Tạo issue trên GitHub
3. Liên hệ qua email

---

**Lưu ý**: Đây là dự án học tập, không sử dụng cho mục đích thương mại.