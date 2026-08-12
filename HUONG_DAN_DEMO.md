# Hướng Dẫn Demo — RAG Legal Assistant

> Hệ thống trợ lý pháp lý thông minh sử dụng kỹ thuật **RAG (Retrieval-Augmented Generation)**, hỗ trợ tra cứu văn bản luật bằng ngôn ngữ tự nhiên, kèm tính năng đọc to câu trả lời bằng giọng nói tiếng Việt (TTS) và nhân bản giọng nói cá nhân.

---

## Mục Lục

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Yêu Cầu & Cài Đặt](#2-yêu-cầu--cài-đặt)
3. [Khởi Động Ứng Dụng](#3-khởi-động-ứng-dụng)
4. [Vai Trò & Tài Khoản](#4-vai-trò--tài-khoản)
5. [Demo Hỏi Đáp Pháp Lý — Vai Trò Học Sinh](#5-demo-hỏi-đáp-pháp-lý--vai-trò-học-sinh)
6. [Demo Nhập Văn Bản Luật (PDF / DOCX) — Vai Trò Giáo Viên](#6-demo-nhập-văn-bản-luật-pdf--docx--vai-trò-giáo-viên)
7. [Demo Nhập Văn Bản Tình Huống (DOCX) — Vai Trò Giáo Viên](#7-demo-nhập-văn-bản-tình-huống-docx--vai-trò-giáo-viên)
8. [Demo Import Dataset Excel — Vai Trò Giáo Viên](#8-demo-import-dataset-excel--vai-trò-giáo-viên)
9. [Demo Đánh Giá Hệ Thống RAG — Vai Trò Giáo Viên](#9-demo-đánh-giá-hệ-thống-rag--vai-trò-giáo-viên)
10. [Demo Quản Lý Tài Khoản — Vai Trò Admin](#10-demo-quản-lý-tài-khoản--vai-trò-admin)
11. [Demo Quản Lý Văn Bản (Manage Law) — Vai Trò Admin](#11-demo-quản-lý-văn-bản-manage-law--vai-trò-admin)
12. [Demo Tính Năng Giọng Nói](#12-demo-tính-năng-giọng-nói)
13. [Câu Hỏi Demo Gợi Ý](#13-câu-hỏi-demo-gợi-ý)
14. [Kiến Trúc Kỹ Thuật](#14-kiến-trúc-kỹ-thuật)
15. [Xử Lý Sự Cố](#15-xử-lý-sự-cố)

---

## 1. Tổng Quan Hệ Thống

### Stack Công Nghệ

| Thành phần | Công nghệ |
|---|---|
| Backend | FastAPI (Python 3.10+), chạy bằng Uvicorn |
| Vector Database | ChromaDB |
| Embedding Model | `BAAI/bge-small-en-v1.5` (HuggingFace) |
| LLM | Groq — `llama-3.1-8b-instant` |
| OCR tiếng Việt | VietOCR (`vgg_transformer`) |
| Trích xuất PDF/DOCX | pypdf (text số) + pdf2image/Poppler (OCR scan) + python-docx |
| Nhận diện dòng văn bản | OpenCV |
| Xử lý Excel | pandas + openpyxl |
| TTS (văn bản → giọng đọc) | edge-TTS (Microsoft Neural), giọng mặc định `vi-VN-NamMinhNeural` |
| Nhân bản giọng nói (tùy chọn) | RVC (Real-time Voice Conversion), huấn luyện/chạy trên Colab (`colab/voice_server.ipynb`) qua tunnel |
| Database ứng dụng | SQLite (`chat.db`) |

### Pipeline RAG (khi học sinh/giáo viên đặt câu hỏi)

```
[1] Câu hỏi người dùng
       ↓
[2] Kiểm tra ngoài phạm vi (ly hôn, hình sự, đất đai, thuế…) → từ chối sớm nếu không thuộc Luật Doanh nghiệp
       ↓
[3] Kiểm tra câu hỏi "meta" về hệ thống (VD: "database đang lưu bao nhiêu điều luật?") → trả lời trực tiếp, bỏ qua RAG
       ↓
[4] Trích xuất chủ đề (topic extraction)
     → Nếu nhận ra chủ đề: bỏ qua bước viết lại → tiết kiệm 1 lần gọi API
       ↓
[5] Viết lại câu hỏi (query rewrite) — chỉ khi không trích được chủ đề
       ↓
[6] Tìm kiếm ngữ nghĩa trong ChromaDB (topic-aware retrieval)
     → Ưu tiên khớp metadata theo chủ đề, fallback sang semantic search có ngưỡng độ tương đồng
       ↓
[7] Xếp hạng lại tài liệu (rerank)
     → Tính điểm từ: từ khóa + cụm từ + số điều luật + nguồn KB
       ↓
[8] Phân loại câu hỏi (definition / condition / procedure / general)
       ↓
[9] Xây dựng prompt có căn cứ pháp lý + gọi Groq LLM (có retry tự động khi rate-limit)
       ↓
[10] Làm sạch câu trả lời (loại bỏ chào hỏi, dòng trùng lặp)
       ↓
[11] Gắn trích dẫn điều luật chính + nguồn tham khảo phụ (📖/📎) + đường dẫn vbpl.vn
        → Trích dẫn (so_ky_hieu) chỉ được in ra nếu đang thực sự có mặt trong ChromaDB
          lúc đó (whitelist CITATION_SOURCE, xem mục 14.4) — chống trích dẫn "ma" từ
          văn bản đã bị xoá hoặc chưa từng được import
       ↓
[12] (Tùy chọn) Người dùng nhấn nút 🔊 đọc to — văn bản câu trả lời được gửi qua
        TTS/RVC (mục 12), không đi lại qua pipeline RAG
```

Hệ thống nạp dữ liệu vào ChromaDB qua **3 luồng import** độc lập, cùng dùng chung một vector store: **Văn bản luật** (mục 6), **Văn bản tình huống** (mục 7), **Dataset Excel** (mục 8). Admin quản lý cả ba qua trang **Manage Law** (mục 11).

---

## 2. Yêu Cầu & Cài Đặt

### 2.1 Phần Mềm Cần Có

- **Python 3.10+**
- **Poppler** — đã có sẵn trong thư mục `poppler/` của dự án (dùng khi OCR PDF scan)
- **ffmpeg** — cần thêm vào PATH hệ thống (dùng bởi edge-TTS/RVC cho tính năng giọng nói)
- **Groq API key** — đăng ký miễn phí tại https://console.groq.com

### 2.2 Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

Thư viện giọng nói (`edge-tts`, `python-docx`, v.v.) đã gộp chung vào `requirements.txt` — không cần file riêng.

> **Bắt buộc: thư mục `clone-voice-client` phải nằm cùng cấp với project này** (`../clone-voice-client`). `requirements.txt` có dòng `-e ../clone-voice-client[local]` — lệnh `pip install` ở trên tự cài thư viện đó (dạng editable) từ thư mục anh em này, không phải từ PyPI. Thiếu thư mục `clone-voice-client`, bước cài đặt trên sẽ báo lỗi ngay, và app không chạy được. Xem đầy đủ các câu lệnh terminal + ngrok cho cả 4 project liên quan (`clone-voice-station`, `clone-voice-client`, `rag-legal-assistant`, `voice-lab-example`) tại [`HUONG_DAN_CHAY_TOAN_HE_THONG.md`](./HUONG_DAN_CHAY_TOAN_HE_THONG.md).

### 2.3 Cấu Hình API Key

Tạo file `groqkey.txt` tại thư mục gốc dự án, dán API key vào:

```
gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 2.4 Kiểm Tra Nhanh

```bash
# Xác nhận Python đúng phiên bản
python --version

# Xác nhận Groq key hợp lệ
python -c "print(open('groqkey.txt').read().strip()[:8] + '...')"
```

---

## 3. Khởi Động Ứng Dụng

```bash
python app.py
```

Nội bộ `app.py` tự gọi Uvicorn (`uvicorn.run("app:app", host="127.0.0.1", port=8000)`), nên không cần chạy lệnh `uvicorn` riêng. Terminal hiển thị:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Mở trình duyệt và truy cập: **http://127.0.0.1:8000**

> **Lần đầu khởi động:** Hệ thống tự động tạo file `chat.db` và seed 3 tài khoản mặc định (học sinh, giáo viên, admin) cùng các giọng đọc TTS dựng sẵn (`voice_profiles`, kind=`builtin`, xem mục 12). ChromaDB cũng sẽ tải embedding model từ HuggingFace (cần kết nối Internet lần đầu). `engine/rag_engine.py` còn tự chạy 2 tác vụ bảo trì lúc khởi động: làm mới whitelist trích dẫn (`refresh_citation_sources`) và gắn nhãn nguồn gốc cho các đoạn dữ liệu cũ chưa được đánh dấu (`backfill_import_source_tags`, xem mục 14.4) — cả hai đều an toàn khi chạy lại nhiều lần.

---

## 4. Vai Trò & Tài Khoản

Hệ thống có **3 vai trò**, phân biệt bằng cột `role` trong bảng `users` (0 = Student, 1 = Teacher, 2 = Admin). Admin kế thừa toàn bộ quyền của Teacher.

### Tài Khoản Mặc Định

| Vai trò | Tên đăng nhập | Mật khẩu | Quyền |
|---|---|---|---|
| Học sinh | `testStudent1` | `123456P@ss` | Hỏi đáp RAG + đọc to câu trả lời + tạo giọng nói cá nhân |
| Giáo viên | `teacher1` | `Teacher@123` | Toàn bộ quyền Học sinh + Import văn bản luật/tình huống/dataset + Đánh giá RAG |
| Admin | `admin1` | `Admin@123` | Toàn bộ quyền Giáo viên + Quản lý tài khoản + Quản lý văn bản (xoá dữ liệu đã import) + Quản lý giọng nói (duyệt/huấn luyện lại/vô hiệu hóa giọng nhân bản) |

> Các tài khoản mặc định chỉ được tạo **một lần**, khi bảng `users` hoàn toàn trống lúc khởi động. Nếu bảng đã có dữ liệu, cần dùng chức năng **Import tài khoản** hoặc thêm thủ công (mục 10) để có tài khoản admin.

### Các Bước Demo Đăng Nhập

1. Truy cập `http://127.0.0.1:8000`
2. Nhập tên đăng nhập và mật khẩu
3. Nhấn **Đăng nhập**
4. Hệ thống tự động điều hướng theo vai trò:
   - **Học sinh** → giao diện chat hỏi đáp + nút **"🎙 Giọng nói của tôi"** trên sidebar (mọi vai trò đều thấy)
   - **Giáo viên** → giao diện chat + nút **"Import Law"** + nút **"🎙 Giọng nói của tôi"** + badge **"Giảng viên"**
   - **Admin** → giao diện chat + nút **"Import Law"** + nút **"Manage Account"** + nút **"Manage Law"** + nút **"🎙 Giọng nói của tôi"** + nút quản lý giọng nói admin + badge **"Quản trị viên"**
5. Trên trang đăng nhập có link **"Đổi mật khẩu"** để tự đổi mật khẩu (yêu cầu tối thiểu 8 ký tự, có chữ hoa, chữ thường, số và ký tự đặc biệt)

![Trang đăng nhập](login.png)

---

## 5. Demo Hỏi Đáp Pháp Lý — Vai Trò Học Sinh

### Bước 1: Tạo Cuộc Trò Chuyện Mới

1. Đăng nhập bằng tài khoản học sinh (`testStudent1`)
2. Nhấn nút **"+ New Chat"** ở thanh bên trái
3. Một cuộc trò chuyện mới được tạo, sẵn sàng nhận câu hỏi

### Bước 2: Đặt Câu Hỏi

- Gõ câu hỏi vào ô nhập liệu ở cuối trang
- Nhấn **Enter** hoặc nút gửi
- Hệ thống xử lý theo pipeline ở mục 1, sau đó hiển thị:
  - Câu trả lời phù hợp với loại câu hỏi (định nghĩa / điều kiện / thủ tục)
  - **📖 Nguồn chính:** trích dẫn điều luật cụ thể
  - **📎 Nguồn tham khảo:** tối đa 3 điều luật liên quan khác (nếu có)
  - **🔗 Link:** đường dẫn vbpl.vn (nếu có)
  - Nút **🔊** cạnh mỗi câu trả lời để đọc to bằng giọng đã chọn (mục 12)

![Giao diện chat học sinh](chat_index_student.png)

### Bước 3: Quản Lý Cuộc Trò Chuyện

| Thao tác | Cách thực hiện |
|---|---|
| Đổi tên chat | Nhấp đúp vào tiêu đề chat ở sidebar |
| Xóa chat | Nhấn icon thùng rác bên cạnh tên chat |
| Chuyển chat | Nhấn vào tên chat khác trong sidebar |
| Xem lại lịch sử | Nhấn vào chat có sẵn — tin nhắn được load từ SQLite |

> Chat của học sinh và giáo viên tách biệt hoàn toàn (cột `role` trong bảng `chats`), kể cả khi cùng một `user_id`.

---

## 6. Demo Nhập Văn Bản Luật (PDF / DOCX) — Vai Trò Giáo Viên

Tính năng này cho phép giáo viên (hoặc admin) tải lên file **PDF hoặc DOCX**. Hệ thống tự động trích xuất văn bản (hoặc OCR nếu là bản scan), phân đoạn theo điều khoản và lập chỉ mục vào ChromaDB.

### Bước 1: Truy Cập Trang Import

1. Đăng nhập bằng tài khoản giáo viên (`teacher1`) hoặc admin
2. Sidebar hiển thị nút **"Import Law"** và badge vai trò ở header
3. Nhấn nút **"Import Law"** — trang `/import` mở ra, tab mặc định **"📄 Upload PDF / DOCX"**

![Giao diện giáo viên — sidebar có nút Import Law và chat "Import new law"](chat_index_teacher.png)

### Bước 2: Điền Thông Tin và Upload

| Trường | Ví dụ | Mô tả |
|---|---|---|
| Số ký hiệu | `59/2020/QH14` | Số hiệu chính thức của văn bản — dùng để chống import trùng |
| Loại văn bản | `Luật Doanh nghiệp` | Tên/loại văn bản pháp luật |
| Nguồn thu thập | `vbpl.vn` | Nguồn gốc tài liệu |
| File | _(chọn file .pdf hoặc .docx)_ | Hỗ trợ PDF (scan hoặc số) và DOCX |

Nhấn **"🚀 Tải lên & Xử lý AI"** để bắt đầu.

![Trang import PDF](import_pdf.png)

### Bước 3: Quy Trình Xử Lý Nền

```
1. Nhận file → lưu tạm thời vào uploads_tmp/
2. Nếu DOCX: trích xuất văn bản trực tiếp (python-docx, gồm cả bảng)
3. Nếu PDF:
   a. Thử trích xuất text trực tiếp bằng pypdf (PDF văn bản số)
   b. Nếu quá ít ký tự (< 150 ký tự/trang trung bình) → coi là bản scan, chuyển sang OCR:
      - Tải mô hình VietOCR (vgg_transformer), tự phát hiện CPU/GPU
      - Chuyển PDF → ảnh (DPI 250, từng batch 5 trang)
      - Tiền xử lý ảnh (tăng độ tương phản, làm nét)
      - Nhận dạng dòng văn bản bằng OpenCV
      - OCR từng dòng bằng VietOCR (beam search)
4. Phân đoạn theo "Điều X." trong văn bản
5. Tạo vector embedding và thêm vào ChromaDB (bỏ qua nếu Số ký hiệu đã tồn tại)
   — mỗi đoạn được gắn nhãn import_source="law" để trang Manage Law (mục 11)
     nhận diện đúng nguồn gốc
6. Tạo/cập nhật chat "Import new law" với thông báo kết quả
```

### Bước 4: Theo Dõi Tiến Trình

- Giao diện hiển thị **thanh tiến trình realtime** và trạng thái từng bước
- Trạng thái: `running` → `done` / `failed`
- Khi hoàn tất, kết quả xuất hiện trong chat **"Import new law"** ở sidebar
  - Thành công: `✅ Hoàn tất! Đã thêm X đoạn vào ChromaDB (bỏ qua Y đoạn trùng lặp).`
  - Thất bại: `❌ Lỗi: <chi tiết>`

![Kết quả import trong chat "Import new law"](import_result.png)

### Lưu Ý Quan Trọng

- Chỉ chấp nhận định dạng **PDF** hoặc **DOCX**
- Văn bản có **số ký hiệu trùng** sẽ bị bỏ qua tự động (không import lại)
- PDF văn bản số (có thể chọn/copy chữ) được trích xuất trực tiếp — **nhanh, không cần OCR**
- OCR chỉ chạy khi phát hiện PDF là bản scan, mặc định trên **CPU** — file 50 trang có thể mất 10–30 phút
- Nếu hệ thống **không nhận diện được ranh giới "Điều X."** trong văn bản (văn bản không có tiêu đề điều rõ ràng), quy trình sẽ dùng cắt đoạn dự phòng (3000 ký tự/đoạn) và **báo cảnh báo rõ ràng** trong tiến trình + trong chat "Import new law" — các đoạn này sẽ không có trích dẫn số Điều (thay vì âm thầm gán số Điều sai như trước)
- Để dùng **GPU** (nhanh hơn đáng kể), tạo file `.device_config` tại thư mục gốc:
  ```
  DEVICE=cuda
  ```
- Muốn xoá một văn bản đã import (VD nhập nhầm số ký hiệu)? Dùng trang **Manage Law** (mục 11, chỉ Admin) thay vì import chồng lên.

---

## 7. Demo Nhập Văn Bản Tình Huống (DOCX) — Vai Trò Giáo Viên

Ngoài văn bản luật gốc, giáo viên có thể nạp một **bộ tình huống pháp lý mẫu** (dạng phân tích IRAC — Issue/Rule/Application/Conclusion) để làm phong phú câu trả lời cho các câu hỏi tình huống thực tế.

### Bước 1: Chuyển Sang Tab Tình Huống

Tại trang `/import`, nhấn tab **"📚 Tình huống"**.

### Bước 2: Chuẩn Bị File & Upload

- File `.docx` phải theo đúng cấu trúc cố định — mỗi tình huống là một mục **Heading 1** dạng `Tình huống NN. <Chủ đề>`, gồm các mục con:

| Mục | Nội dung |
|---|---|
| Dòng đầu (sau Heading 1) | `Mã: <mã tình huống>   Độ khó: <Dễ/Trung bình/Khó>` |
| `1. Đề bài` | `Tình huống: …` và `Câu hỏi: …` |
| `2. Câu hỏi dẫn dắt xác định vấn đề pháp lý` | Mỗi câu hỏi dẫn dắt một dòng riêng |
| `3. Đáp án theo phương pháp IRAC` | `I – Issue:`, `R – Rule:`, `A – Application:`, `C – Conclusion:` |
| `4. Căn cứ pháp lý` | Mỗi căn cứ pháp lý một dòng riêng (có thể trích nhiều điều/nhiều văn bản khác nhau) |
| `5. Dữ liệu hỗ trợ truy xuất chatbot` | `Từ khóa: k1; k2; k3` và `Câu hỏi tương đương: q1 \| q2` |

- Tải file mẫu qua link **"⬇️ Tải example_scenario.docx"** ở panel bên phải để xem đúng cấu trúc (có sẵn 2 tình huống ví dụ minh hoạ).
- Chọn/kéo thả file `.docx` đã điền, nhấn **"📥 Tải lên & Xử lý"**.

### Bước 3: Quy Trình Xử Lý Nền

```
1. Đọc toàn bộ paragraph trong file .docx, tách theo từng khối "Tình huống NN."
2. Với mỗi tình huống: parse Mã/Độ khó, Đề bài, câu hỏi dẫn dắt, 4 thành phần IRAC,
   căn cứ pháp lý, từ khóa, câu hỏi tương đương
3. Gộp thành một đoạn văn bản có cấu trúc cho mỗi tình huống, gắn nhãn
   doc_type="scenario_qa", import_source="scenario", nguon_thu_thap=<tên file gốc>
4. Tạo vector embedding, thêm vào ChromaDB — bỏ qua nếu "Mã" tình huống đã tồn tại
   (import lại cùng file sẽ không tạo trùng)
5. Tạo/cập nhật chat "Nhập văn bản tình huống" với thông báo kết quả
```

### Lưu Ý Quan Trọng

- **Không** gán số ký hiệu (`so_ky_hieu`) cho các đoạn tình huống — một tình huống có thể trích nhiều điều luật từ nhiều văn bản khác nhau cùng lúc (VD vừa Luật Doanh nghiệp vừa Nghị định 168/2025/NĐ-CP), nên hệ thống không tự gán một mã văn bản duy nhất để tránh trích dẫn sai nguồn. Khi trả lời, các đoạn này vẫn được dùng để truy xuất ngữ nghĩa bình thường, chỉ không tự sinh dòng "📖 Nguồn chính" theo số ký hiệu cho riêng chúng.
- Xoá một bộ tình huống đã import: dùng trang **Manage Law** (mục 11) → tab **Tình huống**, xoá theo tên file gốc.

---

## 8. Demo Import Dataset Excel — Vai Trò Giáo Viên

Ngoài việc import từng văn bản PDF/DOCX/tình huống, giáo viên có thể nạp nhanh một **bộ dataset Excel** (câu hỏi mẫu + điều luật + metadata) thẳng vào ChromaDB.

### Bước 1: Chuyển Sang Tab Dataset

Tại trang `/import`, nhấn tab **"📊 Import Dataset"**.

### Bước 2: Upload File

- Chọn/kéo thả file `.xlsx`
- Hệ thống **tự nhận diện định dạng** theo tên sheet có trong file:
  - Sheet `KB_Articles_Updated` + `Dataset_200` → định dạng "200-updated" (mới nhất, có thêm `Legal_Update_2025`, `KB_Articles_Updated`)
  - Sheet `KB_Articles` + `Dataset_150` → định dạng "150" (cũ hơn)
- Nhấn **"📥 Import vào ChromaDB"**

### Bước 3: Quy Trình Xử Lý Nền

```
1. Đọc toàn bộ sheet trong file .xlsx
2. Ưu tiên xử lý KB_Articles_Updated (nếu có), fallback KB_Articles
3. Xử lý Legal_Update_2025 (nếu có) — các thay đổi pháp lý 2025
4. Xử lý sheet Dataset_* (ưu tiên Dataset_200 > Dataset_150) — cặp câu hỏi/trả lời mẫu
5. Loại trùng lặp (theo doc_id hoặc Số ký hiệu + nguồn KB_Articles đã có)
6. Mỗi đoạn được gắn nhãn import_source="dataset" + source_file=<tên file .xlsx
   vừa upload> — đây là điều kiện để trang Manage Law (mục 11) nhóm và xoá đúng
   theo từng file, thay vì gộp chung mọi lần import dataset thành một khối
7. Tạo vector embedding, thêm vào ChromaDB theo từng batch 32 tài liệu
8. Lưu lại file .xlsx gốc vừa upload vào thư mục Dataset/ (không xoá) — nhờ vậy file
   này tự động xuất hiện trong dropdown chọn dataset ở mục 9 (Đánh giá hệ thống RAG)
   mà không cần copy tay. Nếu trùng tên với file đã có sẵn, hệ thống tự thêm hậu tố
   thời gian vào tên file để không ghi đè.
```

- Kết quả trả về: số tài liệu thêm mới theo từng sheet, số bỏ qua do trùng, tổng số tài liệu hiện có trong ChromaDB
- ChromaDB **tích lũy dữ liệu** — import dataset mới không xóa dữ liệu cũ đã có
- Tất cả file dataset (upload qua đây hoặc đặt thủ công) đều nằm trong thư mục
  **`Dataset/`** ở gốc dự án — đây là nơi duy nhất hệ thống quét để tìm file cho
  tính năng Đánh giá RAG (mục 9)
- File `Dataset/example_sheet.xlsx` là **file mẫu/template** (phục vụ nút tải mẫu ở mục 9), luôn bị loại khỏi dropdown đánh giá — không phải dữ liệu thật

---

## 9. Demo Đánh Giá Hệ Thống RAG — Vai Trò Giáo Viên

Ngay bên dưới phần Import Dataset (cùng trang `/import`, tab **"📊 Import Dataset"**) là khu vực **Đánh giá hệ thống RAG**, dùng để đo chất lượng câu trả lời so với đáp án mẫu.

### Bước 0: Kết Quả Lần Gần Nhất Tự Hiện Khi Mở Tab

Ngay khi mở tab, hệ thống tự gọi `GET /latest_eval_result` và hiển thị luôn score card của **lần đánh giá gần nhất** (đánh dấu 📌 "Kết quả lần đánh giá gần nhất") — không cần chạy lại mới thấy điểm. Nếu chưa từng chạy đánh giá nào, khu vực này để trống cho tới khi bấm Quick/Full Evaluation lần đầu.

### Bước 1: Chọn File Dataset

- Dropdown **"File dataset dùng để đánh giá"** liệt kê **mọi file `.xlsx`** trong thư mục **`Dataset/`** (ở gốc dự án) có chứa ít nhất một sheet `Dataset_*` hoặc `Demo_*` — tự động lấy qua `GET /list_datasets`, không cần khai báo tên file cứng trong code (trừ `example_sheet.xlsx`, luôn bị loại vì là file mẫu). Đặt (hoặc để hệ thống tự lưu, xem mục 8) một file mới vào `Dataset/` (VD dataset 300 câu tương lai) là dropdown tự nhận diện ngay, miễn sheet đặt tên đúng quy ước `Dataset_*` / `Demo_*`.
- Mặc định chọn sẵn `enterprise_law_full_rag_chatbot_dataset_200_updated.xlsx` nếu có trong danh sách.
- Ngay dưới dropdown hiển thị gợi ý các sheet Demo/Dataset tìm thấy trong file đang chọn; nếu file không có sheet Demo, nút **Quick Evaluation** tự động bị mờ/disable kèm cảnh báo.

### Bước 2: Chọn Chế Độ Đánh Giá

| Chế độ | Dữ liệu dùng | Cách chấm | Tốc độ |
|---|---|---|---|
| **⚡ Quick Evaluation** | **Toàn bộ** sheet `Demo_*` có trong file đã chọn, gộp lại và loại trùng theo cột `id` (VD file có cả `Demo_30` và `Demo_50` → gộp thành 50 câu duy nhất) | `auto` — so khớp từ khóa/trích dẫn (offline, không cần Groq) | Nhanh |
| **🔬 Full Evaluation** | **Toàn bộ** sheet `Dataset_*` có trong file đã chọn, gộp lại và loại trùng theo cột `id` — đây là **bộ dữ liệu đầy đủ** của file (VD file có cả `Dataset_150` và `Dataset_200` → gộp thành 200 câu duy nhất), không phải bản rút gọn | `llm` — chấm bằng Groq LLM theo rubric | Chậm hơn (~3s/câu do throttle Groq) |

File `Dataset/example_sheet.xlsx` minh hoạ đúng quy ước đặt tên: sheet **`Demo_Quick_example`** (tiền tố `Demo_`) cho Quick Evaluation, sheet **`Dataset_example`** (tiền tố `Dataset_`) cho Full Evaluation. Đổi tên sheet sang tiền tố khác sẽ khiến sheet đó biến mất khỏi cả dropdown lẫn 2 nút đánh giá — `list_available_datasets()`/`run_evaluation()` trong `engine/evaluate_engine.py` chỉ quét đúng theo 2 tiền tố này.

> Nếu file được chọn **không có sheet Demo** mà vẫn bấm Quick Evaluation (hoặc gọi thẳng API), hệ thống báo lỗi: `❌ File '<tên file>' không có sheet Demo — không thể chạy Quick Evaluation cho file này.` Tương tự với Full Evaluation và sheet Dataset. Đây là bộ đánh giá đọc trực tiếp từ file `.xlsx` đã chọn trên đĩa, **không** liên quan tới `chat.db`/ChromaDB.

### Bước 3: Theo Dõi Tiến Trình & Kết Quả

- Thanh tiến trình hiển thị số câu đã xử lý / tổng số câu
- Sau khi hoàn tất, hiển thị:
  - **Điểm tổng** (thang 100), theo rubric 5 tiêu chí có trọng số:

| Tiêu chí | Trọng số |
|---|---|
| Độ chính xác pháp lý | 40% |
| Trích dẫn điều luật | 20% |
| Mức độ liên quan ngữ cảnh | 20% |
| Kiểm soát bịa đặt (hallucination) | 15% |
| Rõ ràng, dễ hiểu | 5% |

  - Điểm chi tiết theo **loại câu hỏi** (definition/condition/procedure/general) và theo **độ khó**
  - Tên file dataset và danh sách sheet đã dùng để đánh giá (VD: `Demo_30, Demo_50`)
- Kết quả chi tiết từng câu được xuất ra file `eval_results_<tên_dataset>_<split>_<mode>_<timestamp>.xlsx` tại thư mục gốc dự án (không phải trong `Dataset/` — đây là file kết quả, không phải file input)
- Nút **"⬇️ Tải sheet kết quả"** xuất hiện ngay trong score card sau khi đánh giá xong — tải trực tiếp file `eval_results_*.xlsx` nói trên qua `GET /download_eval_result/<tên_file>` mà không cần vào thư mục dự án tìm thủ công
- Sau mỗi lần chạy, hệ thống **chỉ giữ lại 2 file `eval_results_*.xlsx` mới nhất** trên đĩa (tự xoá các file cũ hơn) và lưu tóm tắt lần chạy gần nhất vào `eval_results_latest.json` — đây là dữ liệu Bước 0 dùng để hiển thị lại khi mở tab, không cần giữ toàn bộ lịch sử vì giao diện chưa có màn hình duyệt kết quả cũ.

### Lưu Ý Về Độ Chính Xác Của Chấm Điểm `auto` Mode

Chế độ `auto` so khớp trích dẫn bằng cách trích số Điều từ `article_reference` của bộ câu hỏi rồi tìm chuỗi `"điều N"` trong câu trả lời. Với các trích dẫn phức tạp hơn (VD `"Khoản 35 Điều 4"`, `"Điều 17 Nghị định 168/2025/NĐ-CP"`, hoặc nhiều điều gộp `"Điều 27; Điều 38"`), hệ thống chỉ lấy đúng số theo sau từ "Điều" trong chuỗi tham chiếu — **không** gộp lẫn số Khoản/số Nghị định/số năm vào cùng một số Điều như trước (lỗi đã sửa ngày 2026-07-21, xem mục 15). Điểm `auto` vẫn là ước lượng nhanh dựa trên từ khóa, không thay thế được chấm `llm` (Full Evaluation) khi cần độ chính xác cao.

---

## 10. Demo Quản Lý Tài Khoản — Vai Trò Admin

Chỉ tài khoản có vai trò **Admin** mới truy cập được các tính năng này.

### Bước 1: Truy Cập Trang Quản Lý

1. Đăng nhập bằng tài khoản admin (`admin1`)
2. Nhấn nút **"👤 Manage Account"** trên sidebar — trang `/manage_accounts` mở ra
3. Bảng hiển thị toàn bộ tài khoản: tên đăng nhập, vai trò (pill màu), trạng thái (Đang hoạt động / Đã vô hiệu hóa)

### Bước 2: Thao Tác Trên Từng Tài Khoản

| Thao tác | Cách thực hiện | Ghi chú |
|---|---|---|
| Vô hiệu hóa | Nhấn **"Vô hiệu hóa"** | Tài khoản bị khóa đăng nhập, không xóa dữ liệu |
| Kích hoạt lại | Nhấn **"Kích hoạt"** | Cho phép đăng nhập trở lại |
| Xoá tài khoản | Nhấn **"Xoá"** → xác nhận | Xoá vĩnh viễn tài khoản + toàn bộ chat/lịch sử liên quan |

> Admin **không thể tự vô hiệu hóa hoặc tự xoá** chính tài khoản đang đăng nhập — nút tương ứng sẽ bị disable.

### Bước 3: Import Hàng Loạt Tài Khoản

1. Nhấn **"📥 Import tài khoản"** ở góc trên bảng — trang `/import_account` mở ra
2. Tải file mẫu qua link **"⬇️ Tải file Excel mẫu"** (2 cột: `Account Name`, `Role`)
3. Chọn/kéo thả file `.xlsx` đã điền, nhấn **"📥 Import tài khoản"**
4. Mỗi dòng hợp lệ được tạo với mật khẩu mặc định theo vai trò:

| Role trong file | Mật khẩu mặc định |
|---|---|
| Student | `123456P@ss` |
| Teacher | `Teacher@123` |

5. Sau khi import xong, hệ thống báo cáo: **Tạo mới**, **Bỏ qua (trùng tên)**, **Bỏ qua (không hợp lệ)**
6. Nếu có dòng lỗi (trùng tên / thiếu thông tin / role không hợp lệ), nút **"⚠️ Tải tài khoản lỗi"** xuất hiện — tải về file `.xlsx` liệt kê từng dòng lỗi kèm cột thứ 3 **"Nguyên nhân lỗi"**

---

## 11. Demo Quản Lý Văn Bản (Manage Law) — Vai Trò Admin

Trang quản lý tập trung cho cả **3 luồng import** vào ChromaDB (Văn bản pháp luật — mục 6, Dataset — mục 8, Tình huống — mục 7). Chỉ tài khoản **Admin** mới truy cập được — khác với các trang Import (mục 6–8), vốn mở cho cả Giáo viên.

### Bước 1: Truy Cập Trang Quản Lý

1. Đăng nhập bằng tài khoản admin (`admin1`)
2. Nhấn nút **"🗂️ Manage Law"** trên sidebar — trang `/manage_law` mở ra
3. Trang có 3 tab, cùng cấu trúc bảng: **Tên** / **Đoạn trong database** / **Xóa**

### Bước 2: Chuyển Tab & Xoá Dữ Liệu

| Tab | Nhóm theo | Ghi chú |
|---|---|---|
| **📖 Văn bản pháp luật** | Số ký hiệu (`so_ky_hieu`) | Xoá toàn bộ đoạn có cùng số ký hiệu — vd xoá `59/2020/QH14` sẽ gỡ hết các đoạn thuộc văn bản đó |
| **📊 Dataset** | Tên file `.xlsx` đã upload (`source_file`) | Cần import qua giao diện **sau khi** tính năng theo dõi tệp được thêm (2026-07-21) mới có tên file chính xác — dữ liệu import từ trước ngày đó gộp chung vào mục "(không rõ tệp…)" |
| **📚 Tình huống** | Tên file `.docx` đã upload (`nguon_thu_thap`) | Xoá theo từng file — không ảnh hưởng các file tình huống khác |

Nhấn **"Xoá"** ở dòng tương ứng → xác nhận → hệ thống xoá toàn bộ đoạn khớp khỏi ChromaDB, đồng thời làm mới whitelist trích dẫn (`CITATION_SOURCE`) ngay lập tức để không còn trích dẫn tới nguồn đã xoá.

> **Không thể hoàn tác.** Với văn bản pháp luật/dataset, nếu xoá nhầm thì phải import lại từ file gốc.

---

## 12. Demo Tính Năng Giọng Nói

Tính năng giọng nói gồm 2 phần độc lập: **đọc to câu trả lời** (TTS, mọi vai trò dùng được ngay) và **nhân bản giọng nói cá nhân** (tùy chọn, huấn luyện qua RVC trên Colab, cần Admin duyệt trước khi dùng).

### 12.1 Đọc To Câu Trả Lời (TTS)

1. Trong khung chat, nhấn nút **🔊** cạnh câu trả lời của bot
2. Hệ thống gọi `POST /voice/speak` với văn bản câu trả lời (đã bỏ phần trích dẫn 📖/📎) và `profile_id` của giọng đang chọn
3. Nếu giọng đang chọn là giọng dựng sẵn (`kind="builtin"`) → phát trực tiếp qua **edge-TTS**
4. Nếu là giọng nhân bản của người dùng (`kind="cloned"`, trạng thái `ready`) → edge-TTS sinh giọng nền trước, sau đó chuyển qua **RVC** (gọi tới Colab tunnel) để chuyển thành giọng cá nhân

Giọng dựng sẵn có thể chọn (`voice_profiles`, kind=`builtin`, không gắn `user_id`):

| Tên | Giọng TTS |
|---|---|
| HoaiMy (Nữ) | `vi-VN-HoaiMyNeural` |
| NamMinh (Nam) | `vi-VN-NamMinhNeural` (mặc định) |
| Jenny (EN) | `en-US-JennyNeural` |
| F5-TTS demo (VN) | `f5tts:default` |

### 12.2 Tạo Giọng Nói Cá Nhân (RVC)

1. Nhấn nút **"🎙 Giọng nói của tôi"** trên sidebar — trang `/voice` mở ra (mọi vai trò truy cập được)
2. Lần đầu dùng: xác nhận **đồng ý thu âm** (`POST /voice/consent`) — bắt buộc trước khi tạo giọng nhân bản
3. Nhấn **tạo giọng mới** (`POST /voice/profiles`) — tối đa **2 giọng nhân bản mỗi người dùng** (giới hạn tải cho GPU Colab miễn phí giai đoạn thử nghiệm); phải xoá bớt giọng cũ nếu muốn tạo thêm
4. Đọc và ghi âm đủ tối thiểu **5 đoạn mẫu** theo kịch bản đọc sẵn (`GET /voice/scripts`), upload từng đoạn qua `POST /voice/profiles/{id}/samples`
5. Nhấn **"Huấn luyện"** (`POST /voice/profiles/{id}/train`) — job nền tải mẫu lên Colab RVC server, tự poll trạng thái mỗi 15s (tối đa 2 giờ) qua `GET /voice/profiles/{id}/status`
6. Trạng thái giọng chuyển tuần tự: `new` → `collecting` → `training` → `ready` (hoặc `failed` nếu lỗi) — chỉ giọng `ready` mới chọn được để đọc to câu trả lời

### 12.3 Quản Lý Giọng Nói — Vai Trò Admin

1. Nhấn nút quản lý giọng nói trên sidebar (chỉ Admin thấy) — trang `/admin/voice_models` mở ra
2. Xem danh sách toàn bộ giọng nhân bản của mọi người dùng (`GET /list_voice_models`)
3. Có thể **huấn luyện lại** (`POST /admin/voice_models/{id}/retrain`), **vô hiệu hóa** (`POST /admin/voice_models/{id}/disable`) hoặc **xoá hẳn** (`DELETE /admin/voice_models/{id}`) một giọng bất kỳ

### 12.4 Cấu Hình Endpoint RVC (Colab)

RVC chạy trên Colab (`colab/voice_server.ipynb`), lộ ra ngoài qua tunnel (VD ngrok). Endpoint được lưu trong bảng `app_settings` (key `rvc_endpoint`) — seed từ biến môi trường `RVC_ENDPOINT` khi khởi động lần đầu, sau đó admin có thể sửa trực tiếp qua giao diện quản lý giọng nói mà không cần khởi động lại app:

```bash
# Windows — chỉ cần khi seed lần đầu, sau đó sửa qua UI
set RVC_ENDPOINT=https://xxxx.ngrok.io
python app.py
```

---

## 13. Câu Hỏi Demo Gợi Ý

Sử dụng các câu hỏi sau để minh họa từng tính năng trong buổi demo:

### Câu Hỏi Định Nghĩa (`definition`)

```
Quy định về công ty hợp danh là gì?
Quy định về doanh nghiệp tư nhân là gì?
Thành viên hợp danh là gì theo Luật Doanh nghiệp?
Khái niệm công ty TNHH một thành viên là gì?
```

### Câu Hỏi Điều Kiện (`condition`)

```
Điều kiện để thành lập công ty TNHH là gì?
Yêu cầu về vốn góp trong công ty cổ phần như thế nào?
Ai được phép là người đại diện theo pháp luật?
Tên doanh nghiệp bị cấm trong những trường hợp nào?
```

### Câu Hỏi Thủ Tục (`procedure`)

```
Thủ tục đăng ký thành lập doanh nghiệp tư nhân gồm các bước nào?
Hồ sơ đăng ký thành lập công ty TNHH cần những gì?
Quy trình thay đổi đăng ký kinh doanh như thế nào?
Nộp hồ sơ đăng ký doanh nghiệp ở đâu?
```

### Câu Hỏi Tổng Quát (`general`)

```
Công ty cổ phần khác công ty TNHH như thế nào?
Quyền và nghĩa vụ của thành viên góp vốn là gì?
Trách nhiệm của thành viên hợp danh trong công ty hợp danh?
```

### Câu Hỏi Tình Huống (minh hoạ dữ liệu từ mục 7)

```
Nam 17 tuổi có thể tự đứng tên thành lập công ty TNHH một thành viên không?
Công chức có được đứng tên thành lập và làm Giám đốc công ty không?
```

### Câu Hỏi Meta Về Hệ Thống

```
Database đang lưu bao nhiêu điều luật?
Luật Doanh nghiệp có bao nhiêu điều?
```

### Câu Hỏi Ngoài Phạm Vi (để minh họa cơ chế từ chối)

```
Thủ tục ly hôn cần giấy tờ gì?
```

> **Mẹo demo:** Bắt đầu bằng câu hỏi định nghĩa để thấy hệ thống khớp chính xác điều luật. Sau đó chuyển sang câu hỏi thủ tục để thấy định dạng danh sách bước. Thử một câu hỏi ngoài phạm vi để minh họa cơ chế từ chối. Nhấn nút 🔊 để nghe câu trả lời đọc to (mục 12). Cuối cùng nhập PDF/DOCX/tình huống mới hoặc chạy Đánh giá RAG để minh họa các tính năng cho giáo viên/admin.

---

## 14. Kiến Trúc Kỹ Thuật

### 14.1 Cấu Trúc Thư Mục

```
rag-legal-assistant/
├── app.py                          # FastAPI app — tất cả routes (RAG + import + đánh giá + giọng nói)
├── engine/
│   ├── rag_engine.py                # Pipeline RAG hỏi đáp + whitelist trích dẫn + quản lý nguồn (Manage Law)
│   ├── import_law_engine.py         # Pipeline import PDF/DOCX: extract/OCR → phân đoạn → embedding
│   ├── import_scenario_engine.py    # Pipeline import DOCX tình huống (IRAC) → ChromaDB
│   ├── import_dataset_engine.py     # Import dataset Excel (150/200-updated) → ChromaDB
│   ├── import_account_engine.py     # Import tài khoản hàng loạt từ Excel
│   └── evaluate_engine.py           # Đánh giá chất lượng RAG (auto/llm, demo/all/test)
├── database/
│   ├── database.py                  # SQLite: users (3 role), chats, messages, voice_profiles, voice_samples, app_settings, const
│   └── reference_source.py          # Script rời — thêm thủ công vài điều luật tham khảo (không qua UI)
├── voice/
│   ├── voice_engine.py              # Điều phối TTS (edge-TTS) + huấn luyện giọng nhân bản (RVC)
│   ├── tts.py                       # edge-TTS (giọng dựng sẵn, xem mục 12.1)
│   ├── rvc_client.py                # Gọi RVC server chạy trên Colab qua tunnel
│   └── scripts.py                   # Kịch bản đọc mẫu dùng khi thu âm giọng nhân bản
├── colab/
│   └── voice_server.ipynb           # Notebook chạy RVC server trên Colab GPU miễn phí
├── templates/
│   ├── login.html                   # Đăng nhập + đổi mật khẩu
│   ├── index.html                   # Giao diện chat chính
│   ├── import_law.html              # Import PDF/DOCX + Import Tình huống + Import Dataset + Đánh giá RAG
│   ├── manage_accounts.html         # Quản lý tài khoản (Admin)
│   ├── manage_law.html              # Quản lý văn bản đã import — 3 tab (Admin)
│   ├── import_account.html          # Import tài khoản hàng loạt (Admin)
│   ├── voice_profile.html           # Trang "Giọng nói của tôi" (mọi vai trò)
│   └── admin_voice_models.html      # Trang quản lý giọng nói nhân bản (Admin)
├── chroma_db/                       # Vector database (ChromaDB)
├── Dataset/                          # Mọi file .xlsx dùng để đánh giá RAG (mục 9) — nơi duy nhất /list_datasets quét
│   ├── enterprise_law_full_rag_chatbot_dataset_200_updated.xlsx   # Dataset mới nhất (200 câu, cập nhật pháp lý 2025)
│   ├── example_sheet.xlsx            # File mẫu cấu trúc Dataset — không dùng để đánh giá
│   └── example_scenario.docx         # File mẫu cấu trúc Tình huống (mục 7)
├── voice_samples/                   # Đoạn ghi âm mẫu người dùng upload khi tạo giọng nhân bản
├── voice_storage/                   # Model giọng nhân bản đã huấn luyện xong (tải về từ Colab)
├── uploads_tmp/                     # Lưu file tạm thời khi import (bị xoá/di chuyển ngay sau khi xử lý xong)
├── eval_results_*.xlsx               # Kết quả đánh giá RAG chi tiết từng câu — chỉ giữ 2 file mới nhất (mục 9)
├── eval_results_latest.json          # Tóm tắt lần đánh giá gần nhất — hiển thị khi mở tab Đánh giá (mục 9)
├── chat.db                          # SQLite database
├── groqkey.txt                      # Groq API key (không commit lên git)
└── requirements.txt                  # Thư viện (gồm cả thư viện giọng nói)
```

### 14.2 API Endpoints

| Endpoint | Method | Mô tả | Quyền |
|---|---|---|---|
| `/` | GET | Trang chủ / đăng nhập | Public |
| `/login` | POST | Đăng nhập | Public |
| `/logout` | POST | Đăng xuất | Logged in |
| `/session_info` | GET | Thông tin phiên hiện tại | Public |
| `/change_password` | POST | Tự đổi mật khẩu | Public (cần đúng mật khẩu cũ) |
| `/get` | POST | Hỏi đáp RAG | Logged in |
| `/create_chat` | POST | Tạo chat mới | Logged in |
| `/list_chats` | GET | Danh sách chats | Logged in |
| `/get_chat_messages` | GET | Lịch sử tin nhắn | Logged in |
| `/rename_chat` | POST | Đổi tên chat | Logged in |
| `/delete_chat` | POST | Xóa chat | Logged in |
| `/import` | GET | Trang import văn bản luật / tình huống / dataset | Teacher, Admin |
| `/import_law` | POST | Upload PDF/DOCX để import | Teacher, Admin |
| `/import_status/{job_id}` | GET | Tiến trình import PDF/DOCX | Teacher, Admin |
| `/import_scenario` | POST | Upload DOCX tình huống để import | Teacher, Admin |
| `/import_scenario_status/{job_id}` | GET | Tiến trình import tình huống | Teacher, Admin |
| `/download_scenario_example` | GET | Tải file DOCX mẫu cho tình huống | Teacher, Admin |
| `/import_dataset` | POST | Upload dataset Excel | Teacher, Admin |
| `/import_dataset_status/{job_id}` | GET | Tiến trình import dataset | Teacher, Admin |
| `/list_datasets` | GET | Danh sách file `.xlsx` có thể dùng để đánh giá | Teacher, Admin |
| `/download_dataset_example` | GET | Tải file Excel mẫu cho dataset | Teacher, Admin |
| `/evaluate` | POST | Chạy đánh giá RAG (mode/split/dataset_file) | Teacher, Admin |
| `/evaluate_status/{job_id}` | GET | Tiến trình đánh giá | Teacher, Admin |
| `/latest_eval_result` | GET | Kết quả tóm tắt lần đánh giá gần nhất | Teacher, Admin |
| `/download_eval_result/{filename}` | GET | Tải file kết quả đánh giá (`eval_results_*.xlsx`) | Teacher, Admin |
| `/manage_accounts` | GET | Trang quản lý tài khoản | Admin |
| `/list_users` | GET | Danh sách tài khoản | Admin |
| `/toggle_user_status` | POST | Vô hiệu hóa / kích hoạt tài khoản | Admin |
| `/delete_user` | POST | Xoá tài khoản | Admin |
| `/import_account` | GET / POST | Trang & xử lý import tài khoản hàng loạt | Admin |
| `/download_account_template` | GET | Tải file Excel mẫu import tài khoản | Admin |
| `/manage_law` | GET | Trang quản lý văn bản đã import (3 tab) | Admin |
| `/list_law_sources` | GET | Danh sách văn bản pháp luật (nhóm theo số ký hiệu) | Admin |
| `/delete_law_source` | POST | Xoá một văn bản pháp luật theo số ký hiệu | Admin |
| `/list_dataset_sources` | GET | Danh sách dataset đã import (nhóm theo tên file) | Admin |
| `/delete_dataset_source` | POST | Xoá một dataset theo tên file | Admin |
| `/list_scenario_sources` | GET | Danh sách bộ tình huống đã import (nhóm theo tên file) | Admin |
| `/delete_scenario_source` | POST | Xoá một bộ tình huống theo tên file | Admin |
| `/voice` | GET | Trang "Giọng nói của tôi" | Logged in |
| `/voice/scripts` | GET | Kịch bản đọc mẫu để thu âm | Logged in |
| `/voice/profiles` | GET / POST | Danh sách / tạo giọng nói của bản thân | Logged in |
| `/voice/consent` | POST | Xác nhận đồng ý thu âm (bắt buộc trước khi tạo giọng) | Logged in |
| `/voice/profiles/{id}` | PUT / DELETE | Đổi tên, đặt mặc định / xoá giọng của bản thân | Logged in (chủ sở hữu) |
| `/voice/profiles/{id}/samples` | GET / POST / DELETE | Quản lý mẫu ghi âm của một giọng | Logged in (chủ sở hữu) |
| `/voice/profiles/{id}/train` | POST | Bắt đầu huấn luyện giọng nhân bản qua RVC | Logged in (chủ sở hữu) |
| `/voice/profiles/{id}/status` | GET | Tiến trình huấn luyện | Logged in (chủ sở hữu) |
| `/voice/speak` | POST | Đọc to một đoạn văn bản bằng giọng đã chọn (TTS/RVC) | Logged in |
| `/admin/voice_models` | GET | Trang quản lý mọi giọng nhân bản | Admin |
| `/list_voice_models` | GET | Danh sách mọi giọng nhân bản của mọi người dùng | Admin |
| `/admin/voice_models/{id}/retrain` | POST | Huấn luyện lại một giọng | Admin |
| `/admin/voice_models/{id}/disable` | POST | Vô hiệu hóa một giọng | Admin |
| `/admin/voice_models/{id}` | DELETE | Xoá hẳn một giọng | Admin |

### 14.3 Phân Loại Câu Hỏi RAG

| Loại | Từ khóa nhận dạng | Định dạng trả lời |
|---|---|---|
| `procedure` | trình tự, thủ tục, quy trình, các bước, hồ sơ, nộp ở đâu | Danh sách bước 1, 2, 3… |
| `condition` | điều kiện, yêu cầu, cần có, phải có | Liệt kê điều kiện |
| `definition` | là gì, khái niệm, định nghĩa, quy định về | Ngắn gọn + căn cứ điều luật |
| `general` | _(các câu hỏi khác)_ | Tự do, nêu đủ căn cứ pháp lý |

### 14.4 Whitelist Trích Dẫn (`CITATION_SOURCE`) & Nhãn Nguồn Gốc (`import_source`)

- **`CITATION_SOURCE`** (lưu trong `chat.db`, bảng `const`) là danh sách mọi `so_ky_hieu` **đang thực sự có mặt** trong ChromaDB tại thời điểm làm mới gần nhất. `build_citation()` trong `rag_engine.py` từ chối in ra bất kỳ số ký hiệu nào không nằm trong danh sách này — chặn trường hợp trích dẫn tới văn bản đã bị xoá hoặc metadata bị hỏng/giả mạo. Danh sách này tự làm mới sau mỗi lần import (mục 6, 8) và sau mỗi lần xoá (mục 11).
- Đoạn dữ liệu từ **Văn bản tình huống** (mục 7) không có `so_ky_hieu` nên không nằm trong whitelist — đây là chủ đích, không phải thiếu sót (xem giải thích ở mục 7).
- **`import_source`** là nhãn nội bộ (`"law"` / `"dataset"` / `"scenario"`) gắn vào từng đoạn dữ liệu để trang Manage Law (mục 11) biết đoạn đó thuộc luồng import nào. Dữ liệu import **trước khi** nhãn này tồn tại được tự động gắn nhãn suy luận lúc khởi động ứng dụng (`backfill_import_source_tags()`, chạy một lần, an toàn khi gọi lại nhiều lần) — dựa trên các dấu hiệu sẵn có trong metadata (VD: có `segment_index` → `"law"`; `doc_type="scenario_qa"` → `"scenario"`; `so_ky_hieu` khớp mã dataset mặc định và không có `segment_index` → `"dataset"`). Vài đoạn tham khảo thêm bằng tay qua `database/reference_source.py` (ngoài 3 luồng UI) không được gắn nhãn này, nên sẽ không xuất hiện ở tab Dataset/Tình huống của Manage Law — nhưng vẫn xuất hiện đúng ở tab Văn bản pháp luật (nhóm theo `so_ky_hieu`).

### 14.5 Giọng Nói: `voice_profiles` / `voice_samples` / `app_settings`

- **`voice_profiles`** lưu cả giọng dựng sẵn (`kind="builtin"`, `user_id` NULL, xem bảng ở mục 12.1) lẫn giọng nhân bản của người dùng (`kind="cloned"`), kèm trạng thái huấn luyện (`status`: `new`→`collecting`→`training`→`ready`/`failed`).
- **`voice_samples`** lưu đường dẫn từng file ghi âm mẫu, gắn với `profile_id` và kịch bản đọc (`script_id`, xem `voice/scripts.py`) — là dữ liệu đầu vào để huấn luyện RVC.
- **`app_settings`** là key/value store dùng cho cấu hình có thể đổi lúc chạy mà không cần restart app — hiện chỉ có `rvc_endpoint` (mục 12.4).
- Cả 3 bảng này độc lập với whitelist `CITATION_SOURCE`/`import_source` ở mục 14.4 — giọng nói không liên quan tới ChromaDB hay trích dẫn điều luật.

### 14.6 Schema Database SQLite

```
users          (user_id, user_name, password, role[0=student,1=teacher,2=admin], status[0=active,1=disabled])
chats          (id, student_id, title, created_at, role[0=student chat, 1=teacher chat])
messages       (id, chat_id, role[user|assistant], text, timestamp)
const          (name, content)   # key/value — VD name="CITATION_SOURCE" (xem mục 14.4)
voice_profiles (id, user_id, name, kind[builtin|cloned], base_tts_voice, speaker_id, status, is_default, error_message, model_local_path, created_at)
voice_samples  (id, profile_id, script_id, file_path, created_at)
app_settings   (key, value)      # key/value — VD key="rvc_endpoint" (xem mục 12.4)
```

Chat giáo viên và học sinh được tách biệt hoàn toàn theo cột `role`, ngay cả khi `user_id` trùng nhau. Admin dùng chung không gian chat với vai trò Teacher (`role=1`).

---

## 15. Xử Lý Sự Cố

### Lỗi Không Kết Nối Groq

**Triệu chứng:** Câu trả lời trả về `❌ Lỗi hệ thống.`

**Kiểm tra:**
- File `groqkey.txt` tồn tại và chứa API key hợp lệ (bắt đầu bằng `gsk_`)
- Có kết nối Internet
- API key chưa hết hạn / hết quota tháng

**Lưu ý:** Groq có tự động retry 3 lần (5s / 10s / 15s) khi gặp lỗi rate limit / timeout.

---

### ChromaDB Trống — Không Tìm Thấy Kết Quả

**Triệu chứng:** `⚠️ Không tìm thấy thông tin đủ liên quan trong cơ sở dữ liệu.`

**Nguyên nhân:** Chưa có dữ liệu trong ChromaDB.

**Khắc phục:**
```bash
# Import trực tiếp từ file PDF/DOCX qua giao diện giáo viên (xem mục 6)
# Hoặc import bộ tình huống DOCX (xem mục 7)
# Hoặc import nhanh từ dataset Excel có sẵn qua giao diện (xem mục 8)
python database/build_db_from_dataset_updated.py
```

---

### OCR Chạy Chậm

**Nguyên nhân:** Mặc định dùng CPU, chỉ kích hoạt khi PDF là bản scan. File 50 trang mất 10–30 phút.

**Tăng tốc với GPU (nếu có NVIDIA CUDA):**

Tạo file `.device_config` tại thư mục gốc:
```
DEVICE=cuda
```

---

### Lỗi Poppler Không Tìm Thấy

**Triệu chứng:** `PDFPageCountError` hoặc `Unable to get page count`

**Khắc phục:** Kiểm tra thư mục `poppler/Library/bin/` tồn tại và có các file binary (pdftoppm.exe, pdfinfo.exe…). Nếu thiếu, tải Poppler cho Windows từ https://github.com/oschwartz10612/poppler-windows/releases và giải nén vào đúng đường dẫn.

---

### Lỗi Đăng Nhập Thất Bại

**Triệu chứng:** Thông báo "Đăng nhập thất bại" hoặc "Tài khoản đã bị vô hiệu hóa"

**Kiểm tra:**
- Tên đăng nhập và mật khẩu phân biệt chữ hoa/thường
- Tài khoản mặc định (student/teacher/admin) chỉ được tạo khi bảng `users` **trống hoàn toàn** (lần khởi động đầu tiên) — nếu thiếu tài khoản admin, dùng Import tài khoản (mục 10) hoặc thêm thủ công vào bảng `users`
- Tài khoản bị Admin vô hiệu hóa sẽ không đăng nhập được cho tới khi được kích hoạt lại

---

### Đánh Giá RAG Dùng Sai Dataset

**Triệu chứng:** Kết quả đánh giá không phản ánh dataset mong muốn

**Nguyên nhân:** `/evaluate` đọc từ file `.xlsx` được chọn ở dropdown "File dataset dùng để đánh giá" (mục 9, bước 1), **không** phải từ dữ liệu vừa import vào ChromaDB qua mục 8 — đây là hai nguồn hoàn toàn khác nhau (file trên đĩa vs vector DB).

**Khắc phục:** Kiểm tra lại dropdown đã chọn đúng file mong muốn trước khi bấm Quick/Full Evaluation. Nếu file mới không xuất hiện trong dropdown, xác nhận file `.xlsx` đã nằm trong thư mục **`Dataset/`** (không phải thư mục gốc dự án) và có ít nhất một sheet đặt tên đúng quy ước `Dataset_*` hoặc `Demo_*`.

---

### Quick Evaluation Bị Disable / Báo Lỗi "Không Có Sheet Demo"

**Triệu chứng:** Nút Quick Evaluation bị mờ, hoặc chạy báo `❌ File '...' không có sheet Demo…`

**Nguyên nhân:** File dataset đang chọn chỉ có sheet `Dataset_*` (dùng được cho Full Evaluation) nhưng thiếu sheet `Demo_*`.

**Khắc phục:** Chọn file khác có sẵn sheet `Demo_*` trong dropdown, hoặc thêm một sheet đặt tên `Demo_<số>` vào file Excel đó rồi tải lại trang.

---

### Trả Lời Sai Nội Dung / Trích Dẫn Sai Điều Luật (VD: hỏi "Điều 143" ra nội dung Điều khác)

**Triệu chứng:** Đặt câu hỏi nêu rõ số Điều (VD: "Điều 143") hoặc một từ khóa ngắn (VD: "Tập đoàn") nhưng câu trả lời/trích dẫn không khớp với Điều luật thực sự nói về nội dung đó.

**Nguyên nhân đã phát hiện (2026-07-06):** Dữ liệu văn bản 67/VBHN-VPQH trong ChromaDB từng bị import bằng `database/build_db_doc.py` với regex tách "Điều X." gõ nhầm ký tự (`Dieu` ASCII thay vì `Điều` tiếng Việt) — regex này không bao giờ khớp, khiến toàn bộ văn bản bị cắt cứng thành từng đoạn 3000 ký tự bất kể ranh giới Điều luật, và metadata `article_number` chỉ là số thứ tự đoạn (không phải số Điều thật). Hệ quả: một đoạn có thể chứa nội dung của 2 Điều liền kề, và tra cứu theo số Điều/từ khóa ngắn trả về sai.

**Đã khắc phục:**
- Sửa regex trong `database/build_db_doc.py` và `engine/import_law_engine.py` để khớp đúng "Điều X." (ký tự Đ tiếng Việt)
- Cả 2 script giờ **cảnh báo rõ ràng** thay vì âm thầm gán số Điều giả nếu vẫn rơi vào fallback
- `engine/rag_engine.py` được thêm 2 cơ chế truy xuất mới: (1) tra thẳng theo metadata khi câu hỏi có dạng "Điều N", (2) quét từ khóa trực tiếp cho câu hỏi ngắn (≤5 từ, VD "Tập đoàn") — thay vì chỉ dựa vào tìm kiếm ngữ nghĩa

**Nếu ChromaDB hiện tại vẫn còn dữ liệu luật bị chunk sai** (dấu hiệu: metadata `article_number` trùng với `segment_index + 1`), chạy:
```bash
python database/rebuild_law_from_docx.py
```
Script này xóa các đoạn văn bản 67/VBHN-VPQH bị chunk sai và build lại đúng theo từng Điều từ file DOCX gốc (`67_VBHN-VPQH_671127 (1).docx`), **không đụng** tới dữ liệu Q&A/KB_Articles từ Dataset Excel.

---

### Điểm Đánh Giá `auto` Mode Thấp Bất Thường Dù Câu Trả Lời Đúng

**Triệu chứng:** Chạy Quick Evaluation, nhiều câu có `citation_correct = 0` và `hallucination` thấp dù xem thủ công thấy câu trả lời **trích dẫn đúng** điều luật yêu cầu.

**Nguyên nhân đã phát hiện (2026-07-21):** `_auto_score()` trong `engine/evaluate_engine.py` từng lấy số Điều bằng cách xoá hết ký tự không phải số trong toàn bộ chuỗi `article_reference` — với tham chiếu chỉ có một số (VD `"Điều 17"`) thì đúng, nhưng với tham chiếu có nhiều số (VD `"Khoản 35 Điều 4"` → gộp thành `"354"`, `"Điều 17 Nghị định 168/2025/NĐ-CP"` → gộp thành `"171682025"`) thì số bị gộp sai hoàn toàn và không bao giờ khớp được với văn bản trả lời thật, dù trả lời đúng 100%. Lỗi này ảnh hưởng 23/50 câu (46%) trong bộ dữ liệu `enterprise_law_full_rag_chatbot_dataset_200_updated.xlsx`, kéo điểm tổng từ mức thực tế ~73/100 xuống còn 66.7/100.

**Đã khắc phục:** Đổi sang trích riêng số theo sau từ khoá "Điều" (`_extract_article_numbers`), có fallback theo số hiệu văn bản/nghị định khi tham chiếu không chứa "Điều" (VD chỉ có `"76/2025/QH15"`). Xác minh trên dữ liệu thật: 14/50 câu chuyển từ chấm sai (0 điểm) sang chấm đúng (3 điểm), không có câu nào bị chấm sai theo chiều ngược lại.

**Nếu vẫn thấy điểm bất thường:** Kiểm tra định dạng cột `article_reference` trong file dataset — chấm điểm chỉ nhận diện được số Điều đứng ngay sau từ "Điều" (không phân biệt hoa/thường), các dạng viết tắt khác (VD chỉ ghi số Điều mà không có chữ "Điều") sẽ không được nhận diện.

---

### Không Tạo Được / Không Huấn Luyện Được Giọng Nói Cá Nhân

**Triệu chứng:** Nhấn "Huấn luyện" ở trang `/voice` nhưng trạng thái mãi ở `training` không chuyển sang `ready`, hoặc báo `failed`.

**Kiểm tra:**
- `app_settings.rvc_endpoint` đã được cấu hình đúng và Colab notebook (`colab/voice_server.ipynb`) đang chạy — RVC không chạy cục bộ, phụ thuộc hoàn toàn vào tunnel này còn sống
- Đã ghi đủ tối thiểu **5 mẫu** trước khi nhấn Huấn luyện
- Job huấn luyện tự poll tối đa 2 giờ (`MAX_TRAIN_WAIT_SEC`) — Colab notebook bị ngắt kết nối (hết phiên miễn phí) giữa chừng sẽ khiến job không bao giờ hoàn tất; cần khởi động lại notebook và nhấn Huấn luyện lại
- Đã đạt giới hạn **2 giọng nhân bản/người dùng** — phải xoá bớt giọng cũ trước khi tạo giọng mới

**Khắc phục:** Admin có thể chủ động **huấn luyện lại** giọng bị lỗi qua trang `/admin/voice_models` (mục 12.3) mà không cần người dùng ghi âm lại từ đầu.

---

## Ghi Chú Thêm

- Hệ thống hỗ trợ **đa phiên đồng thời** — nhiều người dùng có thể truy cập cùng lúc
- Dữ liệu chat được **lưu vĩnh viễn** trong `chat.db`; không mất khi restart
- ChromaDB **tích lũy dữ liệu** — import thêm văn bản/tình huống/dataset mới không xóa dữ liệu cũ, trừ khi admin chủ động xoá qua trang Manage Law (mục 11)
- Mọi câu trả lời đều kèm **📖 trích dẫn điều luật chính** và có thể có **📎 nguồn tham khảo phụ** + **🔗 link nguồn**, cùng nút **🔊 đọc to** (mục 12)
- Hệ thống có cơ chế **retry tự động** khi Groq bị rate limit (3 lần, backoff 5s/10s/15s)
- Admin quản lý được vòng đời tài khoản (kích hoạt/vô hiệu hóa/xoá), import tài khoản hàng loạt kèm báo cáo lỗi chi tiết, quản lý/xoá dữ liệu đã import theo cả 3 luồng (văn bản luật, dataset, tình huống), và quản lý giọng nói nhân bản của mọi người dùng
