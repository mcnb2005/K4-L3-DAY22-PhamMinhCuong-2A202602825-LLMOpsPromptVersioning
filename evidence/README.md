# Evidence checklist

Thư mục này chỉ chứa bằng chứng được tạo từ lần chạy thật trên tài khoản của học viên.
Ba tệp PNG bắt buộc là ảnh chụp trực tiếp do học viên thực hiện từ LangSmith và
PowerShell; không dùng ảnh kết xuất hoặc ảnh sinh tự động.

- Học viên: **Phạm Minh Cương**
- MSSV: **2A202602825**

## Đã tạo từ lần chạy local

- `01_langsmith_trace_counts.txt`: API LangSmith xác nhận 50 trace `rag-query` và
  50 trace `ab-rag-query` trong project `day22-lab`.
- `01_langsmith_traces.png`: ảnh chụp trực tiếp giao diện LangSmith đăng nhập với
  project `day22-lab`, bộ lọc `name:"rag-query"` và `Stats · 50 traces`.
- `02_prompt_hub_verification.txt`: API LangSmith xác nhận cả hai prompt cá nhân
  tồn tại và đều nhận đúng hai biến `context`, `question`.
- `02_prompt_hub.png`: ảnh chụp trực tiếp danh sách Prompt Hub, hiển thị hai prompt,
  kiểu `ChatPromptTemplate`, chủ sở hữu và commit gần nhất.
- `02_ab_routing_log.txt`: đủ 50 câu; routing tất định V1=19 và V2=31.
- `03_ragas_report.json`: báo cáo thật; cả V1 và V2 đều đủ 50 mẫu cho bốn metric.
- `03_ragas_scores.txt`: bảng điểm dạng text được trích từ báo cáo hoàn chỉnh.
- `03_ragas_scores.png`: ảnh chụp trực tiếp PowerShell khi hiển thị bảng điểm thật
  từ `03_ragas_scores.txt`; có tên học viên, MSSV và trạng thái đạt mục tiêu.
- `04_pii_demo_log.txt`: đủ sáu ca PII; email, phone, SSN và thẻ đều được che.
- `04_json_demo_log.txt`: đủ năm ca JSON; các lỗi sửa được và fallback đều pass.

## Kết quả RAGAS hiện có

- V1: faithfulness `0.9633`, answer relevancy `0.9051`, context recall `1.0000`,
  context precision `0.9417` (đủ 50 mẫu/metric).
- V2: faithfulness `0.9463`, answer relevancy `0.8819`, context recall `1.0000`,
  context precision `0.9450` (đủ 50 mẫu/metric).
- Mục tiêu faithfulness ≥ `0.8` đã đạt bằng kết quả V1 đầy đủ.

## Trạng thái evidence

Đã có đủ bảy tệp evidence bắt buộc theo cấu trúc nộp bài.

## Phân tích V1 và V2

V1 ưu tiên câu trả lời ngắn; V2 ưu tiên cách trình bày có cấu trúc. Trên 50 mẫu cho
mỗi phiên bản, V1 tốt hơn về faithfulness và answer relevancy; V2 tốt hơn nhẹ về
context precision; context recall hòa nhau ở `1.0000`. V1 là lựa chọn tổng thể tốt hơn.
