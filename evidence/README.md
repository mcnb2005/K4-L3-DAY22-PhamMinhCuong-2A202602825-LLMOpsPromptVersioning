# Evidence checklist

Thư mục này chỉ chứa bằng chứng được tạo từ lần chạy thật trên tài khoản của học viên.
Các ảnh được kết xuất tự động từ log/JSON xác minh; không chỉnh sửa thủ công số liệu.

- Học viên: **Phạm Minh Cương**
- MSSV: **2A202602825**

## Đã tạo từ lần chạy local

- `01_langsmith_trace_counts.txt`: API LangSmith xác nhận 50 trace `rag-query` và
  50 trace `ab-rag-query` trong project `day22-lab`.
- `01_langsmith_traces.png`: bảng bằng chứng được kết xuất từ log API ở trên;
  số liệu đã được đối chiếu trên giao diện LangSmith đăng nhập với bộ lọc
  `name:"rag-query"` hiển thị `Stats · 50 traces`.
- `02_prompt_hub_verification.txt`: API LangSmith xác nhận cả hai prompt cá nhân
  tồn tại và đều nhận đúng hai biến `context`, `question`.
- `02_prompt_hub.png`: bảng bằng chứng được kết xuất từ log API ở trên; hai prompt,
  commit và chủ sở hữu đã được đối chiếu trên giao diện Prompt Hub đăng nhập.
- `02_ab_routing_log.txt`: đủ 50 câu; routing tất định V1=19 và V2=31.
- `03_ragas_report.json`: báo cáo thật; cả V1 và V2 đều đủ 50 mẫu cho bốn metric.
- `03_ragas_scores.txt`: bảng điểm dạng text được trích từ báo cáo hoàn chỉnh.
- `03_ragas_scores.png`: ảnh bảng điểm được kết xuất trực tiếp từ
  `03_ragas_report.json`; có tên học viên, MSSV và trạng thái đạt mục tiêu.
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
