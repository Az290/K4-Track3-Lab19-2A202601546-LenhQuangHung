# Reflection Cá Nhân — Mapping Bài Giảng & Action Plan — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Lệnh Quang Hưng
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

---

## 1. Mapping Bài giảng vào Code

| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()` | Chạy ổn định trên 400 chunks (~427-450s), nhưng do phần lớn chunk chỉ có 1 câu (dataset là description ngắn, không phải full-text), giá trị thực tế của bước này bị giới hạn — ít cơ hội để có đại từ cần phân giải. |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Hoạt động đúng thiết kế: `extraction_errors_df` rỗng (0 lỗi), toàn bộ 176 triples đều nằm trong allowlist — LLM tuân thủ tốt JSON schema khi được ràng buộc rõ trong system prompt. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | Chạy thành công với `UNWIND`, insert 362 nodes + 253 edges chỉ trong 1.7–3.8 giây — xác nhận đúng lợi ích của bulk insert so với insert từng dòng. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `UF` | Gộp đúng 5 cặp biến thể tên công ty (suffix Inc., viết tắt AWS), nhưng quy mô dữ liệu quá nhỏ để kiểm chứng đầy đủ khả năng chặn false-merge của lexical guard (0 trường hợp REJECT_GUARD thật). |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()` | Chưa được kích hoạt trong thực nghiệm (degree tối đa chỉ 5 << ngưỡng 100) — cần chạy ở quy mô lớn hơn để kiểm chứng cơ chế cắt tỉa cạnh hoạt động đúng. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()` | Chạy ổn định trên OpenAI `gpt-4o-mini`, sinh rationale rõ ràng và nhất quán với điểm số — hữu ích cho việc audit thủ công các ca lỗi (xem `reports/failure_analysis.md`). |
| **Global Search / Community Detection (Bonus)** | Bonus | `build_communities()` | Chạy thành công, tìm được 136 communities từ 362 nodes/253 edges bằng `nx.algorithms.community.greedy_modularity_communities()` — xem chi tiết trong `reports/technical_defense.md` mục 6. |

---

## 2. Quá trình Debugging & Bài học

- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Kết nối Neo4j AuraDB liên tục thất bại với lỗi `ServiceUnavailable: Unable to retrieve routing information`, dù URI/password đúng và cả 2 instance (cũ và mới tạo) đều gặp cùng lỗi. Quá trình debug trải qua nhiều bước sai lầm liên tiếp:
  1. Nghi ngờ instance Aura bị pause → resume không giải quyết.
  2. Nghi ngờ phần mềm `BestProxyVPN.exe`/certificate iMyFone can thiệp TLS → gỡ cài đặt + restart máy không giải quyết.
  3. Test bằng `openssl s_client -showcerts` phát hiện server thực ra gửi đủ chain hợp lệ (`Verify return code: 0 (ok)`) — chứng minh vấn đề không nằm ở server hay ở việc bị "man-in-the-middle".
  4. Test riêng Python `ssl.create_default_context(cafile=certifi.where())` thành công ngay lập tức → xác định chính xác nguyên nhân là **Windows system CA store thiếu/lỗi khi build chain cho domain neo4j.io cụ thể**, trong khi bundled `certifi` package của Python hoạt động độc lập với Windows store và không bị ảnh hưởng.

- **Cách xử lý thành công:** Thêm `os.environ.setdefault("SSL_CERT_FILE", certifi.where())` ngay đầu cell config (trước mọi import liên quan tới network: `neo4j`, `groq`, `openai`), buộc toàn bộ HTTPS/TLS trong notebook dùng CA bundle độc lập của certifi thay vì phụ thuộc vào Windows trust store. Bài học lớn nhất: **khi một lỗi "self-signed certificate" xuất hiện, đừng vội kết luận là bị chặn mạng/MITM — luôn kiểm chứng bằng công cụ ở tầng thấp hơn (openssl) trước khi debug ở tầng ứng dụng**, vì có thể vấn đề chỉ nằm ở cấu hình CA bundle của chính client.

- **Lỗi thứ hai đáng chú ý — Rate limit Groq (8000 TPM):** Ban đầu dùng model `openai/gpt-oss-120b` qua Groq cho toàn bộ pipeline coref/extraction/answer generation. Model reasoning này tiêu tốn nhiều "hidden reasoning tokens" (quan sát được `reasoning_tokens=503` cho 1 request nhỏ), khiến 80 batch coref liên tiếp nhanh chóng chạm giới hạn 8000 TPM của free tier, gây retry backoff cộng dồn (tổng thời gian ước tính vượt 20+ phút thay vì ~4 phút lý thuyết). Giải pháp: chuyển extraction pipeline sang OpenAI `gpt-4o-mini` (non-reasoning, tiêu ít token hơn, rate limit cao hơn) — xác nhận qua test độc lập chạy ổn định 427.5s cho đủ 80 batch không lỗi.

---

## 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)

- **Tên đồ án / Dự án:** Hệ thống trợ lý tra cứu tài liệu nội bộ cho doanh nghiệp vừa và nhỏ (tạm gọi "InternalDocs Assistant").

- **Đặc thù bài toán & Lý do chọn giải pháp:** Bài toán tra cứu tài liệu nội bộ (chính sách nhân sự, quy trình kỹ thuật, hợp đồng) thường có đặc điểm: (1) khối lượng tài liệu vừa phải (hàng trăm–hàng nghìn văn bản, không phải hàng triệu), (2) câu hỏi thực tế phần lớn là factoid/lookup đơn giản, chỉ một tỷ lệ nhỏ là multi-hop (ví dụ "quy trình nghỉ phép áp dụng cho nhân viên mới ký hợp đồng loại nào, và ai là người phê duyệt cuối cùng theo cấp bậc quản lý?"). Dựa trên bài học từ lab này — **GraphRAG chỉ có giá trị khi graph đủ dày và câu hỏi thực sự cần multi-hop** (kết quả thực nghiệm: GraphRAG chỉ thắng ở nhóm factoid, thua ở nhóm multi-hop do graph thưa) — tôi sẽ triển khai **kiến trúc Hybrid có điều kiện (Adaptive Routing)**: mặc định dùng Flat RAG (rẻ, nhanh, đã đủ tốt cho phần lớn truy vấn factoid theo dữ liệu thực nghiệm), chỉ kích hoạt GraphRAG khi seed-entity matching phát hiện câu hỏi có ≥2 entity liên quan cần nối quan hệ.

- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Document`, `Policy`, `Department`, `Employee`, `Process` (base label `Entity`, tương tự pattern `ALLOWED_NODE_TYPES` trong lab).
  - Relations: `APPROVED_BY`, `APPLIES_TO`, `SUPERSEDES` (văn bản mới thay thế văn bản cũ — quan trọng để tránh trả lời theo chính sách đã lỗi thời), `BELONGS_TO`, `REFERENCES`.

- **Chiến lược xử lý Super-node & Entity Resolution:** Do `Department`/`Policy` gốc (ví dụ "Phòng Nhân sự") sẽ là super-node tự nhiên (liên kết tới hàng trăm tài liệu), áp dụng cùng cơ chế `SUPER_NODE_DEGREE` + temporal cap như lab, nhưng ưu tiên theo **`effective_date` (ngày hiệu lực) thay vì `published_date`** để đảm bảo luôn ưu tiên chính sách hiện hành thay vì chính sách mới nhất về mặt đăng tải (một tài liệu cũ có thể được "republish" nhưng không có nghĩa là chính sách mới). Entity Resolution sẽ cần lexical guard chặt hơn cho tên phòng ban viết tắt (ví dụ "HR" vs "IT" dễ bị nhầm nếu chỉ dựa cosine similarity của embedding tiếng Anh chung) — dựa trên kinh nghiệm lab, ngưỡng `0.90` cosine + `0.72` SequenceMatcher là điểm khởi đầu hợp lý nhưng cần tinh chỉnh riêng cho domain tiếng Việt/song ngữ.

---

## 🎯 TỰ ĐÁNH GIÁ

| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Hiểu rõ kiến trúc, nhưng cần dữ liệu lớn hơn để kiểm chứng đầy đủ giá trị của super-node mitigation trong thực tế |
| Khả năng kiểm soát AI Coding Agent | 4 | Phát hiện và sửa đúng 2 lỗi hạ tầng nghiêm trọng (SSL CA store, rate limit model reasoning) thông qua kiểm chứng độc lập, không chấp nhận giả thuyết đầu tiên chưa được xác minh |
| Chất lượng đồ thị tri thức xây dựng | 3 | Graph hoạt động đúng kỹ thuật (0 invalid provenance, entity resolution chính xác) nhưng quá thưa (362 nodes/253 edges) để phát huy lợi thế GraphRAG so với Flat RAG trong benchmark |
| Khả năng phân tích và debug hệ thống | 5 | Truy vết thành công 2 sự cố hạ tầng phức tạp (Neo4j SSL, Groq rate limit) bằng phương pháp loại trừ có hệ thống (openssl, test độc lập từng thành phần) |
