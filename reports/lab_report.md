# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Lệnh Quang Hưng
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

**Phạm vi dữ liệu thực nghiệm:** 5.000 dòng đầu tiên của `HackerNoon/tech-company-news-data-dump` (khớp với golden dataset `graphrag_golden_50_first5000.csv` do giảng viên cung cấp) → sau exact-dedup còn **2.105 bài báo** → **1.500 chunks** đưa vào Flat RAG index, **400 chunks** đưa vào extraction pipeline. Graph kết quả: **348 nodes / 232 edges / 0 invalid_provenance_edges**.

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)
> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*
- **Ví dụ từ dữ liệu:** Nhiều chunk trong dataset chỉ có 1 câu duy nhất (do nguồn `description` của HackerNoon là đoạn tóm tắt ngắn, không phải full-text bài báo — ví dụ chunk `4bd7afdba71243b0dbcd::c0000`: *"Stewart Information Services Corporation is in the property and casualty insurance industry."*). Với các chunk 1 câu như vậy, không có đại từ nào để phân giải nên coref không tạo lỗi trực tiếp — nhưng đây chính là **nguồn gốc gián tiếp của lỗi ở bước Extraction phía sau**: vì mỗi chunk là một đoạn cực ngắn, tách rời khỏi ngữ cảnh đầy đủ của bài báo gốc, LLM ở bước NER+RE (Cell 2.1) đôi khi phải "đoán" quan hệ dựa trên rất ít tín hiệu.
- **Hiện tượng cụ thể quan sát được ở bước Answer Generation (không phải coref, mà là hệ quả domain-shift tương tự):** Với câu hỏi `G5000-26` ("What external technology provider is named inside Amazon's July AI-service expansion..."), GraphRAG trả lời sai hoàn toàn: gán nhầm nhà cung cấp công nghệ là **"Advanced Micro Devices Inc."** thay vì đáp án đúng **"Cohere"**. Nguyên nhân gốc: chunk `ec3ce18d568e33867724::c0000` (nói về AMD) và chunk chứa thông tin Cohere là hai chunk khác nhau nhưng đều liên quan tới Amazon; seed-entity matching đã trỏ context tới sai chunk vì không đủ tín hiệu phân biệt trong đoạn text ngắn.
- **Hậu quả đối với Graph:** Nếu coref phân giải sai đại từ như `"it"` hoặc `"the company"` trong một chunk có nhiều thực thể xuất hiện gần nhau (ví dụ 1 chunk vừa nhắc AWS vừa nhắc AMD), hệ quả trực tiếp là **False Edge** — quan hệ đúng ra thuộc về thực thể A lại bị gán nhầm cho thực thể B trong bảng `raw_triples_df`, sau đó lan truyền y nguyên vào Neo4j vì không có bước kiểm chứng lại evidence ở tầng entity resolution.

---

### 2. Entity Resolution Threshold & Lexical Guard
> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*
- **Ngưỡng cosine similarity:** `threshold = 0.90` (giá trị mặc định của `build_resolution_map()`), kết hợp `SequenceMatcher.ratio() >= 0.72` làm lexical guard.
- **Kết quả thực tế trên 400 chunks extraction:** `entity_resolution_audit_df` chỉ có **5 dòng**, toàn bộ đều là `MERGE_VECTOR` — **không có cặp nào bị `REJECT_GUARD`** trong lượt chạy này. Đây là điểm cần lưu ý: với quy mô 400 chunks / 176 triples, số lượng entity trùng lặp gần giống nhau (near-duplicate mentions) tự nhiên đã thấp, nên guard chưa có cơ hội can thiệp vào trường hợp nguy hiểm nào (ví dụ `Apple` vs `Apple Watch`, `Sam Altman` vs `Steve Altman` — các case kinh điển từ đề bài).
- **5 cặp MERGE_VECTOR thực tế đã gộp thành công:**
  | Entity A | Entity B | Similarity | Quyết định |
  |---|---|---|---|
  | Stewart Information Services Corporation | Stewart Information Services | 0.9228 | MERGE_VECTOR |
  | Information Services Group | Information Services Group Inc. | 0.9172 | MERGE_VECTOR |
  | Amazon Web Services (AWS) | Amazon Web Services | 0.9287 | MERGE_VECTOR |
  | Fidelity National Information Services | Fidelity National Information Services Inc. | 0.9245 | MERGE_VECTOR |
  | X90A | X90 | 0.9206 | MERGE_VECTOR |
  Cả 5 cặp đều là biến thể hậu tố hợp lệ (thêm/bớt "Inc.", viết tắt trong ngoặc) → guard đúng đắn cho gộp.
- **Đánh giá & khuyến nghị (Bonus Challenge B):** Vì scope 5000-dòng chưa sinh ra case REJECT_GUARD thật, tôi mô phỏng thủ công bằng cách chạy `merge_guard()` ngoài pipeline với cặp `"Apple"` vs `"Apple Watch"`: `strip_suffix()` giữ nguyên cả hai chuỗi (không có suffix công ty để cắt), `SequenceMatcher(None, "apple", "apple watch").ratio()` ≈ 0.63 — **thấp hơn ngưỡng 0.72** nên tự động bị từ chối dù cosine similarity của embedding có thể > 0.85 (do "Apple Watch" vẫn mang nhiều ngữ nghĩa liên quan tới "Apple"). Đây chính là lý do lexical guard cần thiết: **embedding similarity cao không đảm bảo cùng một thực thể** khi một chuỗi là sản phẩm/con của công ty kia.

---

### 3. Đồ thị & Super-node Mitigation
> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*
- **Top Super-node đo được (test_supernode_policy() thực tế):**

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
|------|--------------|---------------------|----------------------|
| 1 | Information Services Group | Company | 5 |

  Trong lượt chạy trên scope 348 nodes / 232 edges (5000 dòng, 400 chunks extraction), **top-1 node chỉ đạt degree = 5** — thấp hơn rất nhiều so với ngưỡng `SUPER_NODE_DEGREE = 100` được thiết kế trong pipeline. Điều này có nghĩa **cơ chế super-node mitigation chưa từng được kích hoạt thật sự** trong phạm vi dữ liệu hiện tại; test `assert len(edges) <= 50` chỉ chạy nhánh else (`limit = 1000`, không cắt).
- **Nguyên nhân:** Với `EXTRACTION_MAX_CHUNKS = 400` (một phần rất nhỏ so với 2.105 bài báo đã dedup), số lượng edge quy tụ về một entity trung tâm (ví dụ Microsoft, Google, OpenAI) chưa đủ lớn để tạo bậc cao. Trong bản đầy đủ (~350MB, ~100.000 bài báo), các công ty lớn chắc chắn sẽ vượt ngưỡng 100 vì xuất hiện trong hàng nghìn bài viết.
- **Ưu điểm & Rủi ro của Temporal Mitigation (phân tích dựa trên thiết kế, áp dụng khi scale lớn):**
  - *Ưu điểm:* Giữ context đồ thị tập trung vào diễn biến gần nhất — với câu hỏi kiểu "Microsoft đầu tư gì gần đây?", 50 cạnh mới nhất trả lời chính xác hơn 1000+ cạnh cũ trộn lẫn ngẫu nhiên, đồng thời giữ `MAX_GRAPH_CONTEXT_CHARS` trong tầm kiểm soát, tránh tràn token khi gọi LLM.
  - *Rủi ro:* Nếu câu hỏi liên quan tới sự kiện lịch sử xa (ví dụ "Microsoft mua LinkedIn năm nào?" — sự kiện 2016), cạnh đó có thể bị cắt khỏi top-50 mới nhất nếu công ty có nhiều tin tức gần đây hơn, dẫn tới GraphRAG "quên" thông tin lịch sử quan trọng dù nó tồn tại trong graph.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge, trung bình theo nhóm, dữ liệu thật từ `outputs/graphrag_vs_flatrag_summary.csv`):

| Nhóm câu hỏi | Tiêu chí đánh giá | Flat RAG | GraphRAG | Δ (Graph − Flat) | Nhận xét |
|---|---|---|---|---|---|
| cross-doc | Comprehensiveness | 2.545 | 2.182 | −0.363 | Hai phương pháp gần nhau |
| cross-doc | Faithfulness | 3.000 | 2.727 | −0.273 | Hai phương pháp gần nhau |
| cross-doc | Multi-hop reasoning | 2.545 | 2.182 | −0.363 | Hai phương pháp gần nhau |
| cross-doc | Latency (s) | 1.930 | 1.765 | −0.165 | GraphRAG không đắt hơn |
| cross-doc | Token usage | 636.2 | 594.9 | −41.3 | GraphRAG không đắt hơn |
| factoid | Comprehensiveness | 3.000 | 2.500 | −0.500 | Flat RAG tốt hơn |
| factoid | Faithfulness | 3.000 | 3.000 | 0.000 | Ngang nhau |
| factoid | Multi-hop reasoning | 3.000 | 2.500 | −0.500 | Flat RAG tốt hơn |
| factoid | Latency (s) | 1.739 | 1.202 | −0.537 | GraphRAG không đắt hơn |
| factoid | Token usage | 599.0 | 488.5 | −110.5 | GraphRAG không đắt hơn |
| multi-hop | Comprehensiveness | 2.250 | 1.917 | −0.333 | Hai phương pháp gần nhau |
| multi-hop | Faithfulness | 2.333 | 2.167 | −0.166 | Hai phương pháp gần nhau |
| multi-hop | Multi-hop reasoning | 2.250 | 1.917 | −0.333 | Hai phương pháp gần nhau |
| multi-hop | Latency (s) | 2.826 | 2.389 | −0.437 | GraphRAG không đắt hơn |
| multi-hop | Token usage | 670.1 | 655.1 | −15.0 | GraphRAG không đắt hơn |

**Nhận xét tổng quát bất ngờ:** Trái với kỳ vọng lý thuyết, **GraphRAG không vượt trội Flat RAG ở bất kỳ nhóm câu hỏi nào trong thực nghiệm này**, kể cả nhóm `multi-hop` — nơi GraphRAG lẽ ra phải có lợi thế lớn nhất. Nguyên nhân chính (phân tích ở mục 3): graph quá thưa (348 nodes / 232 edges, degree tối đa chỉ 5) do `EXTRACTION_MAX_CHUNKS=400` quá nhỏ so với 2.105 bài báo — seed-entity matching thường không tìm đủ node liên quan trong graph, khiến `retrieve_graph_context()` trả về context nghèo nàn hơn Flat RAG (vốn luôn truy xuất được top-k chunk gần nghĩa nhất từ toàn bộ 1.500 chunks). Đây là bài học quan trọng: **GraphRAG chỉ phát huy giá trị khi đồ thị đủ dày** (coverage đủ lớn so với corpus).

#### Phân tích 2 Ca lỗi Điển hình (trích trực tiếp từ `outputs/graphrag_eval_results.csv`):

1. **Ca lỗi GraphRAG thất bại nặng (Flat RAG thành công) — `G5000-26` (multi-hop):**
   - *Câu hỏi:* "What external technology provider is named inside Amazon's July AI-service expansion, and what other new AI capability is mentioned alongside it?"
   - *Điểm số:* Flat RAG = 5/5/5 (Comp/Faith/MultiHop) vs GraphRAG = 1/1/1.
   - *Tại sao GraphRAG thất bại?* Graph trả lời sai hoàn toàn: gán nhầm nhà cung cấp là **"Advanced Micro Devices Inc."** thay vì đáp án đúng **"Cohere"**. Judge rationale: *"The candidate incorrectly identifies the external technology provider as Advanced Micro Devices Inc., whereas the reference states it is Cohere."* Nguyên nhân gốc: seed-entity extraction từ câu hỏi trích ra "Amazon" làm seed chính, nhưng BFS traversal (`max_hops=2`) từ node Amazon trong graph thưa đã đi lạc sang cạnh `Amazon–AMD` (do cả hai đều xuất hiện trong cùng vùng dữ liệu về chip AI) thay vì cạnh `Amazon–Cohere` — vì graph chỉ có 232 edges nên không đủ granularity để phân biệt đúng ngữ cảnh "July AI-service expansion".
   - *Flat RAG thành công như thế nào?* Vector search trực tiếp trên 1.500 chunks tìm đúng chunk chứa cả "Cohere" và "conversational customer-service agents" nhờ embedding similarity ngữ nghĩa tốt, không phụ thuộc vào độ đầy đủ của graph.

2. **Ca lỗi GraphRAG thành công (Flat RAG thất bại) — `G5000-35` (cross-doc):**
   - *Câu hỏi:* "Contrast AWS's AMD-chip posture with HPE's AI-cloud posture. Which is a tentative hardware sourcing decision and which is a service offering?"
   - *Điểm số:* Flat RAG = 2/2/2 vs GraphRAG = 4/5/4.
   - *Nguyên nhân Flat RAG thất bại:* Flat RAG lấy nhầm chunk về HPE — trả lời HPE cung cấp *"AI-powered software solutions for supply chain planning"* thay vì đúng dịch vụ **AI-cloud cho LLM**. Judge: *"it inaccurately describes HPE's offering... the candidate mentions supply chain planning, which is not explicitly stated in the reference."* Đây là hệ quả kinh điển của vector search: top-k theo cosine similarity chọn nhầm chunk "gần nghĩa" (cùng nói về HPE + AI) nhưng sai chủ đề cụ thể.
   - *GraphRAG giải quyết ra sao?* Nhờ có cả context đồ thị (chỉ rõ quan hệ `AWS –(considering)→ AMD chip`) lẫn context vector bổ trợ, GraphRAG tổng hợp đúng: AWS là "tentative hardware sourcing", HPE là "service offering" — dù vẫn thiếu chi tiết cụ thể ("HPE's offerings are not provided in the context"), câu trả lời đúng chủ đề chính. Đây là minh chứng cho việc **kết hợp GRAPH + VECTOR context** (thiết kế hybrid của `answer_graph_rag()`) giúp giảm rủi ro "lạc đề" hoàn toàn so với Flat RAG thuần.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent
> **Trade-offs, Agent Control & Scale 350MB:**
> - So sánh sự đánh đổi giữa GraphRAG vs Flat RAG về Latency, Token và Indexing Overhead.
> - Trong lúc làm bài, AI Coding Agent từng đề xuất điều gì mà bạn **từ chối áp dụng**? Tại sao?
> - Nếu scale lên toàn bộ 350MB (~100,000 bài báo), bottleneck đầu tiên ở đâu và giải pháp xử lý là gì?

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency (dữ liệu thật):** Bất ngờ là trong thực nghiệm này, **GraphRAG có latency thấp hơn** Flat RAG ở cả 3 nhóm (ví dụ factoid: 1.20s vs 1.74s) và **token usage thấp hơn** (factoid: 488.5 vs 599.0 tokens). Điều này đi ngược trực giác thông thường ("GraphRAG luôn đắt hơn vì có thêm bước graph traversal") — lý do là graph quá thưa nên `retrieve_graph_context()` trả về context ngắn (thường 0 supernode events, ít edge), khiến tổng context đưa vào LLM answer generation nhỏ hơn cả Flat RAG (vốn luôn lấy đủ k=6 chunk). Chi phí *indexing overhead* thực sự của GraphRAG nằm ở **pha ingest**: coref (424–450s cho 400 chunks) + extraction (241–259s) + entity resolution — tổng cộng ~11-12 phút mỗi lần build lại graph, so với Flat RAG chỉ cần build FAISS index trong 24-45s.
- **Quyết định từ chối AI Coding Agent (chính là bản thân tôi trong vai trò agent hỗ trợ):** Trong quá trình chạy notebook, khi phát hiện Neo4j bị chặn kết nối TLS, giả thuyết đầu tiên (sai) là do phần mềm VPN lạ (`BestProxyVPN.exe`/iMyFone certificate) trên máy. Tôi đã **từ chối kết luận vội** dù bằng chứng ban đầu (process đang chạy, root cert lạ) khá thuyết phục — thay vào đó tiếp tục kiểm chứng bằng `openssl s_client` để xác nhận server gửi đủ chain hợp lệ, rồi mới phát hiện nguyên nhân thật là **Windows system CA store không build được chain cho domain `*.databases.neo4j.io`**, trong khi Python's bundled `certifi` package build được — giải pháp là set `SSL_CERT_FILE=certifi.where()`, không liên quan gì tới VPN. Việc gỡ cài VPN vẫn hợp lý về bảo mật nhưng không phải nguyên nhân gốc — bài học: **không dừng lại ở bằng chứng đầu tiên có vẻ hợp lý, luôn kiểm chứng bằng công cụ độc lập (openssl) trước khi kết luận nguyên nhân gốc rễ**.
- **Giải pháp scale 350MB (~100.000 bài báo):**
  1. **Bottleneck đầu tiên: LLM extraction tuần tự.** Với `EXTRACTION_MAX_CHUNKS` tăng lên hàng chục nghìn, pipeline hiện tại (for-loop tuần tự qua từng batch, mỗi batch chờ response LLM) sẽ mất hàng chục giờ. Giải pháp: **async batch extraction với worker pool** (ví dụ `asyncio.gather` giới hạn concurrency 10-20 request đồng thời) kết hợp rate-limit backoff thông minh (đọc `Retry-After` header thay vì `2^attempt` cố định — bài học rút ra trực tiếp từ sự cố rate-limit Groq 8000 TPM gặp trong lab này).
  2. **Bottleneck thứ hai: Entity Resolution O(N log N) qua FAISS nhưng vẫn cần build toàn bộ embedding matrix trong RAM.** Với hàng trăm nghìn entity mentions, cần chuyển sang **HNSW index với disk-backed storage** (ví dụ `faiss.IndexHNSWFlat` ghi ra đĩa) thay vì giữ toàn bộ trong RAM, hoặc **blocking theo entity_type + first-letter prefix** trước khi chạy ANN để giảm số lượng candidate cần so sánh.
  3. **Bottleneck thứ ba: Neo4j bulk insert với batch_size cố định 1000.** Ở quy mô lớn, cần **transaction batching động** theo kích thước dữ liệu thực tế và cân nhắc dùng `neo4j-admin import` cho lần nạp ban đầu (offline bulk import nhanh hơn nhiều so với Cypher `UNWIND` qua driver, dù mất khả năng transactional).

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coref_batch()` | Chạy ổn định trên 400 chunks (424–450s), nhưng do phần lớn chunk chỉ có 1 câu (dataset là description ngắn, không phải full-text), giá trị thực tế của bước này bị giới hạn — ít cơ hội để có đại từ cần phân giải. |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `ALLOWED_RELATIONS` | Hoạt động đúng thiết kế: `extraction_errors_df` rỗng (0 lỗi), toàn bộ 176 triples đều nằm trong allowlist — LLM tuân thủ tốt JSON schema khi được ràng buộc rõ trong system prompt. |
| **Bulk Cypher Ingestion** | Module 2 | `bulk_insert_nodes()`, `bulk_insert_edges()` | Chạy thành công với `UNWIND`, insert 348 nodes + 232 edges chỉ trong 1.7–3.8 giây — xác nhận đúng lợi ích của bulk insert so với insert từng dòng. |
| **Entity Resolution & Union-Find** | Module 3 | `build_resolution_map()`, `UF` | Gộp đúng 5 cặp biến thể tên công ty (suffix Inc., viết tắt AWS), nhưng quy mô dữ liệu quá nhỏ để kiểm chứng đầy đủ khả năng chặn false-merge của lexical guard. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()` | Chưa được kích hoạt trong thực nghiệm (degree tối đa chỉ 5 << ngưỡng 100) — cần chạy ở quy mô lớn hơn để kiểm chứng cơ chế cắt tỉa cạnh hoạt động đúng. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()` | Chạy ổn định trên OpenAI `gpt-4o-mini`, sinh rationale rõ ràng và nhất quán với điểm số — hữu ích cho việc audit thủ công các case lỗi (như G5000-26, G5000-35 phân tích ở trên). |

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Kết nối Neo4j AuraDB liên tục thất bại với lỗi `ServiceUnavailable: Unable to retrieve routing information`, dù URI/password đúng và cả 2 instance (cũ và mới tạo) đều gặp cùng lỗi. Quá trình debug trải qua nhiều bước sai lầm liên tiếp: (1) nghi ngờ instance Aura bị pause → resume không giải quyết; (2) nghi ngờ phần mềm `BestProxyVPN.exe`/certificate iMyFone → gỡ cài đặt + restart không giải quyết; (3) test bằng `openssl s_client -showcerts` phát hiện server thực ra gửi đủ chain hợp lệ (`Verify return code: 0 (ok)`) — chứng minh vấn đề không nằm ở server hay ở việc bị "man-in-the-middle"; (4) test riêng Python `ssl.create_default_context(cafile=certifi.where())` thành công ngay lập tức → xác định chính xác nguyên nhân là **Windows system CA store thiếu/lỗi khi build chain cho domain neo4j.io cụ thể**, trong khi bundled `certifi` package của Python hoạt động độc lập với Windows store và không bị ảnh hưởng.
- **Cách xử lý thành công:** Thêm `os.environ.setdefault("SSL_CERT_FILE", certifi.where())` ngay đầu cell config (trước mọi import liên quan tới network: `neo4j`, `groq`, `openai`), buộc toàn bộ HTTPS/TLS trong notebook dùng CA bundle độc lập của certifi thay vì phụ thuộc vào Windows trust store. Bài học lớn nhất: **khi một lỗi "self-signed certificate" xuất hiện, đừng vội kết luận là bị chặn mạng/MITM — luôn kiểm chứng bằng công cụ ở tầng thấp hơn (openssl) trước khi debug ở tầng ứng dụng**, vì có thể vấn đề chỉ nằm ở cấu hình CA bundle của chính client.
- **Lỗi thứ hai đáng chú ý — Rate limit Groq (8000 TPM):** Ban đầu dùng model `openai/gpt-oss-120b` qua Groq cho toàn bộ pipeline coref/extraction/answer generation. Model reasoning này tiêu tốn nhiều "hidden reasoning tokens" (quan sát được `reasoning_tokens=503` cho 1 request nhỏ), khiến 80 batch coref liên tiếp nhanh chóng chạm giới hạn 8000 TPM của free tier, gây retry backoff cộng dồn (tổng thời gian ước tính vượt 20+ phút thay vì ~4 phút lý thuyết). Giải pháp: chuyển extraction pipeline sang OpenAI `gpt-4o-mini` (non-reasoning, tiêu ít token hơn, rate limit cao hơn) — xác nhận qua test độc lập chạy ổn định 427.5s cho đủ 80 batch không lỗi.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án / Dự án:** Hệ thống trợ lý tra cứu tài liệu nội bộ cho doanh nghiệp vừa và nhỏ (tạm gọi "InternalDocs Assistant").
- **Đặc thù bài toán & Lý do chọn giải pháp:** Bài toán tra cứu tài liệu nội bộ (chính sách nhân sự, quy trình kỹ thuật, hợp đồng) thường có đặc điểm: (1) khối lượng tài liệu vừa phải (hàng trăm–hàng nghìn văn bản, không phải hàng triệu), (2) câu hỏi thực tế phần lớn là factoid/lookup đơn giản, chỉ một tỷ lệ nhỏ là multi-hop (ví dụ "quy trình nghỉ phép áp dụng cho nhân viên mới ký hợp đồng loại nào, và ai là người phê duyệt cuối cùng theo cấp bậc quản lý?"). Dựa trên bài học từ lab này — **GraphRAG chỉ có giá trị khi graph đủ dày và câu hỏi thực sự cần multi-hop** — tôi sẽ triển khai **kiến trúc Hybrid có điều kiện (Adaptive Routing)**: mặc định dùng Flat RAG (rẻ, nhanh, đã đủ tốt cho phần lớn truy vấn factoid theo dữ liệu thực nghiệm), chỉ kích hoạt GraphRAG khi seed-entity matching phát hiện câu hỏi có ≥2 entity liên quan cần nối quan hệ.
- **Cấu trúc Node & Relation dự kiến:**
  - Nodes: `Document`, `Policy`, `Department`, `Employee`, `Process` (base label `Entity`, tương tự pattern `ALLOWED_NODE_TYPES` trong lab).
  - Relations: `APPROVED_BY`, `APPLIES_TO`, `SUPERSEDES` (văn bản mới thay thế văn bản cũ — quan trọng để tránh trả lời theo chính sách đã lỗi thời), `BELONGS_TO`, `REFERENCES`.
- **Chiến lược xử lý Super-node & Entity Resolution:** Do `Department`/`Policy` gốc (ví dụ "Phòng Nhân sự") sẽ là super-node tự nhiên (liên kết tới hàng trăm tài liệu), áp dụng cùng cơ chế `SUPER_NODE_DEGREE` + temporal cap như lab, nhưng ưu tiên theo **`effective_date` (ngày hiệu lực) thay vì `published_date`** để đảm bảo luôn ưu tiên chính sách hiện hành thay vì chính sách mới nhất về mặt đăng tải (một tài liệu cũ có thể được "republish" nhưng không có nghĩa là chính sách mới). Entity Resolution sẽ cần lexical guard chặt hơn cho tên phòng ban viết tắt (ví dụ "HR" vs "IT" dễ bị nhầm nếu chỉ dựa cosine similarity của embedding tiếng Anh chung).

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 4 | Hiểu rõ kiến trúc, nhưng cần dữ liệu lớn hơn để kiểm chứng đầy đủ giá trị của super-node mitigation trong thực tế |
| Khả năng kiểm soát AI Coding Agent | 4 | Phát hiện và sửa đúng 2 lỗi hạ tầng nghiêm trọng (SSL CA store, rate limit model reasoning) thông qua kiểm chứng độc lập, không chấp nhận giả thuyết đầu tiên chưa được xác minh |
| Chất lượng đồ thị tri thức xây dựng | 3 | Graph hoạt động đúng kỹ thuật (0 invalid provenance, entity resolution chính xác) nhưng quá thưa (348 nodes/232 edges) để phát huy lợi thế GraphRAG so với Flat RAG trong benchmark |
| Khả năng phân tích và debug hệ thống | 5 | Truy vết thành công 2 sự cố hạ tầng phức tạp (Neo4j SSL, Groq rate limit) bằng phương pháp loại trừ có hệ thống (openssl, test độc lập từng thành phần) |
