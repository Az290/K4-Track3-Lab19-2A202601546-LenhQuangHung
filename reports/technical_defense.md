# Technical Defense — 10 Câu Hỏi Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Lệnh Quang Hưng
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

**Phạm vi dữ liệu thực nghiệm:** 5.000 dòng đầu tiên của `HackerNoon/tech-company-news-data-dump` (khớp với golden dataset `graphrag_golden_50_first5000.csv` do giảng viên cung cấp) → sau exact-dedup còn **2.105 bài báo** → **1.500 chunks** đưa vào Flat RAG index, **400 chunks** đưa vào extraction pipeline. Graph kết quả (lần chạy cuối cùng dùng để xuất `outputs/`): **362 nodes / 253 edges / 0 invalid_provenance_edges**.

---

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

  Trong lượt chạy trên scope 362 nodes / 253 edges (5000 dòng, 400 chunks extraction), **top-1 node chỉ đạt degree = 5** — thấp hơn rất nhiều so với ngưỡng `SUPER_NODE_DEGREE = 100` được thiết kế trong pipeline. Điều này có nghĩa **cơ chế super-node mitigation chưa từng được kích hoạt thật sự** trong phạm vi dữ liệu hiện tại; test `assert len(edges) <= 50` chỉ chạy nhánh else (`limit = 1000`, không cắt).
- **Nguyên nhân:** Với `EXTRACTION_MAX_CHUNKS = 400` (một phần rất nhỏ so với 2.105 bài báo đã dedup), số lượng edge quy tụ về một entity trung tâm (ví dụ Microsoft, Google, OpenAI) chưa đủ lớn để tạo bậc cao. Trong bản đầy đủ (~350MB, ~100.000 bài báo), các công ty lớn chắc chắn sẽ vượt ngưỡng 100 vì xuất hiện trong hàng nghìn bài viết.
- **Ưu điểm & Rủi ro của Temporal Mitigation (phân tích dựa trên thiết kế, áp dụng khi scale lớn):**
  - *Ưu điểm:* Giữ context đồ thị tập trung vào diễn biến gần nhất — với câu hỏi kiểu "Microsoft đầu tư gì gần đây?", 50 cạnh mới nhất trả lời chính xác hơn 1000+ cạnh cũ trộn lẫn ngẫu nhiên, đồng thời giữ `MAX_GRAPH_CONTEXT_CHARS` trong tầm kiểm soát, tránh tràn token khi gọi LLM.
  - *Rủi ro:* Nếu câu hỏi liên quan tới sự kiện lịch sử xa (ví dụ "Microsoft mua LinkedIn năm nào?" — sự kiện 2016), cạnh đó có thể bị cắt khỏi top-50 mới nhất nếu công ty có nhiều tin tức gần đây hơn, dẫn tới GraphRAG "quên" thông tin lịch sử quan trọng dù nó tồn tại trong graph.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge, trung bình theo nhóm, dữ liệu thật từ `outputs/graphrag_vs_flatrag_summary.csv`):

| Nhóm câu hỏi | Tiêu chí đánh giá | Flat RAG | GraphRAG | Δ (Graph − Flat) | Nhận xét |
|---|---|---|---|---|---|
| cross-doc | Comprehensiveness | 2.545 | 2.364 | −0.181 | Hai phương pháp gần nhau |
| cross-doc | Faithfulness | 2.909 | 2.727 | −0.182 | Hai phương pháp gần nhau |
| cross-doc | Multi-hop reasoning | 2.545 | 2.364 | −0.181 | Hai phương pháp gần nhau |
| cross-doc | Latency (s) | 2.201 | 1.943 | −0.258 | GraphRAG không đắt hơn |
| cross-doc | Token usage | 641.3 | 596.0 | −45.3 | GraphRAG không đắt hơn |
| factoid | Comprehensiveness | 2.500 | 3.000 | +0.500 | GraphRAG tốt hơn |
| factoid | Faithfulness | 3.000 | 3.000 | 0.000 | Ngang nhau |
| factoid | Multi-hop reasoning | 2.500 | 3.000 | +0.500 | GraphRAG tốt hơn |
| factoid | Latency (s) | 1.758 | 1.674 | −0.084 | GraphRAG không đắt hơn |
| factoid | Token usage | 602.5 | 502.5 | −100.0 | GraphRAG không đắt hơn |
| multi-hop | Comprehensiveness | 2.333 | 1.917 | −0.416 | Flat RAG tốt hơn |
| multi-hop | Faithfulness | 2.417 | 2.167 | −0.250 | Hai phương pháp gần nhau |
| multi-hop | Multi-hop reasoning | 2.333 | 1.917 | −0.416 | Flat RAG tốt hơn |
| multi-hop | Latency (s) | 2.749 | 2.827 | +0.078 | Flat RAG rẻ/nhanh hơn |
| multi-hop | Token usage | 676.3 | 742.2 | +65.9 | Flat RAG rẻ/nhanh hơn |

**Nhận xét tổng quát bất ngờ:** Trái với kỳ vọng lý thuyết, **GraphRAG chỉ vượt trội Flat RAG ở nhóm `factoid`** (Comprehensiveness/Multi-hop reasoning +0.5), trong khi ở nhóm `multi-hop` — nơi GraphRAG lẽ ra phải có lợi thế lớn nhất — **Flat RAG lại thắng rõ rệt** (−0.416 comprehensiveness, và lần đầu tiên GraphRAG cũng đắt hơn cả về latency lẫn token). Nguyên nhân chính (phân tích ở mục 3): graph quá thưa (362 nodes / 253 edges, degree tối đa chỉ 5) do `EXTRACTION_MAX_CHUNKS=400` quá nhỏ so với 2.105 bài báo — seed-entity matching thường không tìm đủ node liên quan trong graph, khiến `retrieve_graph_context()` trả về context nghèo nàn hơn Flat RAG (vốn luôn truy xuất được top-k chunk gần nghĩa nhất từ toàn bộ 1.500 chunks) đúng ở những câu hỏi multi-hop phức tạp cần nhiều bước suy luận. Đây là bài học quan trọng: **GraphRAG chỉ phát huy giá trị khi đồ thị đủ dày** (coverage đủ lớn so với corpus) — với factoid đơn giản, một vài cạnh chính xác vẫn đủ để GraphRAG thắng, nhưng multi-hop cần độ phủ sâu hơn nhiều mà quy mô 400 chunks chưa đáp ứng được.

*(Lưu ý: kết quả LLM-as-a-Judge không hoàn toàn deterministic giữa các lần chạy do `temperature=0.0` không đảm bảo 100% tái lập với model API — số liệu trên là từ lần chạy cuối cùng dùng để xuất `outputs/graphrag_eval_results.csv` nộp bài; xu hướng tổng thể — GraphRAG không vượt trội rõ rệt do graph thưa — nhất quán qua các lần chạy.)*

*(2 ca lỗi điển hình được phân tích chi tiết root-cause riêng trong `reports/failure_analysis.md`.)*

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent
> **Trade-offs, Agent Control & Scale 350MB:**
> - So sánh sự đánh đổi giữa GraphRAG vs Flat RAG về Latency, Token và Indexing Overhead.
> - Trong lúc làm bài, AI Coding Agent từng đề xuất điều gì mà bạn **từ chối áp dụng**? Tại sao?
> - Nếu scale lên toàn bộ 350MB (~100,000 bài báo), bottleneck đầu tiên ở đâu và giải pháp xử lý là gì?

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency (dữ liệu thật):** Bất ngờ là trong thực nghiệm này, **GraphRAG có latency thấp hơn** Flat RAG ở nhóm cross-doc và factoid (ví dụ factoid: 1.67s vs 1.76s) và **token usage thấp hơn** (factoid: 502.5 vs 602.5 tokens), nhưng ở nhóm multi-hop thì đảo ngược (GraphRAG đắt hơn cả latency lẫn token). Lý do là graph quá thưa nên `retrieve_graph_context()` trả về context ngắn (thường 0 supernode events, ít edge) ở phần lớn câu hỏi, nhưng với câu hỏi multi-hop phức tạp, BFS traversal + context vector bổ trợ lại làm tăng chi phí mà không tăng tương ứng độ chính xác. Chi phí *indexing overhead* thực sự của GraphRAG nằm ở **pha ingest**: coref (~427-450s cho 400 chunks) + extraction (~241-263s) + entity resolution — tổng cộng ~11-12 phút mỗi lần build lại graph, so với Flat RAG chỉ cần build FAISS index trong 24-53s.
- **Quyết định từ chối AI Coding Agent (chính là bản thân tôi trong vai trò agent hỗ trợ):** Trong quá trình chạy notebook, khi phát hiện Neo4j bị chặn kết nối TLS, giả thuyết đầu tiên (sai) là do phần mềm VPN lạ (`BestProxyVPN.exe`/iMyFone certificate) trên máy. Tôi đã **từ chối kết luận vội** dù bằng chứng ban đầu (process đang chạy, root cert lạ) khá thuyết phục — thay vào đó tiếp tục kiểm chứng bằng `openssl s_client` để xác nhận server gửi đủ chain hợp lệ, rồi mới phát hiện nguyên nhân thật là **Windows system CA store không build được chain cho domain `*.databases.neo4j.io`**, trong khi Python's bundled `certifi` package build được — giải pháp là set `SSL_CERT_FILE=certifi.where()`, không liên quan gì tới VPN. Việc gỡ cài VPN vẫn hợp lý về bảo mật nhưng không phải nguyên nhân gốc — bài học: **không dừng lại ở bằng chứng đầu tiên có vẻ hợp lý, luôn kiểm chứng bằng công cụ độc lập (openssl) trước khi kết luận nguyên nhân gốc rễ**.
- **Giải pháp scale 350MB (~100.000 bài báo):**
  1. **Bottleneck đầu tiên: LLM extraction tuần tự.** Với `EXTRACTION_MAX_CHUNKS` tăng lên hàng chục nghìn, pipeline hiện tại (for-loop tuần tự qua từng batch, mỗi batch chờ response LLM) sẽ mất hàng chục giờ. Giải pháp: **async batch extraction với worker pool** (ví dụ `asyncio.gather` giới hạn concurrency 10-20 request đồng thời) kết hợp rate-limit backoff thông minh (đọc `Retry-After` header thay vì `2^attempt` cố định — bài học rút ra trực tiếp từ sự cố rate-limit Groq 8000 TPM gặp trong lab này).
  2. **Bottleneck thứ hai: Entity Resolution O(N log N) qua FAISS nhưng vẫn cần build toàn bộ embedding matrix trong RAM.** Với hàng trăm nghìn entity mentions, cần chuyển sang **HNSW index với disk-backed storage** (ví dụ `faiss.IndexHNSWFlat` ghi ra đĩa) thay vì giữ toàn bộ trong RAM, hoặc **blocking theo entity_type + first-letter prefix** trước khi chạy ANN để giảm số lượng candidate cần so sánh.
  3. **Bottleneck thứ ba: Neo4j bulk insert với batch_size cố định 1000.** Ở quy mô lớn, cần **transaction batching động** theo kích thước dữ liệu thực tế và cân nhắc dùng `neo4j-admin import` cho lần nạp ban đầu (offline bulk import nhanh hơn nhiều so với Cypher `UNWIND` qua driver, dù mất khả năng transactional).

---

### 6. Bonus: Global Search via Community Reports (+5)
> Áp dụng NetworkX phân cụm cộng đồng, nạp `community_id`, sinh tóm tắt cộng đồng.

*Trả lời:*
- **Thực thi:** Gọi `build_communities()` (cell "Bonus — NetworkX community fallback") trên đồ thị Neo4j thật (362 nodes, 253 directed edges sau khi convert sang undirected). Dùng `nx.algorithms.community.greedy_modularity_communities()` để phân cụm, sau đó ghi `community_id` trở lại từng node qua `UNWIND` batch update.
- **Kết quả thật:** Tìm được **136 communities**. Phân bố kích thước: 1 cộng đồng lớn nhất có 7 thành viên, giảm dần xuống phần lớn (108/136) chỉ có 2 thành viên — phản ánh đúng bản chất đồ thị thưa (362 nodes/253 edges, xây từ chỉ 400 chunks extraction).
- **Ví dụ "Community Report" tự sinh từ 2 cộng đồng lớn nhất:**
  - **Community 0** (7 members): `institutions`, `57% stake institutions` (Company & Person — dấu hiệu lỗi extraction nhầm loại thực thể), `Information Services Group Inc.`, `Barry Holt`, `Information Services Group`, `GovernX` → chủ đề chung: **cấu trúc sở hữu và sản phẩm của Information Services Group** (công ty này cũng là top-1 super-node ở mục 3).
  - **Community 1** (6 members): `Claroty`, `ServiceNow`, `CLogic`, `Service Graph Connector`, `Vulnerability Response`, `generative AI` → chủ đề chung: **hệ sinh thái bảo mật OT/security tích hợp với ServiceNow**.
- **Đánh giá:** Cơ chế hoạt động đúng kỹ thuật (ghi `community_id` thành công, verify qua Cypher aggregation query khớp 362/362 node), nhưng giá trị thực tế của "Global Search" bị giới hạn bởi cùng nguyên nhân đã nêu ở mục 3 — đồ thị quá thưa nên phần lớn "cộng đồng" chỉ là các cặp/nhóm nhỏ 2-3 node, chưa đủ lớn để sinh ra "community report" tổng hợp có ý nghĩa như thiết kế gốc của GraphRAG Global Search (Microsoft's paper). Ở quy mô đầy đủ (~100.000 bài báo), kỹ thuật này sẽ phát huy giá trị rõ rệt hơn nhiều vì các công ty lớn sẽ tạo thành cộng đồng hàng chục-hàng trăm node với chủ đề rõ ràng (ví dụ toàn bộ hệ sinh thái AI của một Big Tech).
