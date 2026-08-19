# Failure Analysis — Phân Tích Ca Lỗi (Root-Cause) — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Lệnh Quang Hưng
**Khóa học:** AICB-K34 · Track 3: GraphRAG
**Ngày thực hiện:** 19/08/2026

Nguồn dữ liệu: trích trực tiếp từ `outputs/graphrag_eval_results.csv` (25 câu hỏi golden, LLM-as-a-Judge trên OpenAI `gpt-4o-mini`), scope thực nghiệm 5.000 dòng đầu `HackerNoon/tech-company-news-data-dump` → graph 362 nodes / 253 edges.

---

## Ca lỗi 1: GraphRAG thất bại nặng (Flat RAG thành công) — `G5000-26` (multi-hop)

**Câu hỏi:** "What external technology provider is named inside Amazon's July AI-service expansion, and what other new AI capability is mentioned alongside it?"

**Reference answer:** Amazon's AI-service story names access to technology from **Cohere**. It also mentions a program for building more conversational customer-service agents.

**Điểm số:** Flat RAG = 5/5/5 (Comprehensiveness/Faithfulness/Multi-hop reasoning) vs GraphRAG = 1/1/1 — chênh lệch lớn nhất trong toàn bộ 25 câu (Δ = 4.0).

### Root-cause analysis

1. **Triệu chứng:** GraphRAG trả lời sai hoàn toàn — gán nhầm nhà cung cấp công nghệ là **"Advanced Micro Devices Inc." (AMD)** thay vì đáp án đúng **"Cohere"**.

2. **Bằng chứng từ Judge rationale:** *"The candidate incorrectly identifies the external technology provider as Advanced Micro Devices Inc., whereas the reference states it is Cohere. Additionally, the candidate mentions a program for creating and deploying AI products, which does not align with the reference's mention of conversational customer-service agents..."*

3. **Truy vết nguyên nhân theo pipeline:**
   - **Bước Seed Extraction** (`extract_seeds()`): từ câu hỏi, LLM trích ra "Amazon" làm seed entity chính — hợp lý, đúng.
   - **Bước Seed Matching** (`match_seeds()`): node "Amazon" (hoặc biến thể "Amazon Web Services (AWS)") được match thành công trong graph.
   - **Bước BFS Traversal** (`retrieve_graph_context()`, `max_hops=2`): đây là điểm gãy. Từ node Amazon, BFS đi tới các cạnh liên quan — nhưng vì graph chỉ có 253 edges tổng (rất thưa), các cạnh liên quan tới Amazon lẫn lộn giữa nhiều chủ đề khác nhau (đầu tư AI, chip AMD, dịch vụ cloud...). Cạnh `Amazon–AMD` (từ chunk `ec3ce18d568e33867724::c0000`, nói về AWS cân nhắc dùng chip AMD) và cạnh `Amazon–Cohere` (nói về AI service tháng 7) đều nằm trong bán kính 2-hop, nhưng traversal ưu tiên theo `published_date DESC` (`recent_edges()`) — nếu chunk AMD có ngày xuất bản gần hơn, nó sẽ được ưu tiên đưa vào context trước.
   - **Bước Context Textualization:** với `GLOBAL_EDGE_CAP=250` và graph chỉ có 253 edges, gần như toàn bộ graph liên quan bị nhồi vào context, khiến LLM answer generation nhận được tín hiệu nhiễu (cả AMD lẫn Cohere đều xuất hiện) và chọn sai — có thể do AMD được nhắc tới sớm hơn/nổi bật hơn trong đoạn context textualized.

4. **Vì sao Flat RAG thành công:** Vector search (`retrieve_flat_context()`, k=6) tìm trực tiếp theo cosine similarity ngữ nghĩa của toàn câu hỏi (không tách seed entity), nên chunk chứa đúng cả "Cohere" và "conversational customer-service agents" được xếp hạng cao nhất nhờ độ tương đồng ngữ nghĩa tổng thể với câu hỏi — không bị nhiễu bởi các cạnh đồ thị không liên quan.

### Đề xuất khắc phục
- Tăng `EXTRACTION_MAX_CHUNKS` để graph dày hơn, giảm tỷ lệ nhiễu giữa các cạnh không liên quan cùng chạm 1 node trung tâm.
- Cải tiến seed matching để trích nhiều seed hơn (không chỉ "Amazon" mà cả "July", "AI-service" như một cụm ngữ cảnh) nhằm thu hẹp phạm vi traversal.
- Cân nhắc rerank các cạnh theo cosine similarity với câu hỏi gốc (không chỉ theo `published_date`) trước khi đưa vào context.

---

## Ca lỗi 2: GraphRAG thành công (Flat RAG thất bại) — `G5000-35` (cross-doc)

**Câu hỏi:** "Contrast AWS's AMD-chip posture with HPE's AI-cloud posture. Which is a tentative hardware sourcing decision and which is a service offering?"

**Reference answer:** AWS was only considering AMD's new AI chips (tentative). HPE offers a cloud computing service for LLMs (service offering).

**Điểm số:** Flat RAG = 2/2/2 vs GraphRAG = 4/5/4 — GraphRAG thắng rõ rệt (Δ = −1.33, tính theo Flat − Graph).

### Root-cause analysis

1. **Triệu chứng:** Flat RAG lấy nhầm chunk mô tả HPE — trả lời HPE cung cấp *"AI-powered software solutions for supply chain planning"* thay vì đúng dịch vụ **AI-cloud cho LLM (large language models)**.

2. **Bằng chứng từ Judge rationale (Flat):** *"...it inaccurately describes HPE's offering. HPE's focus is on AI-cloud services, but the candidate mentions supply chain planning, which is not explicitly stated in the reference. The reasoning lacks depth and clarity..."*

3. **Truy vết nguyên nhân — vì sao Flat RAG thất bại:** Đây là hệ quả kinh điển của vector search top-k (k=6): câu hỏi đề cập "HPE" + "AI-cloud", và trong corpus có ít nhất 2 bài báo khác nhau về HPE liên quan tới AI (một về cloud LLM, một về supply-chain AI). Cosine similarity giữa embedding câu hỏi và cả hai chunk đều cao (cùng chứa "HPE" + "AI"), nhưng thuật toán top-k chọn nhầm chunk "gần nghĩa nhất theo embedding tổng thể" chứ không phải chunk "đúng chủ đề cụ thể nhất" — vector search không có cơ chế phân biệt ngữ nghĩa tinh vi giữa 2 chủ đề AI khác nhau trong cùng công ty.

4. **Vì sao GraphRAG thành công:** GraphRAG có 2 nguồn context bổ trợ nhau (`answer_graph_rag()` kết hợp `=== GRAPH ===` + `=== VECTOR ===`):
   - **Context đồ thị** cung cấp quan hệ tường minh `AWS –(considering)→ AMD chip` — một cạnh trích xuất trực tiếp từ NER+RE với `evidence` cụ thể, không mơ hồ như embedding similarity.
   - **Context vector bổ trợ** (k=4, ít hơn Flat RAG thuần k=6) vẫn cung cấp thông tin về HPE, nhưng vì đã có "mỏ neo" đúng chủ đề từ context đồ thị (AWS + AMD = tentative sourcing), LLM answer generation tổng hợp đúng hướng: AWS là "tentative hardware sourcing", HPE là "service offering" — dù không nêu chi tiết cụ thể HPE offering là gì ("HPE's offerings are not provided in the context" — Judge ghi nhận).

5. **Bài học:** Đây là minh chứng rõ ràng nhất trong toàn bộ 25 câu cho giá trị thiết kế **hybrid GRAPH + VECTOR context**: quan hệ tường minh trong graph (dù chỉ 1 cạnh) có thể "neo" đúng chủ đề và tránh việc LLM bị lạc hướng bởi vector search chọn nhầm chunk gần nghĩa nhưng sai trọng tâm — kể cả khi bản thân graph context không đầy đủ.

### Đề xuất khắc phục (cho việc GraphRAG vẫn thiếu chi tiết HPE)
- Tăng `edge_limit` hoặc `max_hops` để lấy thêm cạnh liên quan tới HPE cụ thể hơn.
- Bổ sung thêm entity "AI-cloud" hoặc "LLM service" như 1 node Technology riêng để graph có thể phân biệt rõ 2 chủ đề AI khác nhau của cùng 1 công ty.

---

## Kết luận chung từ 2 ca lỗi

Cả 2 ca lỗi đều bắt nguồn từ cùng một nguyên nhân gốc: **graph quá thưa** (362 nodes / 253 edges, xây từ chỉ 400/2.105 bài báo). Với graph thưa:
- Khi seed entity là node trung tâm có nhiều cạnh không liên quan (Amazon), BFS traversal dễ "lạc hướng" vào cạnh sai chủ đề → **GraphRAG thất bại** (ca lỗi 1).
- Khi graph có ít nhất 1 cạnh trực tiếp đúng chủ đề cần hỏi, nó đóng vai trò "mỏ neo" giúp tránh lỗi mơ hồ ngữ nghĩa của vector search thuần → **GraphRAG thành công** (ca lỗi 2).

Điều này khẳng định: **chất lượng GraphRAG phụ thuộc trực tiếp vào độ phủ (coverage) và độ chính xác (precision) của quá trình extraction**, không phải chỉ vào kiến trúc retrieval. Ở quy mô đầy đủ (~100.000 bài báo), graph dày hơn nhiều lần sẽ giảm đáng kể tỷ lệ ca lỗi kiểu 1 và tăng tỷ lệ ca lỗi kiểu 2.
