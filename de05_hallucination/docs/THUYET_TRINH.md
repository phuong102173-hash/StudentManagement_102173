# Dàn ý thuyết trình đề 05 — khoảng 10 phút

## Slide 1 — Vấn đề (45 giây)

**Tên:** Phát hiện hallucination trong đầu ra LLM.

Ví dụ: câu trả lời “Việt Nam gia nhập ASEAN năm 2000” so với nguồn ghi 1995. Văn bản trôi chảy không đồng nghĩa đúng. Mục tiêu: chỉ ra phần cần kiểm chứng và bằng chứng, không chỉ gắn nhãn.

## Slide 2 — Định nghĩa và đầu vào (45 giây)

Phân biệt factual error, unfaithful summary, fabricated citation. Nhấn mạnh unsupported khác contradicted. Input gồm claim, context, optional question/samples; output có score, quyết định ba trạng thái và quote.

## Slide 3 — Ba công trình nền tảng (60 giây)

FActScore: atomic facts và nguồn. SelfCheckGPT: bất nhất giữa samples. HaluEval: dữ liệu cặp đúng/sai. Dẫn paper/repo ở cuối báo cáo. Giải thích bài làm triển khai lại ý tưởng, không nhận là chạy nguyên pipeline gốc.

## Slide 4 — Kiến trúc (60 giây)

Trình bày sơ đồ trong `BAO_CAO.md`: frontend → FastAPI → detector → evidence validation → JSON/UI. CLI chuẩn hóa dữ liệu và đánh giá chạy độc lập. LLM chỉ được trả schema hợp lệ, quote phải có trong nguồn.

## Slide 5 — Năm detector và baseline (75 giây)

So sánh atomic checking, consistency sampling, HHEM, Qwen judge, NLI. Nêu điều kiện chạy từng phương pháp. Baseline lexical chỉ là mốc đối chiếu, không giả làm HHEM hay Qwen. Ensemble không bỏ qua detector lỗi.

## Slide 6 — Giao thức dữ liệu (60 giây)

HaluEval QA: 10K nguồn → 20K candidate; nhóm đúng/sai cùng split; test 2.980 candidate. Nhãn hallucination=1. TruthfulQA cần nguồn độc lập, không đưa gold answer vào context. HallusionBench là dữ liệu ảnh–văn bản, chưa được đánh giá text-only.

## Slide 7 — Kết quả thực tế và giới hạn (60 giây)

Baseline QA test: Precision 91,85%, Recall 91,54%, F1 91,70%, AUC 0,9115; coverage 100%. Chiếu ma trận nhầm lẫn. Nói rõ đây là baseline theo thiết lập đối chiếu nguồn, chưa có bảng thực nghiệm của năm model, không suy rộng sang TruthfulQA/HallusionBench.

## Slide 8 — Demo trực tiếp (120 giây)

1. Chạy server theo README; chọn **Sai số liệu**, phương pháp **Baseline minh họa**; kiểm chứng.
2. Chỉ vào điểm, mệnh đề và nguồn ghi 1995; tải báo cáo JSON.
3. Chọn **Có căn cứ**; kết quả dưới ngưỡng.
4. Xóa context, chạy lại; kết quả “chưa đủ bằng chứng”, không phải “không hallucination”.
5. Chọn LLM judge khi chưa cấu hình; chỉ ra thông báo thiếu cấu hình thay vì một điểm AI giả.
6. Nếu đã có endpoint/model thật trước buổi bảo vệ, chạy lại một mẫu với judge và trình bày khác biệt. Không gọi demo mock là kết quả model thật.

## Slide 9 — Calibration và human evaluation (45 giây)

Hiệu chỉnh trên validation, giữ test độc lập theo group. Kappa model–gold khác human–human. Chưa có dữ liệu hai người chấm thật; cần bổ sung trước khi báo cáo human evaluation. Confidence hiện chưa được chứng minh là xác suất đáng tin cậy.

## Slide 10 — Kết luận và hướng tiếp (30 giây)

Đã có app và pipeline chạy được, baseline đo trên dữ liệu thật, cơ chế quote/abstention. Còn cần chạy model thật, nguồn độc lập TruthfulQA, VLM cho HallusionBench và gán nhãn người. Không trình bày các hạng mục này như đã hoàn thành.

## Câu hỏi phản biện thường gặp

**Tại sao baseline đạt hơn 90%?** HaluEval QA có cặp đáp án và nguồn rõ; dấu hiệu từ vựng có thể đủ phân biệt nhiều mẫu. Cần kiểm chứng thêm trên paraphrase/domain khác; chưa có bằng chứng baseline hơn model.

**Không có trong context thì có chắc là sai?** Không. Đó là thiếu hỗ trợ; có thể nguồn không đầy đủ. Hệ thống phân biệt unsupported, contradicted và unknown.

**Tại sao không luôn dùng Qwen judge?** Chi phí, độ trễ, thiên lệch và nguồn có thể thiếu. Cần so sánh với model nhỏ và kiểm tra bằng chứng.

**SelfCheckGPT nhất quán có chắc đúng?** Không. Cùng model có thể lặp lại cùng ngộ nhận; kết quả nhất quán chỉ là một tín hiệu.

**Hệ thống xác thực bài báo/DOI thật chưa?** Chưa. Phân loại trích dẫn chỉ dựa trên nguồn được cung cấp; kiểm chứng thư mục là hướng mở rộng.

**Đã đạt trọn vẹn yêu cầu đề chưa?** Đã có bản triển khai và pipeline nền tảng; chưa có thực nghiệm model thật đủ năm phương pháp, ba benchmark và human evaluation. Các phần còn lại được nêu rõ trong báo cáo.
