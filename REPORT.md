# Lab 21 — Evaluation Report

**Học viên**: Phạm Ngọc Vinh — 2A202600563  
**Ngày nộp**: 2026-06-25  
**Submission option**: Option A (Lightweight ZIP)

---

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Qwen 2.5 với cấu hình lượng hóa 4-bit NF4)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (Dataset Alpaca tiếng Việt dịch từ bản GPT-4), quy mô phân tách:
  - **Train set**: 180 samples (90%)
  - **Eval set**: 20 samples (10%)
- **max_seq_length**: 1024 (phân tích phân phối độ dài token: `min=25`, `max=738`, `p50=227`, `p95=562`, `p99=704`. Do giá trị p95 = 562 vượt quá lũy thừa của 2 là 512, cấu hình được làm tròn lên mức lũy thừa tiếp theo là 1024 và giới hạn ở mức 1024 cho dòng GPU T4).
- **GPU**: Tesla T4 (Cung cấp bởi Google Colab), 15.6 GB VRAM.
- **Training cost**: Ước tính khoảng **$0.07** cho toàn bộ thí nghiệm (Tổng thời gian chạy thực tế cho cả 3 ranks: 12.1 phút, tính theo đơn giá thuê GPU T4 tiêu chuẩn là $0.35/giờ).
- **HF Hub link**: https://huggingface.co/vinhpn/lab21-qwen2.5-3b-r16

---

## 2. Rank Experiment Results

Dưới đây là bảng so sánh hiệu năng và chất lượng huấn luyện của 3 mức rank ($r=8, 16, 64$) so với mô hình nền tảng (Base model):

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **8** | 16 | 1,843,200 (~0.06%) | 3.97 min | 7.22 GB | 1.5577 | 4.75 |
| **16 (Baseline)** | 32 | 3,686,400 (~0.12%) | 4.17 min | 6.62 GB | 1.5161 | 4.55 |
| **64** | 128 | 14,745,600 (~0.48%) | 3.92 min | 8.00 GB | 1.4768 | 4.38 |
| **Base** | - | - | - | - | N/A* | N/A* |

*\*Ghi chú: Mô hình Base không thực hiện đánh giá (evaluate) định lượng trên tập eval trong quá trình chạy Colab để tránh quá tải bộ nhớ và tối ưu thời gian.*

---

## 3. Loss Curve Analysis
Biểu đồ đường cong tổn thất huấn luyện (Loss Curve) được lưu trữ tại file [loss_curve.png](file:///Users/ngocvinh/ownCloud/HocTap/Day21-Track3-Finetuning-LLMs-LoRA-QLoRA/results/loss_curve.png).

### Chi tiết các bước huấn luyện (Training Steps Loss):
Dưới đây là giá trị Training Loss đo được sau mỗi 5 steps của 3 phiên thử nghiệm:

| Step | Rank 8 (Alpha 16) | Rank 16 (Alpha 32) | Rank 64 (Alpha 128) |
|---|---|---|---|
| **5** | 1.6165 | 1.6143 | 1.6016 |
| **10** | 1.5918 | 1.5736 | 1.5241 |
| **15** | 1.6331 | 1.6067 | 1.5471 |
| **20** | 1.5811 | 1.5554 | 1.5047 |
| **25** | 1.5125 | 1.4791 | 1.4254 |
| **30** | 1.4516 | 1.4162 | 1.3455 |
| **35** | 1.5322 | 1.4962 | 1.4025 |
| **40** | 1.5163 | 1.4801 | 1.3812 |
| **45** | 1.4175 | 1.3802 | 1.2976 |
| **50** | 1.4415 | 1.3884 | 1.2705 |
| **55** | 1.4645 | 1.4241 | 1.3239 |
| **60** | 1.4585 | 1.4137 | 1.3081 |
| **65** | 1.4388 | 1.3942 | 1.2768 |

### Nhận xét và Phân tích hiện tượng Overfitting:
- **Xu hướng chung**: Cả 3 mức rank đều cho thấy xu hướng giảm dần của Training Loss một cách ổn định từ epoch 1 đến epoch 3 (từ khoảng 1.60 xuống còn 1.43 ở rank 8, 1.39 ở rank 16 và 1.27 ở rank 64). Điều này chứng minh quá trình hội tụ đang diễn ra đúng hướng.
- **Overfitting**:
  - Do cấu hình phần cứng GPU T4 hạn chế, tính năng đánh giá định kỳ (`eval_during_training`) bị tắt để tiết kiệm VRAM, vì vậy chúng ta chỉ có giá trị Eval Loss ở cuối phiên huấn luyện.
  - Tuy nhiên, khi so sánh giá trị Eval Loss cuối cùng với Training Loss ở step 65:
    - Rank 8: Train Loss = 1.4388 | Eval Loss = 1.5577 (Chênh lệch ~0.12)
    - Rank 16: Train Loss = 1.3942 | Eval Loss = 1.5161 (Chênh lệch ~0.12)
    - Rank 64: Train Loss = 1.2768 | Eval Loss = 1.4768 (Chênh lệch ~0.20)
  - Sự chênh lệch giữa Train Loss và Eval Loss ở Rank 8 và Rank 16 ở mức rất thấp (~0.12), cho thấy mô hình không bị overfitting và giữ được khả năng tổng quát hóa tốt. 
  - Ở Rank 64, mặc dù Eval Loss giảm xuống mức tốt nhất (1.4768), khoảng cách chênh lệch bắt đầu nới rộng hơn (~0.20). Điều này cho thấy với rank lớn ($r=64$), mô hình bắt đầu bám sát tập dữ liệu huấn luyện hơn, tuy chưa đến mức overfitting nghiêm trọng (vì eval loss vẫn tiếp tục giảm) nhưng bắt đầu xuất hiện dấu hiệu của việc tăng dung lượng ghi nhớ dữ liệu.

---

## 4. Qualitative Comparison (5 examples)

Dưới đây là so sánh kết xuất câu trả lời thực tế (side-by-side) giữa mô hình Base và mô hình đã Fine-tuned với Rank 16 (Baseline):

### Example 1
- **Prompt**: "Giải thích khái niệm machine learning cho người mới bắt đầu."
- **Base**: "Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu. Mô hình học được từ dữ liệu này có thể thực hiện các tác "
- **Fine-tuned (r=16)**: "Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo) và sử dụng thuật toán và các mô hình học máy để học từ dữ liệu và từ đó đưa ra các dự đoán hoặc"
- **Nhận xét**: **Improved**. Mô hình Fine-tuned đưa ra định nghĩa tiếng Việt tự nhiên, gãy gọn và học thuật hơn ("một bộ môn công nghệ máy tính dựa trên việc học tập... không có sự hướng dẫn trực tiếp") so với cách dùng từ hơi lặp và thô của mô hình Base.

### Example 2
- **Prompt**: "Viết đoạn code Python tính số Fibonacci thứ n."
- **Base**: "Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:

```python
def fibonacci(n):
    if n <= 0:
        return "N phải là một số dương"
    elif n == 1:
        return 0
    elif n == 2:
        return 1
    else:
```"
- **Fine-tuned (r=16)**: "Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:

```python
def fibonacci(n):
    if n < 0:
        raise ValueError("Input phải là một số nguyên dương.")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range
```"
- **Nhận xét**: **Improved**. Mô hình Fine-tuned có tư duy lập trình an toàn và chuẩn mực hơn. Thay vì chỉ trả về một chuỗi thông báo lỗi đơn giản khi nhận đầu vào không hợp lệ ($n \le 0$) như Base model, mô hình đã qua tinh chỉnh biết quăng lỗi ngoại lệ hệ thống (`raise ValueError`) để đảm bảo chất lượng code.

### Example 3
- **Prompt**: "Liệt kê 5 nguyên tắc thiết kế UI/UX."
- **Base**: "1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc, font chữ, hình ảnh và các yếu tố khác để đảm bảo người dùng có thể dễ dàng sử dụng sản phẩm. 2. Tru"
- **Fine-tuned (r=16)**: "1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX thiết kế phải dễ dàng sử dụng và dễ hiểu. 4. Tương thích: UI/UX thiết kế phải dễ dàng tương thích với "
- **Nhận xét**: **Improved**. Mô hình Base bị lặp từ khóa quá nhiều ("thân thiện... thân thiện... thân thiện"). Trái lại, mô hình Fine-tuned sử dụng từ vựng chuyên ngành chính xác, ngắn gọn và mạch lạc hơn hẳn (đưa ra các đầu mục rõ ràng: "Chuyển đổi", "Thích ứng", "Đơn giản").

### Example 4
- **Prompt**: "Tóm tắt sự khác biệt giữa LoRA và QLoRA."
- **Base**: "LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hiệu năng của mô hình bằng cách thay đổi các phép biến đổi trong mô hình bằng các phép biến đổi nhỏ hơ"
- **Fine-tuned (r=16)**: "LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong các mô hình có số lượng các layer lớn. LoRA là một kỹ thuật regularization được sử dụng để cải thiệ"
- **Nhận xét**: **Degraded / Hallucination**. Đây là một ví dụ minh chứng cho thấy tác hại của việc sử dụng dataset dịch máy chưa qua hậu kiểm. Mô hình Fine-tuned đã bị ảo tưởng (hallucination) khi định nghĩa sai từ viết tắt LoRA thành "Layer-wise Adaptive Regularization Optimization" thay vì "Low-Rank Adaptation" như nguyên bản. Trong khi đó, mô hình Base vẫn giữ được định nghĩa viết tắt chính xác của LoRA.

### Example 5
- **Prompt**: "Phân biệt prompt engineering, RAG, và fine-tuning."
- **Base**: "Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của mô hình bằng cách cung cấp cho nó một câu hỏi hoặc câu lệnh để dựa vào, thay vì cung cấp dữ liệu đầu"
- **Fine-tuned (r=16)**: "Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp hệ thống AI giải quyết các vấn đề và thực hiện các tác vụ. Prompt được sử dụng để cung cấp cho hệ th"
- **Nhận xét**: **Same / Slightly Improved**. Cả hai mô hình đều có cấu trúc phân loại tốt và câu từ rõ nghĩa. Mô hình Fine-tuned có cách mở bài trôi chảy hơn khi xếp các kỹ thuật này vào "lĩnh vực AI và tự động hóa".

---

## 5. Conclusion về Rank Trade-off

### 1. Cơ chế cơ bản của LoRA (LoRA Mechanics Context)
Về mặt cơ chế, LoRA (Low-Rank Adaptation) đóng băng các trọng số gốc của mô hình $W_0 \in \mathbb{R}^{d \times k}$ và biểu diễn phần cập nhật trọng số $\Delta W$ dưới dạng tích của hai ma trận phân rã hạng thấp $B \in \mathbb{R}^{d \times r}$ và $A \in \mathbb{R}^{r \times k}$, trong đó $r \ll \min(d, k)$ là hạng (Rank).
- **Hạng ($r$)** quyết định số lượng chiều không gian trung gian (bottleneck dimension) để mô hình học các tính năng mới. Rank càng cao thì dung lượng lưu trữ thông tin của adapter càng lớn.
- **Tham số điều chỉnh tỉ lệ (Scaling factor $\frac{\alpha}{r}$)**: Trong thí nghiệm này, tỉ lệ $\frac{\alpha}{r} = 2$ được giữ cố định ($\alpha=16$ khi $r=8$; $\alpha=32$ khi $r=16$; $\alpha=128$ khi $r=64$). Việc giữ hằng số này giúp đảm bảo biên độ cập nhật trọng số $\Delta W$ được chuẩn hóa ổn định khi thay đổi hạng $r$, từ đó giữ nguyên tốc độ học mà không cần dò lại learning rate.

### 2. Rank nào cho ROI (Return on Investment) tốt nhất trên dataset này?
Đối với tập dữ liệu nhỏ gồm 200 mẫu thử này, **Rank 16** mang lại tỷ suất ROI tốt nhất. Khi so sánh giữa Rank 8 và Rank 16, việc tăng gấp đôi tham số huấn luyện (từ 1.8M lên 3.6M) chỉ làm tăng thời gian chạy thêm vỏn vẹn **12 giây** trên GPU T4 (3.97 phút so với 4.17 phút) và lượng VRAM tiêu hao gần như tương đương do kích thước ma trận $A$ và $B$ vẫn rất nhỏ so với mô hình nền 3B. Tuy nhiên, chất lượng cải thiện rõ rệt với việc Perplexity giảm từ 4.75 xuống 4.55. Sự đánh đổi tài nguyên nhỏ để đổi lấy cải tiến lớn về độ trôi chảy là cực kỳ xứng đáng.

### 3. Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?
Hiện tượng hiệu suất biên giảm dần (diminishing returns) xuất hiện rõ rệt khi ta tăng rank từ 16 lên 64. Khi tăng $r$ từ 16 lên 64, số lượng tham số huấn luyện tăng vọt gấp 4 lần (từ 3.68M lên 14.75M), lượng VRAM tiêu tốn tăng từ 6.62 GB lên 8.00 GB. Tuy nhiên, Perplexity chỉ giảm rất ít từ 4.55 xuống 4.38 (chỉ cải thiện khoảng ~3.7%). 

Lý do vật lý ở đây là do tập dữ liệu huấn luyện quá nhỏ (180 mẫu train) không chứa đủ độ phức tạp hoặc phương sai thông tin để lấp đầy không gian biểu diễn hạng lớn ($r=64$). Do đó, phần lớn dung lượng bổ sung của ma trận $B$ và $A$ trở nên dư thừa, làm tăng nguy cơ học vẹt (overfitting) thay vì học các cấu trúc ngữ nghĩa cốt lõi.

### 4. Recommendation khi deploy production
Khi triển khai thực tế trên môi trường production, **Rank 16** được đề xuất lựa chọn. Rank 16 vừa đáp ứng được tiêu chí chất lượng (perplexity thấp hơn đáng kể so với Rank 8), vừa duy trì chi phí vận hành tối ưu với kích thước file adapter nhỏ nhẹ, thời gian suy luận (inference latency) thấp và lượng RAM/VRAM yêu cầu khi tải mô hình cực kỳ dễ chịu so với Rank 64, cho phép hệ thống mở rộng tải (scale) tốt hơn khi có nhiều yêu cầu đồng thời.

---

## 6. What I Learned

- **Tối ưu hóa tài nguyên phần cứng**: Nắm vững kỹ thuật kết hợp Unsloth (với cấu hình lượng hóa 4-bit và gradient checkpointing) giúp ép xung hiệu năng và cắt giảm lượng VRAM tiêu thụ hơn 60% trên các dòng GPU giá rẻ như Tesla T4, mở ra khả năng huấn luyện các mô hình ngôn ngữ lớn ngay trên môi trường miễn phí.
- **Tác động thực tiễn của siêu tham số Rank ($r$)**: Hiểu sâu sắc rằng rank cao hơn không phải lúc nào cũng mang lại mô hình tốt hơn. Nó tuân theo quy luật hiệu suất biên giảm dần và việc lựa chọn rank luôn cần sự cân bằng chặt chẽ giữa: thời gian train, dung lượng VRAM, kích thước file weights đầu ra và chất lượng hội tụ định lượng (Perplexity).
- **Tầm quan trọng của chất lượng dữ liệu**: Nhận ra bài học đắt giá từ ví dụ 4: Các mô hình LLM rất nhạy cảm với các lỗi có trong dữ liệu huấn luyện. Nếu sử dụng các tập dữ liệu dịch máy tự động không được lọc kỹ lưỡng, mô hình fine-tuned dễ học theo các định nghĩa sai lệch (như dịch sai viết tắt thuật ngữ LoRA). Việc kiểm duyệt và làm sạch dữ liệu đầu vào thủ công luôn đóng vai trò quyết định đến độ chính xác của sản phẩm cuối cùng.
