import os
from huggingface_hub import HfApi, login

# ==========================================
# CẤU HÌNH THÔNG TIN HUGGINGFACE CỦA BẠN
# ==========================================
# 1. Tạo Access Token có quyền "Write" tại: https://huggingface.co/settings/tokens
# Bạn có thể điền vào đây hoặc set biến môi trường: export HF_TOKEN="your_token"
HF_WRITE_TOKEN = os.environ.get("HF_TOKEN", "ĐIỀN_TOKEN_WRITE_CỦA_BẠN_VÀO_ĐÂY")

# 2. Tên tài khoản Hugging Face của bạn
HF_USERNAME = "vinhpn"

# 3. Tên repo bạn muốn tạo (ví dụ: lab21-qwen2.5-3b-r16)
REPO_NAME = "lab21-qwen2.5-3b-r16"
# ==========================================

if HF_WRITE_TOKEN == "ĐIỀN_TOKEN_WRITE_CỦA_BẠN_VÀO_ĐÂY" or HF_USERNAME == "username_cua_ban":
    print("❌ Vui lòng điền HF_WRITE_TOKEN vào file push_to_hf.py (hoặc chạy: export HF_TOKEN=...)!")
    exit(1)

# Thư mục chứa adapter cần push (theo chuẩn Option A)
local_dir = os.path.abspath("./adapters/r16")

if not os.path.exists(local_dir):
    # Nếu không tìm thấy ở adapters/r16, thử kết quả gốc ở results/r16
    local_dir = os.path.abspath("./results/r16")

if not os.path.exists(local_dir):
    print(f"❌ Không tìm thấy thư mục chứa adapter tại {local_dir}")
    exit(1)

repo_id = f"{HF_USERNAME}/{REPO_NAME}"

print(f"🔄 Đang đăng nhập HuggingFace...")
login(token=HF_WRITE_TOKEN)

print(f"🔄 Đang tạo repository và upload adapter từ: {local_dir} lên https://huggingface.co/{repo_id} ...")
api = HfApi()

try:
    print(f"🔄 Đang tự động khởi tạo repository {repo_id} trên HuggingFace...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    # Upload toàn bộ thư mục adapter lên HF Hub
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        # Chỉ đẩy lên các file weights chính, bỏ qua checkpoint phụ hoặc cache
        ignore_patterns=["checkpoint-*", "*.bin", "runs/*", "logs/*"]
    )
    print("\n🎉 THÀNH CÔNG!")
    print(f"Adapter của bạn đã được push publicly tại: https://huggingface.co/{repo_id}")
    print("👉 Hãy copy link này và điền vào mục 'HF Hub link' trong file REPORT.md để nhận +5 điểm bonus!")
except Exception as e:
    print(f"❌ Có lỗi xảy ra trong quá trình upload: {e}")
