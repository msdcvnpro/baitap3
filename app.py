import os
from datetime import datetime

import streamlit as st
from openai import OpenAI


def load_env_file(path: str = ".env") -> None:
    """
    Tải biến môi trường từ file .env (tự cài, không cần thư viện bên ngoài).
    Mỗi dòng dạng: KEY=VALUE, bỏ qua dòng trống và comment (#).
    """
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        # Nếu có lỗi đọc .env thì cũng không làm crash app
        pass


def get_client() -> OpenAI:
    """
    Create an OpenAI client using the API key from environment variables
    hoặc từ file .env (nếu có).
    """
    # Thử load từ file .env (nếu tồn tại)
    load_env_file()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY chưa được thiết lập. "
            "Bạn có thể:\n"
            "1) Tạo file .env cùng thư mục app.py, thêm dòng:\n"
            "   OPENAI_API_KEY=your_key_here\n"
            "hoặc\n"
            "2) Thiết lập biến môi trường OPENAI_API_KEY trong hệ điều hành."
        )
    return OpenAI(api_key=api_key)


def build_system_prompt(expertise: str) -> str:
    base = (
        "Bạn là một trợ lý AI chuyên gia, giải thích rõ ràng, logic, có ví dụ dễ hiểu. "
        "Luôn trả lời bằng tiếng Việt, giọng điệu chuyên nghiệp nhưng dễ gần. "
    )

    if expertise == "Trí tuệ nhân tạo (AI) tổng quát":
        detail = (
            "Tập trung vào các khái niệm nền tảng AI, lịch sử phát triển, ứng dụng "
            "thực tế và xu hướng mới."
        )
    elif expertise == "Machine Learning & Deep Learning":
        detail = (
            "Tập trung vào supervised / unsupervised learning, kiến trúc mạng nơ-ron, "
            "overfitting, regularization, tối ưu, và pipeline huấn luyện mô hình."
        )
    elif expertise == "Xử lý ngôn ngữ tự nhiên (NLP)":
        detail = (
            "Tập trung vào mô hình ngôn ngữ, tokenization, embeddings, transformers, "
            "và ứng dụng NLP trong thực tế."
        )
    elif expertise == "Mô hình sinh (Generative AI)":
        detail = (
            "Tập trung vào LLMs, diffusion models, prompt engineering, và các vấn đề "
            "đạo đức, an toàn trong AI sinh nội dung."
        )
    else:
        detail = "Hãy trả lời như một chuyên gia AI đa lĩnh vực."

    return base + detail


def ask_ai(
    client: OpenAI,
    system_prompt: str,
    question: str,
    temperature: float = 0.3,
    max_tokens: int = 800,
) -> str:
    """
    Gửi câu hỏi tới mô hình OpenAI và nhận câu trả lời.
    """
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return completion.choices[0].message.content.strip()


def setup_page():
    """
    Cấu hình giao diện tổng thể: nền đen, chữ trắng đậm, layout rộng.
    """
    st.set_page_config(
        page_title="TRỢ LÝ MR VĂN",
        page_icon="🤖",
        layout="wide",
    )

    # CSS tùy chỉnh cho nền đen, chữ trắng đậm, giao diện chuyên nghiệp
    custom_css = """
    <style>
        /* Toàn bộ nền và chữ */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #050505;
            color: #f5f5f5;
        }

        [data-testid="stSidebar"] {
            background-color: #050505;
            border-right: 1px solid #333333;
        }

        /* Tiêu đề & text */
        h1, h2, h3, h4, h5, h6, p, span, label {
            color: #ffffff !important;
            font-weight: 600;
        }

        /* Input, textarea, select */
        .stTextInput > div > div > input,
        .stTextArea > div > textarea {
            background-color: #111111 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            border: 1px solid #333333 !important;
        }

        .stSelectbox > div > div {
            background-color: #111111 !important;
            color: #ffffff !important;
            border-radius: 8px !important;
            border: 1px solid #333333 !important;
        }

        /* Nút bấm */
        button[kind="primary"], .stButton > button {
            background: linear-gradient(135deg, #00c6ff, #0072ff);
            color: #ffffff;
            border-radius: 999px;
            border: none;
            padding: 0.5rem 1.5rem;
            font-weight: 700;
        }

        button[kind="primary"]:hover, .stButton > button:hover {
            filter: brightness(1.1);
        }

        /* Khung chat */
        .user-bubble {
            background-color: #1c1c1c;
            padding: 0.8rem 1rem;
            border-radius: 14px;
            border: 1px solid #333333;
            margin-bottom: 0.5rem;
        }

        .assistant-bubble {
            background-color: #0b0b0b;
            padding: 0.8rem 1rem;
            border-radius: 14px;
            border: 1px solid #444444;
            margin-bottom: 1rem;
        }

        .role-badge {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #9ca3af;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def main():
    setup_page()

    # Sidebar cấu hình
    with st.sidebar:
        st.markdown("## ⚙️ Cấu hình trợ lý")
        expertise = st.selectbox(
            "Chuyên môn chính",
            [
                "Trí tuệ nhân tạo (AI) tổng quát",
                "Machine Learning & Deep Learning",
                "Xử lý ngôn ngữ tự nhiên (NLP)",
                "Mô hình sinh (Generative AI)",
                "Khác / Tổng hợp",
            ],
        )

        temperature = st.slider(
            "Mức độ sáng tạo (temperature)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
        )

        st.markdown("---")
        st.caption(
            "Lưu ý: để sử dụng được trợ lý, bạn cần đặt biến môi trường "
            "`OPENAI_API_KEY` (hoặc file `.env`)."
        )

    # Header
    col_left, col_right = st.columns([0.8, 0.2])
    with col_left:
        st.markdown(
            "<h2 style='margin-bottom: 0.2rem;'>Trợ lý Mr Văn</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='color:#9ca3af;margin-top:0;'>Trợ lý thông thái – luôn luôn bên bạn.</p>",
            unsafe_allow_html=True,
        )
    with col_right:
        now = datetime.now()
        formatted_datetime = now.strftime("%H:%M:%S - %d/%m/%Y")
        st.markdown(
            f"<p style='text-align:right;color:#4b5563;'>{formatted_datetime}</p>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Khởi tạo session_state cho chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị lịch sử hội thoại
    for msg in st.session_state.messages:
        role, content = msg["role"], msg["content"]
        if role == "user":
            st.markdown(
                f"<div class='user-bubble'>"
                f"<div class='role-badge'>Bạn</div>"
                f"<div>{content}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='assistant-bubble'>"
                f"<div class='role-badge'>Trợ lý AI</div>"
                f"<div>{content}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("### 💬 Đặt câu hỏi cho trợ lý AI")

    # Ô nhập câu hỏi
    question = st.text_area(
        "Nhập câu hỏi của bạn về AI, machine learning, mô hình sinh,...",
        height=120,
        placeholder="Ví dụ: Giải thích giúp mình sự khác nhau giữa supervised learning và unsupervised learning...",
    )

    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        ask_button = st.button("Hỏi trợ lý", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("Xóa lịch sử", use_container_width=True)

    if clear_button:
        st.session_state.messages = []
        st.experimental_rerun()

    if ask_button and question.strip():
        # Lưu câu hỏi người dùng
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Trợ lý đang suy nghĩ..."):
            try:
                client = get_client()
                system_prompt = build_system_prompt(expertise)

                full_history = [
                    {"role": "system", "content": system_prompt},
                ]
                for m in st.session_state.messages:
                    full_history.append(
                        {"role": m["role"], "content": m["content"]}
                    )

                completion = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=full_history,
                    temperature=temperature,
                    max_tokens=1000,
                )

                answer = completion.choices[0].message.content.strip()

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as e:
                st.error(f"Đã xảy ra lỗi: {e}")

        st.experimental_rerun()


if __name__ == "__main__":
    main()


