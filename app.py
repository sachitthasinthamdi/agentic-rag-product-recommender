# -*- coding: utf-8 -*-
"""Streamlit chat UI — เดโมหลักสำหรับสอน (ธีมมืดสไตล์ Claude / ChatGPT)

จุดเด่นเพื่อการสอน: side panel ด้านซ้ายโชว์ "ความคิด" ของ agent แบบสดๆ
(action ที่ planner เลือก, profile ที่สะสม, คะแนน rerank ของแต่ละ candidate)

รัน:  streamlit run app.py
"""
import streamlit as st

from src import catalog, llm
from src.agent import ProductRecAgent

st.set_page_config(page_title="AI Assistant — Product Rec Agent", layout="wide")

# ---------- ธีมมืดสไตล์ Claude / ChatGPT (inject CSS) ----------
st.markdown("""
<style>
/* ---- ซ่อน chrome ของ Streamlit ให้ดูเหมือนแอปแชท ---- */
#MainMenu, footer, header,
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display: none !important; }

/* ---- พื้นหลังหลัก ---- */
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stBottom"] { background: #1a1a1a; }

/* ---- คอลัมน์แชทตรงกลาง แคบแบบ ChatGPT/Claude ---- */
[data-testid="stMainBlockContainer"] {
  max-width: 46rem; margin: 0 auto;
  padding-top: 2.5rem; padding-bottom: 7rem;
}
[data-testid="stBottomBlockContainer"] {
  max-width: 46rem; margin: 0 auto; background: transparent;
}

/* ---- หัวเรื่อง ---- */
h1 { font-size: 1.4rem !important; font-weight: 600 !important; letter-spacing: .2px; }

/* ---- ข้อความแชท: พื้นโปร่ง เว้นระยะสบายตา ---- */
[data-testid="stChatMessage"] {
  background: transparent; padding: .35rem 0; gap: .85rem;
}
[data-testid="stChatMessageContent"] {
  color: #ececec; font-size: 1rem; line-height: 1.7;
}
[data-testid="stChatMessage"] > div:first-child { display: none; }  /* ซ่อน avatar icon */

/* ---- ข้อความผู้ใช้: bubble ชิดขวาแบบ ChatGPT (แยกด้วย aria-label) ---- */
[data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {
  justify-content: flex-end;
}
[data-testid="stChatMessageContent"][aria-label="Chat message from user"] {
  background: #2f2f2f; padding: .7rem 1.05rem; border-radius: 1.35rem;
  max-width: 85%; margin-left: auto !important; margin-right: 0 !important;
}

/* ---- กล่องพิมพ์ข้อความ: ทรงมนแบบ pill ---- */
[data-testid="stChatInput"] {
  background: #2a2a2a; border: 1px solid #3a3a3a;
  border-radius: 1.6rem; box-shadow: 0 2px 12px rgba(0,0,0,.35);
}
[data-testid="stChatInput"]:focus-within { border-color: #d97757; }
[data-testid="stChatInput"] textarea { color: #ececec; }

/* ---- sidebar (ความคิด agent) ---- */
[data-testid="stSidebar"] {
  background: #171717; border-right: 1px solid #2a2a2a;
}
[data-testid="stSidebar"] h1 { font-size: 1.25rem !important; }

/* ---- ปุ่ม ---- */
.stButton > button {
  background: #2a2a2a; color: #ececec; border: 1px solid #3a3a3a;
  border-radius: .7rem; font-weight: 500; transition: all .15s ease;
}
.stButton > button:hover { border-color: #d97757; color: #fff; background: #303030; }

/* ---- การ์ดสินค้า: รูปมุมมน ---- */
[data-testid="stImage"] img { border-radius: .6rem; }

/* ---- แถบเลื่อน ---- */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #4a4a4a; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init():
    """เตรียมระบบครั้งเดียว: เช็คโมเดล + สร้าง index"""
    missing = llm.check_models()
    if missing:
        return None, missing
    n = catalog.build_index()
    return n, []


n_products, missing = init()
if missing:
    st.error("ยังไม่ได้ดาวน์โหลดโมเดล: " + ", ".join(missing))
    st.code("\n".join(f"ollama pull {m}" for m in missing))
    st.stop()

if "agent" not in st.session_state:
    st.session_state.agent = ProductRecAgent()
    st.session_state.messages = []   # [(role, text)]
    st.session_state.last_trace = None

# ---------- Side panel: มองเห็นความคิดของ agent ----------
PIPELINE = {
    "Agentic RAG (ระยะ 2)": "Memory → Planner → Retriever → Reranker → Generator",
    "Conversational RAG (ระยะ 1)": "Memory → Retriever → Generator",
}
with st.sidebar:
    st.title("AI Assistant")
    st.caption(f"AI Agent + CRS + RAG · คลังสินค้า {n_products} ชิ้น · รัน local 100%")

    mode = st.radio("โหมดการทำงาน", list(PIPELINE.keys()),
                    help="สลับเปรียบเทียบระยะการพัฒนา: ระยะ 1 ยังไม่มี Planner/Reranker")
    st.caption("Pipeline: " + PIPELINE[mode])
    agentic = mode.startswith("Agentic")

    if st.button("เริ่มบทสนทนาใหม่", use_container_width=True):
        st.session_state.agent = ProductRecAgent()
        st.session_state.messages = []
        st.session_state.last_trace = None
        st.rerun()

    trace = st.session_state.last_trace
    if trace:
        st.divider()
        st.subheader("ความคิดของ Agent")
        st.caption(f"(turn ล่าสุด · โหมด {trace.get('mode', 'agentic')})")
        badge = {"ask": "ask — ถามเพิ่ม", "recommend": "recommend — แนะนำ",
                 "chat": "chat — คุยทั่วไป"}
        st.markdown(f"**Action:** {badge.get(trace['action'], trace['action'])}")
        st.caption(f"เหตุผล: {trace['plan_reason']}")

        st.markdown("**User Profile (สะสมจากบทสนทนา)**")
        st.json(trace["profile"], expanded=False)

        if trace.get("query"):
            st.markdown("**Query ที่ใช้ค้น vector DB**")
            st.caption(trace["query"])

        if trace.get("picked"):
            if trace.get("mode") == "agentic":
                st.markdown("**ผล Rerank (ARAG-lite)**")
                for m in trace["picked"]:
                    st.markdown(
                        f"- **{m['title'][:40]}** · vector {m['score']:.2f} → "
                        f"fit **{m['fit_score']:.0f}/10**\n  \n  _{m['fit_reason']}_")
            else:
                st.markdown("**ผลค้น (จัดอันดับด้วย cosine — ไม่มี Reranker)**")
                for m in trace["picked"]:
                    st.markdown(f"- **{m['title'][:40]}** · vector {m['score']:.2f}")

# ---------- หน้าต่างแชท ----------
st.title("คุยกับ AI Assistant")

if not st.session_state.messages:
    st.chat_message("assistant").write(
        "สวัสดีค่า~ วันนี้อยากได้สินค้าอะไรดีคะ บอกหมวด งบประมาณ หรือเอาไว้ใช้ทำอะไรมาได้เลย")

# render ประวัติ + การ์ดสินค้า จาก session_state (การ์ดต้องมาจากประวัติจึง persist หลัง rerun)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["text"])
        picked = msg.get("picked")
        if picked:
            cols = st.columns(len(picked))
            for col, m in zip(cols, picked):
                with col:
                    if m.get("image"):
                        st.image(m["image"], use_container_width=True)
                    st.markdown(
                        f"**{m['title'][:50]}**  \n"
                        f"**${m['price']}** · {m.get('brand', '-')}  \n"
                        f"{m['category']} · {m.get('rating', '-')} "
                        f"({m.get('rating_count', 0)} รีวิว)")

if user_input := st.chat_input("พิมพ์ที่นี่... เช่น หาหูฟังไร้สายงบไม่เกิน 30"):
    st.session_state.messages.append({"role": "user", "text": user_input})
    with st.spinner("กำลังคิด..."):
        trace = st.session_state.agent.respond(user_input, agentic=agentic)
    st.session_state.messages.append(
        {"role": "assistant", "text": trace["reply"], "picked": trace.get("picked", [])})
    st.session_state.last_trace = trace
    st.rerun()
