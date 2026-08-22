# -*- coding: utf-8 -*-
"""Streamlit chat UI — เดโมหลักสำหรับสอน

จุดเด่นเพื่อการสอน: side panel ด้านซ้ายโชว์ "ความคิด" ของ agent แบบสดๆ
(action ที่ planner เลือก, profile ที่สะสม, คะแนน rerank ของแต่ละ candidate)

รัน:  streamlit run app.py
"""
import streamlit as st

from src import catalog, llm
from src.agent import ProductRecAgent

st.set_page_config(page_title="น้องช้อป — Product Rec Agent", page_icon="🛒",
                   layout="wide")


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
with st.sidebar:
    st.title("🛒 น้องช้อป")
    st.caption(f"AI Agent + CRS + RAG · คลังสินค้า {n_products} ชิ้น · รัน local 100%")

    if st.button("🔄 เริ่มบทสนทนาใหม่"):
        st.session_state.agent = ProductRecAgent()
        st.session_state.messages = []
        st.session_state.last_trace = None
        st.rerun()

    trace = st.session_state.last_trace
    if trace:
        st.subheader("🧠 ความคิดของ Agent (turn ล่าสุด)")
        badge = {"ask": "🟡 ask — ถามเพิ่ม", "recommend": "🟢 recommend — แนะนำ",
                 "chat": "🔵 chat — คุยทั่วไป"}
        st.markdown(f"**Action:** {badge.get(trace['action'], trace['action'])}")
        st.caption(f"เหตุผล: {trace['plan_reason']}")

        st.markdown("**User Profile (สะสมจากบทสนทนา)**")
        st.json(trace["profile"], expanded=False)

        if trace.get("query"):
            st.markdown("**Query ที่ใช้ค้น vector DB**")
            st.caption(trace["query"])

        if trace.get("picked"):
            st.markdown("**ผล Rerank (ARAG-lite)**")
            for m in trace["picked"]:
                st.markdown(
                    f"- **{m['title'][:40]}** · vector {m['score']:.2f} → "
                    f"fit **{m['fit_score']:.0f}/10**\n  \n  _{m['fit_reason']}_")

# ---------- หน้าต่างแชท ----------
st.title("คุยกับน้องช้อป 🛍️")

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
                        f"💵 **${m['price']}** · {m.get('brand', '-')}  \n"
                        f"🏷️ {m['category']} · ⭐ {m.get('rating', '-')} "
                        f"({m.get('rating_count', 0)} รีวิว)")

if user_input := st.chat_input("พิมพ์ที่นี่... เช่น หาหูฟังไร้สายงบไม่เกิน 30"):
    st.session_state.messages.append({"role": "user", "text": user_input})
    with st.spinner("กำลังคิด... (LLM รันในเครื่อง อาจใช้เวลาหน่อยนะ)"):
        trace = st.session_state.agent.respond(user_input)
    st.session_state.messages.append(
        {"role": "assistant", "text": trace["reply"], "picked": trace.get("picked", [])})
    st.session_state.last_trace = trace
    st.rerun()
