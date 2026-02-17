import os
import streamlit as st
from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="News Writer Agent (LangGraph)", page_icon="📰", layout="wide")

st.title("📰 News Writer Agent (LangGraph)")
st.caption("Flujo: Search → Tools (Tavily) ↔ Search → Outliner → Writer")

with st.sidebar:
    st.header("🔐 Claves")
    google_key = st.text_input("Gemini / Google API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")

    st.divider()
    st.header("⚙️ Ajustes")
    model_name = st.text_input("Modelo", value="gemini-2.5-flash")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.2, 0.05)
    max_results = st.slider("Tavily max_results", 1, 10, 5, 1)

    st.divider()
    if st.button("🧹 Borrar trazas / estado"):
        st.session_state.clear()
        st.rerun()

if not google_key or not tavily_key:
    st.info("Introduce **Gemini API Key** y **Tavily API Key** en la barra lateral para empezar.")
    st.stop()

# Set env vars (recomendado por integraciones)
os.environ["GOOGLE_API_KEY"] = google_key
os.environ["TAVILY_API_KEY"] = tavily_key


# ---------------------------
# LangGraph: State
# ---------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------
# Prompt templates (igual que el notebook)
# ---------------------------
search_template = """Your job is to search the web for related news that would be relevant to generate the article described by the user.

NOTE: Do not write the article. Just search the web for related news if needed and then forward that news to the outliner node.
"""

outliner_template = """Your job is to take as input a list of articles from the web along with users instruction on what article they want to write and generate an outline
for the article.
"""

writer_template = """Your job is to write an article, do it in this format:

TITLE: <title>
BODY: <body>

NOTE: Do not copy the outline. You need to write the article with the info provided by the outline.
"""


def create_agent(llm, tools, system_message: str):
    """
    Crea un 'agente' como runnable:
    - Prompt con system + MessagesPlaceholder(messages)
    - Si hay tools, bind_tools(tools)
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "{system_message}"),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ).partial(system_message=system_message)

    if tools:
        return prompt | llm.bind_tools(tools)
    return prompt | llm


def agent_node(state: AgentState, agent, name: str):
    """
    Nodo genérico: invoca el runnable del agente con el state completo.
    Devuelve un delta de estado con un mensaje nuevo (se acumula por add_messages).
    """
    result = agent.invoke(state)  # result suele ser AIMessage
    return {"messages": [result]}


def should_search(state: AgentState) -> Literal["tools", "outliner"]:
    """
    Routing:
    - si el último AIMessage trae tool_calls -> "tools"
    - si no -> "outliner"
    """
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        if last_message.tool_calls:
            return "tools"
    return "outliner"


def build_graph(model: str, temp: float, tavily_max_results: int):
    # LLM base
    llm = ChatGoogleGenerativeAI(model=model, temperature=temp)

    # Tools (solo Tavily)
    tools = [TavilySearchResults(max_results=tavily_max_results)]
    tool_node = ToolNode(tools)

    # Agents
    search_agent = create_agent(llm, tools, search_template)
    outliner_agent = create_agent(llm, [], outliner_template)
    writer_agent = create_agent(llm, [], writer_template)

    # Nodes
    import functools
    search_node = functools.partial(agent_node, agent=search_agent, name="Search Agent")
    outliner_node = functools.partial(agent_node, agent=outliner_agent, name="Outliner Agent")
    writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer Agent")

    # Graph
    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("outliner", outliner_node)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("search")
    workflow.add_conditional_edges("search", should_search)
    workflow.add_edge("tools", "search")
    workflow.add_edge("outliner", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile()


def render_message(m: BaseMessage):
    """Render robusto para AI/Human/Tool messages."""
    if isinstance(m, HumanMessage):
        st.markdown(f"**👤 Human:** {m.content}")
        return

    if isinstance(m, ToolMessage):
        st.markdown("**🧰 Tool output:**")
        st.code(m.content if isinstance(m.content, str) else str(m.content))
        return

    if isinstance(m, AIMessage):
        st.markdown("**🤖 AI:**")
        # content puede ser str o lista de “parts”
        if isinstance(m.content, str):
            st.markdown(m.content)
        elif isinstance(m.content, list):
            # intenta extraer texto
            parts = []
            for p in m.content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text", ""))
            st.markdown("\n".join([x for x in parts if x]) or str(m.content))
        else:
            st.markdown(str(m.content))

        # tool_calls (si los hay) para trazabilidad
        tc = getattr(m, "tool_calls", None)
        if tc:
            st.markdown("**🔧 Tool calls solicitadas por el modelo:**")
            st.code(str(tc))
        return

    # fallback
    st.markdown(str(m))


# ---------------------------
# Main inputs
# ---------------------------
col1, col2 = st.columns([2, 1], vertical_alignment="top")

with col1:
    user_instruction = st.text_area(
        "🧾 Instrucción del artículo (lo que escribirías al agente)",
        value="Write an article about the latest trends on AI. Keep it practical, include examples, and end with 3 key takeaways.",
        height=140,
    )

with col2:
    st.markdown("### ▶️ Ejecutar")
    run = st.button("Generar artículo", type="primary", use_container_width=True)
    st.caption("Se guardarán trazas por nodo en pestañas.")

# Persist traces
if "traces" not in st.session_state:
    st.session_state.traces = {"search": [], "tools": [], "outliner": [], "writer": []}
if "raw_updates" not in st.session_state:
    st.session_state.raw_updates = []

# ---------------------------
# Run graph + capture traces
# ---------------------------
if run:
    # reset traces for this run
    st.session_state.traces = {"search": [], "tools": [], "outliner": [], "writer": []}
    st.session_state.raw_updates = []

    graph = build_graph(model_name, temperature, max_results)

    input_message = HumanMessage(content=user_instruction)
    initial_state = {"messages": [input_message]}

    # Captura “por nodo” usando stream_mode="updates"
    # updates suele venir como: {"search": {"messages": [AIMessage(...)]}}
    # ToolNode añade ToolMessage(s) en "tools"
    try:
        for update in graph.stream(initial_state, stream_mode="updates"):
            st.session_state.raw_updates.append(update)

            for node_name, partial_state in update.items():
                if node_name not in st.session_state.traces:
                    continue
                if isinstance(partial_state, dict) and "messages" in partial_state:
                    # Añadimos todos los mensajes nuevos de ese nodo
                    new_msgs = partial_state["messages"]
                    if isinstance(new_msgs, list):
                        st.session_state.traces[node_name].extend(new_msgs)

    except TypeError:
        # Fallback por si tu versión no soporta stream_mode="updates"
        # En ese caso, capturamos por "values" y volcamos todo a raw trace
        for state in graph.stream(initial_state, stream_mode="values"):
            st.session_state.raw_updates.append(state)
        st.warning("Tu versión de LangGraph no devolvió updates por nodo. Mira la pestaña 'Raw trace'.")

# ---------------------------
# Tabs: result + trace per node
# ---------------------------
tabs = st.tabs(["📝 Artículo", "🔎 search", "🧰 tools", "🧱 outliner", "✍️ writer", "🪵 Raw trace"])

# Artículo final: normalmente está en el último mensaje del nodo writer
with tabs[0]:
    writer_msgs = st.session_state.traces.get("writer", [])
    if writer_msgs:
        st.subheader("Resultado final (writer)")
        render_message(writer_msgs[-1])
    else:
        st.info("Ejecuta una generación para ver el artículo aquí.")

with tabs[1]:
    st.subheader("Trazabilidad: nodo search")
    msgs = st.session_state.traces.get("search", [])
    if not msgs:
        st.info("Sin trazas aún.")
    else:
        for i, m in enumerate(msgs, start=1):
            st.markdown(f"---\n#### Paso search #{i}")
            render_message(m)

with tabs[2]:
    st.subheader("Trazabilidad: nodo tools (Tavily)")
    msgs = st.session_state.traces.get("tools", [])
    if not msgs:
        st.info("Sin trazas aún.")
    else:
        for i, m in enumerate(msgs, start=1):
            st.markdown(f"---\n#### Paso tools #{i}")
            render_message(m)

with tabs[3]:
    st.subheader("Trazabilidad: nodo outliner")
    msgs = st.session_state.traces.get("outliner", [])
    if not msgs:
        st.info("Sin trazas aún.")
    else:
        for i, m in enumerate(msgs, start=1):
            st.markdown(f"---\n#### Paso outliner #{i}")
            render_message(m)

with tabs[4]:
    st.subheader("Trazabilidad: nodo writer")
    msgs = st.session_state.traces.get("writer", [])
    if not msgs:
        st.info("Sin trazas aún.")
    else:
        for i, m in enumerate(msgs, start=1):
            st.markdown(f"---\n#### Paso writer #{i}")
            render_message(m)

with tabs[5]:
    st.subheader("Raw trace (debug)")
    if not st.session_state.raw_updates:
        st.info("Sin trazas aún.")
    else:
        st.code(str(st.session_state.raw_updates[:50]))  # muestra primeras 50 entradas para no petar la UI
        if len(st.session_state.raw_updates) > 50:
            st.caption(f"Mostrando 50/{len(st.session_state.raw_updates)} entradas.")
