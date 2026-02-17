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
st.set_page_config(page_title="� ¡HOLA! Revista del Corazón", page_icon="💕", layout="wide")

st.markdown("""
<div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #ff6b9d 0%, #c06c84 100%); border-radius: 10px; margin-bottom: 20px;'>
    <h1 style='color: white; font-size: 3em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
        💕 ¡HOLA! 💕
    </h1>
    <p style='color: white; font-size: 1.5em; margin: 10px 0 0 0; font-style: italic;'>
        Tu Revista Digital del Corazón
    </p>
    <p style='color: #ffe6f0; font-size: 1em; margin: 5px 0 0 0;'>
        ✨ Las noticias más exclusivas de tus celebridades favoritas ✨
    </p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔐 Configuración de Redacción")
    google_key = st.text_input("🔑 Gemini / Google API Key", type="password", help="Tu clave API de Google Gemini")
    tavily_key = st.text_input("🔍 Tavily API Key", type="password", help="Tu clave API de Tavily para búsquedas")

    st.divider()
    st.markdown("### ⚙️ Ajustes del Editor")
    model_name = st.text_input("🤖 Modelo IA", value="gemini-2.5-flash")
    temperature = st.slider("🌡️ Creatividad", 0.0, 1.0, 0.3, 0.05, help="Mayor valor = más creativo")
    max_results = st.slider("📰 Cantidad de noticias", 1, 10, 5, 1, help="Número de fuentes a consultar")

    st.divider()
    st.markdown("### 💝 Secciones Populares")
    st.markdown("""
    - 💑 **Romances y Parejas**
    - 💍 **Bodas y Compromisos**
    - 👶 **Bebés y Embarazos**
    - 💔 **Rupturas y Divorcios**
    - ⭐ **Escándalos y Polémicas**
    - 👗 **Moda y Glamour**
    """)
    
    st.divider()
    if st.button("🧹 Nueva Sesión", use_container_width=True):
        st.session_state.clear()
        st.rerun()

if not google_key or not tavily_key:
    st.markdown("""
    <div style='background-color: #fff0f5; padding: 20px; border-radius: 10px; border-left: 5px solid #ff69b4;'>
        <h3 style='color: #c71585; margin-top: 0;'>� ¡Bienvenida/o a tu Revista del Corazón!</h3>
        <p style='color: #8b008b;'>
            Para comenzar a generar artículos exclusivos sobre tus celebridades favoritas, 
            introduce tus <strong>API Keys</strong> en la barra lateral. 
        </p>
        <p style='color: #8b008b;'>
            💡 <em>¿No tienes las claves? Consigue tu API de Google Gemini y Tavily para empezar.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
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
# Prompt templates - Especializados en noticias del corazón
# ---------------------------
search_template = """Eres un experto buscador de noticias del corazón, prensa rosa y celebridades.

Tu trabajo es buscar en la web noticias relacionadas con famosos, celebridades, parejas, relaciones, escándalos, bodas, divorcios, 
embarazos, rumores y todo lo relacionado con el mundo del espectáculo y la prensa del corazón que sea relevante para el artículo 
que el usuario quiere generar.

IMPORTANTE: 
- Busca solo información relacionada con celebridades y noticias del corazón
- NO escribas el artículo, solo busca las noticias
- Enfócate en contenido de actualidad rosa y famosos
- Pasa la información al siguiente nodo para crear el esquema

NOTA: Las búsquedas deben ser en español cuando sea posible, o traducir el contexto al español.
"""

outliner_template = """Eres un experto editor de revistas del corazón y prensa rosa.

DEBES crear un esquema estructurado y detallado para un artículo de noticias del corazón basándote en las noticias proporcionadas.

GENERA un esquema que incluya:

**TÍTULO PROPUESTO:** [Título atractivo y llamativo estilo prensa rosa]

**ESTRUCTURA DEL ARTÍCULO:**

1. **INTRODUCCIÓN/GANCHO:**
   - Dato más impactante o exclusivo que enganche al lector
   
2. **CONTEXTO DE LA HISTORIA:**
   - Antecedentes de la relación/situación
   - Quiénes son los protagonistas
   
3. **DESARROLLO:**
   - Eventos recientes y cronología
   - Declaraciones y reacciones
   - Detalles jugosos y datos exclusivos
   
4. **REACCIÓN DEL PÚBLICO:**
   - Qué dicen los fans
   - Impacto en redes sociales
   
5. **CIERRE:**
   - Perspectivas a futuro
   - Pregunta o reflexión final

**PUNTOS CLAVE A INCLUIR:** [Lista de datos específicos, fechas, lugares, nombres]

IMPORTANTE: Genera este esquema AHORA con toda la información proporcionada. NO digas que lo harás, HAZLO.
"""

writer_template = """Eres un redactor profesional de noticias del corazón. 

ESCRIBE AHORA un artículo completo en español basándote en el esquema proporcionado.

**INSTRUCCIONES OBLIGATORIAS:**

1. Usa este formato exacto:

TÍTULO: [Título atractivo]

[Párrafo introductorio impactante]

[Desarrollo del artículo en 4-6 párrafos]

[Cierre emotivo]

2. ESTILO REQUERIDO:
   ✓ Todo EN ESPAÑOL
   ✓ Tono cercano y emocionante
   ✓ Usa expresiones de prensa rosa: "se rumorea", "fuentes cercanas revelan", "en exclusiva", "¡bombazo!"
   ✓ Incluye detalles específicos: fechas, lugares, nombres
   ✓ Crea conexión emocional con el lector
   ✓ Mínimo 400 palabras

3. PROHIBIDO:
   ✗ NO copies el esquema tal cual
   ✗ NO uses viñetas ni listas
   ✗ NO dejes secciones vacías
   ✗ NO escribas en inglés

ESCRIBE EL ARTÍCULO COMPLETO AHORA. EMPIEZA CON "TÍTULO:" y continúa con el texto.
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
st.markdown("""
<div style='background: linear-gradient(to right, #ffeef8, #ffe6f0); padding: 15px; border-radius: 10px; margin: 20px 0; border: 2px dashed #ff69b4;'>
    <h3 style='color: #c71585; margin-top: 0; text-align: center;'>💫 Genera tu Artículo Exclusivo 💫</h3>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1], vertical_alignment="top")

with col1:
    user_instruction = st.text_area(
        "� ¿Qué exclusiva quieres revelar?",
        value="Escribe un artículo sobre las últimas noticias de Bad Bunny y su vida amorosa. Incluye rumores recientes, declaraciones y reacciones de sus seguidores.",
        height=140,
        placeholder="Ej: La boda secreta de Shakira, el romance de Rosalía, ¿reconciliación a la vista?, el escándalo que sacude Hollywood...",
        help="Describe el tema sobre el que quieres el artículo"
    )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("✨ ¡Crear Exclusiva!", type="primary", use_container_width=True)
    st.caption("📱 Artículo generado en segundos")
    st.caption("🔥 Con las últimas noticias")

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
st.markdown("<br>", unsafe_allow_html=True)
tabs = st.tabs(["💕 TU EXCLUSIVA", "🔎 Investigación", "📰 Fuentes", "📋 Borrador", "✍️ Redacción Final", "🔧 Detalles Técnicos"])

# Artículo final: normalmente está en el último mensaje del nodo writer
with tabs[0]:
    writer_msgs = st.session_state.traces.get("writer", [])
    if writer_msgs:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #ff6b9d 0%, #c06c84 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h2 style='color: white; text-align: center; margin: 0;'>⭐ ¡EXCLUSIVA! ⭐</h2>
            <p style='color: white; text-align: center; margin: 5px 0 0 0;'>Tu artículo del corazón está listo</p>
        </div>
        """, unsafe_allow_html=True)
        render_message(writer_msgs[-1])
        st.markdown("---")
        st.markdown("💝 *Comparte esta exclusiva con tus amigas* 📱")
    else:
        st.markdown("""
        <div style='text-align: center; padding: 40px;'>
            <h2 style='color: #ff69b4;'>� ¿Lista para tu exclusiva?</h2>
            <p style='color: #c71585; font-size: 1.2em;'>
                Escribe sobre qué celebridad quieres saber y haz clic en <strong>"✨ ¡Crear Exclusiva!"</strong>
            </p>
            <p style='color: #db7093;'>
                🌟 Romances secretos • 💔 Rupturas inesperadas • 💍 Bodas de ensueño • 👶 Bebés en camino
            </p>
        </div>
        """, unsafe_allow_html=True)

with tabs[1]:
    st.markdown("### 🔎 Fase de Investigación")
    st.caption("Nuestro equipo busca las últimas noticias sobre tu celebridad favorita")
    msgs = st.session_state.traces.get("search", [])
    if not msgs:
        st.info("⏳ La investigación comenzará cuando solicites un artículo...")
    else:
        for i, m in enumerate(msgs, start=1):
            st.markdown(f"---\n#### 🔍 Investigación #{i}")
            render_message(m)

with tabs[2]:
    st.markdown("### 📰 Fuentes y Referencias")
    st.caption("Artículos y noticias consultadas de medios especializados")
    msgs = st.session_state.traces.get("tools", [])
    if not msgs:
        st.info("📚 Las fuentes aparecerán aquí durante la investigación...")
    else:
        for i, m in enumerate(msgs, start=1):
            st.markdown(f"---\n#### 📄 Fuente #{i}")
            render_message(m)

with tabs[3]:
    st.markdown("### 📋 Borrador y Estructura")
    st.caption("El esquema preliminar de tu artículo exclusivo")
    msgs = st.session_state.traces.get("outliner", [])
    if not msgs:
        st.info("✏️ El borrador se creará después de recopilar las noticias...")
    else:
        for i, m in enumerate(msgs, start=1):
            st.markdown(f"---\n#### 📝 Esquema #{i}")
            render_message(m)

with tabs[4]:
    st.markdown("### ✍️ Redacción Final")
    st.caption("El artículo completo siendo elaborado por nuestros redactores")
    msgs = st.session_state.traces.get("writer", [])
    if not msgs:
        st.info("📃 La redacción comenzará una vez terminado el borrador...")
    else:
        for i, m in enumerate(msgs, start=1):
            st.markdown(f"---\n#### ✨ Versión #{i}")
            render_message(m)

with tabs[5]:
    st.markdown("### 🔧 Información Técnica")
    st.caption("Detalles del proceso de generación (para desarrolladores)")
    if not st.session_state.raw_updates:
        st.info("⚙️ Los detalles técnicos aparecerán durante el proceso...")
    else:
        with st.expander("📊 Ver trazas completas"):
            st.code(str(st.session_state.raw_updates[:50]))
            if len(st.session_state.raw_updates) > 50:
                st.caption(f"Mostrando 50 de {len(st.session_state.raw_updates)} entradas técnicas.")
