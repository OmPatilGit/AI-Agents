from langgraph.graph import StateGraph, START, END
from langchain_tavily import TavilySearch
from typing import TypedDict
from rich.console import Console
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import model
import prompts


# ------------------------------ Loading models ------------------------------ #
llm = model.ChatOpenAIModel()
tavily = TavilySearch()
console = Console()
sqlite_conn = sqlite3.connect("checkpoint.sqlite")
memory = SqliteSaver(conn=sqlite_conn)

# ---------------------- Defining the state of the Agent --------------------- #
class AgentState(TypedDict):
    query : str 
    plan : str
    research : str
    summary : str
    critique : str
    decision : str
    revise_count : int = 0
    
    
# ----------------------------- Defining Nodes ------------------------------- #

def planner_node(state : AgentState) -> AgentState:
    """Plans what to research for the given topic.
    Args : AgentState -> CUrrent state of the agent."""
    
    query = state['query']
    prompt = prompts.planner_node_template.format_prompt(topic=query)
    
    result = llm.invoke(prompt)
    
    return {'plan' : result.content}


def research_node(state : AgentState) -> AgentState:
    """Researches the 'plan' from the state.
       Uses tavily research tool for web searches.
       Args : AgentState -> CUrrent state of the agent."""
       
    plan = state['plan']
    queries = [line.strip() for line in plan.strip().split('\n') if line.strip()]
    
    final_results = []
    
    for query in queries:
        result = tavily.invoke(query)
        final_results.extend(result)
        
    urls = set()
    formatted_results = []
    
    for res in final_results:
        if res['url'] not in urls:
            formatted_results.append(f"URL : {res['url']}\nContent : {res['content']}")
            urls.add(res['url'])
            
    return {'research' : "\n\n".join(formatted_results)}


def summarize_node(state : AgentState) -> AgentState:
    """Summarizes the topic based on the topic and the query results.
       Args : AgentState -> CUrrent state of the agent."""
    
    topic = state['query']
    web_results = state['research']
    
    prompt = prompts.summary_node_template.format_prompt(web_results=web_results, topic=topic)
    
    result = llm.invoke(prompt)
    
    return {'summary' : result.content}

def critique_node(state : AgentState) -> AgentState:
    """Critizes and cross questions the summary to generate the best output.
       Decides whether to revise the content or output to user.
       Args : AgentState -> CUrrent state of the agent."""
       
    topic = state['query']
    summary = state['summary']
    
    prompt = prompts.critique_node_template.format_prompt(topic=topic, summary=summary)
    
    result = llm.invoke(prompt)
    
    print(f"Critique : {result.content}")    
    
    if "revise" in result.content.strip().lower():
        return {"critique": result.content, "decision": "revise"}
    else:
        return {"critique": result.content, "decision": "approve"}
    

def should_continue(state : AgentState):
    """Conditional Edge : Whether to continue or no."""
    
    if state['decision'] == "approve":
        return "END"
    
    else:
        state['revise_count'] += 1
        if state['revise_count'] > 3:
            return "END"
        return "REVISE"
    
# ---------------------------- Designing The Graph --------------------------- #

graph = StateGraph(state_schema=AgentState)

graph.add_node("plan",planner_node)
graph.add_node("research",research_node)
graph.add_node("summarize",summarize_node)
graph.add_node("critic",critique_node)

graph.add_edge(START, "plan")
graph.add_edge("plan", "research")
graph.add_edge("research", "summarize")
graph.add_edge("summarize", "critic")


graph.add_conditional_edges(
    source="critic",
    path=should_continue,
    path_map={
        "REVISE" : "plan",
        "END" : END
    }
)

app = graph.compile(checkpointer=memory)
