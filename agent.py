from langgraph.graph import StateGraph, START, END
from langchain_tavily import TavilySearch
from typing import TypedDict
from langgraph.checkpoint.sqlite import SqliteSaver
import model
import prompts  
from dotenv import load_dotenv
import sqlite3
load_dotenv()

# ------------------------------ Loading models ------------------------------ #
llm = model.ChatOpenAIModel()
tavily = TavilySearch()
sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)
config = {"configurable" : {"thread_id" : 1}}
# ---------------------- Defining the state of the Agent --------------------- #
class AgentState(TypedDict):
    query : str 
    plan : str
    research : str
    summary : str
    critique : str
    decision : str
    revise_count : int
    
    
# ----------------------------- Defining Nodes ------------------------------- #

def planner_node(state: AgentState) -> AgentState:
    """Plans what to research for the given topic, incorporating critique if it exists."""
    
    query = state['query']
    
    # --- THIS IS THE FIX ---
    # Use .get() to safely access the 'critique' key.
    # If it doesn't exist (like on the first run), it will use the default string.
    critique = state.get('critique', 'No critique has been provided yet.')
    
    prompt = prompts.planner_node_template.format_prompt(
        topic=query,
        critic=critique
    )
    
    result = llm.invoke(prompt)
    
    return {'plan': result.content}


def research_node(state : AgentState) -> AgentState:
    """Researches the 'plan' from the state.
       Uses tavily research tool for web searches.
       Args : AgentState -> CUrrent state of the agent."""
    print("\n----- Researching Content -----\n")   
    plan = state['plan']
    queries = [line.strip() for line in plan.strip().split('\n') if line.strip()]
        
    final_results = []
    
    for query in queries:
        result = tavily.invoke(query)
        final_results.append(result)

        
    urls = set()
    formatted_results = []

# This loop goes through the list of API responses (you might only have one)
    for api_response in final_results:
    
    # Safely get the list of search results from inside the response
        search_results = api_response.get('results', [])
    
    # Now, loop directly through each individual search result
        for result_item in search_results:
        # 'result_item' is now one of the dictionaries with 'url' and 'content'
        
            url = result_item.get('url') # Safely get the url
        
        # Check if we have a URL and if we haven't seen it before
            if url and url not in urls:
                content = result_item.get('content', 'No content provided.')
            
            # Append the formatted string
                formatted_results.append(f"URL : {url}\nContent : {content}")
            
            # Add the URL to our set to avoid duplicates
                urls.add(url)

           
    if not formatted_results:
        return {"research": "No relevant information was found on the web for the given research plan."} 
           
    return {'research' : "\n\n".join(formatted_results)}


def summarize_node(state : AgentState) -> AgentState:
    """Summarizes the topic based on the topic and the query results.
       Args : AgentState -> CUrrent state of the agent."""
    print("----- Summarizing Content -----")
    topic = state['query']
    web_results = state['research']
    
    prompt = prompts.summary_node_template.format_prompt(web_results=web_results, topic=topic)
    
    result = llm.invoke(prompt)
    print(f"\n----- Summary : {result.content}\n")
    return {'summary' : result.content}

def critique_node(state : AgentState) -> AgentState:
    """Critizes and cross questions the summary to generate the best output.
       Decides whether to revise the content or output to user.
       Args : AgentState -> CUrrent state of the agent."""
    print("----- Critiquing Content -----")   
    topic = state['query']
    summary = state['summary']
    
    prompt = prompts.critique_node_template.format_prompt(topic=topic, summary=summary)
    
    result = llm.invoke(prompt)
    
    print(f"\nCritique : {result.content}\n")    
    
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

result = app.invoke({'query' : "Perplexity AI in India", 'revise_count' : 0}, config=config)
print("\n\n----- Final Summary -----\n\n")
print(result['summary'])
