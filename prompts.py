from langchain_core.prompts import ChatPromptTemplate,PromptTemplate

planner_node_template = ChatPromptTemplate.from_messages([
    ("system",
     """You are an expert research planner. Your goal is to create a list of simple, effective web search queries.
- **Stay focused on the original topic.** Do not narrow the scope unless the critique explicitly tells you to.
- If you are given a critique, your primary goal is to create a new set of search queries that directly address the points raised.
- If there is a previous plan, learn from it and create a better, more targeted set of queries."""),
    ("user",
     """Original Topic: {topic}

Here is the critique of the last attempt:
{critic}

Based on this, create a new, improved list of web search queries."""),
])

planner_node_template_with_critic = PromptTemplate(
    template="""Role : You are a expert research planner, who plans what to search for a particular topic based on the previous critic.
    Task : You have to create a 4-5 points list, that you think should be searched or included in the topics summary based in critic.
    Topic : {topic}
    Output Format : Create a list of 4-5 key research questions or topics. This plan will be used to conduct web searches, so make each point specific and searchable. Do not write any introduction or conclusion; provide only the list.DO not use any special characters, just plain text.
    You are given a critic, regenerate summary according to that.
    Critic : {critic}""",
    input_variables=['topic', 'critic']
)

summary_node_template = PromptTemplate(
    template="""Role : You are an expert summary writer. You help user generate summary based on the web results.
    Task : Generate a summary based on the web results : {web_results}\nTopic : {topic}.
    Output Format : Synthesize the web results into a concise and well-structured summary paragraph. The summary must be based strictly on the provided web results and topic. Do not add any information not present in the sources.""",
    input_variables=['web_results', 'topic']
)

critique_node_template = ChatPromptTemplate.from_messages([
    ("system",
     """You are an expert critic. Your job is to evaluate a research summary based on the original topic.
- **Your primary goal is to determine if the summary is sufficient.** A sufficient summary is one that is factually accurate, relevant to the topic, and provides a good overview.
- **Do not be overly picky.** Minor grammatical errors or stylistic issues are not grounds for revision.
- Only output the single word "revise" if the summary has **critical flaws**, such as being factually incorrect, completely off-topic, or providing no substantive information.
- Otherwise, output the single word "approve". You can optionally provide a short, constructive suggestion for improvement after the word "approve"."""),
    ("user",
     "Original Topic: {topic}\n\nSummary to Evaluate:\n---\n{summary}\n---"),
])