from langchain_core.prompts import PromptTemplate

planner_node_template = PromptTemplate(
    template="""Role : You are a expert research planner, who plans what to search for a particular topic.
    Task : You have to create a 4-5 bulleted points list, that you think should be searched or included in the topics summary.
    Topic : {topic}
    Output Format : Create a bulleted list of 4-5 key research questions or topics. This plan will be used to conduct web searches, so make each point specific and searchable. Do not write any introduction or conclusion; provide only the bulleted list.
    
    Example : 
    Topic : Democracy
    Output : 
    - Origins and evolution of democracy: from ancient Athens to modern representative systems.
    - Core principles: popular sovereignty, rule of law, equality, liberty, and accountability.
    - Types of democracy: direct, representative, parliamentary, presidential, and hybrid models.
    - Strengths and challenges: inclusivity, stability, corruption, populism, polarization, and threats from authoritarianism.
    - Role of institutions and civic participation: elections, media, judiciary, and civil society in sustaining democracy.
    """,
    input_variables=['topic']
)

summary_node_template = PromptTemplate(
    template="""Role : You are an expert summary writer. You help user generate summary based on the web results.
    Task : Generate a summary based on the web results : {web_results}\nTopic : {topic}.
    Output Format : Synthesize the web results into a concise and well-structured summary paragraph. The summary must be based strictly on the provided web results and topic. Do not add any information not present in the sources. 
    
    Example:
    Topic : Democray 
    Web Results : Origin : Rooted in Athens; reforms by Cleisthenes in ~508 BCE laid democratic foundations.
                  Core Principles : Sovereignty, law, equality, liberty, accountability.
                  Democracy Types : Direct, representative, constitutional, parliamentary, presidential, hybrid, monitory.
                  Strengths & Challenges : Inclusivity and accountability vs populism, corruption, polarization, democratic erosion.
                  Institutions & Civic Role : Elections, media, courts, civil society, and active citizenship sustain democratic health.
                  
    Output : Democracy began as a radical experiment in ancient Athens—a limited system yet visionary in giving citizens a voice. Over centuries, it has morphed into complex, institutionalized systems worldwide, grounded in universal values like equality, liberty, and accountability. Yet, despite its adaptability and strengths, democracy remains fragile—constantly tested by internal flaws, external pressures, and the erosion of civic norms. Upholding it demands vigilant institutions, informed public participation, and unwavering adherence to democratic principles.""",
    input_variables=['web_results', 'topic']
)

critique_node_template = PromptTemplate(
    template="""Role: You are an expert critic.You review the summary for the given topic in a very expert and comprehensive way.
    Task : Review the given summary based on the topic given. Review based on the summary's quality, clarity, and completeness, decide if it's sufficient.
    Topic : {topic}
    Summary : {summary}
    Output Format : If you are satisfied with summary respond with 'approve'.
                    If not satisfied with summary respond with 'revise', and provide brief, actionable critique.
                    
    Example 1: 
    Topic : Democracy 
    Summary : Democracy began as a radical experiment in ancient Athens—a limited system yet visionary in giving citizens a voice. Over centuries, it has morphed into complex, institutionalized systems worldwide, grounded in universal values like equality, liberty, and accountability. Yet, despite its adaptability and strengths, democracy remains fragile—constantly tested by internal flaws, external pressures, and the erosion of civic norms. Upholding it demands vigilant institutions, informed public participation, and unwavering adherence to democratic principles.
    
    Output : revise - It omits key aspects such as the role of representation vs. direct democracy, global variations (liberal, social, participatory models), and the importance of rule of law and human rights.
    
    Example 2: 
    Topic : Democracy
    Summary : Democracy, first emerging in ancient Athens as a radical but limited experiment in direct citizen participation, has evolved into diverse systems across the globe. Modern democracies take many forms—liberal, representative, participatory, and social—yet all rest on universal principles of equality, liberty, accountability, and the rule of law. Human rights, separation of powers, and protection of minorities are central to its functioning. Despite its adaptability, democracy remains vulnerable to populism, corruption, disinformation, and external pressures. Safeguarding it requires resilient institutions, civic education, informed participation, and steadfast commitment to democratic values.
    
    Output : approve"""
)