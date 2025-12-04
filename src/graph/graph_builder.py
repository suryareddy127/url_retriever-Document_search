
from langgraph.graph import StateGraph,END
from src.state.rag_state import RagState
from src.nodes.reactnode import RAGNodes


class GraphBuilder:
    """
    Manages the building and compilation of the RAG LangGraph.
    """
    def __init__(self,retriever,llm):
        "Initialize the grapgh builder"
        
        self.nodes=RAGNodes(retriever,llm)
        self.graph=None
        
    def build(self):
        """Build the RAG workflow LangGraph."""
        
        #create state graph
        builder = StateGraph(RagState)
        
        #add nodes
        builder.add_node("retriever",self.nodes.retrieve_docs)
        builder.add_node("response",self.nodes.generate_answer)
        
        builder.set_entry_point('retriever')
        
        # add edges
        builder.add_edge("retriever","response")
        builder.add_edge("response",END)
        
        self.graph=builder.compile()
        return self.graph
    
    def run(self,question:str) -> dict:
        """Run the RAG workflow with the given question."""
        
        if self.graph is None:
            self.build()
        
        initial_state = RagState(question=question)
        return self.graph.invoke(initial_state)