from typing import List, Optional
from src.state.rag_state import RagState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun


class RAGNodes:
    """Node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        """Initialize the RAG nodes"""
        self.retriever = retriever
        self.llm = llm
        self._agent = None

    def retrieve_docs(self, state: RagState) -> RagState:
        """Retriever node"""
        docs = self.retriever.invoke(state.question)
        return RagState(question=state.question, retrieved_docs=docs)

    def build_tools(self, retrieved_docs: List[Document]) -> List[Tool]:
        """Build retriever tool from pre-fetched docs + Wikipedia tool"""

        def document_retriever_tool(query: str) -> str:
            """A tool that can search through the documents retrieved for the user's question."""
            if not retrieved_docs:
                return "No documents found"
            merged = []
            for i, d in enumerate(retrieved_docs[:8], start=1):
                meta = getattr(d, "metadata", {})
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n".join(merged)

        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from user-provided documents",
            func=document_retriever_tool,
        )

        wiki = WikipediaQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang="en")
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge",
            func=wiki.run,
        )

        return [retriever_tool, wikipedia_tool]

    def _build_agent(self, retrieved_docs: List[Document]):
        """Build the ReAct agent"""
        tools = self.build_tools(retrieved_docs)
        system_prompt = "You are a helpful RAG agent. Prefer using the retriever tool..."

        self._agent = create_react_agent(
        model=self.llm,
        tools=tools,
        messages=[{"role": "system", "content": system_prompt}]
         )


    def generate_answer(self, state: RagState) -> RagState:
       
        if self._agent is None:
            # Pass the retrieved documents from the state to the agent builder
            self._build_agent(state.retrieved_docs)

        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        # Safely extract the last message's content as the answer
        messages = result.get("messages", []) 
        answer: Optional[str] = None
        if messages and hasattr(messages[-1], "content"):
            answer_msg = messages[-1]
            answer = answer_msg.content

        return RagState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=str(answer) if answer is not None else "Sorry, I couldn't find an answer.",
        )