
from src.state.rag_state import RagState

class RagNodes:
    """ node function for rag workflow"""
    
    def __init__(self,retriever,llm):
        "Intialize the rag nodes"
        
        self.retirever =retriever
        self.llm=llm
        
    def retrieve_docs(self,state:RagState)->RagState:
        "Retrieve the document node"
        
        docs=self.retirever.invoke(state.question)
        return RagState(question = state.question, retrieved_docs = docs)
    
    def generate_answer(self, state :RagState) -> RagState:
        "Generate the answer node"
        
        #combinng the retrieved docs into context
        context = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        
        #create prompt
        prompt=f""" Answer the question based on the context.
        
context:
{context}
        
question:{state.question}"""
        
        #generate answer
        response=self.llm.invoke(prompt)
        
        return RagState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )
        