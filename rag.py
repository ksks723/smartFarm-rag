from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

print("📄 문서 불러오는 중...")
loader = TextLoader("farming_data.txt")
documents = loader.load()

print("✂️  문서 쪼개는 중...")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print("🧠 문서 학습시키는 중...")
embeddings = OllamaEmbeddings(model="llama3.2")
vectorstore = Chroma.from_documents(chunks, embeddings)

# 핵심: 한국어로만 답하고 문서 내용만 사용하도록 지시
prompt_template = """
당신은 스마트팜 환경 관리 전문가입니다.
반드시 한국어로만 답변하세요.
아래 참고 문서의 내용만 사용하여 답변하세요.
문서에 없는 내용은 "매뉴얼에 해당 내용이 없습니다"라고 답하세요.

참고 문서:
{context}

질문: {question}

답변:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

print("🤖 AI 연결 중...")
llm = Ollama(model="llama3.2")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT}
)

print("\n✅ 스마트팜 AI 준비 완료!")
print("종료하려면 'quit' 입력\n")

while True:
    question = input("질문: ")
    if question.lower() == "quit":
        break
    print("\nAI 답변 중...\n")
    answer = qa.invoke(question)
    print(f"답변: {answer['result']}\n")
    print("-" * 50 + "\n")
