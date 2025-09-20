from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

# 加载环境变量
load_dotenv()                                    # 加载.env文件

# 设置页面标题和布局
st.set_page_config(page_title="简易RAG文档问答系统", layout="wide")    # 设置页面标题和布局
st.title("📄 简易RAG文档问答系统")                                # 设置页面标题

# 初始化session state
if "processed_files" not in st.session_state:                   # 初始化是否处理过文件
    st.session_state.processed_files = False
if "vectorstore" not in st.session_state:                       # 初始化向量数据库
    st.session_state.vectorstore = None
if "messages" not in st.session_state:                         # 初始化聊天历史
    st.session_state.messages = [
        {"role": "assistant", "content": "您好，我可以回答关于您文档的问题，或者进行一般性对话！"}]

# 文件上传组件
uploaded_files = st.sidebar.file_uploader(
    "上传TXT文档",
    type=["txt"],
    accept_multiple_files=True
)

# 处理上传的文件
if uploaded_files:                                                           # 处理上传的文件
    with st.spinner("正在处理文档，请稍候..."):                                   # 显示加载动画
        docs = []                                                             # 保存处理后的文档
        temp_dir = tempfile.TemporaryDirectory()                              # 创建临时目录

        for file in uploaded_files:                                             # 遍历上传的文件
            # 保存上传的文件到临时目录
            temp_filepath = os.path.join(temp_dir.name, file.name)            # 保存文件到临时目录
            with open(temp_filepath, "wb") as f:                              # 保存文件到临时目录
                f.write(file.getvalue())                                     # 保存文件到临时目录

            # 使用TextLoader加载文本文件
            loader = TextLoader(temp_filepath, encoding="utf-8")             # 加载文本文件
            docs.extend(loader.load())                                       # 加载文档

        # 文档分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)

        # 创建向量数据库
        embeddings = HuggingFaceEmbeddings()

        st.session_state.vectorstore = FAISS.from_documents(
            documents=splits,                                         # 文档分割
            embedding=embeddings                                     # 使用HuggingFaceEmbeddings进行向量化
        )

        st.session_state.processed_files = True
        st.success("文档处理完成！")

# 显示聊天历史
for msg in st.session_state.messages:                                       # 显示聊天历史
    st.chat_message(msg["role"]).write(msg["content"])                      # 显示聊天历史

# 聊天输入框 - 无论是否上传文件都可以提问
user_query = st.chat_input("请输入您的问题...")                                # 聊天输入框

if user_query:                                                              # 处理用户输入
    # 添加用户消息到历史
    st.session_state.messages.append({"role": "user", "content": user_query})   # 添加用户消息到历史
    st.chat_message("user").write(user_query)                                   # 显示用户消息

    with st.chat_message("assistant"):                                         # 显示机器人消息
        with st.spinner("正在思考..."):                                          # 显示加载动画
            if st.session_state.processed_files and st.session_state.vectorstore:    # 处理过文件
                # 使用RAG模式（有上传文档）
                try:
                    # 创建LLM
                    llm = ChatOpenAI(
                        model="deepseek-chat",
                        base_url='https://api.deepseek.com/v1',
                        temperature=0.1
                    )

                    # 创建记忆模块
                    memory = ConversationBufferMemory(
                        return_messages=True,
                        memory_key='chat_history',
                        output_key='answer'
                    )

                    # 创建检索器和链条
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={'k': 3})
                    chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        memory=memory,
                        retriever=retriever,
                        return_source_documents=True
                    )

                    # 调用链条
                    response = chain.invoke({
                        'question': user_query,
                        'chat_history': []
                    })

                    answer = response['answer']

                except Exception as e:
                    answer = f"处理文档时出错：{str(e)}"
            else:
                # 使用普通对话模式（无上传文档）
                try:
                    llm = ChatOpenAI(
                        model="deepseek-chat",
                        base_url='https://api.deepseek.com/v1',
                        temperature=0.7
                    )
                    response = llm.invoke(user_query)
                    answer = response.content
                except Exception as e:
                    answer = f"生成回答时出错：{str(e)}"

            # 显示答案
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})