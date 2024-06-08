# from langchain.llms import GooglePalm
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI

import os
from dotenv import load_dotenv
from few_shots import few_shots
load_dotenv()

def get_few_shot_db_chain():

    # llm = GooglePalm(google_api_key= os.environ['google_api_key'], temperature=0.2)
    llm = GoogleGenerativeAI(model="models/text-bison-001", google_api_key = 'AIzaSyCGSdPyLPMq1CNtuhmU4XTVOSZNymJvKHE', temperature=0.1)



    # Connecting to the my-sql workbench:
    db_user = "root"
    db_password = "admin"
    db_host = "localhost"
    db_name = "atliq_tshirts"

    cs  = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    db_engine = create_engine(cs)
    db = SQLDatabase(db_engine)

    model_name1 = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name = model_name1)

    to_vectorize = [''.join(qns.values()) for qns in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore = vectorstore,
        k  = 2,
    )

    example_prompt = PromptTemplate(
        input_variables = ['Question','SQLQuery','SQLResult','Answer'],
        template = "\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector = example_selector,
        example_prompt = example_prompt,
        prefix = _mysql_prompt,
        suffix = PROMPT_SUFFIX,
        input_variables = ["input", "table_info", "top_k"],
    )

    chain = SQLDatabaseChain.from_llm(llm, db, verbose = True, prompt = few_shot_prompt)
    return chain

if __name__ == "__main__":
    chain = get_few_shot_db_chain()
    print(chain.run("How many t-shirts are there of red color?"))
