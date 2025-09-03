import os
pwd = os.getcwd()
DB_PARAMS = {
    'host': os.environ.get("POSTGRES_HOST",'localhost'),
    'port':  os.environ.get("POSTGRES_PORT",'5432'),
    'dbname': os.environ.get("POSTGRES_DB",'postgres'), # 사용할 데이터베이스 이름
    'user': os.environ.get("POSTGRES_USER",'postgres'),     # 데이터베이스 사용자 이름
    'password': os.environ.get("POSTGRES_PWD",'test1234'),  # 데이터베이스 비밀번호
}
embedding_model_name = "BAAI/bge-m3"
project_name =  os.environ.get("PROJECT_NAME",'Test_Demo_App')
env_path = "/".join(pwd.split("/")[:-2]) + "/.env"