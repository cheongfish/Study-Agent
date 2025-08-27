import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
def setup_database_and_insert_data(
    _DB_PARAMS,
    dim:int=1024
    ):
    """
    데이터베이스에 연결하여 테이블을 생성하고, JSON 파일의 데이터를 삽입합니다.
    """
    conn = None
    try:
        # 데이터베이스에 연결
        print("데이터베이스에 연결 중...")
        conn = psycopg2.connect(**_DB_PARAMS)
        cursor = conn.cursor()
        print("연결 성공.")

        # 1. pgvector 확장 기능 활성화
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("pgvector 확장 기능이 활성화되었습니다.")

        # 2. 테이블 생성 (테이블이 이미 존재하면 실행되지 않음)
        # 'order'는 SQL 예약어이므로 'order_val'로 변경합니다.
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS curriculum (
            id SERIAL PRIMARY KEY,
            basecode TEXT UNIQUE,
            school_level TEXT,
            grade TEXT,
            domain TEXT,
            category TEXT,
            content TEXT,
            order_val INT,
            embedding VECTOR({dim})
        );
        """
        cursor.execute(create_table_query)
        print("'curriculum' 테이블이 준비되었습니다.")

        # 3. JSON 파일 읽기
        json_path = '/home/cheongwoon/workspace/Study-Agent/Demo/Data/curriculum_with_embeddings.json'
        df = pd.read_json(json_path)
        # NaN 값을 None으로 변환하여 DB에 NULL로 입력되도록 함
        df = df.where(pd.notnull(df), None)
        print(f"'{json_path}' 파일에서 {len(df)}개의 데이터를 읽었습니다.")
        
        # --- ✨✨✨ 오류 해결을 위한 코드 수정 부분 ✨✨✨ ---
        # 4. 데이터 삽입을 위해 튜플 리스트로 변환
        # embedding 리스트의 각 요소를 numpy.float32에서 python float으로 변환합니다.
        data_tuples = []
        for row in df.itertuples(index=False):
            # embedding 필드가 리스트인지 확인하고, 각 요소를 float으로 변환
            embedding_list = [e for e in row.embedding] if isinstance(row.embedding, list) else None
            
            # order 필드를 int로 변환 (None이 아닌 경우)
            order_val = int(row.order) if row.order is not None else None

            data_tuples.append((
                row.basecode, row.school_level, row.grade, row.domain, 
                row.category, row.content, order_val, embedding_list
            ))

        # 4. 데이터 삽입
        # ON CONFLICT를 사용하여 gu_bun_value가 중복될 경우 삽입을 건너뛰도록 함
        insert_query = """
        INSERT INTO curriculum (basecode, school_level, grade, domain, category, content, order_val, embedding)
        VALUES %s
        ON CONFLICT (basecode) DO NOTHING;
        """
        
        # DataFrame을 튜플 리스트로 변환
        data_tuples = [tuple(row) for row in df[['basecode', 'school_level', 'grade', 'domain', 'category', 'content', 'order', 'embedding']].itertuples(index=False)]
        
        print("데이터 삽입을 시작합니다...")
        execute_values(cursor, insert_query, data_tuples)
        conn.commit()
        print(f"{len(data_tuples)}개의 데이터 삽입이 완료되었습니다.")

    except psycopg2.Error as e:
        print(f"데이터베이스 오류: {e}")
    finally:
        if conn:
            cursor.close()
            conn.close()
            print("데이터베이스 연결이 종료되었습니다.")