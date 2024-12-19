import streamlit as st
from Bio import Entrez
import time
import pandas as pd
import os
import boto3
from langchain_community.chat_models import BedrockChat

def is_valid_email(email):
    import re
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

def initialize_bedrock():
    required_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        raise ValueError(f"必要な環境変数が見つかりません: {', '.join(missing_vars)}")
    
    bedrock = boto3.client('bedrock-runtime')
    llm = BedrockChat(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        client=bedrock
    )
    return llm

def generate_introduction(llm, articles_df):
    combined_text = "以下の論文を先行研究として、研究イントロダクション（背景）のドラフトを作成してください：\n\n"
    
    for _, row in articles_df.iterrows():
        combined_text += f"論文タイトル: {row['Title']}\n"
        combined_text += f"著者: {row['Authors']}\n"
        combined_text += f"要旨: {row['Abstract']}\n\n"
    
    prompt = f"""{combined_text}

イントロダクション作成の要件：

1. 構成
- 一般的な内容から具体的なトピックへの論理的な展開
- 研究分野の重要性と社会的意義の説明
- 現在までの研究動向と課題の整理
- 新規研究の必要性と意義の提示

2. 先行研究の引用方法
- 研究の文脈に沿った有機的な構成
- 複数の研究結果の統合的な提示
- 対立する見解がある場合の公平な提示
- 最新の研究動向の反映

3. 表現上の注意点
- 建設的な表現の使用
- 論理的な文章展開
- 明確な課題提起
- 学術的な文体の維持

出力形式：
- 段落分けされた文章形式（約1000字）
- 各段落の冒頭に簡単な説明コメント
"""

    response = llm.predict(prompt)
    return response

def fetch_pubmed_articles(pmids, email):
    Entrez.email = email
    articles_data = []

    for pmid in pmids:
        try:
            record = Entrez.read(Entrez.efetch(db="pubmed", id=pmid, retmode="xml"))
            article = record['PubmedArticle'][0]['MedlineCitation']['Article']

            article_info = {
                'PMID': pmid,
                'Title': article['ArticleTitle'],
                'Journal': article['Journal']['Title'],
                'Year': article['Journal'].get('JournalIssue', {}).get('PubDate', {}).get('Year', ''),
                'Authors': '; '.join([
                    f"{author.get('LastName', '')} {author.get('ForeName', '')}"
                    for author in article.get('AuthorList', [])
                ]),
                'Abstract': article.get('Abstract', {}).get('AbstractText', [''])[0] if isinstance(
                    article.get('Abstract', {}).get('AbstractText', ['']), list
                ) else article.get('Abstract', {}).get('AbstractText', ''),
                'Keywords': '; '.join(
                    record['PubmedArticle'][0]['MedlineCitation'].get('KeywordList', [[]])[0]
                ) if record['PubmedArticle'][0]['MedlineCitation'].get('KeywordList') else ''
            }

            articles_data.append(article_info)
            time.sleep(1)

        except Exception as e:
            st.error(f"PMID {pmid} の取得中にエラーが発生しました: {e}")
            continue

    return pd.DataFrame(articles_data)

def main():
    st.title("研究イントロダクション作成支援ツール")
    st.markdown("PubMed論文を基に研究背景のドラフトを生成します")
    
    try:
        llm = initialize_bedrock()
    except Exception as e:
        st.error("Bedrock初期化エラー: " + str(e))
        return
    
    with st.container():
        st.markdown("### 基本情報入力")
        email = st.text_input("メールアドレス:")
        
        st.markdown("### 文献番号（PMID）入力")
        st.markdown("※最大3件まで入力可能です")
        pmid1 = st.text_input("PMID 1:")
        pmid2 = st.text_input("PMID 2 (任意):")
        pmid3 = st.text_input("PMID 3 (任意):")
    
    # PMIDタグの表示
    if pmid1 or pmid2 or pmid3:
        st.markdown("### 選択した文献番号")
        
        # カスタムCSSの適用
        st.markdown("""
            <style>
            .pmid-tag {
                background-color: #ff4b4b;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                margin: 2px;
                display: inline-block;
                font-size: 0.9em;
            }
            </style>
            """, unsafe_allow_html=True)
        
        cols = st.columns(3)
        if pmid1:
            cols[0].markdown(f'<div class="pmid-tag">{pmid1}</div>', unsafe_allow_html=True)
        if pmid2:
            cols[1].markdown(f'<div class="pmid-tag">{pmid2}</div>', unsafe_allow_html=True)
        if pmid3:
            cols[2].markdown(f'<div class="pmid-tag">{pmid3}</div>', unsafe_allow_html=True)

    if st.button("イントロダクションを生成"):
        if not email or not is_valid_email(email):
            st.error("有効なメールアドレスを入力してください。")
            return
        
        if not pmid1:
            st.error("少なくとも1つのPMIDを入力してください。")
            return
        
        pmids = [pid.strip() for pid in [pmid1, pmid2, pmid3] if pid.strip()]
        
        with st.spinner('論文情報を取得中...'):
            df = fetch_pubmed_articles(pmids, email)
            
            if not df.empty:
                st.subheader("取得した論文情報")
                st.dataframe(df)
                
                with st.spinner('イントロダクションを作成中...'):
                    try:
                        intro = generate_introduction(llm, df)
                        st.subheader("生成されたイントロダクション案")
                        st.markdown(intro)
                    except Exception as e:
                        st.error(f"イントロダクション生成エラー: {e}")
            else:
                st.error("論文情報を取得できませんでした。")

if __name__ == "__main__":
    main()