import streamlit as st
from Bio import Entrez
import time
import pandas as pd

def is_valid_email(email):
    import re
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return re.match(pattern, email) is not None

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
            st.error(f"Error with PMID {pmid}: {e}")
            continue

    return pd.DataFrame(articles_data)

def main():
    st.title("PubMed Article Fetcher")
    
    # メールアドレス入力
    email = st.text_input("Enter your email address:")
    
    # PMID入力欄（最大3つ）
    pmid1 = st.text_input("Enter PMID 1:")
    pmid2 = st.text_input("Enter PMID 2 (optional):")
    pmid3 = st.text_input("Enter PMID 3 (optional):")
    
    # 入力されたPMIDを表示するエリア
    if pmid1 or pmid2 or pmid3:
        st.write("Selected PMIDs:")
        col1, col2, col3 = st.columns(3)
        
        # カスタムCSSでタグのスタイルを定義
        st.markdown("""
            <style>
            .pmid-tag {
                background-color: #ff4b4b;
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                margin: 2px;
                display: inline-block;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # 入力されたPMIDをタグとして表示
        if pmid1:
            col1.markdown(f'<div class="pmid-tag">{pmid1}</div>', unsafe_allow_html=True)
        if pmid2:
            col2.markdown(f'<div class="pmid-tag">{pmid2}</div>', unsafe_allow_html=True)
        if pmid3:
            col3.markdown(f'<div class="pmid-tag">{pmid3}</div>', unsafe_allow_html=True)
    
    # 実行ボタン
    if st.button("Fetch Articles"):
        # 入力チェック
        if not email or not is_valid_email(email):
            st.error("Please enter a valid email address.")
            return
        
        if not pmid1:
            st.error("Please enter at least one PMID.")
            return
        
        # 有効なPMIDのリストを作成
        pmids = [pid.strip() for pid in [pmid1, pmid2, pmid3] if pid.strip()]
        
        # データ取得と表示
        with st.spinner('Fetching articles...'):
            df = fetch_pubmed_articles(pmids, email)
            if not df.empty:
                st.dataframe(df)
            else:
                st.error("No articles were retrieved.")

if __name__ == "__main__":
    main()