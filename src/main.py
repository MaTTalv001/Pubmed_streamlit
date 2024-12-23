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

def generate_introduction(llm, articles_df, add_text=""):
    combined_text = "以下の論文を先行研究とし、研究イントロダクション（背景）のドラフトを作成してください：\n\n"
    
    for _, row in articles_df.iterrows():
        combined_text += f"論文タイトル: {row['Title']}\n"
        combined_text += f"著者: {row['Authors']}\n"
        combined_text += f"要旨: {row['Abstract']}\n\n"
    
    prompt = f"""{combined_text}

#イントロダクション作成の要件：

## 全体像と導入  
イントロ冒頭では、まず研究分野の全体像と学術的・社会的背景を手短に示す。読者がテーマの意義を瞬時に理解できるよう、背景と目的を簡潔に提示し、「なぜこのトピックが重要なのか」を明確に打ち出すことが重要。研究動向を大まかに触れることで、その後の展開にも自然に誘導する。

## 研究分野の重要性と先行研究の整理 
研究分野の重要性を社会的・産業的観点などから掘り下げ、これまで先行研究がどのような課題に取り組み、どんな成果を出してきたかを要約する。政策との関連や実用的インパクトを示すことで、読者の関心を引きつける。さらに、先行研究の到達点と未解決の問題を指摘し、本研究の必要性を強調する。

## 先行研究の引用と構成 
先行研究の引用は、ただ成果を並べるのではなく、それぞれの関係性や流れを示すことが重要。複数の文献を比較し、自身の研究との接点を示すことで、読者に本研究の位置づけを明確に伝える。最新の研究動向を盛り込むことで、研究の時宜性もアピール可能。引用スタイルは所定の形式を守り、論理的な文脈を意識する。

## 対立する見解と最新研究への配慮
見解が対立する研究がある場合は、どのような根拠や方法論が異なるかを公正に提示する。そのうえで、本研究がその対立をどう整理・解決し、どの立場をとるかを明示する。単なる批判に終わらず、多角的な視点を受け入れる姿勢を示すことで、学問的信頼性と説得力を高める。

## 第五段落：研究課題と結論への導き  
イントロの締めくくりでは、本研究の目的と課題を明確に打ち出し、先行研究とのギャップや新規性を簡潔に示す。研究手法や期待される効果にも軽く触れ、読者が論文全体の方向性を把握できるようにする。こうした締めによって、研究の学術的価値や社会的意義を提示し、本論文の説得力を高める。

# 先行研究の内容に加えユーザーが付け加えたいことことがある場合はそれを加味する
ユーザーが付け加えたいこと：{add_text} (空欄の場合は無視)
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
    st.title("研究イントロ作成支援")
    st.markdown("PubMed論文を基に研究背景のドラフトを生成します")
    
    try:
        llm = initialize_bedrock()
    except Exception as e:
        st.error("Bedrock初期化エラー: " + str(e))
        return
    
    with st.container():
        st.markdown("### 基本情報入力")
        st.write("NCBIのガイドラインに基づき、使用の際にはメールアドレスが必要です")
        email = st.text_input("メールアドレス:")
        
        st.markdown("### 文献番号（PMID）入力")
        st.markdown("※入力後Enterで確定してください(最大3件まで)")
        
        col1, col2, col3 = st.columns(3)

        with col1:
            pmid1 = st.text_input("PMID 1:")
        with col2:
            pmid2 = st.text_input("PMID 2 (任意):")
        with col3:
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
    add_txt = st.text_area("先行研究に加え、イントロを書く上で踏まえたい事項があれば記入してください（任意）")
    
    st.markdown("***")
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
                        intro = generate_introduction(llm, df, add_txt)
                        st.subheader("生成されたイントロダクション案")
                        st.markdown(intro)
                    except Exception as e:
                        st.error(f"イントロダクション生成エラー: {e}")
            else:
                st.error("論文情報を取得できませんでした。")

if __name__ == "__main__":
    main()