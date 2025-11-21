import streamlit as st
import pandas as pd
import pdfplumber
import io
import re
from collections import defaultdict

# --- ツール①：PDFからデータを抽出する関数（変更なし） ---
def extract_tables_from_multiple_pdfs(pdf_files, keywords, start_page, end_page):
    all_rows = []
    if not keywords:
        st.error("❗ キーワードが入力されていません。", icon="🚨")
        return None
    for pdf_file in pdf_files:
        all_rows.append([f"ファイル名: {pdf_file.name}"])
        all_rows.append([])
        found_in_file = False
        try:
            with pdfplumber.open(pdf_file) as pdf:
                start_index = start_page - 1 if start_page else 0
                end_index = end_page if end_page else len(pdf.pages)
                target_pages = pdf.pages[start_index:end_index]
                for page in target_pages:
                    text = page.extract_text() or ""
                    if any(kw in text for kw in keywords):
                        found_in_file = True
                        tables = page.extract_tables()
                        for table_index, table in enumerate(tables):
                            if not table:
                                continue
                            all_rows.append([f"--- ページ {page.page_number} / テーブル {table_index + 1} ---"])
                            for row in table:
                                cleaned_row = ["" if item is None else str(item).replace("\n", " ") for item in row]
                                all_rows.append(cleaned_row)
                            all_rows.append([])
        except Exception as e:
            st.error(f"ファイル「{pdf_file.name}」処理中にエラー: {e}", icon="🔥")
            continue
        if not found_in_file:
            st.warning(f"ファイル「{pdf_file.name}」ではキーワードを含む表が見つかりませんでした。", icon="⚠️")

    if not any(r for r in all_rows if r):
        return None
    return pd.DataFrame(all_rows)

# --- 横方向統合 ---
def horizontal_merge(df_chunk):
    df_chunk = df_chunk.fillna("")
    merged_df = pd.DataFrame()
    # 列ブロック単位で統合
    n_cols = df_chunk.shape[1]
    col_indices = list(range(0, n_cols, 3))  # 3列ずつのブロックを想定（左、中央、右）
    for idx, start_col in enumerate(col_indices):
        left_col = start_col
        right_col = min(start_col + 2, n_cols - 1)
        block_df = df_chunk.iloc[:, [left_col, right_col]].copy()
        block_df.columns = ["項目", f"値_{idx+1}"]
        block_df["項目"] = block_df["項目"].astype(str).str.strip()
        block_df[f"値_{idx+1}"] = pd.to_numeric(block_df[f"値_{idx+1}"].astype(str).str.replace(",", ""), errors='coerce').fillna(0)
        if merged_df.empty:
            merged_df = block_df
        else:
            merged_df = pd.merge(merged_df, block_df, on="項目", how="outer")
    merged_df.fillna(0, inplace=True)
    # 各ブロックの一番上の数値でソート
    first_value_cols = [col for col in merged_df.columns if col.startswith("値_")]
    if first_value_cols:
        merged_df = merged_df.sort_values(by=first_value_cols[0], ascending=True).reset_index(drop=True)
    return merged_df

# --- 縦方向統合（既存の簡略版） ---
def vertical_merge(df_chunk):
    df_chunk = df_chunk.fillna("")
    df_chunk.columns = ["項目", "値"]
    df_chunk["項目"] = df_chunk["項目"].astype(str).str.strip()
    df_chunk["値"] = pd.to_numeric(df_chunk["値"].astype(str).str.replace(",", ""), errors='coerce').fillna(0)
    return df_chunk

# --- ツール②：Excel統合処理 ---
def process_excel(file, direction="縦"):
    try:
        xls = pd.ExcelFile(file)
        sheet_name_to_read = "抽出結果" if "抽出結果" in xls.sheet_names else xls.sheet_names[0]
        df_full = pd.read_excel(xls, sheet_name=sheet_name_to_read, header=None)
    except Exception as e:
        st.error(f"Excelファイル読み込み失敗: {e}")
        return None

    if direction == "縦":
        result_df = vertical_merge(df_full)
    else:
        result_df = horizontal_merge(df_full)
    return result_df

# --- Streamlit UI ---
st.set_page_config(page_title="多機能ツール", layout="wide")
st.title("📄📊 多機能ツール")

# --- ツール① ---
with st.container():
    st.header("ツール①：PDF表データ抽出")
    pdf_files = st.file_uploader("PDFファイルをアップロード（複数可）", type="pdf", accept_multiple_files=True)
    keyword_input_str = st.text_input("検索キーワード（カンマ区切り）")
    col1, col2 = st.columns(2)
    start_page_input = col1.text_input("開始ページ", placeholder="例: 5")
    end_page_input = col2.text_input("終了ページ", placeholder="例: 10")
    if st.button("抽出開始 ▶️"):
        if pdf_files:
            keywords = [kw.strip() for kw in keyword_input_str.split(",") if kw.strip()]
            start_page = int(start_page_input) if start_page_input.isdigit() else None
            end_page = int(end_page_input) if end_page_input.isdigit() else None
            with st.spinner("PDF解析中..."):
                df_result = extract_tables_from_multiple_pdfs(pdf_files, keywords, start_page, end_page)
                if df_result is not None and not df_result.empty:
                    st.success("抽出完了！", icon="✅")
                    st.dataframe(df_result)
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        df_result.to_excel(writer, index=False, header=False, sheet_name="抽出結果")
                    st.download_button(
                        label="📥 Excelファイルをダウンロード",
                        data=output.getvalue(),
                        file_name="抽出結果_まとめ.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
        else:
            st.error("PDFファイルをアップロードしてください。", icon="🚨")

st.divider()

# --- ツール② ---
with st.container():
    st.header("ツール②：Excel統合")
    excel_file = st.file_uploader("Excelファイルをアップロード", type=["xlsx"])
    merge_direction = st.radio("統合方向を選択", ["縦方向", "横方向"])
    if st.button("統合まとめ表を作成 ▶️", disabled=(excel_file is None)):
        with st.spinner("データ整理中..."):
            merged_df = process_excel(excel_file, direction=merge_direction)
            if merged_df is not None:
                st.success("統合完了！", icon="✅")
                st.dataframe(merged_df)
                output_excel = io.BytesIO()
                with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
                    merged_df.to_excel(writer, sheet_name="統合まとめ表", index=False)
                base_name_input = excel_file.name.rsplit('.xlsx', 1)[0]
                download_filename = f"{base_name_input}_統合.xlsx"
                st.download_button(
                    label="📥 統合まとめ表をダウンロード",
                    data=output_excel.getvalue(),
                    file_name=download_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
