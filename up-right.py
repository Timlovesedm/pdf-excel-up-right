import streamlit as st
import pandas as pd
import pdfplumber
import io
import re
from collections import defaultdict

# ==========================================
# --- ツール①：PDFからデータを抽出する関数 ---
# ==========================================
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


# ==========================================
# --- ツール②：共通ユーティリティ ---
# ==========================================

def detect_year_header(cell_value):
    """セル内の文字列から年次ヘッダー(YYYY, YYYYQ1, YYYY/MM等)を検出する"""
    cell_value = str(cell_value).strip()
    # パターン①：「YYYYQZ」 (年+四半期) 形式
    quarter_pat = re.compile(r"^\s*(20\d{2}Q[1-4])\s*$", re.IGNORECASE)
    # パターン②：「(自 YYYY年MM月...」形式
    from_date_pat = re.compile(r"\(自\s*(\d{4})年(\d{1,2})月")
    # パターン③：「(YYYY年MM月...」または「YYYY年MM月」形式
    date_pat = re.compile(r"\(?(\d{4})年(\d{1,2})月") 
    # パターン④：2024 (4桁) または 202401 (6桁) の数値
    year_pat = re.compile(r"^\s*20\d{2}(\d{2})?\s*$")

    match_q = quarter_pat.search(cell_value)
    match1 = from_date_pat.search(cell_value)
    match2 = date_pat.search(cell_value)

    if match_q:
        return match_q.group(1).upper()
    elif match1:
        return f"{match1.group(1)}/{match1.group(2)}"
    elif match2:
        return f"{match2.group(1)}/{match2.group(2)}"
    elif cell_value.isdigit() and year_pat.match(cell_value):
        return cell_value
    return None

# ==========================================
# --- ツール②：【縦方向】統合ロジック (既存) ---
# ==========================================
def tool2_extract_data_vertical(df_chunk):
    """
    既存のロジック: 表の中にヘッダー列があり、その下にデータがある形式
    """
    if df_chunk.empty:
        return None, []
    
    year_cells = []
    for r in range(df_chunk.shape[0]):
        for c in range(df_chunk.shape[1]):
            cell_value = str(df_chunk.iat[r, c])
            year_header = detect_year_header(cell_value)
            
            if year_header:
                year_cells.append({"row": r, "col": c, "year_header": year_header})

    if not year_cells:
        return None, []

    year_cells.sort(key=lambda x: (x["row"], x["col"]))
    processed_years = set()
    
    # 項目列は0列目と仮定
    initial_items = df_chunk[0].astype(str).str.strip().dropna()
    initial_items = initial_items[initial_items != ""]
    # 「その他」重複対策
    is_sonota = initial_items == "その他"
    if is_sonota.any():
        sonota_counts = initial_items.groupby(initial_items).cumcount()
        initial_items.loc[is_sonota] = "その他_temp_" + sonota_counts[is_sonota].astype(str)
    
    all_items_ordered = initial_items.drop_duplicates(keep="first").tolist()
    df_result = pd.DataFrame({"共通項目": all_items_ordered})

    for cell in year_cells:
        year_header = cell["year_header"]
        if year_header in processed_years:
            continue
        processed_years.add(year_header)
        val_col = cell["col"]
        # ヘッダー行の次からデータを取得
        temp_df = df_chunk.iloc[cell["row"] + 1 :, [0, val_col]].copy()
        temp_df.columns = ["共通項目", year_header]
        temp_df["共通項目"] = temp_df["共通項目"].astype(str).str.strip()
        temp_df = temp_df[temp_df["共通項目"] != ""].dropna(subset=["共通項目"])
        
        # 重複処理
        is_sonota = temp_df["共通項目"] == "その他"
        if is_sonota.any():
            sonota_counts = temp_df.groupby("共通項目").cumcount()
            temp_df.loc[is_sonota, "共通項目"] = "その他_temp_" + sonota_counts[is_sonota].astype(str)
            
        temp_df[year_header] = (
            pd.to_numeric(temp_df[year_header].astype(str).str.replace(",", ""), errors='coerce').fillna(0)
        )
        temp_df = temp_df.drop_duplicates(subset=["共通項目"], keep="first")
        df_result = pd.merge(df_result, temp_df, on="共通項目", how="left")

    return df_result, all_items_ordered

# ==========================================
# --- ツール②：【横方向】統合ロジック (新規) ---
# ==========================================
def tool2_extract_data_horizontal(df_chunk):
    """
    新規ロジック: 
    - 左の文字列と一番右の数値のみ統合する
    - 項目(左) | ... | 数値(右) の形式
    - ヘッダー(年次)はこのブロック内のどこか(主に上部)にあると仮定
    """
    if df_chunk.empty:
        return None, []

    # 1. 年次ヘッダーを探す（チャンク内の最初の数行を走査）
    detected_header = None
    for r in range(min(5, df_chunk.shape[0])): # 上から5行以内で探す
        for c in range(df_chunk.shape[1]):
            val = df_chunk.iat[r, c]
            header_cand = detect_year_header(val)
            if header_cand:
                detected_header = header_cand
                break
        if detected_header:
            break
    
    # ヘッダーが見つからない場合は、ダミーまたはファイル名依存になるが、今回はスキップ扱いにするか汎用名にする
    if not detected_header:
        # 明示的な日付がない場合、処理不能としてNoneを返すか、あるいは強制的に取り込むか。
        # ここでは安全のためNoneを返すが、必要に応じて "Unknown" で処理も可能
        return None, []

    # 2. データ抽出（左端列と右端列）
    # 空の列を削除して、確実に端の列を取得する
    clean_chunk = df_chunk.dropna(axis=1, how='all')
    if clean_chunk.shape[1] < 2:
        return None, [] # 列が足りない

    item_col_idx = 0
    val_col_idx = clean_chunk.shape[1] - 1

    # データフレーム構築
    temp_df = clean_chunk.iloc[:, [item_col_idx, val_col_idx]].copy()
    temp_df.columns = ["共通項目", detected_header]
    
    # クレンジング
    temp_df["共通項目"] = temp_df["共通項目"].astype(str).str.strip()
    temp_df = temp_df[temp_df["共通項目"] != ""].dropna(subset=["共通項目"])
    
    # 数値と思われる行のみ残す、あるいは文字列行(ヘッダーなど)を除外するフィルタ
    # シンプルに数値変換できるか、もしくは項目名が長すぎる(文章)場合は除外するなどの処理
    temp_df = temp_df[temp_df["共通項目"].str.len() < 50] # 仮：極端に長い項目は説明文とみなして除外
    
    # 数値変換
    temp_df[detected_header] = (
        pd.to_numeric(temp_df[detected_header].astype(str).str.replace(",", ""), errors='coerce')
    )
    # 数値がNaNになった行（ヘッダー行やゴミデータ）を削除 (0埋めではなく削除)
    temp_df = temp_df.dropna(subset=[detected_header])

    # 「その他」などの重複処理
    is_sonota = temp_df["共通項目"] == "その他"
    if is_sonota.any():
        sonota_counts = temp_df.groupby("共通項目").cumcount()
        temp_df.loc[is_sonota, "共通項目"] = "その他_temp_" + sonota_counts[is_sonota].astype(str)

    # 同じ項目が複数行ある場合は合計する (例: 小計行などがなく単純なリストの場合)
    temp_df = temp_df.groupby("共通項目", as_index=False).sum()

    # 項目リスト（順序保持用）
    item_list = temp_df["共通項目"].tolist()

    return temp_df, item_list


# ==========================================
# --- ツール②：ファイル処理メイン関数 ---
# ==========================================
def process_files_and_tables(excel_file, integration_mode):
    try:
        xls = pd.ExcelFile(excel_file)
        sheet_name_to_read = "抽出結果" if "抽出結果" in xls.sheet_names else xls.sheet_names[0]
        df_full = pd.read_excel(xls, sheet_name=sheet_name_to_read, header=None)
    except Exception as e:
        st.error(f"Excelファイル読み込み失敗: {e}")
        return None

    df_full[0] = df_full[0].astype(str)
    file_indices = df_full[df_full[0].str.contains(r"ファイル名:", na=False)].index.tolist()
    file_chunks = []
    
    # ファイルごとに分割
    if not file_indices:
        file_chunks.append(df_full)
    else:
        for i in range(len(file_indices)):
            start_idx = file_indices[i]
            end_idx = file_indices[i + 1] if i + 1 < len(file_indices) else len(df_full)
            file_chunks.append(df_full.iloc[start_idx:end_idx].reset_index(drop=True))

    grouped_tables = defaultdict(list)
    master_item_order = defaultdict(list)

    # 各ファイルチャンクを処理
    for file_chunk in file_chunks:
        page_indices = file_chunk[file_chunk[0].str.contains(r"--- ページ", na=False)].index.tolist()
        table_chunks = []
        last_idx = 0
        
        # ページ/テーブルごとに分割
        if not page_indices:
            clean_chunk = file_chunk[
                ~file_chunk[0].str.contains(r"ファイル名:|---|^\s*$", na=False, regex=True)
            ].dropna(how="all")
            if not clean_chunk.empty:
                table_chunks.append(clean_chunk)
        else:
            for idx in page_indices:
                chunk = file_chunk.iloc[last_idx:idx]
                if not chunk.empty:
                    table_chunks.append(chunk)
                last_idx = idx
            final_chunk = file_chunk.iloc[last_idx:]
            if not final_chunk.empty:
                table_chunks.append(final_chunk)

        # 各テーブルチャンクを解析
        for i, table_chunk in enumerate(table_chunks):
            clean_table_chunk = table_chunk[
                ~table_chunk[0].str.contains(r"ファイル名:|---", na=False, regex=True)
            ].dropna(how="all")
            
            if clean_table_chunk.empty:
                continue
            
            # --- モードによる分岐 ---
            if integration_mode == "vertical":
                processed_df, item_order = tool2_extract_data_vertical(clean_table_chunk.reset_index(drop=True))
            else: # horizontal
                processed_df, item_order = tool2_extract_data_horizontal(clean_table_chunk.reset_index(drop=True))
            # -----------------------

            if processed_df is not None and not processed_df.empty:
                grouped_tables[i].append(processed_df)
                
                # マスタ項目の順序を更新（和集合を作成しつつ順序維持）
                current_master_order = master_item_order[i]
                if not current_master_order:
                    master_item_order[i].extend(item_order)
                else:
                    last_known_index = -1
                    for item in item_order:
                        if item in current_master_order:
                            last_known_index = current_master_order.index(item)
                        else:
                            # 新出項目は直前の既知項目の後ろに挿入
                            current_master_order.insert(last_known_index + 1, item)
                            last_known_index += 1

    # 最終マージ処理
    final_summaries = []
    for table_index in sorted(grouped_tables.keys()):
        list_of_dfs = grouped_tables[table_index]
        ordered_items = master_item_order[table_index]
        
        if not list_of_dfs:
            continue
            
        result_df = pd.DataFrame({"共通項目": ordered_items})
        
        for df_to_merge in list_of_dfs:
            # 既に存在する列名と重複しないようにマージ
            cols_to_drop = [
                col for col in df_to_merge.columns if col in result_df.columns and col != "共通項目"
            ]
            result_df = pd.merge(
                result_df, df_to_merge.drop(columns=cols_to_drop), on="共通項目", how="left"
            )
            
        result_df.fillna(0, inplace=True)
        
        # 列のソート (YYYY/MM, YYYY, YYYYQZ 対応)
        year_cols = sorted(
            [col for col in result_df.columns if col != "共通項目"],
            key=lambda x: int(str(x).upper().replace('/', '').replace('Q', '0').ljust(6, '0'))
        )
        final_cols = ["共通項目"] + year_cols
        result_df = result_df[final_cols]
        
        # 数値整形
        for col in year_cols:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).astype(int) # 整数表示
            
        # 一時的な項目名（_temp_数字）を元に戻す
        result_df["共通項目"] = result_df["共通項目"].str.replace(r"_temp_\d+$", "", regex=True)
        
        final_summaries.append(result_df)
        
    return final_summaries


# ==========================================
# --- Streamlit UI ---
# ==========================================
st.set_page_config(page_title="多機能ツール", layout="wide")
st.title("📄📊 多機能ツール")

# --- ツール① ---
with st.container(border=True):
    st.header("ツール①：PDF表データ抽出")
    pdf_files = st.file_uploader(
        "PDFファイルをアップロード（複数可）", type="pdf", accept_multiple_files=True
    )
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
                df_result = extract_tables_from_multiple_pdfs(
                    pdf_files, keywords, start_page, end_page
                )
                if df_result is not None and not df_result.empty:
                    st.success("抽出完了！", icon="✅")
                    st.dataframe(df_result)
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        df_result.to_excel(writer, index=False, header=False, sheet_name="抽出結果")
                        workbook = writer.book
                        worksheet = writer.sheets["抽出結果"]
                        bold_format = workbook.add_format({"bold": True, "font_size": 20})
                        for idx, val in enumerate(df_result[0]):
                            if isinstance(val, str) and val.startswith("ファイル名:"):
                                worksheet.set_row(idx, None, bold_format)
                    
                    if keywords:
                        base_name = '_'.join(keywords)
                        download_filename = f"{base_name}_まとめ.xlsx"
                    else:
                        download_filename = "抽出結果_まとめ.xlsx"

                    st.download_button(
                        label="📥 Excelファイルをダウンロード",
                        data=output.getvalue(),
                        file_name=download_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
        else:
            st.error("PDFファイルをアップロードしてください。", icon="🚨")

st.divider()

# --- ツール② ---
with st.container(border=True):
    st.header("ツール②：統合データ作成")
    
    st.info("📝 データの並び方を選択してください")
    # ラジオボタンでモード選択
    integration_mode_label = st.radio(
        "統合モード選択",
        ("縦方向統合 (従来の形式)", "横方向統合 (項目:左 / 数値:右)"),
        help="データが縦に積み上がっている場合は「縦方向」、横並びの年次データを結合する場合は「横方向」を選択してください"
    )
    
    # 内部ロジック用のフラグ変換
    integration_mode = "vertical" if "縦方向" in integration_mode_label else "horizontal"
    
    excel_file = st.file_uploader("Excelファイルをアップロード", type=["xlsx"])
    
    if st.button("統合まとめ表を作成 ▶️", disabled=(excel_file is None)):
        with st.spinner("データ整理中..."):
            # 選択されたモードを関数に渡す
            all_summaries = process_files_and_tables(excel_file, integration_mode)
            
            if all_summaries:
                st.success(f"{len(all_summaries)}個のまとめ表を作成！", icon="✅")
                output_excel = io.BytesIO()
                with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
                    for i, summary_df in enumerate(all_summaries):
                        sheet_name = f"統合まとめ表_{i+1}"
                        summary_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        # 列幅調整などの簡易フォーマット
                        worksheet = writer.sheets[sheet_name]
                        worksheet.set_column(0, 0, 30) # 項目列を広げる

                base_name_input = excel_file.name.rsplit('.xlsx', 1)[0]
                mode_suffix = "_縦統合" if integration_mode == "vertical" else "_横統合"
                if base_name_input.endswith('_まとめ'):
                    base_name_output = base_name_input.removesuffix('_まとめ') + mode_suffix
                else:
                    base_name_output = base_name_input + mode_suffix
                download_filename = f"{base_name_output}.xlsx"

                st.download_button(
                    label="📥 統合まとめ表をダウンロード",
                    data=output_excel.getvalue(),
                    file_name=download_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.warning("有効なデータが見つかりませんでした。モードやファイルを確認してください。", icon="⚠️")