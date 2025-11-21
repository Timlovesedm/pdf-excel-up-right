import streamlit as st
import pandas as pd
import pdfplumber
import io
import re
from collections import defaultdict

# --- ツール①：PDFからデータを抽出する関数（複数キーワード対応） ---
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


# --- ツール②：縦方向統合関数 ---
def tool2_extract_data_from_chunk(df_chunk):
    if df_chunk.empty:
        return None, []
    
    quarter_pat = re.compile(r"^\s*(20\d{2}Q[1-4])\s*$", re.IGNORECASE)
    from_date_pat = re.compile(r"\(自\s*(\d{4})年(\d{1,2})月")
    date_pat = re.compile(r"\((\d{4})年(\d{1,2})月") 
    year_pat = re.compile(r"^\s*20\d{2}(\d{2})?\s*$")

    year_cells = []
    for r in range(df_chunk.shape[0]):
        for c in range(df_chunk.shape[1]):
            cell_value = str(df_chunk.iat[r, c]).strip()
            year_header = None

            match_q = quarter_pat.search(cell_value)
            match1 = from_date_pat.search(cell_value)
            match2 = date_pat.search(cell_value)

            if match_q:
                year_header = match_q.group(1).upper()
            elif match1:
                year = match1.group(1)
                month = match1.group(2)
                year_header = f"{year}/{month}"
            elif match2:
                year = match2.group(1)
                month = match2.group(2)
                year_header = f"{year}/{month}"
            elif cell_value.isdigit() and year_pat.match(cell_value):
                year_header = cell_value

            if year_header:
                year_cells.append({"row": r, "col": c, "year_header": year_header})

    if not year_cells:
        return None, []

    year_cells.sort(key=lambda x: (x["row"], x["col"]))
    processed_years = set()
    initial_items = df_chunk[0].astype(str).str.strip().dropna()
    initial_items = initial_items[initial_items != ""]
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
        temp_df = df_chunk.iloc[cell["row"] + 1 :, [0, val_col]].copy()
        temp_df.columns = ["共通項目", year_header]
        temp_df["共通項目"] = temp_df["共通項目"].astype(str).str.strip()
        temp_df = temp_df[temp_df["共通項目"] != ""].dropna(subset=["共通項目"])
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


# --- ツール②：横方向統合関数（完全版） ---
def tool2_extract_data_horizontal(df_chunk):
    """横方向のデータを統合する関数
    
    想定される構造：
    - 1行目に「共通項目」と年次ヘッダー（2024, 2023, 2022, 2021など）
    - 各行に項目名とその年次ごとの数値
    """
    if df_chunk.empty:
        return None, []
    
    # 1行目から列の構造を解析
    first_row = df_chunk.iloc[0] if len(df_chunk) > 0 else pd.Series()
    
    # 共通項目列を検出（通常は一番左の列）
    item_col_idx = None
    for col_idx in range(len(first_row)):
        cell_value = str(first_row.iloc[col_idx]).strip()
        if "共通項目" in cell_value or col_idx == 0:
            item_col_idx = col_idx
            break
    
    if item_col_idx is None:
        item_col_idx = 0  # デフォルトで一番左の列を項目列とする
    
    # 年次列を検出（4桁の数字）
    year_columns = []
    for col_idx in range(len(first_row)):
        if col_idx == item_col_idx:
            continue
        cell_value = str(first_row.iloc[col_idx]).strip()
        # 4桁の年次を検出（2000年代）
        if cell_value.isdigit() and len(cell_value) == 4 and cell_value.startswith('20'):
            year_columns.append({
                "col_idx": col_idx,
                "year": cell_value
            })
    
    if not year_columns:
        return None, []
    
    # 年次で昇順ソート（古い年が左に）
    year_columns.sort(key=lambda x: int(x["year"]))
    
    # データを抽出（1行目はヘッダーなのでスキップ）
    items = []
    data_by_year = {yc["year"]: [] for yc in year_columns}
    
    for row_idx in range(1, df_chunk.shape[0]):
        # 項目名を取得
        item = str(df_chunk.iloc[row_idx, item_col_idx]).strip()
        
        # 項目が有効な場合のみ追加
        if item and item != "nan" and item != "":
            items.append(item)
            
            # 各年次の値を取得
            for yc in year_columns:
                value = str(df_chunk.iloc[row_idx, yc["col_idx"]]).strip()
                data_by_year[yc["year"]].append(value)
    
    if not items:
        return None, []
    
    # 結果DataFrameを構築
    result_dict = {"共通項目": items}
    
    for yc in year_columns:
        year = yc["year"]
        year_values = []
        
        for value_str in data_by_year[year]:
            # 数値変換
            clean_value = str(value_str).replace(",", "").strip()
            try:
                if clean_value and clean_value != "nan" and clean_value != "":
                    # マイナス記号の処理
                    if clean_value.startswith("-") or "△" in clean_value:
                        clean_value = clean_value.replace("△", "-")
                        value = float(clean_value)
                    else:
                        value = float(clean_value)
                    # 整数に変換できる場合は整数に
                    if value == int(value):
                        value = int(value)
                else:
                    value = 0
            except:
                value = 0
            
            year_values.append(value)
        
        result_dict[year] = year_values
    
    result_df = pd.DataFrame(result_dict)
    
    # 項目の順序を保存
    item_order = result_df["共通項目"].tolist()
    
    return result_df, item_order


def process_files_and_tables_vertical(excel_file):
    """縦方向の統合処理（元のロジック）"""
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
    if not file_indices:
        file_chunks.append(df_full)
    else:
        for i in range(len(file_indices)):
            start_idx = file_indices[i]
            end_idx = file_indices[i + 1] if i + 1 < len(file_indices) else len(df_full)
            file_chunks.append(df_full.iloc[start_idx:end_idx].reset_index(drop=True))

    grouped_tables = defaultdict(list)
    master_item_order = defaultdict(list)

    for file_chunk in file_chunks:
        page_indices = file_chunk[file_chunk[0].str.contains(r"--- ページ", na=False)].index.tolist()
        table_chunks = []
        last_idx = 0
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

        for i, table_chunk in enumerate(table_chunks):
            clean_table_chunk = table_chunk[
                ~table_chunk[0].str.contains(r"ファイル名:|---", na=False, regex=True)
            ].dropna(how="all")
            if clean_table_chunk.empty:
                continue
            processed_df, item_order = tool2_extract_data_from_chunk(
                clean_table_chunk.reset_index(drop=True)
            )
            if processed_df is not None and not processed_df.empty:
                grouped_tables[i].append(processed_df)
                current_master_order = master_item_order[i]
                if not current_master_order:
                    master_item_order[i].extend(item_order)
                else:
                    last_known_index = -1
                    for item in item_order:
                        if item in current_master_order:
                            last_known_index = current_master_order.index(item)
                        else:
                            current_master_order.insert(last_known_index + 1, item)
                            last_known_index += 1

    final_summaries = []
    for table_index in sorted(grouped_tables.keys()):
        list_of_dfs = grouped_tables[table_index]
        ordered_items = master_item_order[table_index]
        if not list_of_dfs:
            continue
        result_df = pd.DataFrame({"共通項目": ordered_items})
        for df_to_merge in list_of_dfs:
            cols_to_drop = [
                col for col in df_to_merge.columns if col in result_df.columns and col != "共通項目"
            ]
            result_df = pd.merge(
                result_df, df_to_merge.drop(columns=cols_to_drop), on="共通項目", how="left"
            )
        result_df.fillna(0, inplace=True)
        year_cols = sorted(
            [col for col in result_df.columns if col != "共通項目"],
            key=lambda x: int(str(x).upper().replace('/', '').replace('Q', '0').ljust(6, '0'))
        )
        final_cols = ["共通項目"] + year_cols
        result_df = result_df[final_cols]
        for col in year_cols:
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).astype(int)
        result_df["共通項目"] = result_df["共通項目"].str.replace(r"_temp_\d+$", "", regex=True)
        final_summaries.append(result_df)
    return final_summaries


def process_files_and_tables_horizontal(excel_file):
    """横方向の統合処理"""
    try:
        xls = pd.ExcelFile(excel_file)
        sheet_name_to_read = "抽出結果" if "抽出結果" in xls.sheet_names else xls.sheet_names[0]
        df_full = pd.read_excel(xls, sheet_name=sheet_name_to_read, header=None)
    except Exception as e:
        st.error(f"Excelファイル読み込み失敗: {e}")
        return None
    
    df_full = df_full.astype(str)
    
    # ファイル名でチャンクを分割
    file_indices = df_full[df_full[0].str.contains(r"ファイル名:", na=False)].index.tolist()
    file_chunks = []
    if not file_indices:
        file_chunks.append(df_full)
    else:
        for i in range(len(file_indices)):
            start_idx = file_indices[i]
            end_idx = file_indices[i + 1] if i + 1 < len(file_indices) else len(df_full)
            file_chunks.append(df_full.iloc[start_idx:end_idx].reset_index(drop=True))

    all_table_results = []

    for file_chunk in file_chunks:
        # ページ区切りを検出
        page_indices = file_chunk[file_chunk[0].str.contains(r"--- ページ", na=False)].index.tolist()
        table_chunks = []
        
        if not page_indices:
            clean_chunk = file_chunk[
                ~file_chunk[0].str.contains(r"ファイル名:|---|^\s*$", na=False, regex=True)
            ].dropna(how="all")
            if not clean_chunk.empty:
                table_chunks.append(clean_chunk)
        else:
            last_idx = 0
            for idx in page_indices:
                chunk = file_chunk.iloc[last_idx:idx]
                clean_chunk = chunk[
                    ~chunk[0].str.contains(r"ファイル名:|---", na=False, regex=True)
                ].dropna(how="all")
                if not clean_chunk.empty:
                    table_chunks.append(clean_chunk)
                last_idx = idx
            
            final_chunk = file_chunk.iloc[last_idx:]
            clean_chunk = final_chunk[
                ~final_chunk[0].str.contains(r"ファイル名:|---", na=False, regex=True)
            ].dropna(how="all")
            if not clean_chunk.empty:
                table_chunks.append(clean_chunk)

        # 各テーブルチャンクを処理
        for table_chunk in table_chunks:
            if table_chunk.empty:
                continue
            
            processed_df, item_order = tool2_extract_data_horizontal(
                table_chunk.reset_index(drop=True)
            )
            
            if processed_df is not None and not processed_df.empty:
                all_table_results.append(processed_df)

    if not all_table_results:
        return None

    return all_table_results


# --- Streamlit UI ---
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
    excel_file = st.file_uploader("Excelファイルをアップロード", type=["xlsx"])
    
    # 統合方向の選択（ラジオボタン）
    st.subheader("統合方向を選択")
    direction = st.radio(
        "データの統合方向：",
        options=["縦方向", "横方向"],
        horizontal=True,
        help="縦方向：年次データが縦に並んでいる場合 / 横方向：表が横に複数並んでいる場合（1行目に年次、各ブロックに項目列と数値列）"
    )
    
    if st.button("統合まとめ表を作成 ▶️", disabled=(excel_file is None)):
        with st.spinner("データ整理中..."):
            if direction == "縦方向":
                all_summaries = process_files_and_tables_vertical(excel_file)
            else:  # 横方向
                all_summaries = process_files_and_tables_horizontal(excel_file)
            
            if all_summaries:
                st.success(f"✅ {len(all_summaries)}個のまとめ表を作成！", icon="🎉")
                
                # プレビュー表示
                for i, summary_df in enumerate(all_summaries):
                    st.subheader(f"📊 統合まとめ表_{i+1}")
                    st.dataframe(summary_df, use_container_width=True)
                
                output_excel = io.BytesIO()
                with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
                    for i, summary_df in enumerate(all_summaries):
                        summary_df.to_excel(
                            writer, sheet_name=f"統合まとめ表_{i+1}", index=False
                        )

                base_name_input = excel_file.name.rsplit('.xlsx', 1)[0]
                if base_name_input.endswith('_まとめ'):
                    base_name_output = base_name_input.removesuffix('_まとめ') + '_統合'
                else:
                    base_name_output = base_name_input + '_統合'
                download_filename = f"{base_name_output}.xlsx"

                st.download_button(
                    label="📥 統合まとめ表をダウンロード",
                    data=output_excel.getvalue(),
                    file_name=download_filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.warning("⚠️ 統合可能なデータが見つかりませんでした。", icon="⚠️")
