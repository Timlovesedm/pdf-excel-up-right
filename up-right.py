import streamlit as st
import pandas as pd
import pdfplumber
import io
import re
from collections import defaultdict

# ==========================================
# 共通ユーティリティ・PDF抽出ロジック (Tool 1)
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
# ツール②：縦方向統合ロジック (Vertical Integration)
# ==========================================

def tool2_extract_data_from_chunk_vertical(df_chunk):
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


def process_files_and_tables_vertical(excel_file):
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
            processed_df, item_order = tool2_extract_data_from_chunk_vertical(
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


# ==========================================
# ツール②：横方向統合ロジック (Horizontal) - 再構築版
# ==========================================

def split_into_horizontal_blocks(df):
    """
    空白列を区切りとして、DataFrameを複数の「ブロック（表）」に分割する関数
    """
    blocks = []
    current_cols = []
    
    # 列ごとにチェック
    for col in df.columns:
        # 列が全て空（NaNまたは空文字）かチェック
        is_empty_col = df[col].astype(str).str.strip().replace("nan", "").eq("").all()
        
        if is_empty_col:
            if current_cols:
                blocks.append(df[current_cols].copy())
                current_cols = []
        else:
            current_cols.append(col)
            
    # 最後のブロックを追加
    if current_cols:
        blocks.append(df[current_cols].copy())
        
    return blocks

def extract_data_from_block(block_df):
    """
    1つのブロックから {Year: {Item: Value}} のデータを抽出する
    """
    # 正規表現パターンの定義（縦方向と同じ強力なパターン）
    patterns = [
        re.compile(r"^\s*(20\d{2}Q[1-4])\s*$", re.IGNORECASE),
        re.compile(r"\(自\s*(\d{4})年(\d{1,2})月"),
        re.compile(r"\((\d{4})年(\d{1,2})月"),
        re.compile(r"20\d{2}") # シンプルな年号
    ]
    
    year_header = None
    value_col_idx = None
    item_col_idx = None
    
    # 1. 年次ヘッダー（数値列）を探す
    # ブロック内の最初の数行をスキャン
    for r in range(min(5, len(block_df))):
        for c in range(len(block_df.columns)):
            cell_val = str(block_df.iat[r, c]).strip()
            
            for pat in patterns:
                match = pat.search(cell_val)
                if match:
                    # 年号が見つかった
                    found_year = match.group(0)
                    # 正規表現の結果からきれいな年号文字列を作る
                    nums = re.findall(r"20\d{2}", found_year)
                    if nums:
                        year_header = nums[0]
                    else:
                        year_header = found_year # Q1などをそのまま使う場合
                    
                    value_col_idx = c
                    break
            if year_header: break
        if year_header: break
        
    if not year_header:
        return None, None
        
    # 2. 項目列を探す（数値列の左側にあると仮定）
    # 基本的に一番左の列、もしくは数値列の1つ左
    if value_col_idx > 0:
        item_col_idx = 0 # ブロックの一番左を項目列とするのが一般的
    else:
        return None, None # 項目列がない
        
    # 3. データ抽出
    items = []
    values = []
    item_counter = defaultdict(int)
    
    # ヘッダー行の次からスキャン
    start_row = r + 1
    
    for i in range(start_row, len(block_df)):
        raw_item = str(block_df.iat[i, item_col_idx]).strip()
        raw_val = str(block_df.iat[i, value_col_idx]).strip()
        
        if raw_item and raw_item != "nan":
            # 重複処理（A, B, その他など）
            count = item_counter[raw_item]
            item_counter[raw_item] += 1
            
            if raw_item == "その他":
                item_name = f"{raw_item}_temp_{count}"
            elif count > 0:
                item_name = f"{raw_item}_{count}"
            else:
                item_name = raw_item
                
            # 数値処理
            clean_val = raw_val.replace(",", "").replace("△", "-").replace("▲", "-").strip()
            try:
                val = float(clean_val)
                if val.is_integer(): val = int(val)
            except:
                val = 0
            
            items.append(item_name)
            values.append(val)
            
    return year_header, pd.DataFrame({"共通項目": items, year_header: values})


def process_files_and_tables_horizontal(excel_file):
    """
    横方向統合のメイン関数
    ブロック分割 -> 各抽出 -> マージ（順序保持）
    """
    try:
        xls = pd.ExcelFile(excel_file)
        sheet_name_to_read = "抽出結果" if "抽出結果" in xls.sheet_names else xls.sheet_names[0]
        df_full = pd.read_excel(xls, sheet_name=sheet_name_to_read, header=None)
    except Exception as e:
        st.error(f"Excelファイル読み込み失敗: {e}")
        return None

    # 1. ファイル名行などで大きなチャンクに分ける（Tool 1の出力形式依存）
    # ただし、横方向の場合は1シートにまとめて貼り付けられていることが多いので
    # まずは行ごとの区切り（ファイル名）で分割し、その中でさらに「ブロック」を探す
    
    df_full = df_full.astype(str)
    file_indices = df_full[df_full[0].str.contains(r"ファイル名:", na=False)].index.tolist()
    
    file_chunks = []
    if not file_indices:
        file_chunks.append(df_full)
    else:
        for i in range(len(file_indices)):
            start_idx = file_indices[i]
            # ファイル名行自体はデータではないのでスキップしたいが、
            # チャンクとして渡して後で除外する
            end_idx = file_indices[i + 1] if i + 1 < len(file_indices) else len(df_full)
            file_chunks.append(df_full.iloc[start_idx:end_idx].reset_index(drop=True))

    all_extracted_dfs = []
    all_item_orders = []

    for chunk in file_chunks:
        # "ファイル名:" や "--- ページ" などのメタデータ行を除外して純粋な表データにする
        clean_rows = []
        for idx, row in chunk.iterrows():
            row_txt = row.astype(str).str.cat()
            if "ファイル名:" in str(row[0]) or "--- ページ" in str(row[0]):
                continue
            clean_rows.append(row)
        
        if not clean_rows:
            continue
            
        df_clean = pd.DataFrame(clean_rows)
        
        # 2. 空白列でブロック分割
        blocks = split_into_horizontal_blocks(df_clean)
        
        # 3. 各ブロックからデータを抽出
        for block in blocks:
            if block.empty: continue
            year, df_data = extract_data_from_block(block)
            
            if year and df_data is not None:
                all_extracted_dfs.append(df_data)
                all_item_orders.append(df_data["共通項目"].tolist())

    if not all_extracted_dfs:
        return None

    # 4. 統合ロジック（マスタ項目の作成・順序保持）
    master_items = []
    
    # 全ての項目リストを巡回してマスターリストを育てる
    for items in all_item_orders:
        if not master_items:
            master_items = list(items)
            continue
        
        # 相対位置を学習しながら挿入
        last_known_idx = -1
        for item in items:
            if item in master_items:
                last_known_idx = master_items.index(item)
            else:
                # 未知の項目（Bなど）は、直前の既知項目の後ろに挿入
                master_items.insert(last_known_idx + 1, item)
                last_known_idx += 1

    # 5. マスターフレームの作成とマージ
    final_df = pd.DataFrame({"共通項目": master_items})
    
    for df in all_extracted_dfs:
        # 年次カラム名を取得
        year_col = [c for c in df.columns if c != "共通項目"][0]
        
        # マージ
        merged = pd.merge(final_df, df, on="共通項目", how="left")
        
        # 既に同じ年がある場合は update (combine_first)
        if year_col in final_df.columns:
             final_df[year_col] = final_df[year_col].combine_first(merged[year_col])
        else:
             final_df[year_col] = merged[year_col]

    # 6. 0埋めと整形
    final_df = final_df.fillna(0)
    
    # 年次順に並べ替え（降順）
    cols = [c for c in final_df.columns if c != "共通項目"]
    cols.sort(key=lambda x: float(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else 0, reverse=True)
    
    final_df = final_df[["共通項目"] + cols]
    
    # 表示用にサフィックス削除
    final_df["共通項目"] = final_df["共通項目"].str.replace(r"_temp_\d+$", "", regex=True)
    final_df["共通項目"] = final_df["共通項目"].str.replace(r"_\d+$", "", regex=True)

    return [final_df]


# ==========================================
# Streamlit UI
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
    excel_file = st.file_uploader("Excelファイルをアップロード", type=["xlsx"])
    
    # 統合方向の選択
    st.subheader("統合方向を選択")
    direction = st.radio(
        "データの統合方向：",
        options=["縦方向", "横方向"],
        horizontal=True,
        help="縦方向：年次データが縦に並んでいる場合 / 横方向：表が横に複数並んでいる場合"
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
