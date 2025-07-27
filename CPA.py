# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import japanize_matplotlib
import warnings
import io
from pytrends.request import TrendReq

# 警告を非表示にする
warnings.filterwarnings('ignore')

# Matplotlibのスタイルを設定
plt.style.use('seaborn-v0_8-whitegrid')

# ============== Streamlit アプリケーションのUI設定 ==============
st.set_page_config(
    page_title="CPA予測モデル (Googleトレンド対応版)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- カスタムCSSで見た目を調整 ---
st.markdown("""
<style>
    .stMetric {
        border-radius: 10px;
        padding: 15px;
        background-color: #f0f2f6;
    }
    .st-emotion-cache-1r4qj8v {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============== 関数の定義 ==============

@st.cache_data
def load_and_preprocess_data(file_all, file_30d):
    if file_all is None or file_30d is None:
        return None, None, "ファイルがアップロードされていません。サイドバーから2つのCSVファイルをアップロードしてください。"
    log_messages = []
    def read_csv_robust(file):
        encodings = ['shift_jis', 'cp932', 'utf-8']
        file.seek(0)
        for encoding in encodings:
            try:
                return pd.read_csv(file, encoding=encoding)
            except UnicodeDecodeError:
                file.seek(0)
                continue
        raise UnicodeDecodeError(f"ファイル '{file.name}' をどのエンコーディングでも読み込めませんでした。")
    try:
        df_all = read_csv_robust(file_all)
        df_30d = read_csv_robust(file_30d)
        log_messages.append("✅ CSVファイルの読み込みに成功しました。")
    except Exception as e:
        return None, None, f"❌ ファイル読み込みエラー: {e}"
    
    df_all['weight'] = 1.0
    df_30d['weight'] = 2.0
    log_messages.append(f"✅ モデルが直近のデータ({file_30d.name})を重視するように、重みを2.0に調整しました。")
    df = pd.concat([df_all, df_30d], ignore_index=True)
    df.columns = df.columns.str.replace(' ', '').str.replace('　', '')

    if '日' not in df.columns:
        return None, None, "❌ 必要なカラム '日' が見つかりません。CSVに `YYYY-MM-DD` 形式の '日' カラムを追加してください。"
    df['日'] = pd.to_datetime(df['日'], errors='coerce')
    df.dropna(subset=['日'], inplace=True)
    log_messages.append("✅ 日付データを認識しました。")

    try:
        pytrends = TrendReq(hl='ja-JP', tz=540)
        kw_list = ["塾講師 バイト", "塾 バイト"]
        start_date = df['日'].min().strftime('%Y-%m-%d')
        end_date = df['日'].max().strftime('%Y-%m-%d')
        timeframe = f'{start_date} {end_date}'
        pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='JP', gprop='')
        trends_df = pytrends.interest_over_time()

        if trends_df.empty or 'isPartial' in trends_df.columns and trends_df['isPartial'].any():
             log_messages.append("⚠️ Googleトレンドのデータが取得できないか、不完全でした。デフォルト値(50)で補完します。")
             df['google_trend'] = 50
        else:
            trends_df['google_trend'] = trends_df[kw_list].mean(axis=1)
            trends_df = trends_df.reset_index().rename(columns={'date': '日'})
            trends_df = trends_df[['日', 'google_trend']]
            trends_df['日'] = pd.to_datetime(trends_df['日'])
            df = pd.merge(df, trends_df, on='日', how='left')
            df['google_trend'].fillna(method='ffill', inplace=True)
            df['google_trend'].fillna(method='bfill', inplace=True)
            log_messages.append(f"✅ Googleトレンドのデータ({', '.join(kw_list)})を取得し、平均値を結合しました。")

    except Exception as e:
        log_messages.append(f"⚠️ Googleトレンドの取得に失敗しました ({e})。デフォルト値(50)で補完します。")
        df['google_trend'] = 50

    required_columns = ['日', 'インプレッション数', 'クリック数', 'クリック率', 'コスト', 'コンバージョン数', 'インプレッションシェア', 'ページ最上部のインプレッションシェア', 'google_trend']
    required_columns_with_weight = required_columns + ['weight']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return None, None, f"❌ 必要なカラムが見つかりません: {missing_cols}。利用可能なカラム: {df.columns.tolist()}"
    
    for col in [c for c in required_columns if c not in ['日', 'google_trend']]:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.replace(',', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=required_columns_with_weight, inplace=True)
    log_messages.append("✅ データ型のクレンジングと欠損値の除去を行いました。")
    Q1, Q3 = df['コスト'].quantile(0.25), df['コスト'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    original_rows = len(df)
    df_cleaned = df[(df['コスト'] >= lower_bound) & (df['コスト'] <= upper_bound)].copy()
    log_messages.append(f"✅ 外れ値を除去しました。(元のデータ数: {original_rows}, 除去後: {len(df_cleaned)})")
    if len(df_cleaned) < 10:
         return None, None, "❌ 前処理後のデータが少なすぎます。十分なデータ量のあるCSVをアップロードしてください。"
    log_messages.append("✅ データの前処理が完了しました。")
    return df_cleaned, log_messages, None

@st.cache_resource
def train_models(_df_cleaned):
    features_for_sub_models = ['コスト', 'google_trend']
    sub_models = {}
    dependent_features = ['インプレッション数', 'クリック数', 'インプレッションシェア', 'ページ最上部のインプレッションシェア']
    for feature in dependent_features:
        X_sub, y_sub, weights_sub = _df_cleaned[features_for_sub_models], _df_cleaned[feature], _df_cleaned['weight']
        sub_model = lgb.LGBMRegressor(random_state=42, objective='regression_l1')
        sub_model.fit(X_sub, y_sub, sample_weight=weights_sub)
        sub_models[feature] = sub_model
    main_features = ['コスト', 'google_trend'] + dependent_features
    target = 'コンバージョン数'
    X_main, y_main, weights_main = _df_cleaned[main_features], _df_cleaned[target], _df_cleaned['weight']
    main_model = LinearRegression()
    main_model.fit(X_main, y_main, sample_weight=weights_main)
    coefficients = pd.DataFrame(main_model.coef_, X_main.columns, columns=['回帰係数'])
    return sub_models, main_model, coefficients

def run_simulation(df_cleaned, sub_models, main_model, input_budget, input_trend):
    min_cost_data = df_cleaned['コスト'].min()
    max_cost_data = df_cleaned['コスト'].max()
    graph_max_cost = max(max_cost_data, input_budget) * 1.1
    num_steps = max(200, int((graph_max_cost - min_cost_data) / 1000))
    cost_range = np.linspace(min_cost_data, graph_max_cost, num_steps)
    sim_results = []
    main_features = ['コスト', 'google_trend'] + list(sub_models.keys())
    for cost in cost_range:
        sim_data = {'コスト': cost, 'google_trend': input_trend}
        for feature, model in sub_models.items():
            predict_df = pd.DataFrame({'コスト': [cost], 'google_trend': [input_trend]})
            sim_data[feature] = model.predict(predict_df)[0]
        sim_df_row = pd.DataFrame([sim_data])
        predicted_cv = main_model.predict(sim_df_row[main_features])[0]
        sim_results.append({'コスト': cost, '予測コンバージョン数': predicted_cv})
    sim_df = pd.DataFrame(sim_results)
    sim_df['予測コンバージョン数'] = sim_df['予測コンバージョン数'].clip(lower=0)
    sim_df['予測CPA'] = sim_df.apply(lambda r: r['コスト'] / r['予測コンバージョン数'] if r['予測コンバージョン数'] > 0.01 else np.nan, axis=1)
    sim_df.dropna(subset=['予測CPA'], inplace=True)
    return sim_df

# ### 変更 ### : グラフ描画関数に最適点の引数を追加
def create_plot(sim_df, input_budget, predicted_cv, predicted_cpa, optimal_budget=None, optimal_cv=None, optimal_cpa=None):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    color = 'royalblue'
    ax1.set_xlabel('予算（コスト）[円]', fontsize=14)
    ax1.set_ylabel('予測コンバージョン数', fontsize=14, color=color)
    ax1.plot(sim_df['コスト'], sim_df['予測コンバージョン数'], linestyle='-', color=color, label='予測コンバージョン数', linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2 = ax1.twinx()
    color = 'mediumseagreen'
    ax2.set_ylabel('予測CPA [円]', fontsize=14, color=color)
    ax2.plot(sim_df['コスト'], sim_df['予測CPA'], linestyle='--', color=color, label='予測CPA', linewidth=2.5)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # ユーザー入力の予算をプロット
    if predicted_cv is not None and predicted_cpa is not None:
        ax1.axvline(x=input_budget, color='tomato', linestyle=':', linewidth=2, label=f"入力予算: {input_budget:,.0f}円")
        ax1.plot(input_budget, predicted_cv, 'o', color='tomato', markersize=10, markeredgecolor='white', markeredgewidth=1.5)
        ax2.plot(input_budget, predicted_cpa, 'o', color='tomato', markersize=10, markeredgecolor='white', markeredgewidth=1.5)

    # ### 追加 ### : 最適予算の点をプロット
    if optimal_budget is not None:
        ax1.axvline(x=optimal_budget, color='gold', linestyle=':', linewidth=2, label=f"最適予算: {optimal_budget:,.0f}円")
        ax1.plot(optimal_budget, optimal_cv, 'o', color='gold', markersize=10, markeredgecolor='black', markeredgewidth=1.5)
        ax2.plot(optimal_budget, optimal_cpa, 'o', color='gold', markersize=10, markeredgecolor='black', markeredgewidth=1.5)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=12, frameon=True, shadow=True)
    fig.tight_layout()
    return fig


# ============== UI: サイドバー ==============
with st.sidebar:
    st.markdown("## ⚙️ 設定パネル")
    with st.expander("📁 データアップロード", expanded=True):
        uploaded_file_all = st.file_uploader(
            "1. 全期間データ (CSV)",
            type="csv",
            help="過去の全ての広告パフォーマンスデータを含むCSVファイルをアップロードしてください。`日`カラムが必要です。"
        )
        uploaded_file_30d = st.file_uploader(
            "2. 直近データ (CSV)",
            type="csv",
            help="予測で特に重視したい直近期間（例: 過去30日間）のデータファイルをアップロードしてください。`日`カラムが必要です。"
        )
    st.markdown("---")
    st.markdown("## 予算シミュレーション")
    input_budget = st.number_input(
        "予測したい予算（円）を入力",
        min_value=0,
        value=None,
        placeholder="例: 50000",
        step=1000,
        help="任意の予算額を円単位で入力してください。"
    )
    st.markdown("## 外部要因の設定")
    input_trend = st.slider(
        "📈 将来のトレンド指数 (任意)",
        min_value=0,
        max_value=100,
        value=50,
        help="「塾講師 バイト」「塾 バイト」のGoogleトレンド検索数を想定して設定します。100が最大関心時です。"
    )
    recommendation_placeholder = st.empty()


# ============== UI: メインページ ==============
st.title('📈 CPA予測モデル (Googleトレンド対応版)')
st.markdown("過去の広告パフォーマンスデータとGoogleトレンドの検索量を元に、最適な広告予算を見つけ出しましょう。")

with st.expander("💡 使い方ガイド"):
    st.markdown("""
    1.  **データの準備**: `コスト`, `コンバージョン数`などに加え、`YYYY-MM-DD`形式の**`日`**カラムを含むCSVファイルを2種類（全期間・直近）用意します。
    2.  **データアップロード**: サイドバーの「データアップロード」セクションから、用意した2つのファイルをアップロードします。
    3.  **条件の設定**: サイドバーで、予測したい「予算」と、将来の「トレンド指数」をスライダーで設定します。
    4.  **結果の確認**: 入力後、即座に予測結果がメイン画面に表示されます。**あなたの設定した予算での予測**と、**費用対効果が最も良い「最適予算」**が提案されます。
    """)
st.markdown("---")

# ============== メイン処理 ==============
if uploaded_file_all and uploaded_file_30d:
    with st.spinner('データを読み込み、モデルを準備中です...'):
        df_cleaned, log_messages, error_message = load_and_preprocess_data(uploaded_file_all, uploaded_file_30d)

    if error_message:
        st.error(error_message)
    else:
        min_cost_data = df_cleaned['コスト'].min()
        max_cost_data = df_cleaned['コスト'].max()
        with recommendation_placeholder.container():
            st.markdown("##### **推奨予算範囲**")
            st.info(f"{min_cost_data:,.0f} 円 〜 {max_cost_data:,.0f} 円")
            st.caption("学習データに基づいた、予測精度が比較的安定している範囲です。")

        with st.spinner('予測モデルを学習中です...'):
            sub_models, main_model, coefficients = train_models(df_cleaned)

        if input_budget is not None and input_budget > 0:
            # ユーザー入力予算のシミュレーション
            sim_data_user = {'コスト': float(input_budget), 'google_trend': float(input_trend)}
            for feature, model in sub_models.items():
                predict_df = pd.DataFrame({'コスト': [input_budget], 'google_trend': [input_trend]})
                sim_data_user[feature] = model.predict(predict_df)[0]
            
            sim_df_user_row = pd.DataFrame([sim_data_user])
            main_features = ['コスト', 'google_trend'] + list(sub_models.keys())
            predicted_cv_user = main_model.predict(sim_df_user_row[main_features])[0]
            predicted_cv_user = max(0, predicted_cv_user)
            predicted_cpa_user = input_budget / predicted_cv_user if predicted_cv_user > 0.01 else float('inf')
            
            st.markdown(f"### 予算 **{input_budget:,.0f}円** / トレンド指数 **{input_trend}** のシミュレーション結果")
            
            if input_budget > max_cost_data:
                st.warning(f"⚠️ **警告:** 入力された予算は、学習データの最大コスト ({max_cost_data:,.0f}円) を超えています。予測精度が低い可能性があります。")

            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="🎯 予測コンバージョン数", value=f"{predicted_cv_user:.2f} 件")
            with col2:
                cpa_display = f"{predicted_cpa_user:,.0f} 円" if predicted_cpa_user != float('inf') else "算出不可"
                st.metric(label="💰 予測CPA", value=cpa_display)
            
            # ▼▼▼【処理順序を修正】▼▼▼
            # 先に全体シミュレーションを実行してsim_dfを確定させる
            with st.spinner('全体シミュレーションと最適点の計算中です...'):
                sim_df = run_simulation(df_cleaned, sub_models, main_model, input_budget, input_trend)
                
                # sim_df を使って最適点を計算
                optimal_budget, optimal_cv, optimal_cpa = None, None, None
                if not sim_df.empty and '予測CPA' in sim_df.columns:
                    optimal_point = sim_df.loc[sim_df['予測CPA'].idxmin()]
                    optimal_budget = optimal_point['コスト']
                    optimal_cv = optimal_point['予測コンバージョン数']
                    optimal_cpa = optimal_point['予測CPA']
            
            # 最適予算の提案を表示
            st.markdown("---")
            st.markdown("### 💡 最適予算の提案")
            if optimal_budget is not None:
                st.success(
                    f"""
                    シミュレーション上、最もCPAが低くなる（費用対効果が高い）のは **予算 {optimal_budget:,.0f} 円** です。
                    - **その時の予測コンバージョン数**: {optimal_cv:.2f} 件
                    - **その時の予測CPA**: **{optimal_cpa:,.0f} 円**
                    """
                )
            else:
                st.info("シミュレーション結果から最適予算を算出できませんでした。")

            # 最後に、確定した情報を使ってグラフを描画
            fig = create_plot(sim_df, input_budget, predicted_cv_user, predicted_cpa_user, optimal_budget, optimal_cv, optimal_cpa)
            
            tab1, tab2, tab3 = st.tabs(["📊 **予測結果の全体像グラフ**", "📄 **学習データ詳細**", "🧠 **モデルの分析情報**"])
            # ▲▲▲【ここまで修正】▲▲▲

            with tab1:
                st.pyplot(fig)
            with tab2:
                st.markdown("モデルの学習に使用されたデータ（外れ値除去・トレンドデータ結合後）のサンプルです。")
                st.dataframe(df_cleaned.head(100))
            with tab3:
                st.markdown("#### 重回帰モデルの回帰係数")
                st.markdown("各指標が1単位増加したときに、コンバージョン数がどれだけ増減するかを示します。`google_trend`が追加されています。")
                st.table(coefficients.style.format("{:.4f}").background_gradient(cmap='viridis'))
                st.markdown("---")
                st.markdown("#### データ処理ログ")
                st.code("\n".join(log_messages))
        else:
            st.info("サイドバーで予算を入力すると、シミュレーション結果が表示されます。")

else:
    st.info('サイドバーの「データアップロード」から、分析用のCSVファイルを2つアップロードしてください。')
