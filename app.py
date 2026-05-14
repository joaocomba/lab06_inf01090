import io
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import gspread
from streamlit_gsheets import GSheetsConnection


# =========================================================
# Metrics
# =========================================================
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


# =========================================================
# Metrics
# =========================================================
def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def f1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average="macro"))


def logloss(y_true, y_pred):
    return float(log_loss(y_true, y_pred))


def roc_auc(y_true, y_pred):
    return float(roc_auc_score(y_true, y_pred))


METRICS = {
    "accuracy": accuracy,
    "f1_score": f1,
    "log_loss": logloss,
    "roc_auc": roc_auc,
}


def compute_all_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    results = {
        "accuracy": accuracy(y_true, y_pred),
        "f1_score": f1(y_true, y_pred),
        "log_loss": logloss(y_true, y_pred),
        "roc_auc": roc_auc(y_true, y_pred),
    }

    return results


# =========================================================
# Config / Secrets
# =========================================================
def get_config():
    if "competition" not in st.secrets:
        raise RuntimeError("Missing [competition] section in secrets.")
    return st.secrets["competition"]


def get_hidden_ground_truth(cfg):
    if "ground_truth" not in st.secrets or "csv" not in st.secrets["ground_truth"]:
        raise RuntimeError("Missing [ground_truth] csv in secrets.")

    csv_text = st.secrets["ground_truth"]["csv"]
    df = pd.read_csv(io.StringIO(csv_text))

    id_col = cfg["id_column"]
    target_col = cfg["target_column"]

    required = {id_col, target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ground truth missing required columns: {sorted(missing)}")

    if df[id_col].duplicated().any():
        raise ValueError("Ground truth contains duplicated ids.")

    return df


def get_instructor_password():
    if "instructor" in st.secrets and "password" in st.secrets["instructor"]:
        return st.secrets["instructor"]["password"]
    return None


# =========================================================
# Google Sheets
# =========================================================
def get_conn():
    return st.connection("gsheets", type=GSheetsConnection)


def get_gspread_client():
    """Get a gspread client using the same credentials as GSheetsConnection."""
    if "connections" not in st.secrets or "gsheets" not in st.secrets["connections"]:
        raise RuntimeError("Missing [connections.gsheets] in secrets.")
    
    # Extract only the keys needed for the service account
    # (gspread might fail if extra keys like 'spreadsheet' are present in some versions)
    creds = {
        k: v for k, v in st.secrets["connections"]["gsheets"].items()
        if k not in ["spreadsheet"]
    }
    return gspread.service_account_from_dict(creds)


def append_to_worksheet(worksheet_name, data_row_dict):
    """Safely append a single row to a worksheet using gspread."""
    try:
        gc = get_gspread_client()
        spreadsheet_url = st.secrets["connections"]["gsheets"]["spreadsheet"]
        sh = gc.open_by_url(spreadsheet_url)
        wks = sh.worksheet(worksheet_name)
        
        # Get headers to ensure correct column order
        headers = wks.row_values(1)
        if not headers:
            # If sheet is totally empty, we can't append reliably by header
            # Fallback to dict keys if we know them, but let's assume headers exist
            raise ValueError(f"Worksheet '{worksheet_name}' has no headers.")
            
        row_to_append = [data_row_dict.get(h, "") for h in headers]
        wks.append_row(row_to_append)
    except Exception as e:
        st.error(f"Failed to append to worksheet '{worksheet_name}': {e}")
        raise e


def load_worksheet(worksheet_name, columns):
    conn = get_conn()
    try:
        df = conn.read(worksheet=worksheet_name, ttl=0)
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return pd.DataFrame(columns=columns)
        
        df = pd.DataFrame(df)
        # Ensure all requested columns exist
        for col in columns:
            if col not in df.columns:
                df[col] = pd.Series(dtype="object")
        return df[columns]
    except Exception as e:
        # If it's a "worksheet not found" or similar, we might want to return empty
        # but for general errors (timeout, etc), we should stop to avoid overwriting later.
        st.error(f"Error loading worksheet '{worksheet_name}': {e}")
        raise e


def save_worksheet(worksheet_name, df, allow_empty=False):
    if not allow_empty and (df is None or df.empty):
        # Prevent accidental clearing of the sheet
        st.warning(f"Attempted to save an empty DataFrame to '{worksheet_name}'. Action aborted to prevent data loss. If you really want to clear the sheet, use the Instructor Tools.")
        return
    
    conn = get_conn()
    conn.update(worksheet=worksheet_name, data=df)


def load_leaderboard():
    return load_worksheet(
        "leaderboard",
        ["rank", "team", "score", "metric", "submitted_at", "file", "note"],
    )


def save_leaderboard(df, allow_empty=False):
    save_worksheet("leaderboard", df, allow_empty=allow_empty)


def load_history():
    return load_worksheet(
        "history",
        ["team", "accuracy", "f1_score", "log_loss", "roc_auc", "submitted_at", "file", "note"],
    )


def save_history(df, allow_empty=False):
    save_worksheet("history", df, allow_empty=allow_empty)


def normalize_leaderboard(df, lower_is_better=True):
    columns = ["rank", "team", "score", "metric", "submitted_at", "file", "note"]

    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    df = df.copy()

    if "rank" in df.columns:
        df = df.drop(columns=["rank"])

    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")

    df = df.sort_values("score", ascending=lower_is_better).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    for col in columns:
        if col not in df.columns:
            df[col] = pd.Series(dtype="object")

    return df[columns]


# =========================================================
# Validation / Scoring
# =========================================================
def validate_submission(df_sub, id_col, pred_col):
    required = {id_col, pred_col}
    missing = required - set(df_sub.columns)
    if missing:
        raise ValueError(f"Submission missing required columns: {sorted(missing)}")

    if df_sub[id_col].duplicated().any():
        raise ValueError("Submission contains duplicated ids.")

    if df_sub[pred_col].isna().any():
        n_missing = int(df_sub[pred_col].isna().sum())
        raise ValueError(f"Submission contains missing predictions: {n_missing}")

    try:
        df_sub[pred_col] = pd.to_numeric(df_sub[pred_col])
    except Exception:
        pass  # Keep as string/boolean if it cannot be parsed as numeric

    return df_sub


def score_submission_df(df_sub, team, file_name, note=""):
    cfg = get_config()
    id_col = cfg["id_column"]
    pred_col = cfg["prediction_column"]
    target_col = cfg["target_column"]
    metric_name = cfg["metric"]
    lower_is_better = bool(cfg.get("lower_is_better", True))
    keep_best = bool(cfg.get("keep_best_score_only", True))

    df_gt = get_hidden_ground_truth(cfg)
    df_sub = validate_submission(df_sub, id_col, pred_col)

    merged = df_gt.merge(
        df_sub[[id_col, pred_col]],
        on=id_col,
        how="left",
        validate="one_to_one",
    )

    if merged[pred_col].isna().any():
        missing_count = int(merged[pred_col].isna().sum())
        missing_ids = merged.loc[merged[pred_col].isna(), id_col].head(10).tolist()
        raise ValueError(
            f"Submission does not cover all ids in the hidden test set. "
            f"Missing predictions: {missing_count}. Example missing ids: {missing_ids}"
        )

    all_metrics = compute_all_metrics(
        merged[target_col].values,
        merged[pred_col].values,
    )

    if metric_name not in all_metrics:
        raise ValueError(f"Unsupported metric '{metric_name}'.")

    score = all_metrics[metric_name]

    # -------- leaderboard (best scores only, if configured)
    new_leaderboard_row = {
        "team": team,
        "score": score,
        "metric": metric_name,
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": file_name,
        "note": note,
    }

    # Load current leaderboard with error handling
    try:
        leaderboard = load_leaderboard().drop(columns=["rank"], errors="ignore")
        
        if leaderboard.empty:
            leaderboard = pd.DataFrame([new_leaderboard_row])
        else:
            leaderboard = pd.concat([leaderboard, pd.DataFrame([new_leaderboard_row])], ignore_index=True)

        if keep_best:
            leaderboard["score"] = pd.to_numeric(leaderboard["score"], errors="coerce")
            leaderboard = leaderboard.sort_values("score", ascending=lower_is_better)
            leaderboard = leaderboard.groupby("team", as_index=False).first()

        leaderboard = normalize_leaderboard(leaderboard, lower_is_better=lower_is_better)
        save_leaderboard(leaderboard)
    except Exception as e:
        st.warning(f"Could not update leaderboard view: {e}. However, your submission was saved to history.")

    # -------- full history (keep everything) - USE APPEND FOR SAFETY
    history_row_dict = {
        "team": team,
        "accuracy": all_metrics["accuracy"],
        "f1_score": all_metrics["f1_score"],
        "log_loss": all_metrics["log_loss"],
        "roc_auc": all_metrics["roc_auc"],
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": file_name,
        "note": note,
    }

    append_to_worksheet("history", history_row_dict)

    return score, all_metrics


# =========================================================
# UI Helpers
# =========================================================
def show_podium(leaderboard):
    if leaderboard.empty:
        st.info("No submissions yet.")
        return

    top3 = leaderboard.head(3).copy()

    medals = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)

    for i in range(3):
        with cols[i]:
            if i < len(top3):
                row = top3.iloc[i]
                st.markdown(f"### {medals[i]} {row['team']}")
                st.metric(label="Score", value=f"{float(row['score']):.6f}")
                st.caption(f"Metric: {row['metric']}")
                st.caption(f"File: {row['file']}")
            else:
                st.markdown(f"### {medals[i]} —")
                st.caption("No team yet")


def show_score_evolution(history):
    st.subheader("📈 Score evolution per team")

    if history.empty:
        st.info("No submission history yet.")
        return

    history = history.copy()
    history["submitted_at"] = pd.to_datetime(history["submitted_at"], errors="coerce")

    teams = sorted(history["team"].dropna().unique().tolist())
    default_teams = teams[:3] if len(teams) >= 3 else teams

    selected_teams = st.multiselect(
        "Select team(s)",
        options=teams,
        default=default_teams,
        key="evolution_teams",
    )

    metric_for_history = st.selectbox(
        "Metric for evolution chart",
        ["accuracy", "f1_score", "log_loss", "roc_auc"],
        index=0,
        key="evolution_metric",
    )

    df_plot = history[history["team"].isin(selected_teams)].copy()

    if df_plot.empty:
        st.info("No data for selected teams.")
        return

    chart = alt.Chart(df_plot).mark_line(point=True).encode(
        x=alt.X("submitted_at:T", title="Submission time"),
        y=alt.Y(f"{metric_for_history}:Q", title=metric_for_history.upper()),
        color=alt.Color("team:N", title="Team"),
        tooltip=[
            "team:N",
            "submitted_at:T",
            alt.Tooltip(f"{metric_for_history}:Q", format=".6f"),
            "file:N",
            "note:N",
        ],
    ).properties(
        height=350,
        title=f"Score evolution by team ({metric_for_history})",
    )

    st.altair_chart(chart, width="stretch")


def show_metric_breakdown(history):
    st.subheader("📊 Metric breakdown (Accuracy vs F1-Score vs Log-Loss)")

    if history.empty:
        st.info("No submission history yet.")
        return

    history = history.copy()
    history["submitted_at"] = pd.to_datetime(history["submitted_at"], errors="coerce")

    latest_only = st.checkbox("Show only latest submission per team", value=True)

    df_metrics = history.copy()
    if latest_only:
        df_metrics = df_metrics.sort_values("submitted_at").groupby("team", as_index=False).last()

    df_long = df_metrics.melt(
        id_vars=["team", "submitted_at", "file", "note"],
        value_vars=["accuracy", "f1_score", "log_loss", "roc_auc"],
        var_name="metric",
        value_name="value",
    )

    chart = alt.Chart(df_long).mark_bar().encode(
        x=alt.X("team:N", title="Team"),
        y=alt.Y("value:Q", title="Metric value"),
        color=alt.Color("metric:N", title="Metric"),
        column=alt.Column("metric:N", title=None),
        tooltip=[
            "team:N",
            "metric:N",
            alt.Tooltip("value:Q", format=".6f"),
            "file:N",
            "note:N",
        ],
    ).properties(
        width=180,
        height=320,
        title="Metric breakdown across teams",
    )

    st.altair_chart(chart, width="stretch")


# =========================================================
# App
# =========================================================
st.set_page_config(page_title="Live Classroom Leaderboard", page_icon="🏁", layout="wide")
alt.data_transformers.disable_max_rows()

cfg = get_config()
instructor_password = get_instructor_password()

st.title("🏁 INF01090 - Lab 06 - Live Classification Leaderboard")
st.write(f"**Competition:** {cfg['competition_name']}")
st.write(f"**Primary leaderboard metric:** `{cfg['metric']}`")

left, right = st.columns([1, 1.2])

with left:
    st.subheader("Submit predictions")

    team = st.text_input("Team name")
    note = st.text_input("Optional note")
    uploaded = st.file_uploader("Upload CSV submission", type=["csv"])
    submit = st.button("Score submission")

    if submit:
        if not team.strip():
            st.error("Please enter a team name.")
        elif uploaded is None:
            st.error("Please upload a CSV file.")
        else:
            try:
                df_sub = pd.read_csv(uploaded)
                score, all_metrics = score_submission_df(
                    df_sub,
                    team=team.strip(),
                    file_name=uploaded.name,
                    note=note.strip(),
                )
                st.success(f"Submission scored successfully. {cfg['metric']} = {score:.6f}")
                st.info(
                    f"Accuracy = {all_metrics['accuracy']:.6f} | "
                    f"F1-Score = {all_metrics['f1_score']:.6f} | "
                    f"Log-Loss = {all_metrics['log_loss']:.6f} | "
                    f"ROC-AUC = {all_metrics['roc_auc']:.6f}"
                )
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("Expected submission format")
    st.code(
        f"{cfg['id_column']},{cfg['prediction_column']}\n0013_01,True\n0018_01,False",
        language="csv",
    )

with right:
    st.subheader("🥇 Top 3 ranking")
    leaderboard = load_leaderboard()
    leaderboard = normalize_leaderboard(
        leaderboard,
        lower_is_better=bool(cfg.get("lower_is_better", True)),
    )
    show_podium(leaderboard)

st.divider()

st.subheader("🏆 Full leaderboard")
leaderboard = load_leaderboard()
leaderboard = normalize_leaderboard(
    leaderboard,
    lower_is_better=bool(cfg.get("lower_is_better", True)),
)

if leaderboard.empty:
    st.info("No submissions yet.")
else:
    st.dataframe(leaderboard, width="stretch", hide_index=True)

history = load_history()

st.divider()
show_score_evolution(history)

st.divider()
show_metric_breakdown(history)

st.divider()
with st.expander("🔒 Instructor tools"):
    if instructor_password is None:
        st.warning("No instructor password configured in secrets.")
    else:
        password_input = st.text_input("Instructor password", type="password", key="instr_pwd")

        if password_input == instructor_password:
            st.success("Access granted")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("🧹 Reset leaderboard"):
                    empty_leaderboard = pd.DataFrame(
                        columns=["rank", "team", "score", "metric", "submitted_at", "file", "note"]
                    )
                    save_leaderboard(empty_leaderboard, allow_empty=True)

                    # 🔥 clear UI state
                    for key in ["evolution_teams", "evolution_metric"]:
                        if key in st.session_state:
                            del st.session_state[key]

                    st.success("Leaderboard reset.")
                    st.rerun()

            with col2:
                if st.button("🗑 Reset history"):
                    empty_history = pd.DataFrame(
                        columns=["team", "accuracy", "f1_score", "log_loss", "roc_auc", "submitted_at", "file", "note"]
                    )
                    save_history(empty_history, allow_empty=True)

                    # 🔥 clear UI state
                    for key in ["evolution_teams", "evolution_metric"]:
                        if key in st.session_state:
                            del st.session_state[key]

                    st.success("History reset.")
                    st.rerun()

        elif password_input:
            st.error("Incorrect password")