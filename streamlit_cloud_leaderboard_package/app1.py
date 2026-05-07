import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

#st.write(st.secrets["connections"]["gsheets"]["client_email"])
#st.write(st.secrets["connections"]["gsheets"]["spreadsheet"])
#st.write(st.secrets["connections"]["gsheets"]["type"])

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmse_log(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("rmse_log requires non-negative targets and predictions.")
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


METRICS = {
    "rmse": rmse,
    "rmse_log": rmse_log,
    "mae": mae,
}


def get_config():
    if "competition" not in st.secrets:
        raise RuntimeError("Missing [competition] section in secrets.")
    return st.secrets["competition"]


def get_hidden_ground_truth(cfg):
    csv_text = st.secrets["ground_truth"]["csv"]
    df = pd.read_csv(io.StringIO(csv_text))
    id_col = cfg["id_column"]
    target_col = cfg["target_column"]
    required = {id_col, target_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ground truth missing required columns: {sorted(missing)}")
    return df


def get_conn():
    return st.connection("gsheets", type=GSheetsConnection)


def load_leaderboard():
    conn = get_conn()
    try:
        df = conn.read(worksheet="leaderboard", ttl=0)
        if df is None:
            return pd.DataFrame(columns=["rank", "team", "score", "metric", "submitted_at", "file", "note"])
        return pd.DataFrame(df)
    except Exception:
        return pd.DataFrame(columns=["rank", "team", "score", "metric", "submitted_at", "file", "note"])


def save_leaderboard(df):
    conn = get_conn()
    conn.update(worksheet="leaderboard", data=df)


def normalize_leaderboard(df, lower_is_better=True):
    if df.empty:
        return pd.DataFrame(columns=["rank", "team", "score", "metric", "submitted_at", "file", "note"])
    if "rank" in df.columns:
        df = df.drop(columns=["rank"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.sort_values("score", ascending=lower_is_better).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


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
    df_sub[pred_col] = pd.to_numeric(df_sub[pred_col], errors="raise")
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

    merged = df_gt.merge(df_sub[[id_col, pred_col]], on=id_col, how="left", validate="one_to_one")
    if merged[pred_col].isna().any():
        missing_count = int(merged[pred_col].isna().sum())
        missing_ids = merged.loc[merged[pred_col].isna(), id_col].head(10).tolist()
        raise ValueError(
            f"Submission does not cover all ids in the hidden test set. "
            f"Missing predictions: {missing_count}. Example missing ids: {missing_ids}"
        )

    score = METRICS[metric_name](merged[target_col].values, merged[pred_col].values)

    row = pd.DataFrame([{
        "team": team,
        "score": score,
        "metric": metric_name,
        "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file": file_name,
        "note": note,
    }])

    leaderboard = load_leaderboard()
    leaderboard = pd.concat([leaderboard.drop(columns=["rank"], errors="ignore"), row], ignore_index=True)

    if keep_best:
        leaderboard["score"] = pd.to_numeric(leaderboard["score"], errors="coerce")
        leaderboard = leaderboard.sort_values("score", ascending=lower_is_better)
        leaderboard = leaderboard.groupby("team", as_index=False).first()

    leaderboard = normalize_leaderboard(leaderboard, lower_is_better=lower_is_better)
    save_leaderboard(leaderboard)
    return score


st.set_page_config(page_title="Live Classroom Leaderboard", page_icon="🏁", layout="wide")
cfg = get_config()

st.title("🏁 INF01090 - Lab 05 - Live Regression Leaderboard")
st.write(f"**Competition:** {cfg['competition_name']}")
st.write(f"**Metric:** `{cfg['metric']}`")

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
                score = score_submission_df(df_sub, team.strip(), uploaded.name, note.strip())
                st.success(f"Submission scored successfully. Score = {score:.6f}")
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("Expected submission format")
    st.code(
        f"{cfg['id_column']},{cfg['prediction_column']}\n1461,208500.0\n1462,181500.0",
        language="csv",
    )

with right:
    st.subheader("Leaderboard")
    leaderboard = load_leaderboard()
    leaderboard = normalize_leaderboard(leaderboard, lower_is_better=bool(cfg.get("lower_is_better", True)))
    if leaderboard.empty:
        st.info("No submissions yet.")
    else:
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)

st.divider()
with st.expander("🔒 Instructor tools"):

    password_input = st.text_input("Instructor password", type="password")

    if password_input == st.secrets["instructor"]["password"]:

        st.success("Access granted")

        if st.button("🧹 Reset leaderboard"):
            empty = pd.DataFrame(columns=["rank", "team", "score", "metric", "submitted_at", "file", "note"])
            save_leaderboard(empty)
            st.success("Leaderboard reset.")

    elif password_input:
        st.error("Incorrect password")