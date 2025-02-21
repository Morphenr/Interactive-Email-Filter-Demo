import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Extended synthetic data with "genuine_score"
# Asset-manager-themed subject lines, mixing obviously spammy with genuine marketing content.
EMAILS = [
    # Obviously fake/spammy (with emojis & hype)
    {"subject": "Explosive returns guaranteed 🚀: Double your money FAST!", "genuine_score": 0.1,  "is_spam": True},
    {"subject": "S3cr3t TIP: Multiply your revenue by 10",                 "genuine_score": 0.2,  "is_spam": True},
    {"subject": "Last chance: free crypto tokens for sign-up!",           "genuine_score": 0.05, "is_spam": True},
    {"subject": "Urgent: 50% bonus on new fund invests – Act now!",       "genuine_score": 0.3,  "is_spam": True},
    {"subject": "Confidential: new hedge-fund strategy enclosed!",        "genuine_score": 0.4,  "is_spam": True},
    {"subject": "Re: invoice payment pending – avoid penalty fees!",       "genuine_score": 0.5,  "is_spam": True},
    {"subject": "Exclusive tip: top performing penny-stock insider info", "genuine_score": 0.15, "is_spam": True},
    {"subject": "Manager's private deal – your returns guaranteed!",      "genuine_score": 0.35, "is_spam": True},

    # More subtle spam with moderately high genuineness
    {"subject": "Invest Securely! (Confirm your account here)",           "genuine_score": 0.75, "is_spam": True},
    {"subject": "Ioana said this in a meeting and revenues exploded!!!",  "genuine_score": 0.65, "is_spam": True},
    {"subject": "Your account has been locked – please verify details",   "genuine_score": 0.61, "is_spam": True},

    # Genuine marketing / internal emails
    {"subject": "Q2 Market Outlook: Strategies for Growth",               "genuine_score": 0.68, "is_spam": False},
    {"subject": "Invitation: Client Relationship Workshop Next Week",     "genuine_score": 0.85, "is_spam": False},
    {"subject": "Your Team’s Monthly Campaign Performance Review",        "genuine_score": 0.9,  "is_spam": False},
    {"subject": "ESG Strategy Highlights: Whitepaper Attached",           "genuine_score": 0.8,  "is_spam": False},
    {"subject": "Conference Invite: Global Markets Outlook",              "genuine_score": 0.88, "is_spam": False},
    {"subject": "New Brand Guidelines for Asset Manager Campaign",        "genuine_score": 0.9,  "is_spam": False},
    {"subject": "Coffee Chat with Analytics: Next Steps on ROI Tracking", "genuine_score": 0.95, "is_spam": False},
    {"subject": "Performance Analysis: Our Latest Marketing Efforts",     "genuine_score": 0.7,  "is_spam": False},
    {"subject": "What's New in European Regulatory Environment?",         "genuine_score": 0.6,  "is_spam": False},
    {"subject": "Scheduling Conflict: Can we move next meeting?",         "genuine_score": 0.55, "is_spam": False},
    {"subject": "Lunch & Learn: Asset Allocation Trends",                 "genuine_score": 0.78, "is_spam": False},
    {"subject": "Internal Memo: Updated Reporting Templates",             "genuine_score": 0.73, "is_spam": False},
    {"subject": "Reminder: Compliance Training Due by End of Month",      "genuine_score": 0.85, "is_spam": False},
]

def main():
    # Inject custom CSS for fade-in animations & smaller card style
    st.markdown("""
    <style>
    .fade-in {
      animation: fadeIn 0.4s ease-in;
    }
    @keyframes fadeIn {
      0%   {opacity: 0; transform: translateY(4px);}
      100% {opacity: 1; transform: translateY(0);}
    }
    .email-card {
      border: 1px solid #ccc; 
      border-radius: 3px; 
      padding: 4px 8px; 
      margin-bottom: 5px; 
      font-size: 0.9rem; 
      line-height: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # STEP 1: Check if we've already stored the shuffled DataFrame in session_state
    if "shuffled_df" not in st.session_state:
        # Shuffle once and store in session_state
        df_shuffled = pd.DataFrame(EMAILS).sample(frac=1, random_state=None).reset_index(drop=True)
        st.session_state["shuffled_df"] = df_shuffled

    # Retrieve the already-shuffled dataset from session_state
    df = st.session_state["shuffled_df"]

    # ----- SIDEBAR -----
    st.sidebar.title("Settings")
    show_details = st.sidebar.checkbox("Show advanced details")

    # ----- MAIN PAGE -----
    st.title("Fun Interactive Email Filtering Demo")

    st.markdown("""
    Welcome to this **interactive demo**! Below, you'll find a set of emails—some
    genuine, some spam—each with a **'Genuineness Score'** from our detection model.
    
    **1.** Drag the **slider** to set a threshold for delivery.  
    **2.** Emails with scores **below** this threshold are **Blocked** (treated as spam).  
    **3.** Emails with scores **above** or **equal** to this threshold are **Delivered**.

    Experiment to see how well you can **filter out spam** while **keeping genuine** messages!
    """)

    threshold = st.slider(
        label="Minimum Genuineness Score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help="Emails below this threshold are considered spam (Blocked)."
    )

    # Predict 'delivered' if genuine_score >= threshold
    df["predicted_genuine"] = df["genuine_score"] >= threshold

    # Split into delivered vs blocked
    delivered_df = df[df["predicted_genuine"]]
    blocked_df = df[~df["predicted_genuine"]]

    # Basic counts
    total_spam = df["is_spam"].sum()
    total_genuine = len(df) - total_spam

    # Spam that got delivered
    spam_delivered = delivered_df["is_spam"].sum()
    # Spam that got blocked
    spam_blocked = blocked_df["is_spam"].sum()
    # Genuine that got delivered
    genuine_delivered = len(delivered_df) - spam_delivered
    # Genuine that got blocked
    genuine_blocked = len(blocked_df) - spam_blocked

    # --- Display Key KPIs (Spam/Genuine) ---
    st.markdown("### Key Performance Indicators")
    kpi_cols = st.columns(2)
    kpi_cols[0].metric("Spam Blocked", f"{spam_blocked}/{total_spam}" if total_spam else "0")
    kpi_cols[1].metric("Genuine Blocked", f"{genuine_blocked}/{total_genuine}" if total_genuine else "0")

    # ----- TWO-COLUMN LAYOUT (Delivered vs Blocked) -----
    col_delivered, col_blocked = st.columns(2)

    # =========== Delivered Emails ===========
    col_delivered.markdown("## Delivered Emails")
    if delivered_df.empty:
        col_delivered.info("No emails are delivered at this threshold.")
    else:
        for _, row in delivered_df.iterrows():
            subject = row["subject"]
            is_spam = row["is_spam"]
            genuine_score = row["genuine_score"]

            # Default icon = mail
            icon = "📧"
            # If advanced details are ON and it's spam => show skull
            if is_spam and show_details:
                icon = "💀"

            detail_text = ""
            if show_details:
                score_pct = f"{genuine_score * 100:.0f}%"
                detail_text = f"<br/><em>Model genuineness score: {score_pct}</em>"

            col_delivered.markdown(
                f"""
                <div class="fade-in email-card">
                    <strong>{icon} Subject:</strong> {subject}
                    {detail_text}
                </div>
                """,
                unsafe_allow_html=True
            )

    # =========== Blocked Emails ===========
    col_blocked.markdown("## Blocked Emails")
    if blocked_df.empty:
        col_blocked.info("No emails are blocked at this threshold.")
    else:
        for _, row in blocked_df.iterrows():
            subject = row["subject"]
            is_spam = row["is_spam"]
            genuine_score = row["genuine_score"]

            icon = "📧"
            if is_spam and show_details:
                icon = "💀"

            detail_text = ""
            if show_details:
                score_pct = f"{genuine_score * 100:.0f}%"
                detail_text = f"<br/><em>Model genuineness score: {score_pct}</em>"

            col_blocked.markdown(
                f"""
                <div class="fade-in email-card">
                    <strong>{icon} Subject:</strong> {subject}
                    {detail_text}
                </div>
                """,
                unsafe_allow_html=True
            )

    # ----- Confusion Matrix & ROC Curve (Plotly) -----
    if show_details:
        st.markdown("---")
        st.subheader("Advanced Visuals")

        st.markdown("""
        Here you can see how the **model** is performing behind the scenes:
        
        **Confusion Matrix**  
        - Rows: The *actual* label of the email (Genuine or Spam).  
        - Columns: The *predicted* label (Genuine or Spam).  
        - A perfect filter would put all spam in the “Spam” column and all genuine emails in the “Genuine” column.  

        **ROC Curve**  
        - Shows how well the model can separate spam from genuine over *all possible* thresholds.  
        - The higher the curve, the better!  
        - We highlight your **currently selected threshold** (the red "O") on this curve.
        """)

        # Prepare data for confusion matrix
        y_true = df["is_spam"].astype(int)            # 1=Spam, 0=Genuine
        y_pred_spam = (~df["predicted_genuine"]).astype(int)
        cm = confusion_matrix(y_true, y_pred_spam)

        # Plotly confusion matrix as a square
        labels_x = ["Genuine", "Spam"]
        labels_y = ["Genuine", "Spam"]

        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=[f"Predicted: {lbl}" for lbl in labels_x],
                y=[f"Actually: {lbl}" for lbl in labels_y],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                hovertemplate="Row: %{y}<br>Column: %{x}<br>Count: %{z}<extra></extra>",
                showscale=False
            )
        )
        fig_cm.update_layout(
            title="Confusion Matrix",
            margin=dict(l=70, r=30, t=60, b=50),
            xaxis=dict(title="Model Prediction"),
            yaxis=dict(title="Actual Label", autorange="reversed"),

            # Force a square figure
            width=500,
            height=500
        )
        # Make the axes keep a 1:1 aspect ratio
        fig_cm.update_yaxes(scaleanchor="x", scaleratio=1)

        st.plotly_chart(fig_cm, use_container_width=False)

        # ROC curve with current threshold
        spam_probability = 1 - df["genuine_score"]  # Probability that an email is spam
        fpr, tpr, thresholds = roc_curve(y_true, spam_probability)
        roc_auc = auc(fpr, tpr)

        current_spam_threshold = 1 - threshold
        y_pred_spam_threshold = (spam_probability >= current_spam_threshold).astype(int)
        cm_threshold = confusion_matrix(y_true, y_pred_spam_threshold)
        tp = cm_threshold[1, 1]
        fn = cm_threshold[1, 0]
        fp = cm_threshold[0, 1]
        tn = cm_threshold[0, 0]
        tpr_current = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_current = fp / (fp + tn) if (fp + tn) > 0 else 0

        fig_roc = go.Figure()
        # Main ROC line
        fig_roc.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                line=dict(color="steelblue", width=2),
                name=f"ROC curve (AUC={roc_auc:.2f})"
            )
        )
        # Diagonal "random guess"
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(color="gray", width=1, dash="dash"),
                name="Random Guess"
            )
        )
        # Marker for CURRENT threshold
        fig_roc.add_trace(
            go.Scatter(
                x=[fpr_current],
                y=[tpr_current],
                mode="markers",
                marker=dict(color="red", size=10, symbol="circle-open"),
                name="Current Threshold"
            )
        )
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
            margin=dict(l=70, r=30, t=60, b=50),

            # Force a square figure
            width=500,
            height=500
        )
        fig_roc.update_xaxes(showgrid=False)
        fig_roc.update_yaxes(showgrid=False, scaleanchor="x", scaleratio=1)

        st.plotly_chart(fig_roc, use_container_width=False)


if __name__ == "__main__":
    main()
