import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.figure_factory as ff
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
    /* Adjust the width of the slider */
    div[data-baseweb="slider"] {
        width: 80% !important;  /* Adjust percentage to desired width */
        margin-left: auto;
        margin-right: auto;
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
    st.title("Interactive Email Filtering")

    st.markdown("""
    Welcome to this **interactive demo**! Below, you'll find a set of emails — some
    genuine, some spam—each with a **'Genuineness Score'** from the spam detection model.
    
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
    # ----- Confusion Matrix & ROC Curve (Plotly) -----
    if show_details:
        st.markdown("---")
        st.subheader("Advanced Model Performance Visualizations")

        # Confusion Matrix Explanation
        st.markdown("""
        ### 🔍 Confusion Matrix
        The confusion matrix helps visualize how well the model classifies emails as **Spam** or **Genuine**.
    
        **How to read it:**  
        - **Rows** represent the *actual* email labels (True class).  
        - **Columns** represent the *model's predictions*.  
        - A perfect model would classify all spam as "Spam" and all genuine emails as "Genuine".  

        🔵 **Correct classifications** (diagonal elements) → Good model performance  
        🔴 **Misclassifications** (off-diagonal elements) → Errors in classification  
        """)

        # Compute Confusion Matrix
        y_true = df["is_spam"].astype(int)  # 1=Spam, 0=Genuine
        y_pred_spam = (~df["predicted_genuine"]).astype(int)
        cm = confusion_matrix(y_true, y_pred_spam)

        # Plot Confusion Matrix
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
            width=500,
            height=500
        )
        fig_cm.update_yaxes(scaleanchor="x", scaleratio=1)

        st.plotly_chart(fig_cm, use_container_width=False)

        # ROC Curve Explanation
        st.markdown("""
        ### 📈 ROC Curve
        The **Receiver Operating Characteristic (ROC) curve** shows how well the model differentiates between spam and genuine emails.

        **How to interpret it:**  
        - **X-axis**: False Positive Rate (FPR) → Mistakenly classifying a genuine email as spam  
        - **Y-axis**: True Positive Rate (TPR) → Correctly identifying spam emails  
        - A model with a high curve is better at distinguishing spam from genuine emails.  

        🔴 **Your selected threshold** is marked in red to show its impact.  
        """)

        # Compute ROC Curve
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

        # Plot ROC Curve
        fig_roc = go.Figure()
        fig_roc.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                line=dict(color="steelblue", width=2),
                name=f"ROC curve (AUC={roc_auc:.2f})"
            )
        )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode="lines",
                line=dict(color="gray", width=1, dash="dash"),
                name="Random Guess"
            )
        )
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
            margin=dict(l=20, r=0, t=0, b=20),
            showlegend=False,
            width=500,
            height=500
        )
        fig_roc.update_xaxes(showgrid=False)
        fig_roc.update_yaxes(showgrid=False, scaleanchor="x", scaleratio=1)

        st.plotly_chart(fig_roc, use_container_width=False)

        # Score Distribution Explanation
        st.markdown("""
        ### 📊 Distribution of Model Scores
        This plot shows how the model scores different emails based on their likelihood of being genuine.

        **What to look for:**  
        - The **blue curve** represents genuine emails.  
        - The **orange curve** represents spam emails.  
        - The **red vertical line** is your current classification threshold.  

        A well-separated distribution means the model is effectively distinguishing spam from genuine emails.
        """)

        # Compute Score Distributions
        genuine_scores = df[df["is_spam"] == False]["genuine_score"]
        spam_scores = df[df["is_spam"] == True]["genuine_score"]

        # Plot Score Distributions
        fig_dist = ff.create_distplot(
            [genuine_scores, spam_scores],
            ["Genuine Emails", "Spam Emails"],
            show_hist=True,
            show_rug=False,
            bin_size=0.05,
            curve_type='normal'
        )
        fig_dist.add_vline(
            x=threshold,
            line=dict(color="red", width=2),
            annotation_text="Threshold",
            annotation_position="top left"
        )

        fig_dist.update_layout(
            title="Distribution of Model Scores",
            xaxis_title="Genuineness Score",
            yaxis_title="Density",
            legend_title="Email Type"
        )
        fig_dist.update_xaxes(showgrid=False)
        fig_dist.update_yaxes(showgrid=False)

        st.plotly_chart(fig_dist, use_container_width=True)



if __name__ == "__main__":
    main()
