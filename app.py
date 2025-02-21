import streamlit as st
import pandas as pd

# Extended Synthetic Data with "genuine_score" 
# (including ambiguous records: genuine with low scores, spam with high scores, etc.)
EMAILS = [
    # High-spam-likelihood / Low genuineness
    {"subject": "Claim your prize now! Limited offer!",   "genuine_score": 0.05, "is_spam": True},
    {"subject": "URGENT: Update your account details",    "genuine_score": 0.1,  "is_spam": True},
    {"subject": "Question about your website domain",     "genuine_score": 0.2,  "is_spam": True},
    {"subject": "Congrats! You've been selected for a survey", "genuine_score": 0.15, "is_spam": True},
    {"subject": "Get cheap meds now",                     "genuine_score": 0.25, "is_spam": True},
    {"subject": "One time investment opportunity",        "genuine_score": 0.3,  "is_spam": True},
    {"subject": "Check out these great sale items",       "genuine_score": 0.3,  "is_spam": True},
    {"subject": "Re: invoice payment pending",            "genuine_score": 0.5,  "is_spam": True},
    {"subject": "Don’t miss this exclusive discount!",    "genuine_score": 0.55, "is_spam": True},
    {"subject": "Restricted notice: password needed",     "genuine_score": 0.45, "is_spam": True},

    # Low genuineness but actually genuine
    {"subject": "Staff poll: Which coffee beans to order?", "genuine_score": 0.1,  "is_spam": False},
    {"subject": "Quick question: Remote day tomorrow?",     "genuine_score": 0.2,  "is_spam": False},
    {"subject": "Daily check-in for new hires",             "genuine_score": 0.3,  "is_spam": False},

    # Mid-range
    {"subject": "Scheduling conflict: can we move the meeting?",  "genuine_score": 0.4,  "is_spam": False},
    {"subject": "Your package is on the way",                     "genuine_score": 0.6,  "is_spam": False},
    {"subject": "FW: New deal from vendor",                       "genuine_score": 0.65, "is_spam": False},

    # High genuineness but spam (a sneaky spam)
    {"subject": "Your account is secure! (But confirm here)",      "genuine_score": 0.8,  "is_spam": True},

    # Genuine with reasonably high genuineness
    {"subject": "Meeting agenda for Monday",           "genuine_score": 0.9,  "is_spam": False},
    {"subject": "Reminder: Project deadline",          "genuine_score": 0.85, "is_spam": False},
    {"subject": "Weekly newsletter: Productivity tips","genuine_score": 0.75, "is_spam": False},
    {"subject": "Vacation itinerary",                  "genuine_score": 0.8,  "is_spam": False},
    {"subject": "Invitation: Team building event",     "genuine_score": 0.88, "is_spam": False},
    {"subject": "Coffee catch-up?",                    "genuine_score": 0.95, "is_spam": False},
    {"subject": "Lunch plan with the team",            "genuine_score": 0.9,  "is_spam": False},
]

def main():
    # Inject custom CSS for fade-in animations
    st.markdown("""
    <style>
    .fade-in {
      animation: fadeIn 0.4s ease-in;
    }
    @keyframes fadeIn {
      0%   {opacity: 0; transform: translateY(5px);}
      100% {opacity: 1; transform: translateY(0);}
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Interactive Email Filter Demo (Genuineness Threshold)")

    # === Sidebar Configuration ===
    st.sidebar.title("Filter Settings")
    
    # A subdued toggle for showing advanced details in the sidebar
    show_details = st.sidebar.checkbox("Show advanced details", value=False)
    
    # Slider in the sidebar
    threshold = st.sidebar.slider(
        label="Minimum Genuineness Score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05
    )

    # Turn dataset into DataFrame
    df = pd.DataFrame(EMAILS)

    # Predict 'delivered' if genuine_score >= threshold
    df["predicted_genuine"] = df["genuine_score"] >= threshold

    # Split into delivered vs blocked
    delivered_df = df[df["predicted_genuine"]]
    blocked_df = df[~df["predicted_genuine"]]

    total_emails = len(df)
    emails_delivered = len(delivered_df)
    emails_blocked = len(blocked_df)

    # Count how many are actually spam vs genuine
    total_spam = sum(df["is_spam"])
    total_genuine = total_emails - total_spam

    # Spam that got delivered
    spam_delivered = delivered_df["is_spam"].sum()
    # Spam that got blocked
    spam_blocked = blocked_df["is_spam"].sum()
    # Genuine that got delivered
    genuine_delivered = emails_delivered - spam_delivered
    # Genuine that got blocked
    genuine_blocked = emails_blocked - spam_blocked

    # === KPI Display in Sidebar ===
    st.sidebar.markdown("### Key Performance Indicators")
    st.sidebar.metric(label="Emails Delivered", value=f"{emails_delivered} / {total_emails}")
    st.sidebar.metric(label="Total Spam Emails", value=str(total_spam))
    st.sidebar.metric(label="Total Genuine Emails", value=str(total_genuine))
    st.sidebar.metric(
        label="Spam Delivered",
        value=f"{spam_delivered} / {total_spam}" if total_spam else "0"
    )
    st.sidebar.metric(
        label="Spam Blocked",
        value=f"{spam_blocked} / {total_spam}" if total_spam else "0"
    )
    st.sidebar.metric(
        label="Genuine Delivered",
        value=f"{genuine_delivered} / {total_genuine}" if total_genuine else "0"
    )
    st.sidebar.metric(
        label="Genuine Blocked",
        value=f"{genuine_blocked} / {total_genuine}" if total_genuine else "0"
    )

    # === Main Section: Delivered + Blocked Emails ===
    # Delivered
    st.markdown("## Delivered Emails")
    if emails_delivered == 0:
        st.info("No emails are delivered at this threshold.")
    else:
        for _, row in delivered_df.iterrows():
            subject = row["subject"]
            is_spam = row["is_spam"]
            genuine_score = row["genuine_score"]

            # Subtle highlight if it's actually spam
            bg_color = "#f8d7da" if is_spam else "#ffffff"

            # Show details if user opts in
            detail_text = ""
            if show_details:
                # Show genuineness score as e.g. "55%"
                score_pct = f"{genuine_score*100:.0f}%"
                truth_label = "Spam" if is_spam else "Genuine"
                detail_text = f"<br/><em>Genuineness score: {score_pct}, Ground truth: {truth_label}</em>"

            st.markdown(
                f"""
                <div class="fade-in" style="
                    border:1px solid #ccc; 
                    border-radius:3px; 
                    padding:8px; 
                    margin-bottom:5px; 
                    background-color:{bg_color};">
                    <strong>📧 Subject:</strong> {subject}
                    {detail_text}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Blocked
    st.markdown("## Blocked Emails")
    if emails_blocked == 0:
        st.info("No emails are blocked at this threshold.")
    else:
        for _, row in blocked_df.iterrows():
            subject = row["subject"]
            is_spam = row["is_spam"]
            genuine_score = row["genuine_score"]

            bg_color = "#f8d7da" if is_spam else "#ffffff"

            detail_text = ""
            if show_details:
                score_pct = f"{genuine_score*100:.0f}%"
                truth_label = "Spam" if is_spam else "Genuine"
                detail_text = f"<br/><em>Genuineness score: {score_pct}, Ground truth: {truth_label}</em>"

            st.markdown(
                f"""
                <div class="fade-in" style="
                    border:1px solid #ccc; 
                    border-radius:3px; 
                    padding:8px; 
                    margin-bottom:5px; 
                    background-color:{bg_color};">
                    <strong>📧 Subject:</strong> {subject}
                    {detail_text}
                </div>
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main()
