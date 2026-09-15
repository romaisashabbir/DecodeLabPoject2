import streamlit as st
from recommender import UserProfile, default_engine

st.set_page_config(page_title="AI Recommendation Logic", page_icon="🤖", layout="wide")

st.title("🤖 AI Learning Resource Recommender")
st.caption("DecodeLabs - Artificial Intelligence Project 3 | Explainable content-based recommendation")

engine = default_engine()

all_topics = sorted({topic for r in engine.resources for topic in r.topics})

with st.sidebar:
    st.header("Your preferences")
    interests = st.multiselect(
        "Interests",
        all_topics,
        default=["machine learning", "python"],
        help="Select one or more topics you are interested in."
    )
    difficulty = st.selectbox("Difficulty", ["Any", "beginner", "intermediate", "advanced"])
    format_ = st.selectbox("Format", ["Any", "course", "tutorial", "project", "article", "book"])
    style = st.selectbox("Learning style", ["Any", "hands-on", "balanced", "theory", "visual"])
    top_k = st.slider("Number of recommendations", 3, 10, 5)

profile = UserProfile(
    interests=tuple(interests),
    difficulty=None if difficulty == "Any" else difficulty,
    format=None if format_ == "Any" else format_,
    style=None if style == "Any" else style,
)

if not interests and difficulty == "Any" and format_ == "Any" and style == "Any":
    st.info("Choose at least one preference from the sidebar to generate recommendations.")
else:
    results = engine.recommend(profile, top_k=top_k)

    st.subheader("Recommended for you")
    for rank, item in enumerate(results, 1):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                st.markdown(f"### {rank}. {item['title']}")
                st.write(
                    f"**Difficulty:** {item['difficulty'].title()}  |  "
                    f"**Format:** {item['format'].title()}  |  "
                    f"**Style:** {item['style'].title()}"
                )
                st.write("**Topics:** " + ", ".join(item["topics"]))
                st.write("**Why recommended:** " + " • ".join(item["reasons"]))
            with right:
                st.metric("Match", f"{item['match_percent']}%")
                st.caption(f"Rating: {item['rating']}/5")

    with st.expander("How the recommendation logic works"):
        st.markdown(
            """
            The system is **content-based** and explainable. User preferences and resource
            attributes are represented as weighted feature tokens. Topic matches receive the
            highest weight, while difficulty, format and learning style provide additional
            alignment. **Cosine similarity** measures how closely each resource matches the
            user profile. A tiny rating bonus is used only as a tie-breaker, so preferences
            remain the dominant signal.
            """
        )
