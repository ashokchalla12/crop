"""OptiCrop — Dataset Explorer Page."""

import streamlit as st

from utils.helper import apply_custom_css, init_session_state, load_dataset, render_page_header
from utils.preprocessing import get_dataset_statistics


def render() -> None:
    init_session_state()
    apply_custom_css(st.session_state.get("dark_mode", False))
    render_page_header(
        "Dataset Explorer",
        "Browse, search, filter, and download the crop recommendation dataset.",
        "🗂️",
    )

    df = load_dataset()
    stats = get_dataset_statistics(df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", stats["shape"][0])
    m2.metric("Features", stats["shape"][1] - 1)
    m3.metric("Crop Classes", df["label"].nunique())
    m4.metric("Missing Values", sum(stats["missing"].values()))

    tab_data, tab_stats = st.tabs(["📋 Data Table", "📊 Statistics"])

    with tab_data:
        st.markdown("#### 🔍 Search & Filter")

        f1, f2, f3 = st.columns(3)
        with f1:
            search = st.text_input("🔎 Search (any column)", placeholder="e.g. rice, 90, 6.5")
        with f2:
            crop_filter = st.multiselect(
                "Filter by Crop",
                options=sorted(df["label"].unique()),
                format_func=lambda x: x.title(),
            )
        with f3:
            sort_by = st.selectbox("Sort by", options=list(df.columns))
            sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)

        filtered = df.copy()
        if crop_filter:
            filtered = filtered[filtered["label"].isin(crop_filter)]
        if search:
            mask = filtered.astype(str).apply(
                lambda row: row.str.contains(search, case=False, na=False).any(), axis=1
            )
            filtered = filtered[mask]

        ascending = sort_order == "Ascending"
        filtered = filtered.sort_values(sort_by, ascending=ascending)

        st.markdown(f"**Showing {len(filtered)} of {len(df)} records**")
        st.dataframe(filtered, use_container_width=True, height=400)

        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Filtered CSV",
            data=csv_data,
            file_name="OptiCrop_dataset_export.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tab_stats:
        st.markdown("#### 📊 Descriptive Statistics")
        st.dataframe(stats["describe"], use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Data Types**")
            st.json(stats["dtypes"])
        with c2:
            st.markdown("**Missing Values per Column**")
            st.json(stats["missing"])

        st.markdown("**Crop Count Distribution**")
        import pandas as pd
        crop_counts = pd.DataFrame(
            list(stats["crop_counts"].items()), columns=["Crop", "Count"]
        ).sort_values("Count", ascending=False)
        st.bar_chart(crop_counts.set_index("Crop"))

        st.download_button(
            "📥 Download Full Dataset CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="Crop_recommendation.csv",
            mime="text/csv",
        )
